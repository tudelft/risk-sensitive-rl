import os
import sys
import yaml
import time
import pickle
import argparse
import logging
import gym

import torch
import numpy as np
from collections import deque

from agent import DQNAgent
from utils.util import to_gym_interface_pos, ExpoWeightedAveForcast
import matplotlib.pyplot as plt

import crazyflie_env
from crazyflie_env.envs.utils.action import ActionXY
from crazyflie_env.envs.utils.state import FullState

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default=None, help='Change the model loading directory here')
    parser.add_argument('--env', default='CrazyflieEnv-v0', help='Training environment')
    parser.add_argument('--num_directions', default=4, type=int, help='Discrete directions')
    parser.add_argument('--num_speeds', default=3, type=int, help='Discrete velocities')
    parser.add_argument('--max_velocity', default=1.0, type=float, help='Maximum velocity')
    parser.add_argument('--distortion', default='neutral', help='Which risk distortion measure to use')
    parser.add_argument('--sample_cvar', default=1, type=float, help="Enable cvar value sampling from the uniform distribution")
    parser.add_argument('--cvar', default=0.2, type=float, help="Give the quantile value of the CVaR tail")
    parser.add_argument('--seed', default=5, type=int, help="Random seed")
    parser.add_argument('--update_every', default=1, type=int, help='Update policy network every update_every steps')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--layer_size', default=512, type=int, help='Hidden layer size of neural network')
    parser.add_argument('--n_step', default=1, type=int, help='Number of future steps for Q value evaluation')
    parser.add_argument('--gamma', default=0.99, type=float, help='Gamma discount factor')
    parser.add_argument('--tau', default=1e-2, type=float, help='Tau for soft updating the network weights')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--buffer_size', default=100000, type=int, help='Buffer size of the replay memory')
    parser.add_argument('--init_x', default=0, type=float, help='Initial robot position x')
    parser.add_argument('--init_y', default=-3, type=float, help='Initial robot position y')
    parser.add_argument('--render_mode', default='trajectory', help='Render mode')
    parser.add_argument('--render_density', default=8, type=int, help='Render density')
    parser.add_argument('--obstacle_num', default=3, type=int, help='Number of obstacles set in the env, only functions when random_obstacle is True')
    parser.add_argument('--test_eps', default=0.0, type=float, help='Learning rate')
    parser.add_argument('--random_obstacle', default=1, type=int, help='Enable random obstacle generation or fixed obstacle position')
    parser.add_argument('--variance_samples_n', default=8, type=int, help='Truncated Variance calculation hyperparameter')
    parser.add_argument('--tcv_lr', default=2, type=float, help='Updating rate for TCV controller')
    args = parser.parse_args()

    np.random.seed(args.seed)

    env = gym.make("CrazyflieEnv-v0")
    env.enable_random_obstacle(args.random_obstacle)
    env.set_obstacle_num(args.obstacle_num)
    state, _ = env.reset()

    # if you want to set robot initial position by hand:
    env.robot.set_state(args.init_x, args.init_y, 0, 0, 0.0, 3.0, 0, 0, env.obstacle_segments) # generalize to a slightly modified goal position
    state = env.robot.observe()
    tcv_controller = ExpoWeightedAveForcast(arms=[0.1, 1], lr=args.tcv_lr, init=0.01, use_std=True, window=5)

    print('initial state:', state) #observable state: px, py, vx, vy, radius

    state_size = len(to_gym_interface_pos(state))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    eps=args.test_eps # exploration rate during test
    logger = {}
    logger['tcv'] = []
    logger['cvar_probs'] = []
    logger['adaptive_cvar'] = []
    logger['feedback'] = []

    agent = DQNAgent(state_size=state_size,
                        num_directions=args.num_directions,
                        num_speeds=args.num_speeds,
                        layer_size=args.layer_size,
                        n_step=args.n_step,
                        BATCH_SIZE=args.batch_size,
                        BUFFER_SIZE=args.buffer_size,
                        LR=args.lr,
                        TAU=args.tau,
                        GAMMA=args.gamma,
                        UPDATE_EVERY=args.update_every,
                        device=device,
                        seed=args.seed,
                        distortion=args.distortion,
                        con_val_at_risk=bool(args.sample_cvar),
                        variance_samples_n=args.variance_samples_n)
    
    # load trained model
    dir = './experimentsSim/{}/IQN.pth'.format(args.dir)
    agent.qnetwork_local.load_state_dict(torch.load(dir))
    agent.action_space = agent.build_action_space(args.max_velocity)

    done = False
    score = 0
    tcv_window = deque(maxlen=5)
    old_average_tcv = 0
    counter = 0
    cvar = tcv_controller.get_probs()[1]


    while not done:
        # pass tactical sampled cvar to agent.act()
        #cvar = np.random.uniform(0.1, 0.2)
        #action_id, action = agent.act(to_gym_interface_pos(state), eps, args.cvar)
        action_id, action = agent.act(to_gym_interface_pos(state), eps, cvar)
        next_state, reward, done, info = env.step(action)
        tcv = agent.get_tcv(to_gym_interface_pos(state), action_id)
        tcv_window.append(tcv)
        if counter % 1 == 0 and counter != 0:
            average_tcv = np.sum(tcv_window)
            #tcv_window.append(tcv)
            #feedback = np.sum(last_tcv_window) - np.sum(tcv_window)
            feedback =  old_average_tcv - average_tcv # feedback > 0 means safter
            tcv_controller.update_dists(feedback)
            cvar = tcv_controller.get_probs()[1]
            logger['feedback'].append(feedback)
            old_average_tcv = average_tcv

        probs = tcv_controller.get_probs()
        state = next_state
        score += reward
        logger['tcv'].append(tcv.item())
        logger['adaptive_cvar'].append(cvar)
        logger['cvar_probs'].append(probs) # [0.1, 1]

        logger['global_time'] = env.global_time
        logger['state'] = env.states
        #print("dist {}, reward {}, {}, vx {}, vy {}, tcv {} probs {}".format(round(state.goal_distance, 6), round(reward, 2), info, round(state.velocity[0], 2), round(state.velocity[1], 2), tcv.item(), probs))
        
        #last_tcv_window = tcv_window.copy()
        counter += 1

    print("Episodic return:", score, "Task time", env.global_time)

    #print([[round(obs.centroid[0], 2), round(obs.centroid[1], 2), round(obs.wx, 2), round(obs.wy, 2)] for obs in env.obstacles])
    env.multi_render(mode=args.render_mode, output_file="./overleaf/no_truncated", tcvs=logger['tcv'], cvars=logger['adaptive_cvar'])
    pickle.dump(logger, open("experimentsSim/{}/loggers/logger_no_truncated.pkl".format(args.dir), 'wb'))