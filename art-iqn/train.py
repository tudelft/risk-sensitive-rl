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
from utils.util import eval_runs, computeExperimentID, to_gym_interface_pos
import crazyflie_env

def run(n_episodes, frames, eps_fixed, eps_frames, min_eps):
    """Deep Q-Learning
    Params
    ======
    eps_fixed (int): whether epsilon greedy exploration
    min_eps (float)L minimum epsilon greedy exploration rate
    n_episodes (int): maximum number of training episodes
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores

    # logger
    logger = {}
    logger['scores'] = []
    logger['scores_window'] = []
    logger['success_rate'] = []
    logger['timeout_rate'] = []
    logger['collision_rate'] = []
    logger['losses'] = []
    logger['stage_point'] = []
    logger['new_success_rate'] = []

    frame = 0
    if eps_fixed:
        eps = 0
    else:
        eps = 1

    eps_start = 1
    i_episode = 1
    success = 0
    collision = 0
    timeout = 0
    score = 0                  

    success_rates = deque(maxlen=50)
    for _ in range(50):
        success_rates.append(0)
    new_success_rate = np.sum(success_rates) / len(success_rates)

    state, obs_num = env.reset()
    # sample CVaR as risk-averse level (0, 1) interval excluding both ends
    cvar = 1 - np.random.uniform(0.0, 1.0)

    for frame in range(1, frames+1):
        action_id, action = agent.act(to_gym_interface_pos(state), eps, cvar)
        next_state, reward, done, info = env.step(action)
        #print(done, info)
        loss = agent.update(to_gym_interface_pos(state), action_id, reward, to_gym_interface_pos(next_state), done) # save experience and update network
        logger['losses'].append(loss)
        state = next_state
        score += reward

        # linear annealing to the min epsilon value until eps_frames and from there slowly decease epsilon to 0 until the end of training
        if eps_fixed == False:
            if frame < eps_frames:
                eps = max(eps_start - (frame * (1/eps_frames)), min_eps)
            else:
                eps = max(min_eps - min_eps * ((frame-eps_frames)/(frames-eps_frames)), min_eps)

        # evaluation runs
        #if frame % 5000 == 0:
            #eval_runs(agent, eps, frame)
        
        if done:
            scores_window.append(score)
            logger['scores'].append(score)
            logger['scores_window'].append(np.mean(scores_window))
            logger['success_rate'].append(success / i_episode)
            logger['timeout_rate'].append(timeout / i_episode)
            logger['collision_rate'].append(collision / i_episode)
            print('\rEpo {:5d}\tFrame {:5d}\tAvScore {:.3f}\tS {:.2f}\tC {:.2f}\tT {:.2f}\tEps {:.3f}\tInfo {}\t CVaR {:.3f}\tStage {}\tObsacles {:2d}'.format(i_episode, frame, np.mean(scores_window), success / i_episode, collision / i_episode, timeout / i_episode, eps, info, cvar, env.training_stage, obs_num), end="")

            # if i_episode % 100 == 0:
            #     print('\rEpisode {}\tFrame {}\tAverage Score {:.2f}\tS Rate {:.2f}\tC Rate {:.2f}\tT Rate {:.2f}\teps {:.3f}\tinfo {}'.format(i_episode, frame, np.mean(scores_window), success / i_episode, collision / i_episode, timeout / i_episode, eps, info), end="")
            
            i_episode += 1
            if info == "Timeout":
                timeout += 1
            elif info == "Collision":
                collision += 1
            elif info == "Reached":
                success += 1
            if i_episode == n_episodes:
                break
            
            success_rates.append(1 if info == "Reached" else 0)
            new_success_rate = np.sum(success_rates) / len(success_rates)
            logger['new_success_rate'].append(new_success_rate)
            if success / i_episode >= 0.8:
                env.set_training_stage('second')
                if len(logger['stage_point']) < 1:
                    logger['stage_point'].append(i_episode)
            state, obs_num = env.reset()
            cvar = 1 - np.random.uniform(0.0, 1.0)
            score = 0
    
    pickle.dump(logger, open("{}/{}/logger.pkl".format(args.save_dir, currentExperimentID), 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='experimentsCrazy', help='Change the experiment saving directory here')
    parser.add_argument('--env', default='CrazyflieEnv-v0', help='Training environment')
    parser.add_argument('--random_init', default=1, help='Whether initialize robot at random position')
    parser.add_argument('--num_directions', default=4, type=int, help='Discrete directions')
    parser.add_argument('--num_speeds', default=3, type=int, help='Discrete velocities')
    parser.add_argument('--max_velocity', default=1.0, type=float, help='Maximum velocity')
    parser.add_argument('--distortion', default='neutral', help='Which risk distortion measure to use')
    parser.add_argument('--sample_cvar', default=1, type=float, help="Enable cvar value sampling from the uniform distribution")
    parser.add_argument('--seed', default=5, help=" Random seed")
    parser.add_argument('--update_every', default=1, type=int, help='Update policy network every update_every steps')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--layer_size', default=512, type=int, help='Hidden layer size of neural network') # increase model width
    parser.add_argument('--n_step', default=1, type=int, help='Number of future steps for Q value evaluation')
    parser.add_argument('--gamma', default=0.99, type=float, help='Gamma discount factor')
    parser.add_argument('--tau', default=1e-2, type=float, help='Tau for soft updating the network weights')
    parser.add_argument('--lr', default=2e-4, type=float, help='Learning rate')
    parser.add_argument('--buffer_size', default=60000, type=int, help='Buffer size of the replay memory')
    parser.add_argument('--frames', default=100000, type=int, help='Number of training frames')
    parser.add_argument('--n_episodes', default=2000, type=int, help='Number of episodes of training')
    parser.add_argument('--obstacle_num', default=6, type=int, help='Number of obstacles set in the env')
    parser.add_argument('--random_obstacle', default=1, type=int, help='Enable random obstacle generation or fixed obstacle position')
    parser.add_argument('--variance_samples_n', default=8, type=int, help='Truncated Variance calculation hyperparameter')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    
    currentExperimentID = computeExperimentID(args.save_dir)
    print("Current experiment ID: {}".format(currentExperimentID))
    os.mkdir("{}/{}/".format(args.save_dir, currentExperimentID))

    with open("{}/{}/arguments".format(args.save_dir, currentExperimentID), 'w') as f:
        yaml.dump(args.__dict__, f)

    with open("{}/{}/{}".format(args.save_dir, currentExperimentID, "".join(sys.argv)), 'w') as f:
        pass

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print("Using", device)

    np.random.seed(args.seed)
    env = gym.make(args.env)
    env.enable_random_obstacle(args.random_obstacle)
    env.set_obstacle_num(args.obstacle_num)
    
    #env.seed(args.seed)
    state, obs_num = env.reset() # reset so we can get the dim of states
    state_size = len(to_gym_interface_pos(state))
    print("State size {} Num obstacle {}".format(state_size, env.obstacle_num))

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
    
    max_velocity = args.max_velocity
    agent.action_space = agent.build_action_space(max_velocity)

    # set epsilon frames to 0 so no epsilon exploration
    eps_fixed = False

    # logger for multiple plots

    t_start = time.time()
    run(n_episodes=args.n_episodes, frames=args.frames, eps_fixed=eps_fixed, eps_frames=args.frames / 4, min_eps=0.2)
    t_end = time.time()
    
    print("Training time: {}min".format(round((t_end-t_start) / 60, 2)))
    torch.save(agent.qnetwork_local.state_dict(), "{}/{}/IQN.pth".format(args.save_dir, currentExperimentID))