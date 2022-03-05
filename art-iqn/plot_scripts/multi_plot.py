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

import matplotlib.pyplot as plt

import crazyflie_env
from crazyflie_env.envs.utils.action import ActionXY
from crazyflie_env.envs.utils.state import FullState
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='multi', help='Change the model loading directory here')
    args = parser.parse_args()
    exp_id = 191
    env = gym.make("CrazyflieEnv-v0")
    env.enable_random_obstacle(False)
    state, _ = env.reset()
    env.robot.set_state(0, -3, 0, 0, 0.0, 3.0, 0, 0, env.obstacle_segments)
    loggers = []
    for logger_label in ['1.0', '0.75', '0.5', '0.25', '0.1', 'adaptive']:
        logger = pickle.load(open("../experimentsSim/191/loggers/logger_{}.pkl".format(logger_label), "rb"))
        loggers.append(logger)

    if args.mode == 'multi':
        env.multi_plot(loggers=loggers, output_file="../overleaf/behaviors")
    elif args.mode == 'single':
        env.adaptive_plot(loggers=loggers, output_file="../overleaf/behavior_w_tcv", tcvs=loggers[-1]['tcv'], cvars=loggers[-1]['cvar_probs'])
