import os
import gym
import torch
import random
import numpy as np
from collections import deque, namedtuple
from crazyflie_env.envs.utils.state import FullState, ObservableState
from typing import List, Dict

def weight_init(layers):
    for layer in layers:
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')


def calculate_huber_loss(td_errors, k=1.0):
    """
    Calculate huber loss element-wisely depending on kappa k
    """
    loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))

    return loss
    

def eval_runs(agent, eps, frame):
    """
    Makes an evaluation run with the current epsilon
    """
    env = gym.make("CartPole-v0")
    reward_batch = []
    for i in range(5):
        state = env.reset()
        rewards = 0
        while True:
            action = agent.act(state, eps)
            state, reward, done, _ = env.step(action)
            #env.render()
            rewards += reward
            if done:
                break
        reward_batch.append(rewards)
    #writer.add_scalar("Reward", np.mean(reward_batch), frame)


def to_gym_interface_pos(state):
    state = np.hstack((state.position, state.goal_distance, state.ranger_reflections))
    return state


def to_gym_interface_pomdp(state):
    state = np.hstack((state.goal_distance, state.ranger_reflections))
    return state


def to_gym_interface_vel(state):
    state = np.hstack((state.position, state.velocity, state.goal_distance, state.ranger_reflections))
    return state


def computeExperimentID(save_dir):
    list_of_ids = [int(id) for id in os.listdir(save_dir)]
    return max(list_of_ids) + 1 if len(list_of_ids) else 0


def concatenate_states(state, hist_steps):
    """
    Concatenate states after reset.
    """
    hist_state = deque(maxlen=hist_steps)
    while len(hist_state) < hist_steps:
        hist_state.append(state)

    return hist_state


def to_gym_interface_cat(hist_state):
    position = np.array([state.position for state in hist_state]).T.flatten()
    goal_distance = np.array([state.goal_distance for state in hist_state])
    ranger_reflections = np.array([state.ranger_reflections for state in hist_state]).T.flatten()

    return np.hstack([position, goal_distance, ranger_reflections])


def plotValues(model, state):
    quantiles, taus = model(state)
    print(quantiles, '\n', taus)


class ReplayBuffer():
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, buffer_size, batch_size, device, seed, gamma, n_step=1):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.device = device
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.n_step = n_step
        self.n_step_buffer = deque(maxlen=self.n_step)
    

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        #print("before:", state,action,reward,next_state, done)
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) == self.n_step:
            state, action, reward, next_state, done = self.calc_multistep_return()
            #print("after:",state,action,reward,next_state, done)
            e = self.experience(state, action, reward, next_state, done)
            self.memory.append(e)
    

    def calc_multistep_return(self):
        reward = 0
        for idx in range(self.n_step):
            reward += self.gamma**idx * self.n_step_buffer[idx][2]
        return self.n_step_buffer[0][0], self.n_step_buffer[0][1], reward, self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
        
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)


    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class ExpoWeightedAveForcast():
    def __init__(self, 
                 lr: float = 0.1,
                 window: int = 10, 
                 decay: float = 0.8,
                 init: float = 0.0,
                 use_std: bool = True) -> None:
        """Initialize.
        """
        
        self.w = {0:-3, 1:3.0}
        self.error_buffer = []
        self.window = window
        self.lr = lr
        self.decay = decay
        self.use_std = use_std
        
        self.choices = [self.arm]
        self.data = []

    def get_probs(self) -> List:
        """Get arm probabilities.
        Returns:
            List: probabilities for each arm. 
        """
        p = [np.exp(x) for x in self.w.values()]
        p[1] += (np.sum(p) / 9)
        p /= np.sum(p) # normalize to make it a distribution

        return p

        
    def update_dists(self, feedback: float, norm: float = 1.0) -> None:
        """Update distribution over arms. 
        Args:
            feedback (float): Feedback signal. 
            norm (float, optional): Normalization factor. Defaults to 1.0.
        """

        # Since this is non-stationary, subtract mean of previous self.window errors. 
        self.error_buffer.append(feedback)
        self.error_buffer = self.error_buffer[-self.window:]
        
        # normalize
        feedback -= np.mean(self.error_buffer)
        if self.use_std and len(self.error_buffer) > 1: norm = np.std(self.error_buffer); 
        feedback /= (norm + 1e-4)

        feedback = np.mean(self.error_buffer)

        # update arm weights
        #self.w[self.arm] *= self.decay
        if feedback > 0:

            #self.w[1] += self.lr * (feedback / max(self.get_probs()[1], 0.01))
            self.w[1] += self.lr * feedback #lr=2
            self.w[0] -= self.lr * feedback
        elif feedback < 0:
            self.w[1] += self.lr * feedback
            self.w[0] -= self.lr * feedback
            #self.w[1] += self.lr * (feedback)
        #self.w[self.arm] += self.lr * feedback * elf.get_probs()[self.arm]
        
        self.data.append(feedback)