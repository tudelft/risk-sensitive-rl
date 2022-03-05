import torch
import torch.optim as optim
import numpy as np
import random
from model import IQN
from utils.util import calculate_huber_loss, ReplayBuffer
import crazyflie_env
from crazyflie_env.envs.utils.action import ActionXY

class DQNAgent():
    """Interacts with and learns from the environment."""
    def __init__(self, state_size, num_directions, num_speeds, layer_size, n_step, BATCH_SIZE, BUFFER_SIZE,
                LR, TAU, GAMMA, UPDATE_EVERY, device, seed, distortion, con_val_at_risk, variance_samples_n, eval=False):
        """Initialize an Agent object.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            layer_size (int): size of the hidden layer
            BATCH_SIZE (int): size of the training batch
            BUFFER_SIZE (int): size of the replay memory
            LR (float): learning rate
            TAU (float): tau for soft updating the network weights
            GAMMA (float): discount factor
            UPDATE_EVERY (int): update frequency
            device (str): device that is used for the compute
            seed (int): random seed
            distortion (str): decide which distortion function to use
        """
        self.state_size = state_size
        self.num_directions = num_directions # larger number provide more precise direction control
        self.num_speeds = num_speeds # larger num_speed give more precise speed control
        self.action_size = self.num_directions * self.num_speeds
        self.action_space = None # don't mess it around with env.action_space in gym
        self.eval = eval
        if not self.eval:
            self.seed = random.seed(seed)
        self.device = device
        self.TAU = TAU
        self.GAMMA = GAMMA
        self.UPDATE_EVERY = UPDATE_EVERY
        self.BATCH_SIZE = BATCH_SIZE
        self.n_step = n_step

        self.variance_samples_n = variance_samples_n

        self.action_step = 4
        self.last_action = None

        # IQN-Network
        self.qnetwork_local = IQN(self.state_size, self.action_size, layer_size, n_step, seed, distortion, con_val_at_risk, variance_samples_n).to(device)
        self.qnetwork_target = IQN(self.state_size, self.action_size, layer_size, n_step, seed, distortion, con_val_at_risk, variance_samples_n).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        #print(self.qnetwork_local)
        
        # Replay memory
        if not self.eval:
            self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, device, seed, GAMMA, n_step)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def update(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()
                loss = self.learn(experiences)
                
                return loss


    def build_action_space(self, max_velocity):
        """Build discrete action space according to given number of possible directions and max velocity.
        Return: List of ActionXY (or Action rotation)
        """
        rotations = np.linspace(0, 2 * np.pi, self.num_directions, endpoint=False)
        speeds = [(np.exp((i + 1) / self.num_speeds) - 1) / (np.e - 1) * max_velocity for i in range(self.num_speeds)]
        #action_space = [ActionXY(0, 0)]
        action_space = []
        none_zero_actions = [(r, v) for r in rotations for v in speeds]
        for r, v in none_zero_actions:
            action_space.append(ActionXY(v * np.cos(r), v * np.sin(r)))
        print("Num of discrete actions: {}".format(len(action_space)))
        
        return action_space


    def act(self, state, eps, cvar):
        """Returns action indexes for given state as per current policy. Acting only every 4 frames!
        Params
        ======
            frame: to adjust epsilon
            state (array_like): current state
        """
        assert self.action_space is not None
        if self.action_step == 4:
            state = np.array(state)

            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local.get_qvals(state, cvar)
            self.qnetwork_local.train()

            # epsilon-greedy action selection
            if random.random() > eps:
                action = np.argmax(action_values.cpu().data.numpy())
                self.last_action = action
                return action, self.action_space[action]
            else:
                action = random.choice(np.arange(self.action_size))
                self.last_action = action
                return action, self.action_space[action]
        else:
            self.action_step += 1
            return self.last_action, self.action_space[self.last_action]


    def get_tcv(self, state, action_id):
        state = np.array(state)
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            tcv = self.qnetwork_local.get_tcv(state, action_id)
        
        return tcv


    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones = experiences
        # Get max predicted Q values (for next states) from target model
        Q_targets_next, _ = self.qnetwork_target(next_states)
        Q_targets_next = Q_targets_next.detach().max(2)[0].unsqueeze(1) # (batch_size, 1, N)
        
        # Compute Q targets for current states 
        Q_targets = rewards.unsqueeze(-1) + (self.GAMMA ** self.n_step * Q_targets_next * (1. - dones.unsqueeze(-1)))
        # Get expected Q values from local model
        Q_expected, taus = self.qnetwork_local(states)
        Q_expected = Q_expected.gather(2, actions.unsqueeze(-1).expand(self.BATCH_SIZE, 8, 1))

        # Quantile Huber loss
        td_error = Q_targets - Q_expected
        assert td_error.shape == (self.BATCH_SIZE, 8, 8), "wrong td error shape"
        huber_l = calculate_huber_loss(td_error, 1.0)
        quantil_l = abs(taus -(td_error.detach() < 0).float()) * huber_l / 1.0
        
        loss = quantil_l.sum(dim=1).mean(dim=1) # keepdim=True if per weights get multiple
        loss = loss.mean()

        # minimize the loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 0.5)
        self.optimizer.step()

        # update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        return loss.detach().cpu().numpy()


    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ * θ_local + (1 - τ) * θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data)