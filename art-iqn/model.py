import torch
import torch.nn as nn
import numpy as np
from utils.util import weight_init
class IQN(nn.Module):
    '''implicit quantile network model'''
    def __init__(self, state_size, action_size, layer_size, n_step, seed, distortion, con_val_at_risk, variance_samples_n=8, layer_type="ff"):
        super(IQN, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.seed = torch.manual_seed(seed)
        self.input_shape = state_size
        self.action_size = action_size
        self.K = 32 # for action selection
        self.N = 8 # num tau
        # samples for variance calculation, variance_samples_n = N / 2, N in paper (DRL for efficient exploration, ICML 2019)
        self.variance_samples_n = variance_samples_n
        self.n_cos = 64 # embedding dimension
        self.distortion = distortion
        self.con_val_at_risk = con_val_at_risk
        self.layer_size = layer_size
        self.pis = torch.FloatTensor([np.pi * i for i in range(self.n_cos)]).view(1, 1, self.n_cos).to(self.device)
        self.head = nn.Linear(self.input_shape, layer_size)
        self.cos_embedding = nn.Linear(self.n_cos, layer_size)
        self.hidden_layer = nn.Linear(layer_size, layer_size)
        self.hidden_layer_2 = nn.Linear(layer_size, layer_size)
        self.output_layer = nn.Linear(layer_size, action_size)
        #weight_init([self.head_1, self.ff_1])
        print("Distortion measure: {}\tRandom CVaR: {}".format(self.distortion, 'True' if self.con_val_at_risk else 'False'))


    def calc_cos(self, batch_size, n_tau=8, distortion='neutral', cvar=1.0):
        """
        Calculating the cosinus values depending on the number of tau samples
        """
        taus = torch.rand(batch_size, n_tau).to(self.device).unsqueeze(-1) # (batch_size, n_tau, 1) for broadcast
        #print(taus)

        # distorted quantile sampling
        if distortion == 'CVaR':
            taus = taus * cvar
            #print(taus)
        elif distortion == 'neutral':
            pass
        else:
            raise ValueError('Distortion type not supported.')

        cos = torch.cos(taus * self.pis)
        assert cos.shape == (batch_size, n_tau, self.n_cos), "cos shape is incorrect"
        return cos, taus
    

    def calc_cos_tcv(self, batch_size, n_tau=8):
        """
        Calculating the cosinus values depending on the number of tau samples for TCV calculation
        Fixed taus range
        n_tau: controls how many samples to measure the tcv
        """
        taus = torch.linspace(1/(2 * n_tau), 0.5, n_tau).to(self.device).unsqueeze(-1) # (batch_size, n_tau, 1) for broadcast

        cos = torch.cos(taus * self.pis)
        assert cos.shape == (batch_size, n_tau, self.n_cos), "cos shape is incorrect"
        return cos, taus


    def forward(self, inputs, num_tau=8, distortion='neutral', cvar=1.0, tcv=False):
        """
        Quantile calculation depending on the number of tau
        
        Return:
        quantiles [ shape of (batch_size, num_tau, action_size)]
        taus [shape of ((batch_size, num_tau, 1))]
        
        """
        batch_size = inputs.shape[0]
        
        x = torch.relu(self.head(inputs))
        if not tcv:
            cos, taus = self.calc_cos(batch_size, num_tau, distortion, cvar) # cos shape (batch, num_tau, layer_size)
        else:
            cos, taus = self.calc_cos_tcv(batch_size, num_tau) # fixed tau range, no sampling for tau
        cos = cos.view(batch_size * num_tau, self.n_cos)
        cos_x = torch.relu(self.cos_embedding(cos)).view(batch_size, num_tau, self.layer_size)
        
        # x has shape (batch, layer_size) for multiplication –> reshape to (batch, 1, layer)
        x = (x.unsqueeze(1) * cos_x).view(batch_size * num_tau, self.layer_size)
        
        x = torch.relu(self.hidden_layer(x))
        x = torch.relu(self.hidden_layer_2(x))
        out = self.output_layer(x)
        return out.view(batch_size, num_tau, self.action_size), taus


    def get_qvals(self, inputs, cvar):
        quantiles, _ = self.forward(inputs=inputs, num_tau=self.K, distortion=self.distortion, cvar=cvar)
        qvals = quantiles.mean(dim=1)
        return qvals
    

    def get_tcv(self, inputs, action_id):
        #quantiles [ shape of (batch_size, num_tau, action_size)]
        truncated_quantiles, taus = self.forward(inputs=inputs, num_tau=self.variance_samples_n, tcv=True)
        quantiles_action = truncated_quantiles.transpose(1, 2)[0, action_id]
        #print(quantiles_action.shape, taus) # should be torch.Size([n_tau])
        median = quantiles_action[-1]
        tcv = torch.mean((quantiles_action - median)**2) / 4 # tcv = 1/(2*N)*sum_1^{N/2}(theta_{N/2} - theta_i)^2
    
        return tcv