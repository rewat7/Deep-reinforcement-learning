import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, MultivariateNormal

class CriticNetwork(nn.Module):
    def __init__(self, batch_size, env) -> None:
        super(CriticNetwork, self).__init__()
        self.batch_size = batch_size
        self.n_actions = env.action_space.shape[0]
        self.obs = env.observation_space.shape[0]
        self.fc1 = nn.Linear(self.obs + self.n_actions, 256)
        
        self.fc2 =nn.Linear(256, 512)
        
        self.fc3 = nn.Linear(512, 1)
    

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.003)
        # self.device = torch.device('cuda:0')
        self.device = torch.device('mps')
        self.to(self.device)
        print(f'running on {self.device}')
    
    def forward(self, state, action):
        state = state.reshape(self.batch_size, -1)
        if(len(action.shape) == 1):
            action = action.reshape(self.batch_size, -1)
            
        x = torch.cat((state,action), dim=1)

        x = x.to(self.device)

        x= self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        q= self.fc3(x)
    
        return q.cpu()



class ActorNetwork(CriticNetwork):
    def __init__(self, batch_size, env):
        super().__init__(batch_size, env)
        self.reparam_noise = 1e-6
        self.max_action = env.action_space.high[0]

        self.fc1 = nn.Linear(self.obs, 256)
        
        self.fc2 =nn.Linear(256, 512)
        
        self.mu = nn.Linear(512, self.n_actions)
        self.sigma = nn.Linear(512, self.n_actions)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.003)
        self.device = torch.device('mps')
        self.to(self.device)
        print(f'running on {self.device}')
    
    
    def forward(self, state):

            state = state.to(self.device)
            x= self.fc1(state)
            x = F.relu(x)

            x= self.fc2(x)
            x = F.relu(x)

            mu= self.mu(x)

            log_std= self.sigma(x)
            log_std = torch.clamp(log_std, min=-20, max=2)

            return mu, log_std

            

    def sample_normal(self, state, reparameterize=True):
            mu, log_std = self.forward(state)
            std = log_std.exp()
            probs = Normal(mu, std)
            if reparameterize:
                action = probs.rsample() # reparameterizes the policy
            else:
                action = probs.sample()
            action = torch.tanh(action)*torch.tensor(self.max_action).to(self.device) 
            log_probs = probs.log_prob(action)
            log_probs -= torch.log(1-action.pow(2) + self.reparam_noise)
            log_probs = log_probs.sum(-1, keepdim=True)
            
            return action.cpu(), log_probs.cpu()

    def save_checkpoint(self, checkpoint_dir):
        print('--------------saving checkpoint- -----------------')
        torch .save(self.state_dict(), checkpoint_dir)

    def load_checkpoint(self, checkpoint_dir):
        self.load_state_dict(torch.load(checkpoint_dir))