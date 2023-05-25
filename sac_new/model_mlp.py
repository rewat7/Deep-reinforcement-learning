import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from model import LinearReparameterizationNew
torch.manual_seed(seed=0)

class CriticNetwork(nn.Module):
    def __init__(self, batch_size, env) -> None:
        super(CriticNetwork, self).__init__()
        self.batch_size = batch_size
        self.n_actions = env.action_space.shape[0]
        self.obs = env.observation_space.shape[0]
        self.prior_mu = 0.0
        self.prior_sigma = 1.0
        self.posterior_mu_init = 0.0
        self.posterior_rho_init = -3.0
        self.fc1 = LinearReparameterizationNew(self.obs + self.n_actions, 256, 
                                                prior_mean=self.prior_mu,
                                                prior_variance=self.prior_sigma,
                                                posterior_mu_init=self.posterior_mu_init,
                                                posterior_rho_init=self.posterior_rho_init)

        
        self.fc2 =LinearReparameterizationNew(256, 512, 
                                                prior_mean=self.prior_mu,
                                                prior_variance=self.prior_sigma,
                                                posterior_mu_init=self.posterior_mu_init,
                                                posterior_rho_init=self.posterior_rho_init)
        
        self.fc3 =LinearReparameterizationNew(512, 1, 
                                                prior_mean=self.prior_mu,
                                                prior_variance=self.prior_sigma,
                                                posterior_mu_init=self.posterior_mu_init,
                                                posterior_rho_init=self.posterior_rho_init)
    

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
        jsg_sum = 0
        x = x.to(self.device)

        x, jsg = self.fc1(x)
        jsg_sum += jsg
        x = F.relu(x)

        x, jsg = self.fc2(x)
        jsg_sum += jsg
        x = F.relu(x)

        q, jsg= self.fc3(x)
        jsg_sum += jsg
    
        return q, jsg_sum


    def sample(self, state, action, num_mc):
        output_1 = []
        kl_1 = []
        for mc_run in range(num_mc):
            q1, kl = self.forward(state, action)
            output_1.append(q1)
            kl_1.append(kl)
        q1 = torch.mean(torch.stack(output_1), dim=0)
        kl = torch.mean(torch.stack(kl_1), dim=0)
        
        return q1.cpu(), kl.cpu()


class ActorNetwork(CriticNetwork):
    def __init__(self, batch_size, env):
        super().__init__(batch_size, env)
        self.reparam_noise = 1e-6
        self.max_action = env.action_space.high[0]

        self.fc1 = LinearReparameterizationNew(self.obs, 256, 
                                                prior_mean=self.prior_mu,
                                                prior_variance=self.prior_sigma,
                                                posterior_mu_init=self.posterior_mu_init,
                                                posterior_rho_init=self.posterior_rho_init)
        
        self.fc2 = LinearReparameterizationNew(256, 512, 
                                                prior_mean=self.prior_mu,
                                                prior_variance=self.prior_sigma,
                                                posterior_mu_init=self.posterior_mu_init,
                                                posterior_rho_init=self.posterior_rho_init)
        
        self.mu = LinearReparameterizationNew(512, self.n_actions, 
                                                prior_mean=self.prior_mu,
                                                prior_variance=self.prior_sigma,
                                                posterior_mu_init=self.posterior_mu_init,
                                                posterior_rho_init=self.posterior_rho_init)
        
        self.log_std = LinearReparameterizationNew(512, self.n_actions, 
                                                prior_mean=self.prior_mu,
                                                prior_variance=self.prior_sigma,
                                                posterior_mu_init=self.posterior_mu_init,
                                                posterior_rho_init=self.posterior_rho_init)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.003)
        # # self.device = torch.device('cuda:0')
        self.device = torch.device('mps')
        self.to(self.device)
        print(f'running on {self.device}')
    
    
    def forward(self, state):
            jsg_sum = 0
            state = state.to(self.device)
            x, jsg = self.fc1(state)
            jsg_sum += jsg
            x = F.relu(x)

            x, jsg = self.fc2(x)
            jsg_sum += jsg
            x = F.relu(x)

            mu, jsg = self.mu(x)
            jsg_sum += jsg

            log_std, jsg = self.log_std(x)
            jsg_sum += jsg
            log_std = torch.clamp(log_std, min=-20, max=2)

            return mu, log_std, jsg_sum

            

    def sample_normal(self, state, num_mc, reparameterize=True):
            mu_ = []
            log_std_ = []
            jsg_ = []
            for mc_run in range(num_mc):
                mu, log_std, jsg = self.forward(state)
                mu_.append(mu)
                log_std_.append(log_std)
                jsg_.append(jsg)

            mu = torch.mean(torch.stack(mu_), dim=0)
            log_std = torch.mean(torch.stack(log_std_), dim=0)
            jsg = torch.mean(torch.stack(jsg_), dim=0)

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
            
            return action.cpu(), log_probs.cpu(), jsg.cpu()

    def save_checkpoint(self, checkpoint_dir):
        print('--------------saving checkpoint- -----------------')
        torch .save(self.state_dict(), checkpoint_dir)

    def load_checkpoint(self, checkpoint_dir):
        self.load_state_dict(torch.load(checkpoint_dir))