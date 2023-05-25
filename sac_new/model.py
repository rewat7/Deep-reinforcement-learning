import torch
import torch.nn as nn
import torch.nn.functional as F
from bayesian_torch.layers import BaseVariationalLayer_, Conv2dReparameterization, LinearReparameterization
torch.manual_seed(seed=0)

class BaseVariationalLayerNew(BaseVariationalLayer_):
    def __init__(self):
        super().__init__()

    def kl_div(self, posterior_mu, posterior_sigma, prior_mu, prior_sigma, alpha=0.5):
        # print('using jensons')
        sigma_0_alpha = (posterior_sigma.pow(2) * prior_sigma.pow(2)) / ((1-alpha)*posterior_sigma.pow(2) + alpha*prior_sigma.pow(2))
        mu_0_alpha = sigma_0_alpha * (alpha*posterior_mu/(posterior_sigma.pow(2)) + (1-alpha)*prior_mu/(posterior_sigma.pow(2)))
    
        term1 = ((1-alpha)*posterior_sigma.pow(2) + alpha*prior_sigma.pow(2)) / sigma_0_alpha
        term2 = torch.log(sigma_0_alpha.pow(2) / posterior_sigma.pow(2-2*alpha) * prior_sigma.pow(2*alpha))
        term3 = (1-alpha)*(mu_0_alpha - posterior_mu).pow(2) / sigma_0_alpha
        term4 = alpha*(mu_0_alpha - prior_mu).pow(2) / sigma_0_alpha
    
        js_g_divergence = 0.5 * torch.sum(term1 + term2 + term3 + term4 - 1)
        return js_g_divergence
    
class Conv2dReparameterizationNew(Conv2dReparameterization, BaseVariationalLayerNew):
    def __init__(self, in_channels, out_channels, kernel_size,  prior_mean, prior_variance, posterior_mu_init, posterior_rho_init, stride=1, padding=0, dilation=1, groups=1,bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, prior_mean, prior_variance, posterior_mu_init, posterior_rho_init, bias)

class LinearReparameterizationNew(LinearReparameterization, BaseVariationalLayerNew):
    def __init__(self, in_features, out_features, prior_mean, prior_variance, posterior_mu_init, posterior_rho_init, bias=True):
        super().__init__(in_features, out_features, prior_mean, prior_variance, posterior_mu_init, posterior_rho_init, bias)


class CriticNetwork(nn.Module):
    def __init__(self, env):
        super(CriticNetwork, self).__init__()
        self.n_actions = env.action_space.n
        prior_mu = 0.0
        prior_sigma = 1.0
        posterior_mu_init = 0.0
        posterior_rho_init = -3.0
        
        self.conv1 = Conv2dReparameterizationNew(
            in_channels=4,
            out_channels=64,
            kernel_size=3,
            stride=1,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )

        self.conv2 = Conv2dReparameterizationNew(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )

        self.conv3 = Conv2dReparameterizationNew(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            stride=1,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )

        self.conv4 = Conv2dReparameterizationNew(
            in_channels=256,
            out_channels=512,
            kernel_size=3,
            stride=1,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )

        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = LinearReparameterizationNew(
            in_features=6144,
            out_features=128,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )

        self.fc2 = LinearReparameterizationNew(
            in_features=128,
            out_features=self.n_actions,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.003)
        self.device = torch.device('mps')
        self.to(self.device)
        print(f'running on {self.device}')

    def forward(self, x):
        x = x.reshape(-1, 4, 128, 64).to(self.device)
        
        kl_sum = 0
        x, kl = self.conv1(x)
        kl_sum += kl
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x, kl = self.conv2(x)
        kl_sum += kl
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x, kl = self.conv3(x)
        kl_sum += kl
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x, kl = self.conv4(x)
        kl_sum += kl
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = x.reshape(-1, 6144)
        x, kl = self.fc1(x)
        kl_sum += kl
        x = F.relu(x)

        q, kl = self.fc2(x)
        kl_sum += kl
        return q, kl_sum
    
    def sample(self, state, num_mc):
        output_1 = []
        kl_1 = []
        for mc_run in range(num_mc):
            q1, kl = self.forward(state)
            output_1.append(q1)
            kl_1.append(kl)
        q1 = torch.mean(torch.stack(output_1), dim=0)
        kl = torch.mean(torch.stack(kl_1), dim=0)
        
        return q1.cpu(), kl.cpu()
        
class ActorNetwork(CriticNetwork):
    def __init__(self, env):
        super().__init__(env)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.003)
        self.reparam_noise=1e-6
        self.device = torch.device('mps')
        self.to(self.device)
        print(f'running on {self.device}')
    
    def forward(self, state):
        x, kl_sum = super().forward(state)
        probs = F.softmax(x, dim=1)
        return probs, kl_sum
        
    def sample_categorical(self, state, num_mc):
        probs_ = []
        kl_ = []
        for mc_run in range(num_mc):
            probs, kl = self.forward(state)
            probs_.append(probs)
            kl_.append(kl)
        probs = torch.mean(torch.stack(probs_), dim=0)
        # print(f'probs {probs}')
        kl = torch.mean(torch.stack(kl_), dim=0)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().reshape(-1, 1)
        log_probs = torch.log(probs + self.reparam_noise)
        # print(f'logprobs {log_probs}')
        return action.cpu(), probs.cpu(), log_probs.cpu(), kl.cpu()
     
    def save_checkpoint(self, checkpoint_dir):
        print('--------------saving checkpoint------------------')
        torch.save(self.state_dict(), checkpoint_dir)

    def load_checkpoint(self, checkpoint_dir):
        self.load_state_dict(torch.load(checkpoint_dir))

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, prediction, target):
        loss = F.mse_loss(prediction, target)
        pt = torch.exp(-loss)
        focal_loss = ((1 - pt) ** self.gamma) * loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
    