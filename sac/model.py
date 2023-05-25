import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, MultivariateNormal
torch.manual_seed(0)


class CriticNetwork(nn.Module):
    def __init__(self, env) -> None:
        super(CriticNetwork, self).__init__()
        self.n_actions = env.action_space.n
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(6144, 256)
        self.fc2 = nn.Linear(256, self.n_actions)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.003)
        # self.device = torch.device('cuda:1')
        self.device=torch.device('mps')
        self.to(self.device)
        print(f'running on {self.device}')
    
    def forward(self, state):
        x = state.reshape(-1, 4, 128, 64).to(self.device)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = x.reshape(-1, 6144)
        x = F.relu(self.fc1(x))
        q = self.fc2(x)
        return q.cpu()

class ActorNetwork(CriticNetwork):
    def __init__(self, env):
        super().__init__(env)
        self.reparam_noise = 1e-6

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.003)
        self.device = torch.device('mps')
        # self.device = torch.device('cuda:0')
        self.to(self.device)
        print(f'running on {self.device}')
    
    def forward(self, state):
        x = super().forward(state)
        probs = F.softmax(x, dim=1)
        return probs
        
    def sample_categorical(self, state):
        probs = self.forward(state)
        dist = Categorical(probs)
        actions = dist.sample().reshape(-1, 1)
        log_probs = torch.log(probs + self.reparam_noise)
        return actions.cpu(), probs.cpu(), log_probs.cpu()

    def save_checkpoint(self, checkpoint_dir):
        print('--------------saving checkpoint------------------')
        torch.save(self.state_dict(), checkpoint_dir)

    def load_checkpoint(self, checkpoint_dir):
        self.load_state_dict(torch.load(checkpoint_dir))