import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, env, n_atoms=51, v_min=-10, v_max=10):
        super().__init__()
        self.env = env
        self.n_atoms = n_atoms
        self.register_buffer("atoms", torch.linspace(v_min, v_max, steps=n_atoms))
        self.n = env.action_space.n

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(6144, 256)
        self.fc2 = nn.Linear(256, self.n * n_atoms)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.003)
        self.device = torch.device('mps')
        self.to(self.device)
        print(f'running on {self.device}')

    def forward(self, state):
        x = state.view(-1, 4, 128, 64).to(self.device)
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
        x = self.fc2(x)
        return x

    def get_action(self, x, action=None):
        logits = self.forward(x / 255.0)
        x = x.view(-1, 4, 128, 64).to(self.device)
        # probability mass function for each action
        pmfs = torch.softmax(logits.view(len(x), self.n, self.n_atoms), dim=2)
        q_values = (pmfs * self.atoms).sum(2)
        if action is None:
            action = torch.argmax(q_values, 1)
        return action, pmfs[torch.arange(len(x)), action]