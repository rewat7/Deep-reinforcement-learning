import numpy as np
import torch
import random
from collections import deque
from model import QNetwork
import time 


class Agent():
     
    def __init__(self, envs):
        self.q_network = QNetwork(envs)
        self.target_network = QNetwork(envs)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.terminate = False
        self.minibatch_size = 128
        self.replay_memory_size = 100_000
        self.action_space = envs.action_space
        self.env = envs
        self.replay_memory = deque(maxlen=self.replay_memory_size)
        self.v_min = -10
        self.v_max = 10
        self.n_atoms = 51
        self.device = self.q_network.device
        self.gamma = torch.tensor(0.99).to(device=self.device)

    
    def learn(self):
        if len(self.replay_memory) < self.minibatch_size:
            return

        minibatch = random.sample(self.replay_memory, self.minibatch_size)
        current_states = torch.tensor(np.vstack([transition[0] for transition in minibatch]),  dtype=torch.float32, device=self.device)
        # current_states = current_states/255

        actions = torch.tensor(np.vstack([transition[1] for transition in minibatch]), dtype=torch.int64, device=self.device)
        # assert actions.shape == (self.batch_size)

        rewards=torch.tensor(np.vstack([transition[2] for transition in minibatch]), dtype=torch.float32, device=self.device)
        # assert rewards.shape == (self.batch_size)

        new_states = torch.tensor(np.vstack([transition[3] for transition in minibatch]), dtype=torch.float32, device=self.device)
        # new_states = new_states/255

        done = torch.tensor(np.vstack([int(transition[4]) for transition in minibatch]), device=self.device)
        # with torch.no_grad():
        _, next_pmfs = self.target_network.get_action(new_states)
        next_atoms = rewards + self.gamma * self.target_network.atoms * (1 - done)
        # projection
        delta_z = self.target_network.atoms[1] - self.target_network.atoms[0]
        tz = next_atoms.clamp(self.v_min, self.v_max)

        b = (tz - self.v_min) / delta_z
        l = b.floor().clamp(0, self.n_atoms - 1)
        u = b.ceil().clamp(0, self.n_atoms - 1)
        # (l == u).float() handles the case where bj is exactly an integer
        # example bj = 1, then the upper ceiling should be uj= 2, and lj= 1
        d_m_l = (u + (l == u).float() - b) * next_pmfs
        d_m_u = (b - l) * next_pmfs
        target_pmfs = torch.zeros_like(next_pmfs)
        # print(target_pmfs.shape)
        for i in range(target_pmfs.size(0)):
            target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
            target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])

        _, old_pmfs = self.q_network.get_action(current_states, actions.flatten())
        # print(old_pmfs.shape)
        loss = (-(target_pmfs * old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)).mean()

    # optimize the model
        self.q_network.optimizer.zero_grad()
        loss.backward()
        self.q_network.optimizer.step()

    # update the target network
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def train_loop(self):
        while True:
            if self.terminate:
                return
            self.learn()
            time.sleep(0.01)
    
    def update_replay_memory(self, transition):
        # transition = (curr state, action, reward, new state, done)
        self.replay_memory.append(transition)