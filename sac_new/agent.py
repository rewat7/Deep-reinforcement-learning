import numpy as np
import torch
import torch.nn.functional as F
import random
from collections import deque
from model_mlp import ActorNetwork, CriticNetwork
import time
import math


class Agent():
    def __init__(self, env):
        self.tau = 0.005
        self.discount = 0.99
        self.iters = 0
        self.terminate = False
        self.minibatch_size = 128
        self.replay_memory_size = 100_000
        self.num_mc = 3
        self.env = env
        self.replay_memory = deque(maxlen=self.replay_memory_size)

        self.target_entropy = - torch.tensor(self.env.action_space.shape[0])
        
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=0.003, eps=1e-4)

        self.critic_1 = CriticNetwork(self.minibatch_size, env)
        # self.critic_2 = CriticNetwork(self.minibatch_size, env)
        self.critic_1_target = CriticNetwork(self.minibatch_size, env)
        # self.critic_2_target = CriticNetwork(self.minibatch_size, env)
        
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        # self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        
        self.actor = ActorNetwork(self.minibatch_size, env)

    def Q_target(self, reward, new_states, done):
            with torch.no_grad():
                new_action, log_probs, _ = self.actor.sample_normal(new_states, self.num_mc, reparameterize=False)
                q1, _ = self.critic_1_target.sample(new_states, new_action, self.num_mc)
                # q2, _ = self.critic_2_target.sample(new_states, new_action, self.num_mc)
                # next_q_values = torch.min(q1, q2).view(-1)
                q1 = q1.view(-1)
                log_probs = log_probs.view(-1)
                next_values = q1 - self.alpha*log_probs
                Q_target = reward + (~done)*(self.discount * next_values)
            
            return Q_target
                
    
    def critic_loss(self, curr_states, actions, reward, new_states, done, store=False):
        Q_target = self.Q_target(reward, new_states, done)
        # with torch.autograd.detect_anomaly():
        # for critic in (self.critic_1, self.critic_2):
        q, jsg = self.critic_1.sample(curr_states, actions, self.num_mc)
        q = q.view(-1)
        mse_loss = F.mse_loss(q, Q_target)
        critic_loss = (jsg - mse_loss).to(device=self.critic_1.device)
        self.critic_1.optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_1.optimizer.step()

    def policy_loss(self, curr_states):
        action, log_probs, jsg = self.actor.sample_normal(curr_states, self.num_mc)
        q1, _ = self.critic_1.sample(curr_states, action, self.num_mc)
        # q2, _ = self.critic_2.sample(curr_states, action, self.num_mc)
        # critic_value = torch.min(q1, q2).view(-1)
        policy_loss = jsg - (self.alpha*log_probs - q1).mean()
        # with torch.autograd.detect_anomaly():
        policy_loss = policy_loss.to(device=self.actor.device)
        self.actor.optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

    def sigmoid_decay(self, iters):
        entropy = (3 / (1 + math.exp(0.001*iters))) + self.target_entropy
        return entropy

    def update_alpha(self, state, iters):
        _, log_probs, _ = self.actor.sample_normal(state, self.num_mc)
        # target_entropy = self.sigmoid_decay(iters)
        alpha_loss = - (self.log_alpha * (log_probs + self.target_entropy)).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward(retain_graph=True)
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()
        
    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action, _, _ = self.actor.sample_normal(state, self.num_mc)
        action = action.reshape(-1)
        action = action.detach().cpu().numpy()
        return action

    def learn(self):
        if len(self.replay_memory) < self.minibatch_size:
            return
        
        self.iters += 1
        minibatch = random.sample(self.replay_memory, self.minibatch_size)
    
        current_states = torch.tensor(np.vstack([transition[0] for transition in minibatch]),  dtype=torch.float32)
        # current_states = current_states/255

        actions = torch.tensor(np.vstack([transition[1] for transition in minibatch]), dtype=torch.int64)
        actions = actions.view(-1)
        # assert actions.shape == (self.batch_size)

        rewards=torch.tensor(np.vstack([transition[2] for transition in minibatch]), dtype=torch.float32)
        rewards = rewards.view(-1)
        # assert rewards.shape == (self.batch_size)

        new_states = torch.tensor(np.vstack([transition[3] for transition in minibatch]), dtype=torch.float32)
        # new_states = new_states/255

        done = torch.tensor(np.vstack([int(transition[4]) for transition in minibatch]), dtype=torch.int64)
        done  = done.view(-1)
        # assert done.shape == (self.batch_size)

        self.critic_loss(current_states, actions, rewards, new_states, done)

        self.policy_loss(current_states)

        self.update_alpha(current_states, self.iters)

        self.update_model_params()

    def train_loop(self):
        while True:
            if self.terminate:
                return
            self.learn()
            time.sleep(0.01)
        
    def update_model_params(self):
        for target_param, local_param in zip(self.critic_1_target.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
        
        # for target_param, local_param in zip(self.critic_2_target.parameters(), self.critic_2.parameters()):
        #     target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
            
    def update_replay_memory(self, transition):
        # transition = (curr state, action, reward, new state, done)
        self.replay_memory.append(transition)
