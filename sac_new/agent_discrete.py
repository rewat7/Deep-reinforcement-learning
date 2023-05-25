import numpy as np
import torch
import torch.nn.functional as F
import random
from collections import deque
from model import ActorNetwork, CriticNetwork
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

        self.target_entropy = torch.tensor(self.env.action_space.n)
        self.target_entropy = 0.98*torch.log(self.target_entropy)
    
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=0.003, eps=1e-4)

        self.critic_1 = CriticNetwork(env)
        self.critic_1_target = CriticNetwork(env)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.actor = ActorNetwork(env)

    def Q_target(self, reward, new_states, done):
        with torch.no_grad():
            _, probs, log_probs, _ = self.actor.sample_categorical(new_states, self.num_mc)
            q1, _ = self.critic_1_target.sample(new_states, self.num_mc)
            next_values = (probs*(q1 - self.alpha*log_probs)).sum(dim=1).unsqueeze(-1)
            Q_target = reward + (1-done)*(self.discount*next_values)
            return Q_target

    def take_optimization_step(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()          
    
    def critic_loss(self, curr_states, actions, reward, new_states, done):
        Q_target = self.Q_target(reward, new_states, done)
        q, jsg = self.critic_1.sample(curr_states, self.num_mc)
        q = q.gather(1, actions)
        loss = F.mse_loss(q, Q_target)
        critic_loss = (jsg/self.minibatch_size - loss).to(device=self.critic_1.device)
        self.take_optimization_step(critic_loss, self.critic_1.optimizer)

    def policy_loss(self, curr_states):
        q1, _ = self.critic_1.sample(curr_states, self.num_mc)
        _, probs, log_probs, jsg = self.actor.sample_categorical(curr_states, self.num_mc)
    
        # with torch.autograd.detect_anomaly():
        policy_loss = (probs*(self.alpha*log_probs - q1)).sum(dim=1).mean()
        policy_loss = (jsg/self.minibatch_size - policy_loss).to(self.actor.device)
        self.take_optimization_step(policy_loss, self.actor.optimizer)
        

    def sigmoid_decay(self, iters):
        entropy = (3 / (1 + math.exp(0.001*iters))) + self.target_entropy
        return entropy

    def update_alpha(self, state):
        _, _, log_probs, _ = self.actor.sample_categorical(state, self.num_mc)
        alpha_loss = - (self.log_alpha * (log_probs + self.target_entropy)).mean()
        self.take_optimization_step(alpha_loss, self.alpha_optim)
        self.alpha = self.log_alpha.exp()
        
    def choose_action(self, state):
        state = torch.tensor(state)
        state = state/255
        with torch.no_grad():
            action, _, _, _ = self.actor.sample_categorical(state, self.num_mc)
            action = action.reshape(-1)
            action = action.detach().cpu().numpy()[0]
    #         print(action)
            return action

    def learn(self):
        if len(self.replay_memory) < self.minibatch_size:
            return
        
        self.iters += 1
        minibatch = random.sample(self.replay_memory, self.minibatch_size)
    
        current_states = torch.tensor(np.vstack([transition[0] for transition in minibatch]),  dtype=torch.float32)
        current_states = current_states/255

        actions = torch.tensor(np.vstack([transition[1] for transition in minibatch]), dtype=torch.int64)

        rewards=torch.tensor(np.vstack([transition[2] for transition in minibatch]), dtype=torch.float32)

        new_states = torch.tensor(np.vstack([transition[3] for transition in minibatch]), dtype=torch.float32)
        new_states = new_states/255

        done = torch.tensor(np.vstack([int(transition[4]) for transition in minibatch]), dtype=torch.int64)

        self.critic_loss(current_states, actions, rewards, new_states, done)

        self.policy_loss(current_states)

        self.update_alpha(current_states)
        
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
        
    def update_replay_memory(self, transition):
        # transition = (curr state, action, reward, new state, done)
        self.replay_memory.append(transition)
