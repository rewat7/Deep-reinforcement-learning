import numpy as np
import torch
import torch.nn.functional as F
import random
from collections import deque
from model import ActorNetwork, CriticNetwork
import time


class Agent():
    def __init__(self, env):
        self.tau = 0.005
        self.discount = 0.99
        self.iters = 0
        self.terminate = False
        self.minibatch_size = 128
        self.replay_memory_size = 100_000
        self.env = env
        self.replay_memory = deque(maxlen=self.replay_memory_size)

        self.target_entropy = torch.tensor(self.env.action_space.n)
        self.target_entropy = 0.98*torch.log(self.target_entropy)
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=0.003, eps=1e-4)

        self.actor = ActorNetwork(env)
        self.critic_1 = CriticNetwork(env)
        self.critic_2 = CriticNetwork(env)
        self.critic_1_target = CriticNetwork(env)
        self.critic_2_target = CriticNetwork(env)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        
    def Q_target(self, reward, new_state, done):
        with torch.no_grad():
            _, probs, log_probs = self.actor.sample_categorical(new_state)
            q1 = self.critic_1_target(new_state)
            q2 = self.critic_2_target(new_state)
            next_q_values = torch.min(q1,q2)
            next_values = (probs*(next_q_values - self.alpha*log_probs)).sum(dim=1).unsqueeze(-1)
            Q_target = reward + (1-done)*(self.discount*next_values)
            return Q_target
    
    def critic_loss(self, curr_states, actions, reward, new_states, done: bool):
        Q_target = self.Q_target(reward, new_states, done)
        
        # with torch.autograd.detect_anomaly():
        for critic in (self.critic_1, self.critic_2):
            q = critic(curr_states)
            # q = q.gather(1, actions.unsqueeze(-1)).view(-1)
            q = q.gather(1, actions)
            # print(q.shape)
            mse_loss = F.mse_loss(q, Q_target)
            critic_loss = mse_loss.to(device=critic.device)
            critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            critic.optimizer.step()

    def policy_loss(self, curr_states):
        q1 = self.critic_1(curr_states)
        q2 = self.critic_2(curr_states)
        critic_value = torch.min(q1, q2)
    #         alpha = self.alpha_network.sample(states)
        _, probs, log_probs = self.actor.sample_categorical(curr_states)

        # with torch.autograd.detect_anomaly():
        policy_loss = probs*(self.alpha*log_probs - critic_value)
        policy_loss = policy_loss.sum(dim=1).mean().to(self.actor.device)
        self.actor.optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

    def update_alpha(self, state):

        _, _, log_probs = self.actor.sample_categorical(state)
        # target_entropy = self.sigmoid_decay(iters)
        alpha_loss = - (self.log_alpha * (log_probs + self.target_entropy)).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward(retain_graph=True)
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()
        return self.alpha, _

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        state = state/255
        with torch.no_grad():
            action, _, _, = self.actor.sample_categorical(state)
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
        # actions = actions.view(-1)
        # assert actions.shape == (self.batch_size)

        rewards=torch.tensor(np.vstack([transition[2] for transition in minibatch]), dtype=torch.float32)
        # rewards = rewards.view(-1)
        # assert rewards.shape == (self.batch_size)

        new_states = torch.tensor(np.vstack([transition[3] for transition in minibatch]), dtype=torch.float32)
        new_states = new_states/255

        done = torch.tensor(np.vstack([int(transition[4]) for transition in minibatch]))
        # done  = done.view(-1)
        # assert done.shape == (self.batch_size)

        self.critic_loss(current_states, actions, rewards, new_states, done)
        
        self.policy_loss(current_states)
        
        self.update_alpha(current_states)
        # self.alpha_df = create_Df(self.alpha_df, [self.iters, alpha], ['iters','alpha'])
        
        self.update_model_params()

    def train_loop(self):
        while True:
            if self.terminate:
                # self.alpha_df.to_csv('results/alpha.csv')
                # print('=================== results saved ================')
                return
            self.learn()
            time.sleep(0.01)
        
    def update_model_params(self):
        for target_param, local_param in zip(self.critic_1_target.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
        
        for target_param, local_param in zip(self.critic_2_target.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def update_replay_memory(self, transition):
        # transition = (curr state, action, reward, new state, done)
        self.replay_memory.append(transition)