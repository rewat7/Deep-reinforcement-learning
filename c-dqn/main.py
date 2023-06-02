import os
from threading import Thread
from tqdm import tqdm
import gymnasium as gym
from agent import Agent
import pandas as pd
import random
import numpy as np
import torch

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
        slope = (end_e - start_e) / duration
        return max(slope * t + start_e, end_e)

if __name__ == "__main__":
    ep_rewards = []
    ep_len = []
    episodes = 5000
    model_name = 'DQN'

    env = gym.make("merge-v0")
    env.configure({
    "observation":{
            "type": "GrayscaleObservation",
            "observation_shape": (128, 64),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
            "scaling": 1.75,
        }
    })


    if not os.path.isdir(f'models/{env.spec.id}'):
        os.makedirs(f'models/{env.spec.id}')

    if not os.path.isdir(f'results/{env.spec.id}'):
        os.makedirs(f'results/{env.spec.id}')


    agent = Agent(env)
    trainer_thread = Thread(target=agent.train_loop, daemon=True)
    trainer_thread.start()

    for episode in tqdm(range(1, episodes + 1), unit='episodes'):
        episode_reward = 0
        step = 1
        current_state, _ = env.reset(seed=0)
        done = truncated = False
        epsilon = linear_schedule(1, 0.01, episodes, episode)

        while not (done or truncated):
            if random.random() < epsilon:
                actions = np.array(env.action_space.sample())
            else:
                actions, pmf = agent.q_network.get_action(torch.Tensor(current_state))
                actions = actions.cpu().numpy()

            new_state, reward, done, truncated , info = env.step(actions)
            episode_reward += reward
            agent.update_replay_memory((current_state, actions, reward,
                                        new_state, done))

            current_state = new_state
            step += 1
            # env.render()

        ep_rewards.append([episode, episode_reward])
        ep_len.append([episode, step])
        # if not episode % aggregate_stats_every:
        #     average_reward = sum(ep_rewards[-aggregate_stats_every:]) / len(ep_rewards[-aggregate_stats_every:])
        #     # min_reward = min(ep_rewards[-aggregate_stats_every:])
        #     # max_reward = max(ep_rewards[-aggregate_stats_every:])
        #     avg_rewards = create_Df(avg_rewards, data=[episode, average_reward], columns=['episode','avg reward'])

    rewards = pd.DataFrame(ep_rewards, columns=['episodes', 'rewards'])
    episode_lens = pd.DataFrame(ep_len, columns=['episodes','episode len'])

    agent.terminate = True
    trainer_thread.join()
    rewards.to_csv(f'results/{env.spec.id}/rewards_{env.spec.id}_{model_name}.csv')
    episode_lens.to_csv(f'results/{env.spec.id}/episode_lens_{env.spec.id}_{model_name}.csv')
    print('==================== results saved ================')
    # agent.actor.save_checkpoint(f'models/{env.spec.id}/target_model_{env.spec.id}_{model_name}.pt')