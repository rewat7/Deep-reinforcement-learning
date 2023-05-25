import os
from threading import Thread
from tqdm import tqdm
import gymnasium as gym
from agent import Agent
import pandas as pd

ep_rewards = []
ep_len = []
episodes = 500
model_name = 'SAC_vanilla'

# env = gym.make('Humanoid-v4')
# env = gym.make('HumanoidStandup-v4')
# env = gym.make('HalfCheetah-v4')
# env = gym.make('Swimmer-v4')
# env = gym.make("roundabout-v0")
# env.configure({
#     "observation":{
#             "type": "GrayscaleObservation",
#             "observation_shape": (128, 64),
#             "stack_size": 4,
#             "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
#             "scaling": 1.75,
#     }
# })

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
    current_state, _ = env.reset(seed=1)
    done = truncated = False

    while not (done or truncated):
        action = agent.choose_action(current_state)
        new_state, reward, done, truncated , info = env.step(action)
        episode_reward += reward
        agent.update_replay_memory((current_state, action, reward,
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
agent.actor.save_checkpoint(f'models/{env.spec.id}/target_model_{env.spec.id}_{model_name}.pt')