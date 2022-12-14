# import libraries
# ray
import ray
from ray.rllib.utils.pre_checks.env import check_env
# important to see the config of result
from ray.tune.logger import pretty_print
from ray import tune, rllib, air
import gym
from ipywidgets import Output
from IPython import display
import numpy as np
import random
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 17})

# Customize for pandas
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

# Let's create our Qtable of size (state_space, action_space) and initialized each values at 0 using np.zeros
def initialize_q_table(state_space, action_space):
  Qtable = np.zeros((state_space, action_space))
  return Qtable

def epsilon_greedy_policy(Qtable, state, epsilon, env):
  # Randomly generate a number between 0 and 1
  random_int = random.uniform(0,1)
  # if random_int > greater than epsilon --> exploitation
  if random_int > epsilon:
    # Take the action with the highest value given a state
    # np.argmax can be useful here
    action = np.argmax(Qtable[state])
  # else --> exploration
  else:
    action = env.action_space.sample()
  
  return action

def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable, learning_rate, gamma):
  """
  Reward function:

  Reach goal: +1
  Reach hole: 0
  Reach frozen: 0
  """
  for episode in range(n_training_episodes):
    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
    # Reset the environment
    state = env.reset()
    step = 0
    done = False

    # repeat
    for step in range(max_steps):
      # Choose the action At using epsilon greedy policy
      action = epsilon_greedy_policy(Qtable, state, epsilon, env)

      # Take action At and observe Rt+1 and St+1
      # Take the action (a) and observe the outcome state(s') and reward (r)
      new_state, reward, done, info = env.step(action)

      # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]S
      Qtable[state][action] = round(Qtable[state][action] + learning_rate * (reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action]), 3)   

      # If done, finish the episode
      if done:
        break
      
      # Our state is the new state
      state = new_state
  return Qtable

def evaluate_agent(env, max_steps, n_eval_episodes, Q, seed):
  """
  Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
  :param env: The evaluation environment
  :param n_eval_episodes: Number of episode to evaluate the agent
  :param Q: The Q-table
  :param seed: The evaluation seed array (for taxi-v3)
  """
  episode_rewards = []
  for episode in range(n_eval_episodes):
    if seed:
      state = env.reset(seed=seed[episode])
    else:
      state = env.reset()
    step = 0
    done = False
    total_rewards_ep = 0

    for step in range(max_steps):
      # Take the action (index) that have the maximum expected future reward given that state
      action = np.argmax(Q[state][:])
      new_state, reward, done, info = env.step(action)
      total_rewards_ep += reward

      if done:
        break
      state = new_state
    episode_rewards.append(total_rewards_ep)
  mean_reward = np.mean(episode_rewards)
  std_reward = np.std(episode_rewards)

  return mean_reward, std_reward

def main():
    # Training parameters
    n_training_episodes = 45000  # STotal training episodes
    learning_rate = 0.7          # Learning rate

    # Evaluation parameters
    n_eval_episodes = 100        # Total number of test episodes

    # Environment parameters
    env_id = "FrozenLake-v1"     # Name of the environment
    max_steps = 99               # Max steps per episode
    gamma = 0.95                 # Discounting rate
    eval_seed = []               # The evaluation seed of the environment

    # Exploration parameters
    max_epsilon = 1.0             # Exploration probability at start
    min_epsilon = 0.05            # Minimum exploration probability
    decay_rate = 0.0005            # Exponential decay rate for exploration prob

    # action space is 4 possible actions (LEFT, DOWN, RIGHT, UP)
    # observation space is 4x4, runs from 0 to 15
    env = gym.make(env_id, map_name="4x4", is_slippery=True)
    env.reset()
    print("_____OBSERVATION SPACE_____ \n")
    print("Observation Space", env.observation_space)
    print("Sample observation", env.observation_space.sample()) # Get a random observation

    print("\n _____ACTION SPACE_____ \n")
    print("Action Space Shape", env.action_space.n)
    print("Action Space Sample", env.action_space.sample()) # Take a random action

    state_space = env.observation_space.n
    print("There are ", state_space, " possible states")

    action_space = env.action_space.n
    print("There are ", action_space, " possible actions")

    # We re-initialize the Q-table
    Qtable_frozenlake = initialize_q_table(state_space, action_space)

    print('Qtable_frozenlake before training:')
    print(Qtable_frozenlake)

    Qtable_frozenlake = train(n_training_episodes, min_epsilon,
                          max_epsilon, decay_rate, env, max_steps, Qtable_frozenlake,
                          learning_rate, gamma)

    print('Qtable_frozenlake after training:')
    print(Qtable_frozenlake)

    # Evaluate our Agent
    mean_reward, std_reward = evaluate_agent(env, max_steps, n_eval_episodes, Qtable_frozenlake, eval_seed)
    print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

if __name__ == "__main__":
    main()
