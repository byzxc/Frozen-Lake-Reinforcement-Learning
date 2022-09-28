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
import os
import time

# Customize for pandas
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)


# List all available gym environments
# all_env = list(gym.envs.registry.all())
# print(f'Num Gym Environments: {len(all_env)}')

# You could loop through and list all environments if you wanted
# envs_starting_with_f = [e for e in all_env if str(e).startswith("EnvSpec(Frozen")]
# [EnvSpec(FrozenLake-v0), EnvSpec(FrozenLake8x8-v0)]

# action space is 4 possible actions (LEFT, DOWN, RIGHT, UP)
# observation space is 4x4, runs from 0 to 15
env_name = "FrozenLake-v0"

# Creating gym object
# is_slippery=True specifies the environment is stochastic
# is_slippery=False is the same as "deterministic=True"
# deterministic= current state + selected action dtermines the next state of environment (Chess)
# stochastic= probability of distribution over a set of possible actions （Random visitor to website)
env = gym.make(env_name, is_slippery=False)

# Check for environment errors before continuing
try:
    check_env(env)
    print("All checks passed. No errors found.")
except:
    print("failed")
    exit()

# Testing
"""
# Always Reset env before render for the first time
env.reset()  # set done flag = false
env.render()

# Take an action
new_obs, reward, done, _ = env.step(2)  # Right
env.render()

new_obs, reward, done, _ = env.step(1)  # Down
env.render()
"""

# They allow us to render the env frame-by-frame in-place
# (w/o creating a huge output which we would then have to scroll through).
out = Output()
display.display(out)
with out:

    # Putting the Gym simple API methods together.
    # Here is a pattern for running a bunch of episodes.
    num_episodes = 5  # Number of episodes you want to run the agent
    num_timesteps = 0
    episode_rewards = []  # store every episode reward

    # Loop through episodes
    for ep in range(num_episodes):

        # Reset the environment at the start of each episode
        obs = env.reset()
        done = False
        episode_reward = 0.0

        # Loop through time steps per episode
        while True:

            # take random action, but you can also do something more intelligent
            action = env.action_space.sample()

            # apply the action
            new_obs, reward, done, info = env.step(action)
            episode_reward += reward

            num_timesteps += 1

            # If the epsiode is up, then start another one
            if done:
                episode_rewards.append(episode_reward)
                break

            # Render the env (in place).
            time.sleep(0.3)
            out.clear_output(wait=True)
            print(f"episode: {ep}")
            print(f"obs: {new_obs}, reward: {episode_reward}, done: {done}")
            env.render()
