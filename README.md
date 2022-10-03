# Frozen Lake V1 Reinforcement Learning
 
Learning Reinforcement Learning through OpenAI Frozen Lake Environment.

Policy used: Epsilon Greedy Policy

pip install pipreqs 

Update requirements.txt using "pipreqs --force"

Important Variables to note:

# Training parameters
n_training_episodes = 45000  # Total training episodes

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

Formula for Epsilon Greedy Policy:

![image](https://user-images.githubusercontent.com/59598406/193518442-e3c58769-de55-4330-89fe-92606c87e6b3.png)

Q = Q-table

α alpha = learning rate

γ gamma = discount factor

ε epsilon = action selection procedure
