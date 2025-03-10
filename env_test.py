import gymnasium as gym
import numpy as np
import random

# Create the FrozenLake environment
env = gym.make('FrozenLake-v1', desc=None, map_name='8x8', is_slippery=False)  

EPISODES = 10000
GAMMA = 0.9
EPSILON = 0.9
ALPHA = 0.1
EPSILON_DECAY = 0.95

Q = np.zeros((env.observation_space.n, env.action_space.n))
returns = {(s, a): [] for s in range(env.observation_space.n) for a in range(env.action_space.n)}

def get_epsilon(t, epsilon_max=1.0, epsilon_min=0.01, decay_rate=0.005):
    """
    Get the exploration rate using exponential decay.

    Args:
    t: The current timestep or episode number.
    epsilon_max: The maximum epsilon value (exploration rate).
    epsilon_min: The minimum epsilon value (exploration rate).
    decay_rate: The rate at which epsilon decays.

    Returns:
    The exploration rate for the given timestep.
    """
    return epsilon_min + (epsilon_max - epsilon_min) * np.exp(-decay_rate * t)

# Take a few random steps and render the environment after each step
for episode in range(EPISODES):
    state, info = env.reset()
    episode_data = []
    done = False
    while not done:
        if random.uniform(0,1) < EPSILON:
            action = env.action_space.sample()
            print("random action")
        else:
            action = np.argmax(Q[state])
        next_state, reward, done, truncated, _ = env.step(action)
        #print(f"Action: {action}, Next State: {next_state}, Reward: {reward}, Done: {done}")
        
        if done and reward == 0:
            reward = -1
        episode_data.append((state, int(action), reward))
        state = next_state

    G = 0
    visited_state_action_pairs = set()
    first_occurrence = {}
    
    print(episode_data)
    for t, (state, action, reward) in enumerate(episode_data):
        if (state, action) not in first_occurrence:
            first_occurrence[(state, action)] = t
    for t in reversed(range(len(episode_data))):
        state, action, reward = episode_data[t]
        G = GAMMA * G + reward
        if t == first_occurrence[(state, action)]:
            returns[(state,action)].append(G)
            Q[(state, action)] = np.mean(returns[(state,action)])
    
    print(Q)
    EPSILON = max(0.01, EPSILON * EPSILON_DECAY)

env.close()  # Close the environment when done
