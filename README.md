# FrozenLake Q-Learning with Monte Carlo Updates

This repository contains an implementation of a reinforcement learning algorithm using Q-learning with Monte Carlo updates to solve the FrozenLake environment from OpenAI Gymnasium.

## Environment

The environment used is the `FrozenLake-v1` with an 8x8 grid, where the agent must navigate from the starting position to the goal while avoiding holes.

- **Environment:** `FrozenLake-v1`
- **Grid Size:** `8x8`
- **Slippery:** `False`

## Implementation Details

- The agent follows an **epsilon-greedy policy** for action selection.
- **Monte Carlo updates** are used to estimate the Q-values.
- The exploration rate (**epsilon**) decays over time to balance exploration and exploitation.
- The algorithm stores returns for state-action pairs and updates Q-values based on their average.

## Parameters

- `EPISODES = 10000` : Number of training episodes
- `GAMMA = 0.9` : Discount factor for future rewards
- `EPSILON = 0.9` : Initial exploration rate
- `ALPHA = 0.1` : Learning rate
- `EPSILON_DECAY = 0.95` : Decay rate for exploration probability

## Installation

To run the code, install the required dependencies:

```bash
pip install gymnasium numpy
```

## Running the Code

Execute the script to train the agent:

```bash
python frozen_lake_q_learning.py
```

## Q-Learning Algorithm

1. Initialize Q-values and returns.
2. For each episode:
   - Start from the initial state.
   - Select an action using an epsilon-greedy policy.
   - Execute the action and observe the next state and reward.
   - Store the state-action-reward sequence.
   - Perform Monte Carlo updates on the Q-values.
   - Decay epsilon to reduce exploration over time.
3. Repeat for a defined number of episodes.

## Author

This project was developed as part of reinforcement learning experiments using Gymnasium.

## License

This project is released under the MIT License.

