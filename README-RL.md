# 2Cars Game with Reinforcement Learning

Experimenting with reinforcement learning AI that can play the game automatically. The AI is trained using Playwright to interact with the game in a browser, just like a human player.

## Overview

The reinforcement learning system consists of:

1. **Deep Q-Network (DQN)** - A neural network that learns to predict the best actions to take in any game state
2. **Playwright Integration** - Automated browser interaction to control and observe the game
3. **Experience Replay** - Storage and retraining on past experiences to improve learning efficiency
4. **Training Script** - Controls the training process over many game episodes
5. **Play Script** - Runs the trained model to showcase its performance

## Setup and Installation

1. Install the dependencies:

```bash
bun install
bunx playwright install chromium
```

## Training the AI

1. Start the development server:

```bash
bun run dev
```

2. In a separate terminal, run the training script:

```bash
bun run train-ai
```

The training process:
- Opens a browser window with the game
- Runs 500 episodes (configurable in `playwright-training.js`)
- Saves the trained model in the `models/rl-model` directory
- Outputs training progress in the console

Training parameters can be adjusted in `playwright-training.js`:
- `TRAINING_EPISODES` - Number of game episodes to train for
- `EPSILON_START` - Initial exploration rate
- `EPSILON_DECAY` - Rate at which exploration decreases
- `LEARNING_RATE` - Neural network learning rate
- `GAMMA` - Discount factor for future rewards

## Running the Trained AI

Once training is complete, you can watch the AI play:

```bash
bun run play-ai
```

This will:
- Load the trained model
- Open a browser with the game
- Have the AI play automatically for 5 minutes
- Output game scores in the console

## Model Architecture

The DQN model consists of:
- Input layer: 18 neurons (game state representation)
- Hidden layer 1: 64 neurons with ReLU activation
- Hidden layer 2: 32 neurons with ReLU activation
- Output layer: 4 neurons (one for each possible action)

The state representation includes:
- Position of both cars (which lane they're in)
- Position and type of nearest obstacles in each lane
- The model outputs Q-values for 4 possible actions:
  - Both cars in left lanes
  - Left car in left lane, right car in right lane
  - Left car in right lane, right car in left lane
  - Both cars in right lanes

## How the AI Learns

1. **Exploration vs. Exploitation**:
   - Initially, the AI mostly takes random actions to explore the game (high epsilon)
   - Over time, it increasingly relies on its learned Q-values (decreasing epsilon)

2. **Reward Structure**:
   - +10 points for collecting circles
   - +0.1 points for surviving each step
   - -1.0 points for unnecessary lane changes
   - -100 points for game over

3. **Experience Replay**:
   - Stores experiences (state, action, reward, next state, done)
   - Randomly samples from experiences to break correlations between consecutive samples
   - Updates Q-values based on the Bellman equation

## Future Improvements

Potential enhancements:
- Prioritized experience replay
- Double DQN architecture
- More sophisticated state representation
- Convolutional neural networks for raw pixel input
- Multi-step returns for faster learning

## Troubleshooting

- If the AI is not learning well, try adjusting the reward structure or increasing the number of training episodes
- If the browser crashes during training, reduce the action frequency or batch size
- Ensure the game window remains in focus during training for consistent performance
