import { chromium } from 'playwright';
import * as tf from '@tensorflow/tfjs-node';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODEL_PATH = path.join(__dirname, 'models', 'rl-model');
const MODEL_JSON_PATH = path.join(MODEL_PATH, 'model.json');

// Constants
const TRAINING_EPISODES = 1000;      // Total number of episodes
const EPSILON_START = 1.0;           // Initial epsilon for exploration
const EPSILON_MIN = 0.01;            // Minimum epsilon value
const EPSILON_DECAY = 0.995;         // Epsilon decay rate per episode
const LEARNING_RATE = 0.001;         // Learning rate for the optimizer
const GAMMA = 0.95;                  // Discount factor for future rewards
const BATCH_SIZE = 32;               // Number of experiences per training batch (increased from 8)
const MEMORY_SIZE = 10000;           // Replay buffer size (increased from 1000)
const STATE_SIZE = 18;               // Size of the state vector
const ACTION_SIZE = 4;               // Number of possible actions
const SAVE_INTERVAL = 5;             // Episodes between model saves
const TRAINING_INTERVAL = 8;         // Steps between training iterations
const MAX_STEPS_PER_EPISODE = 500;   // Max steps per episode

// Utility functions
const gc = () => { if (global.gc) global.gc(); };
const cleanupTensors = (tensors) => {
  if (Array.isArray(tensors)) tensors.forEach(t => t?.dispose?.());
  else tensors?.dispose?.();
};

// Neural network model creation
function createModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 24, activation: 'relu', inputShape: [STATE_SIZE] }));
  model.add(tf.layers.dense({ units: 12, activation: 'relu' }));
  model.add(tf.layers.dense({ units: ACTION_SIZE, activation: 'linear' }));
  model.compile({ optimizer: tf.train.adam(LEARNING_RATE), loss: 'meanSquaredError' });
  return model;
}

// Replay buffer for experience replay
class ReplayBuffer {
  constructor(maxSize) {
    this.maxSize = maxSize;
    this.buffer = [];
  }
  add(state, action, reward, nextState, done) {
    if (this.buffer.length >= this.maxSize) this.buffer.shift();
    this.buffer.push([state, action, reward, nextState, done]);
  }
  sample(batchSize) {
    const indices = new Set();
    while (indices.size < Math.min(batchSize, this.buffer.length)) {
      indices.add(Math.floor(Math.random() * this.buffer.length));
    }
    return Array.from(indices).map(i => this.buffer[i]);
  }
  clear() { this.buffer = []; gc(); }
  size() { return this.buffer.length; }
}

// DQN Agent
class DQNAgent {
  constructor(stateSize, actionSize) {
    this.stateSize = stateSize;
    this.actionSize = actionSize;
    this.memory = new ReplayBuffer(MEMORY_SIZE);
    this.gamma = GAMMA;
    this.epsilon = EPSILON_START;
    this.epsilonMin = EPSILON_MIN;
    this.epsilonDecay = EPSILON_DECAY;
    this.model = createModel();
    this.targetModel = createModel();
    this.updateTargetModel();
    this.lastTrainingTime = Date.now();
    this.losses = []; // For tracking training loss
  }

  updateTargetModel() {
    this.targetModel.setWeights(this.model.getWeights());
  }

  act(state) {
    if (Math.random() < this.epsilon) return Math.floor(Math.random() * this.actionSize);
    const stateTensor = tf.tensor2d([state]);
    const qValues = this.model.predict(stateTensor);
    const action = tf.argMax(qValues, 1).dataSync()[0];
    cleanupTensors([stateTensor, qValues]);
    return action;
  }

  remember(state, action, reward, nextState, done) {
    this.memory.add(state, action, reward, nextState, done);
  }

  async replay(batchSize) {
    if (this.memory.size() < batchSize) return;
    const now = Date.now();
    if (now - this.lastTrainingTime < 200) return;
    this.lastTrainingTime = now;

    const batch = this.memory.sample(batchSize);
    const tensors = [];
    try {
      const states = tf.tensor2d(batch.map(exp => exp[0])); tensors.push(states);
      const nextStates = tf.tensor2d(batch.map(exp => exp[3])); tensors.push(nextStates);
      const qValues = this.model.predict(states); tensors.push(qValues);
      const futureQValues = this.targetModel.predict(nextStates); tensors.push(futureQValues);
      const targets = qValues.arraySync();

      for (let i = 0; i < batch.length; i++) {
        const [, action, reward, , done] = batch[i];
        const futureQ = done ? 0 : futureQValues.arraySync()[i][futureQValues.argMax(1).arraySync()[i]];
        targets[i][action] = reward + this.gamma * futureQ;
      }

      const targetTensor = tf.tensor2d(targets); tensors.push(targetTensor);
      const history = await this.model.fit(states, targetTensor, {
        epochs: 1,
        verbose: 0,
        batchSize: batchSize // Use full batch size
      });
      this.losses.push(history.history.loss[0]); // Log loss
    } catch (error) {
      console.error('Error in replay():', error);
    } finally {
      cleanupTensors(tensors);
      gc();
    }
  }

  async saveModel() {
    if (!fs.existsSync(MODEL_PATH)) fs.mkdirSync(MODEL_PATH, { recursive: true });
    await this.model.save(`file://${MODEL_PATH}`);
    console.log(`Model saved to ${MODEL_PATH}`);
  }

  async loadModel() {
    if (fs.existsSync(MODEL_JSON_PATH)) {
      this.model = await tf.loadLayersModel(`file://${MODEL_PATH}/model.json`);
      this.targetModel = await tf.loadLayersModel(`file://${MODEL_PATH}/model.json`);
      this.model.compile({ optimizer: tf.train.adam(LEARNING_RATE), loss: 'meanSquaredError' });
      this.targetModel.compile({ optimizer: tf.train.adam(LEARNING_RATE), loss: 'meanSquaredError' });
      this.epsilon = this.epsilonMin;
      return true;
    }
    return false;
  }
}

// Training function
async function trainAgent() {
  await tf.setBackend('tensorflow');
  await tf.ready();
  console.log('TensorFlow backend initialized:', tf.getBackend());

  const browser = await chromium.launch({ headless: false });
  const context = await browser.newContext();
  const page = await context.newPage();

  try {
    await page.goto('http://localhost:5173/');
    await page.waitForSelector('.game-container');

    const agent = new DQNAgent(STATE_SIZE, ACTION_SIZE);
    try {
      await agent.loadModel();
    } catch (e) {
      console.log('Starting fresh training');
    }

    const getGameState = async () => await page.evaluate(() => window.getState());
    const performAction = async (action) => await page.evaluate((action) => window.setLanesFromAction(action), action);
    const isGameOver = async () => await page.evaluate(() => document.querySelector('.game-over') !== null);
    const getScore = async () => await page.evaluate(() => {
      const scoreElement = document.querySelector('.score');
      return scoreElement ? parseInt(scoreElement.textContent.match(/(\d+)/)[0]) || 0 : 0;
    });
    const restartGame = async () => {
      await page.evaluate(() => {
        const restartButton = document.querySelector('.restart-button');
        if (restartButton) restartButton.click();
      });
      await page.waitForFunction(() => !document.querySelector('.game-over'), { timeout: 5000 })
        .catch(() => console.log('Game restart timeout'));
    };

    for (let episode = 0; episode < TRAINING_EPISODES; episode++) {
      console.log(`Episode ${episode + 1}/${TRAINING_EPISODES} - Epsilon: ${agent.epsilon.toFixed(4)}`);
      if (await isGameOver()) {
        await restartGame();
        await page.waitForTimeout(500);
      }

      let totalReward = 0;
      let steps = 0;
      let gameOver = false;
      let previousScore = await getScore();

      while (!gameOver && steps < MAX_STEPS_PER_EPISODE) {
        try {
          const state = await getGameState();
          const action = await agent.act(state);
          await performAction(action);
          await page.waitForTimeout(50);

          const nextState = await getGameState();
          const currentScore = await getScore();
          const scoreDiff = currentScore - previousScore;

          let reward = 0.1 + 2 * scoreDiff; // Added living reward
          gameOver = await isGameOver();
          if (gameOver) reward -= 10;

          agent.remember(state, action, reward, nextState, gameOver);
          if (steps % TRAINING_INTERVAL === 0) await agent.replay(BATCH_SIZE);

          totalReward += reward;
          previousScore = currentScore;
          steps++;

          if (steps % 100 === 0) gc();
        } catch (error) {
          console.error('Error during game step:', error);
          break;
        }
      }

      agent.updateTargetModel();

      // Log average loss
      let avgLoss = 'N/A';
      if (agent.losses.length > 0) {
        avgLoss = (agent.losses.reduce((a, b) => a + b, 0) / agent.losses.length).toFixed(4);
        agent.losses = [];
      }

      // Decay epsilon per episode
      if (agent.epsilon > agent.epsilonMin) agent.epsilon *= agent.epsilonDecay;

      if ((episode + 1) % SAVE_INTERVAL === 0) {
        await agent.saveModel();
        console.log(`Model saved at episode ${episode + 1}`);
      }

      console.log(`Episode ${episode + 1}/${TRAINING_EPISODES} - Score: ${previousScore}, Steps: ${steps}, Total Reward: ${totalReward.toFixed(2)}, Epsilon: ${agent.epsilon.toFixed(4)}, Avg Loss: ${avgLoss}`);
      gc();
      await page.waitForTimeout(100);
    }

    await agent.saveModel();
    console.log('Training completed!');
  } catch (error) {
    console.error('Training error:', error);
  } finally {
    await browser.close();
  }
}

process.env.NODE_OPTIONS = '--max-old-space-size=8192';
trainAgent().catch(err => {
  console.error('Fatal training error:', err);
  process.exit(1);
});
