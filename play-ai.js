// Script to run the trained RL model on the 2Cars game
import { chromium } from 'playwright';
import * as tf from '@tensorflow/tfjs-node';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODEL_PATH = path.join(__dirname, 'models', 'rl-model');
const MODEL_JSON_PATH = path.join(MODEL_PATH, 'model.json');

// Configure the game options
const GAME_SPEED = 100; // ms between actions
const PLAY_TIME = 300000; // 5 minutes in milliseconds

async function loadModel() {
  if (fs.existsSync(MODEL_JSON_PATH)) {
    const model = await tf.loadLayersModel(`file://${MODEL_PATH}/model.json`);
    console.log('Model loaded successfully from', MODEL_PATH);
    return model;
  } else {
    throw new Error(`No model found at ${MODEL_PATH}. Please train a model first.`);
  }
}

async function predict(model, state) {
  const stateTensor = tf.tensor2d([state]);
  const qValues = model.predict(stateTensor);
  const action = qValues.argMax(1).dataSync()[0];

  stateTensor.dispose();
  qValues.dispose();

  return action;
}

async function playGame() {
  console.log('Loading model...');
  const model = await loadModel();

  console.log('Starting game...');
  const browser = await chromium.launch({ headless: false });
  const context = await browser.newContext();
  const page = await context.newPage();

  try {
    // Navigate to the game
    await page.goto('http://localhost:5173/');
    console.log('Game loaded');

    // Wait for game container to be visible
    await page.waitForSelector('.game-container');

    // Helper functions to interact with the game
    const getGameState = async () => {
      return await page.evaluate(() => {
        return window.getState();
      });
    };

    const performAction = async (action) => {
      await page.evaluate((action) => {
        window.setLanesFromAction(action);
      }, action);
    };

    const isGameOver = async () => {
      return await page.evaluate(() => {
        return document.querySelector('.game-over') !== null;
      });
    };

    const getScore = async () => {
      return await page.evaluate(() => {
        const scoreElement = document.querySelector('.score');
        if (scoreElement) {
          const text = scoreElement.textContent;
          const match = text.match(/(\d+)/);
          return match ? parseInt(match[1]) : 0;
        }
        return 0;
      });
    };

    const restartGame = async () => {
      await page.evaluate(() => {
        const restartButton = document.querySelector('.restart-button');
        if (restartButton) {
          restartButton.click();
        }
      });

      await page.waitForFunction(() => {
        return document.querySelector('.game-over') === null;
      }, { timeout: 5000 }).catch(() => {
        console.log('Game restart timeout');
      });
    };

    // Start playing the game
    console.log('Starting to play...');

    const startTime = Date.now();
    let gameCount = 0;
    let highScore = 0;

    while (Date.now() - startTime < PLAY_TIME) {
      // Check if game is over
      const gameOverStatus = await isGameOver();
      if (gameOverStatus) {
        const score = await getScore();
        console.log(`Game ${++gameCount} over with score: ${score}`);

        if (score > highScore) {
          highScore = score;
          console.log(`New high score: ${highScore}`);
        }

        await restartGame();
        console.log('Game restarted');
        await page.waitForTimeout(1000); // Give game time to fully restart
        continue;
      }

      // Get current state
      const state = await getGameState();

      // Choose action
      const action = await predict(model, state);

      // Perform action
      await performAction(action);

      // Wait before next action
      await page.waitForTimeout(GAME_SPEED);
    }

    console.log(`Playing finished after ${gameCount} games`);
    console.log(`Highest score achieved: ${highScore}`);

  } catch (error) {
    console.error('Error playing game:', error);
  } finally {
    await browser.close();
  }
}

// Run the game
playGame().catch(console.error);
