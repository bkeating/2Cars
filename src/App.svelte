<script>
  /**
   * Game Architecture Overview **********************************************
   *
   * This is a dual-lane endless runner game built in Svelte where a player
   * controls two cars simultaneously. Core mechanics:
   *
   * - Each side (left/right) has two lanes the car can switch between
   * - Obstacles spawn from the top and move downward at increasing speed
   * - Players tap/click left/right side of screen to control respective cars
   * - Score increases based on circles collected, time survived and speed
   * - Game ends if either car collides with an obstacle or misses a circle
   *
   * The game uses requestAnimationFrame for smooth animation and updates obstacle
   * positions each frame based on elapsed time and current speed. Collision
   * detection uses simple radius-based checking between cars and obstacles.
   *
   * State management is handled through Svelte's reactive declarations, with the
   * main game loop updating positions and spawning obstacles based on probability
   * that increases with score/time.
   */

  import { onMount } from 'svelte';
  import * as tf from '@tensorflow/tfjs';
  import './app.css';

  /**
   * Configuration ************************************************************
   */
  const GAME_CONFIG = {
    MAX_WIDTH: 380,
    CAR_OFFSET_Y: 140,
    MIN_VERTICAL_SPACING: 100,
    SIDE_MIN_SPACING: 150,
    CROSS_SIDE_SPACING: 80,
    BASE_SPAWN_CHANCE: 0.02,
    SPEED_MULTIPLIER: 0.004,
    COLLISION_RADIUS: 30,
    LEFT_COLOR: '#f23a63',
    RIGHT_COLOR: '#05aac1'
  };

  /**
   * Game State ***************************************************************
   */
  let score = 0;
  let bestScore = parseInt(localStorage.getItem('bestScore')) || 0;
  let gameOver = false;
  let isPaused = false;
  let gameStartTime = Date.now();
  let elapsedTime = 0;
  let lastFrameTime = performance.now();
  let roadDashOffset = 0;
  let gameHeight, gameWidth, gameContainer;

  /**
   * Reactive Calculations ****************************************************
   */
  $: speed = Math.min(15, 3 + (score * 0.05) + (elapsedTime / 20000));
  $: speedDisplay = speed.toFixed(1);
  $: gameWidth = gameContainer ? Math.min(GAME_CONFIG.MAX_WIDTH, gameContainer.clientWidth) : GAME_CONFIG.MAX_WIDTH;

  $: LANE_POSITIONS = {
    left: { start: gameWidth * 0.12, alt: gameWidth * 0.38 },
    right: { start: gameWidth * 0.625, alt: gameWidth * 0.88 }
  };

  /**
   * Game Objects *************************************************************
   */
  let cars = {
    left: { lane: 'left', y: 0, x: 0 },
    right: { lane: 'left', y: 0, x: 0 }
  };
  let obstacles = [];

  // Update car positions
  $: {
    if (gameHeight) {
      ['left', 'right'].forEach(side => {
        cars[side].x = cars[side].lane === 'left' ? LANE_POSITIONS[side].start : LANE_POSITIONS[side].alt;
        cars[side].y = gameHeight - GAME_CONFIG.CAR_OFFSET_Y;
      });
    }
  }

  /**
   * Game Controls ************************************************************
   */
  const toggleCarLane = (side) => (event) => {
    event?.preventDefault();
    if (!isPaused && !gameOver) {
      cars[side].lane = cars[side].lane === 'left' ? 'right' : 'left';
    }
  };

  const handleKeydown = (event) => {
    const controls = {
      s: () => toggleCarLane('left')(),
      k: () => toggleCarLane('right')(),
      p: togglePause
    };
    controls[event.key]?.();
  };

  /**
   * Obstacle Management ******************************************************
   */
  const spawnObstacle = () => {
    if (obstacles.some(obs => obs.y < GAME_CONFIG.MIN_VERTICAL_SPACING)) return;

    const spawnChance = GAME_CONFIG.BASE_SPAWN_CHANCE + (speed - 3) * GAME_CONFIG.SPEED_MULTIPLIER;
    if (Math.random() > spawnChance) return;

    const side = Math.random() < 0.5 ? 'left' : 'right';
    const otherSide = side === 'left' ? 'right' : 'left';

    // Check spacing
    if (obstacles.some(obs =>
      (obs.side === side && Math.abs(obs.y) < GAME_CONFIG.SIDE_MIN_SPACING) ||
      (obs.side === otherSide && Math.abs(obs.y) < GAME_CONFIG.CROSS_SIDE_SPACING)
    )) return;

    const type = Math.random() < 0.5 ? 'circle' : 'square';
    const lane = Math.random() < 0.5 ? 'left' : 'right';
    const x = lane === 'left' ? LANE_POSITIONS[side].start : LANE_POSITIONS[side].alt;

    obstacles = [...obstacles, { x, y: -20, type, side }];
  };

  const checkCollision = (car, obstacle) =>
    Math.hypot(car.x - obstacle.x, car.y - obstacle.y) < GAME_CONFIG.COLLISION_RADIUS;



  /**
   * Game State Management ****************************************************
   */
  const updateBestScore = () => {
    if (score > bestScore) {
      bestScore = score;
      localStorage.setItem('bestScore', bestScore);
    }
  };


  const restartGame = () => {
    score = 0;
    gameOver = false;
    obstacles = [];
    gameStartTime = Date.now();
    elapsedTime = 0;
    roadDashOffset = 0;
    isPaused = false;
    cars = {
      left: { lane: 'left', y: gameHeight - GAME_CONFIG.CAR_OFFSET_Y, x: 0 },
      right: { lane: 'left', y: gameHeight - GAME_CONFIG.CAR_OFFSET_Y, x: 0 }
    };
    lastFrameTime = performance.now();
    requestAnimationFrame(update);
  };


  // RL Functions
  function getState() {
    const leftCarLane = cars.left.lane === 'left' ? 0 : 1;
    const rightCarLane = cars.right.lane === 'left' ? 0 : 1;
    const carY = cars.left.y;

    function getNextObstacles(side, lane, M) {
      const laneX = lane === 'left' ? LANE_POSITIONS[side].start : LANE_POSITIONS[side].alt;
      const obsInLane = obstacles.filter(obs => obs.side === side && Math.abs(obs.x - laneX) < 1);
      const aboveObs = obsInLane.filter(obs => obs.y < carY).sort((a, b) => b.y - a.y);
      const nextM = aboveObs.slice(0, M);
      const statePart = [];
      for (let i = 0; i < M; i++) {
        if (i < nextM.length) {
          statePart.push(nextM[i].y);
          statePart.push(nextM[i].type === 'circle' ? 1 : 0);
        } else {
          statePart.push(-100);
          statePart.push(-1);
        }
      }

      return statePart;
    }

    const M = 2;
    const leftLeftObs = getNextObstacles('left', 'left', M);
    const leftRightObs = getNextObstacles('left', 'right', M);
    const rightLeftObs = getNextObstacles('right', 'left', M);
    const rightRightObs = getNextObstacles('right', 'right', M);

    return [leftCarLane, rightCarLane, ...leftLeftObs, ...leftRightObs, ...rightLeftObs, ...rightRightObs];
  }


  function setLanesFromAction(action) {
    const laneConfigs = [
      ['left', 'left'],
      ['left', 'right'],
      ['right', 'left'],
      ['right', 'right']
    ];
    const [leftLane, rightLane] = laneConfigs[action];
    cars.left.lane = leftLane;
    cars.right.lane = rightLane;
  }

  class DQNAgent {
  constructor() {
    this.model = createModel();
    this.replayBuffer = [];
    this.epsilon = 1.0;
    this.epsilonMin = 0.01;
    this.epsilonDecay = 0.995;
    this.gamma = 0.99;
    this.batchSize = 32;
    this.maxBufferSize = 10000;
  }

  async act(state) {
    if (Math.random() < this.epsilon) {
      return Math.floor(Math.random() * 4);
    }
    const stateTensor = tf.tensor2d([state]);
    const qValues = this.model.predict(stateTensor);
    const action = qValues.argMax(1).dataSync()[0];
    stateTensor.dispose();
    qValues.dispose();
    return action;
  }

  async remember(state, action, reward, nextState, done) {
    this.replayBuffer.push([state, action, reward, nextState, done]);
    if (this.replayBuffer.length > this.maxBufferSize) {
      this.replayBuffer.shift();
    }
  }

  async replay() {
    if (this.replayBuffer.length < this.batchSize) return;
    const batch = this.sample(this.replayBuffer, this.batchSize);
    const states = batch.map(b => b[0]);
    const actions = batch.map(b => b[1]);
    const rewards = batch.map(b => b[2]);
    const nextStates = batch.map(b => b[3]);
    const dones = batch.map(b => b[4]);

    const stateTensors = tf.tensor2d(states);
    const nextStateTensors = tf.tensor2d(nextStates);
    const qValues = this.model.predict(stateTensors);
    const nextQValues = this.model.predict(nextStateTensors);

    const targets = qValues.arraySync();
    const nextQMax = nextQValues.max(1).dataSync();
    for (let i = 0; i < batch.length; i++) {
      const target = dones[i] ? rewards[i] : rewards[i] + this.gamma * nextQMax[i];
      targets[i][actions[i]] = target;
    }

    await this.model.fit(stateTensors, tf.tensor2d(targets), { epochs: 1, verbose: 0 });
    stateTensors.dispose();
    nextStateTensors.dispose();
    qValues.dispose();
    nextQValues.dispose();

    if (this.epsilon > this.epsilonMin) {
      this.epsilon *= this.epsilonDecay;
    }
  }

  sample(array, size) {
    const shuffled = array.slice().sort(() => Math.random() - 0.5);
    return shuffled.slice(0, size);
  }
}


function createModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [18] }));
  model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 4, activation: 'linear' }));
  model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
  return model;
}

  let agent;
  onMount(async () => {
    agent = new DQNAgent();
    console.log('Agent:', agent);
    const updateHeight = () => {
      if (gameContainer) {
        gameHeight = window.innerHeight;
        gameContainer.style.height = `${gameHeight}px`;
      }
    };
    updateHeight();
    window.addEventListener('resize', updateHeight);
    window.addEventListener('keydown', handleKeydown);
    requestAnimationFrame(async (timestamp) => await update(timestamp));
    return () => {
      window.removeEventListener('resize', updateHeight);
      window.removeEventListener('keydown', handleKeydown);
      gameOver = true;
    };
  });

  let totalRewards = 0;
  let reward = 0;
  let previousAction = null;
  const update = async (timestamp) => {
    if (gameOver || isPaused) return;

    const deltaTime = timestamp - lastFrameTime;
    lastFrameTime = timestamp;
    elapsedTime = Date.now() - gameStartTime;
    roadDashOffset -= speed * (deltaTime / 16);

    let done = false;

    if (!gameOver && !isPaused && agent) {
      const state = getState();
      const action = await agent.act(state);

      const currentLanes = [
        cars.left.lane === 'left' ? 0 : 1,
        cars.right.lane === 'left' ? 0 : 1
      ];
      setLanesFromAction(action); // get small penalty for changing lanes
      const newLanes = [
        cars.left.lane === 'left' ? 0 : 1,
        cars.right.lane === 'left' ? 0 : 1
      ];


      // Determine if a lane switch occurred
      let switchPenalty = 0;
      if (previousAction !== null) {
        const prevLanes = [
          previousAction < 2 ? 0 : 1, // Left car: 0 or 1 based on action 0/1 vs 2/3
          previousAction % 2 === 0 ? 0 : 1 // Right car: 0 or 1 based on action 0/2 vs 1/3
        ];
        if (currentLanes[0] !== prevLanes[0] || currentLanes[1] !== prevLanes[1]) {
          const leftCar = cars.left;
          const rightCar = cars.right;
          const leftObstacles = obstacles.filter(obs => obs.side === 'left' && obs.y > leftCar.y - 100 && obs.y < leftCar.y);
          const rightObstacles = obstacles.filter(obs => obs.side === 'right' && obs.y > rightCar.y - 100 && obs.y < rightCar.y);
          const hasBenefit = leftObstacles.some(obs => obs.type === 'circle' && checkCollision(leftCar, obs)) ||
                             rightObstacles.some(obs => obs.type === 'circle' && checkCollision(rightCar, obs)) ||
                             leftObstacles.some(obs => obs.type === 'square' && !checkCollision(leftCar, obs)) ||
                             rightObstacles.some(obs => obs.type === 'square' && !checkCollision(rightCar, obs));
          switchPenalty = hasBenefit ? -0.1 : -1.0; // Smaller penalty if beneficial, larger otherwise
        }
      }

      obstacles = obstacles.filter(obs => {
        obs.y += speed * (deltaTime / 16);
        const relevantCar = cars[obs.side];
        if (checkCollision(relevantCar, obs)) {
          if (obs.type === 'square') {
            reward -= 10;
            done = true;
            return true;
          } else {
            score++;
            reward += 2;
            return false;
          }
        }
        if (obs.y >= gameHeight) {
          if (obs.type === 'circle') {
            reward -= 10;
            done = true;
          }
          return false;
        }
        return true;
      });

      reward += 0.01; // Survival reward
      reward += switchPenalty;

      const nextState = getState();
      await agent.remember(state, action, reward, nextState, done);
      await agent.replay();

      if (done) {
        gameOver = true;
        totalRewards += reward;
        console.log(`Episode Reward: ${reward} - Total Reward: ${totalRewards}`);
        reward = 0; // Reset for next episode
        updateBestScore();
        setTimeout(restartGame, 1000); // Restart after 1 second for automated training
      }
    }

    spawnObstacle();
    if (!gameOver) requestAnimationFrame(async (timestamp) => await update(timestamp));
  };

  const togglePause = () => {
    isPaused = !isPaused;
    if (!isPaused) {
      lastFrameTime = performance.now();
      requestAnimationFrame(async (timestamp) => await update(timestamp));
    }
  };
</script>

<!-- Game Container & Controls ******************************************** -->
<div class="game-container" bind:this={gameContainer}>
  <button
    class="pause-button {!isPaused && !gameOver ? 'opacity-50' : ''}"
    on:click={togglePause}
    disabled={gameOver}
  >
    {isPaused ? 'Resume' : 'Pause'}
  </button>

  <div class="absolute score-container">
    <div class="score">{score} / {speedDisplay}</div>
  </div>

  <!-- Game Over Overlay ************************************************** -->
  {#if gameOver}
    <div class="absolute overlay flex-center"></div>
    <div class="absolute game-over">
      GAME OVER
      <div class="final-scores">
        <div>Score: {score}</div>
        <div>Best: {bestScore}</div>
      </div>
      <button class="restart-button" on:click={restartGame}>Restart</button>
    </div>
  {/if}

  <!-- Game Canvas ******************************************************** -->
  <svg width={gameWidth} height={gameHeight}>
    {#each ['left', 'right'] as side}
      <!-- Lanes -->
      <rect
        x={side === 'left' ? 0 : gameWidth/2}
        y="0"
        width={gameWidth/2}
        height={gameHeight}
        fill="#253479"
      />
      <!-- Lane borders -->
      <line
        x1={side === 'left' ? 0 : gameWidth/2}
        y1="0"
        x2={side === 'left' ? 0 : gameWidth/2}
        y2={gameHeight}
        stroke="#8297ee"
        stroke-width="2"
      />
      <line
        x1={side === 'left' ? gameWidth/2 : gameWidth}
        y1="0"
        x2={side === 'left' ? gameWidth/2 : gameWidth}
        y2={gameHeight}
        stroke="#8297ee"
        stroke-width="2"
      />
      <!-- Center lane divider -->
      <line
        x1={side === 'left' ? gameWidth * 0.25 : gameWidth * 0.75}
        y1="0"
        x2={side === 'left' ? gameWidth * 0.25 : gameWidth * 0.75}
        y2={gameHeight}
        stroke="#7a95ec"
        stroke-dasharray="20,20"
        stroke-dashoffset={roadDashOffset}
        stroke-width="2"
      />

      <!-- Cars -->
      <g>
        <rect class="car" x={cars[side].x - 15} y={cars[side].y - 25} width="30" height="50" rx="5" ry="5" fill={side === 'left' ? GAME_CONFIG.LEFT_COLOR : GAME_CONFIG.RIGHT_COLOR} />
        <rect class="car" x={cars[side].x - 10} y={cars[side].y - 20} width="20" height="15" rx="2" ry="2" fill="#333" opacity="0.7" />
        <rect class="car" x={cars[side].x - 10} y={cars[side].y + 5} width="20" height="15" rx="2" ry="2" fill="#333" opacity="0.7" />
      </g>
    {/each}

    <!-- Obstacles -->
    {#each obstacles as obstacle}
      <g>
        {#if obstacle.type === 'circle'}
          <circle cx={obstacle.x} cy={obstacle.y} r="15" fill={obstacle.side === 'left' ? GAME_CONFIG.LEFT_COLOR : GAME_CONFIG.RIGHT_COLOR} />
          <circle cx={obstacle.x} cy={obstacle.y} r="13" fill="none" stroke="white" stroke-width="2" />
        {:else}
          <rect x={obstacle.x - 15} y={obstacle.y - 15} width="30" height="30" rx="3" ry="3" fill={obstacle.side === 'left' ? GAME_CONFIG.LEFT_COLOR : GAME_CONFIG.RIGHT_COLOR} />
          <rect x={obstacle.x - 13} y={obstacle.y - 13} width="26" height="26" rx="2" ry="2" fill="none" stroke="white" stroke-width="2" />
        {/if}
      </g>
    {/each}
  </svg>

  <!-- Touch Controls ***************************************************** -->
  <div class="absolute touch-left"
    on:touchstart={toggleCarLane('left')}
    on:mousedown={toggleCarLane('left')}
    role="button"
    tabindex="0"
    aria-label="Control left car"
  ></div>
  <div class="absolute touch-right"
    on:touchstart={toggleCarLane('right')}
    on:mousedown={toggleCarLane('right')}
    role="button"
    tabindex="0"
    aria-label="Control right car"
  ></div>
</div>
