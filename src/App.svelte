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
  import './app.css';

  /**
   * Configuration ************************************************************
   */
  const GAME_CONFIG = {
    MAX_WIDTH: 380,
    CAR_OFFSET_Y: 140,
    MIN_VERTICAL_SPACING: 92,
    SIDE_MIN_SPACING: 138,
    CROSS_SIDE_SPACING: 72,
    BASE_SPAWN_CHANCE: 0.026,
    SPEED_MULTIPLIER: 0.0052,
    REF_H: 560,
    CAR_HW: 15,
    CAR_HH: 25,
    CIRCLE_R: 15,
    SQUARE_HW: 15,
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
  $: speed = Math.min(22, 3 + score * 0.062 + elapsedTime / 14000);
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
    left: { lane: 'left', y: 0, x: 0, tx: 0 },
    right: { lane: 'left', y: 0, x: 0, tx: 0 }
  };
  let obstacles = [];
  let nextObsId = 0;

  // Lane targets (tx) and vertical position from layout; x lerps in update().
  $: if (gameHeight) {
    for (const side of ['left', 'right']) {
      const c = cars[side];
      c.tx = c.lane === 'left' ? LANE_POSITIONS[side].start : LANE_POSITIONS[side].alt;
      c.y = gameHeight - GAME_CONFIG.CAR_OFFSET_Y;
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

    let spawnChance = (GAME_CONFIG.BASE_SPAWN_CHANCE + (speed - 3) * GAME_CONFIG.SPEED_MULTIPLIER) * Math.min(1.38, 0.82 + gameHeight / GAME_CONFIG.REF_H * 0.45);
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

    obstacles = [...obstacles, { id: nextObsId++, x, y: -22, type, side }];
  };

  /**
   * Collision (car center cx,cy vs obstacle center ox,oy) **********************
   */
  const hitCircle = (cx, cy, ox, oy) => {
    const L = cx - GAME_CONFIG.CAR_HW,
      T = cy - GAME_CONFIG.CAR_HH,
      R = cx + GAME_CONFIG.CAR_HW,
      B = cy + GAME_CONFIG.CAR_HH;
    const nx = ox < L ? L : ox > R ? R : ox;
    const ny = oy < T ? T : oy > B ? B : oy;
    const dx = ox - nx,
      dy = oy - ny,
      r = GAME_CONFIG.CIRCLE_R;
    return dx * dx + dy * dy < r * r;
  };

  const hitSquare = (cx, cy, ox, oy) =>
    Math.abs(cx - ox) < GAME_CONFIG.CAR_HW + GAME_CONFIG.SQUARE_HW &&
    Math.abs(cy - oy) < GAME_CONFIG.CAR_HH + GAME_CONFIG.SQUARE_HW;

  /**
   * Game Loop ****************************************************************
   */
  const update = (timestamp) => {
    if (gameOver || isPaused) return;

    const deltaTime = timestamp - lastFrameTime;
    lastFrameTime = timestamp;
    const dt = deltaTime / 16;
    const laneBlend = 1 - Math.exp(-deltaTime / 165);
    elapsedTime = Date.now() - gameStartTime;
    roadDashOffset -= speed * dt;

    for (const side of ['left', 'right']) {
      const c = cars[side];
      c.x += (c.tx - c.x) * laneBlend;
    }

    obstacles = obstacles.filter(obs => {
      obs.y += speed * dt;
      const c = cars[obs.side];

      if (obs.type === 'square' ? hitSquare(c.x, c.y, obs.x, obs.y) : hitCircle(c.x, c.y, obs.x, obs.y)) {
        if (obs.type === 'square') {
          updateBestScore();
          gameOver = true;
          return true;
        }
        score++;
        return false;
      }

      if (obs.y >= gameHeight) {
        if (obs.type === 'circle') {
          updateBestScore();
          gameOver = true;
        }
        return false;
      }
      return true;
    });

    spawnObstacle();
    if (!gameOver) requestAnimationFrame(update);
  };

  /**
   * Game State Management ****************************************************
   */
  const updateBestScore = () => {
    if (score > bestScore) {
      bestScore = score;
      localStorage.setItem('bestScore', bestScore);
    }
  };

  const togglePause = () => {
    isPaused = !isPaused;
    if (!isPaused) {
      lastFrameTime = performance.now();
      requestAnimationFrame(update);
    }
  };

  const restartGame = () => {
    score = 0;
    gameOver = false;
    obstacles = [];
    nextObsId = 0;
    gameStartTime = Date.now();
    elapsedTime = 0;
    roadDashOffset = 0;
    isPaused = false;
    cars = { left: { lane: 'left', y: 0, x: 0, tx: 0 }, right: { lane: 'left', y: 0, x: 0, tx: 0 } };
    for (const side of ['left', 'right']) {
      const t = cars[side].lane === 'left' ? LANE_POSITIONS[side].start : LANE_POSITIONS[side].alt;
      cars[side].y = gameHeight - GAME_CONFIG.CAR_OFFSET_Y;
      cars[side].tx = cars[side].x = t;
    }
    lastFrameTime = performance.now();
    requestAnimationFrame(update);
  };

  /**
   * Initialization ***********************************************************
   */
  onMount(() => {
    const updateHeight = () => {
      if (gameContainer) {
        gameHeight = window.innerHeight;
        gameContainer.style.height = `${gameHeight}px`;
        for (const side of ['left', 'right']) {
          const c = cars[side];
          c.tx = c.lane === 'left' ? LANE_POSITIONS[side].start : LANE_POSITIONS[side].alt;
          c.y = gameHeight - GAME_CONFIG.CAR_OFFSET_Y;
          c.x = c.tx;
        }
      }
    };

    updateHeight();
    window.addEventListener('resize', updateHeight);
    window.addEventListener('keydown', handleKeydown);
    requestAnimationFrame(update);

    return () => {
      window.removeEventListener('resize', updateHeight);
      window.removeEventListener('keydown', handleKeydown);
      gameOver = true;
    };
  });
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
    {#each ['left', 'right'] as side (side)}
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
    {#each obstacles as obstacle (obstacle.id)}
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
