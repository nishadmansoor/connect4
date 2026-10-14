// Connect-4 vs the trained AlphaZero network. Depends on c4.js (window.C4).
(function () {
  'use strict';
  const { ROWS, COLS, emptyBoard, topRow, validMoves, drop, winningLine, terminal, makeNet, think } = window.C4;

  const $ = (id) => document.getElementById(id);
  const boardEl = $('board'), statusEl = $('status'), barsEl = $('bars');

  let net = null;
  let board = emptyBoard();
  let toPlay = 1;          // player 1 (red) always moves first
  let human = 1;
  let sims = 250;
  let over = true;
  let gameId = 0;          // bumps on new game so a stale AI search can't play into it
  let lastMove = -1;

  // ---------------------------------------------------------------- DOM setup

  const discs = [];        // discs[r][c]
  for (let r = 0; r < ROWS; r++) discs.push([]);
  for (let c = 0; c < COLS; c++) {
    const col = document.createElement('div');
    col.className = 'col';
    col.setAttribute('role', 'button');
    col.setAttribute('aria-label', 'Drop in column ' + (c + 1));
    col.addEventListener('click', () => humanMove(c));
    for (let r = 0; r < ROWS; r++) {
      const cell = document.createElement('div');
      cell.className = 'cell';
      const d = document.createElement('div');
      d.className = 'disc';
      cell.appendChild(d);
      col.appendChild(cell);
      discs[r][c] = d;
    }
    boardEl.appendChild(col);
  }
  const bars = [];
  for (let c = 0; c < COLS; c++) {
    const b = document.createElement('div');
    b.className = 'bar';
    barsEl.appendChild(b);
    bars.push(b);
  }

  function segmented(id, onChange) {
    const el = $(id);
    el.addEventListener('click', (e) => {
      const btn = e.target.closest('button');
      if (!btn) return;
      el.querySelectorAll('button').forEach((b) => b.classList.toggle('on', b === btn));
      onChange(btn.dataset.v);
    });
  }
  segmented('first', (v) => { human = v === 'human' ? 1 : 2; newGame(); });
  segmented('level', (v) => { sims = +v; });
  $('new-game').addEventListener('click', newGame);

  // ---------------------------------------------------------------- rendering

  const dot = (p) => `<span class="dot" style="background:var(--p${p})"></span>`;

  function render(animateMove) {
    const line = winningLine(board);
    const winSet = new Set((line || []).map(([r, c]) => r * COLS + c));
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        const v = board[r * COLS + c];
        const d = discs[r][c];
        d.className = 'disc' + (v ? ' p' + v : '') + (winSet.has(r * COLS + c) ? ' win' : '');
        if (!line && lastMove === r * COLS + c) d.classList.add('last');
      }
    }
    if (animateMove != null) {
      const [r, c] = animateMove;
      const d = discs[r][c];
      d.style.setProperty('--drop-from', `${-(r + 1) * 118}%`);
      d.style.setProperty('--drop-ms', `${180 + r * 45}ms`);
      d.classList.add('drop');
    }
    boardEl.classList.toggle('locked', over || toPlay !== human || !net);
  }

  function showSearch(stats, progress) {
    const max = Math.max(1, ...stats.visits);
    const top = stats.visits.indexOf(max);
    stats.visits.forEach((n, c) => {
      bars[c].style.height = (100 * n / max) + '%';
      bars[c].classList.toggle('top', c === top && n > 0);
    });
    $('search-count').textContent = `${Math.round(progress * sims)} / ${sims} sims`;
    const pct = Math.round(stats.winChance * 100);
    $('win-pct').textContent = pct + '%';
    $('win-bar').style.width = pct + '%';
  }

  function clearSearch() {
    bars.forEach((b) => { b.style.height = '0'; b.classList.remove('top'); });
    $('search-count').textContent = '';
  }

  // ---------------------------------------------------------------- game flow

  function place(col) {
    const r = topRow(board, col);
    board = drop(board, col, toPlay);
    lastMove = r * COLS + col;
    toPlay = 3 - toPlay;
    const [done, winner] = terminal(board);
    if (done) {
      over = true;
      if (winner === 0) statusEl.textContent = "It's a draw.";
      else if (winner === human) statusEl.innerHTML = dot(human) + 'You win! Nice.';
      else statusEl.innerHTML = dot(3 - human) + 'The AI wins.';
    }
    render([r, col]);
    return done;
  }

  function humanMove(col) {
    if (over || !net || toPlay !== human || topRow(board, col) < 0) return;
    if (!place(col)) aiMove();
  }

  async function aiMove() {
    const id = gameId;
    statusEl.innerHTML = dot(3 - human) + 'AI is thinking…';
    render();
    const t0 = performance.now();
    const { move, stats } = await think(net, board, toPlay, sims, (st, p) => { if (id === gameId) showSearch(st, p); });
    // Keep a short beat so fast (Easy) moves don't feel instant.
    const wait = Math.max(0, 350 - (performance.now() - t0));
    await new Promise((res) => setTimeout(res, wait));
    if (id !== gameId) return;
    showSearch(stats, 1);
    if (!place(move)) statusEl.innerHTML = dot(human) + 'Your move. Click a column.';
  }

  function newGame() {
    gameId++;
    board = emptyBoard();
    toPlay = 1;
    lastMove = -1;
    over = false;
    clearSearch();
    $('win-pct').textContent = '–';
    $('win-bar').style.width = '50%';
    render();
    if (!net) return;
    if (human === 1) statusEl.innerHTML = dot(1) + 'You are red. Click a column to drop a piece.';
    else aiMove();
  }

  // ---------------------------------------------------------------- load weights

  function showTrainingStats(meta) {
    if (!meta) return;
    const items = [
      [meta.self_play_games.toLocaleString(), 'self-play games'],
      [meta.sims_per_move, 'MCTS simulations per move'],
      [meta.train_minutes + ' min', 'training time (laptop CPU)'],
      [meta.params.toLocaleString(), 'network parameters'],
    ];
    $('train-stats').innerHTML = items.map(([v, k]) => `<div class="stat"><div class="v">${v}</div><div class="k">${k}</div></div>`).join('');
  }

  fetch('weights.json')
    .then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); })
    .then((w) => { net = makeNet(w); showTrainingStats(w.meta); newGame(); })
    .catch((err) => {
      console.error(err);
      statusEl.textContent = 'Could not load the network weights.';
    });
})();
