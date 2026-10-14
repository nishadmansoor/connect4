// Connect-4 engine, policy-value network, and PUCT MCTS for the browser.
// Mirrors model.py: board is 6x7 row-major (row 0 = top), players 1 and 2, 0 = empty.
(function (root) {
  'use strict';

  const ROWS = 6, COLS = 7, CELLS = 42;

  // ------------------------------------------------------------------ game rules

  function emptyBoard() { return new Int8Array(CELLS); }

  function topRow(board, col) {
    for (let r = ROWS - 1; r >= 0; r--) if (board[r * COLS + col] === 0) return r;
    return -1;
  }

  function validMoves(board) {
    const moves = [];
    for (let c = 0; c < COLS; c++) if (board[c] === 0) moves.push(c);
    return moves;
  }

  function drop(board, col, player) {
    const r = topRow(board, col);
    if (r < 0) throw new Error('column is full');
    const next = board.slice();
    next[r * COLS + col] = player;
    return next;
  }

  // Returns the winning cells [[r,c] x4] or null.
  function winningLine(board) {
    const dirs = [[0, 1], [1, 0], [1, 1], [-1, 1]];
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        const p = board[r * COLS + c];
        if (!p) continue;
        for (const [dr, dc] of dirs) {
          const cells = [[r, c]];
          for (let k = 1; k < 4; k++) {
            const rr = r + dr * k, cc = c + dc * k;
            if (rr < 0 || rr >= ROWS || cc >= COLS || board[rr * COLS + cc] !== p) break;
            cells.push([rr, cc]);
          }
          if (cells.length === 4) return cells;
        }
      }
    }
    return null;
  }

  // [done, winner] like is_terminal in model.py
  function terminal(board) {
    const line = winningLine(board);
    if (line) return [true, board[line[0][0] * COLS + line[0][1]]];
    if (validMoves(board).length === 0) return [true, 0];
    return [false, 0];
  }

  // ------------------------------------------------------------------ network

  function makeNet(W) {
    const toF32 = (layer) => ({ ...layer, w: Float32Array.from(layer.w), b: Float32Array.from(layer.b) });
    const backbone = W.backbone.map(toF32);
    const pConv = toF32(W.policy_conv), pFc = toF32(W.policy_fc);
    const vConv = toF32(W.value_conv), vFc1 = toF32(W.value_fc1), vFc2 = toF32(W.value_fc2);

    // 3x3 conv with zero padding: copy input into an 8x9 padded grid so the inner loops have no bounds checks.
    const PR = ROWS + 2, PC = COLS + 2, PCELLS = PR * PC;
    const acc = new Float32Array(CELLS);
    function conv3x3Relu(x, L) {
      const xp = new Float32Array(L.in * PCELLS);
      for (let i = 0; i < L.in; i++)
        for (let r = 0; r < ROWS; r++)
          for (let c = 0; c < COLS; c++) xp[i * PCELLS + (r + 1) * PC + c + 1] = x[i * CELLS + r * COLS + c];
      const out = new Float32Array(L.out * CELLS);
      const w = L.w;
      for (let o = 0; o < L.out; o++) {
        acc.fill(L.b[o]);
        for (let i = 0; i < L.in; i++) {
          const wBase = (o * L.in + i) * 9, xBase = i * PCELLS;
          for (let kr = 0; kr < 3; kr++) {
            for (let kc = 0; kc < 3; kc++) {
              const wv = w[wBase + kr * 3 + kc];
              if (wv === 0) continue;
              for (let r = 0; r < ROWS; r++) {
                const src = xBase + (r + kr) * PC + kc, dst = r * COLS;
                for (let c = 0; c < COLS; c++) acc[dst + c] += wv * xp[src + c];
              }
            }
          }
        }
        const ob = o * CELLS;
        for (let p = 0; p < CELLS; p++) out[ob + p] = acc[p] > 0 ? acc[p] : 0;
      }
      return out;
    }

    function conv1x1Relu(x, L) {
      const out = new Float32Array(L.out * CELLS);
      for (let o = 0; o < L.out; o++) {
        for (let p = 0; p < CELLS; p++) {
          let s = L.b[o];
          for (let i = 0; i < L.in; i++) s += L.w[o * L.in + i] * x[i * CELLS + p];
          out[o * CELLS + p] = s > 0 ? s : 0;
        }
      }
      return out;
    }

    function linear(x, L, act) {
      const out = new Float32Array(L.out);
      for (let j = 0; j < L.out; j++) {
        let s = L.b[j];
        for (let k = 0; k < L.in; k++) s += L.w[j * L.in + k] * x[k];
        out[j] = act === 'relu' ? (s > 0 ? s : 0) : act === 'tanh' ? Math.tanh(s) : s;
      }
      return out;
    }

    // Returns { logits: Float32Array(7), value: number } from `player`'s point of view.
    function forward(board, player) {
      const opp = 3 - player;
      let x = new Float32Array(2 * CELLS);
      for (let p = 0; p < CELLS; p++) {
        if (board[p] === player) x[p] = 1;
        else if (board[p] === opp) x[CELLS + p] = 1;
      }
      for (const L of backbone) x = conv3x3Relu(x, L);
      const logits = linear(conv1x1Relu(x, pConv), pFc);
      const value = linear(linear(conv1x1Relu(x, vConv), vFc1, 'relu'), vFc2, 'tanh')[0];
      return { logits, value };
    }

    // Masked softmax priors over legal columns, plus value.
    function evaluate(board, player) {
      const { logits, value } = forward(board, player);
      const legal = validMoves(board);
      const priors = new Float32Array(COLS);
      let max = -Infinity;
      for (const c of legal) max = Math.max(max, logits[c]);
      let sum = 0;
      for (const c of legal) { priors[c] = Math.exp(logits[c] - max); sum += priors[c]; }
      for (const c of legal) priors[c] /= sum;
      return { priors, value };
    }

    return { forward, evaluate };
  }

  // ------------------------------------------------------------------ MCTS
  // Each node's valueSum is from the perspective of the player who moved INTO it,
  // so a parent simply picks the child with the highest Q + U.

  function node(board, toPlay, prior, parent) {
    return { board, toPlay, prior, parent, N: 0, W: 0, children: null };
  }

  function createSearch(net, board, toPlay, cPuct = 1.5) {
    const rootNode = node(board, toPlay, 1, null);

    function expand(n) {
      const { priors, value } = net.evaluate(n.board, n.toPlay);
      n.children = {};
      for (const c of validMoves(n.board)) {
        n.children[c] = node(drop(n.board, c, n.toPlay), 3 - n.toPlay, priors[c], n);
      }
      return value;
    }

    function simulate() {
      let n = rootNode;
      while (n.children) {
        const sqrtN = Math.sqrt(n.N);
        let best = null, bestScore = -Infinity;
        for (const c in n.children) {
          const ch = n.children[c];
          const q = ch.N ? ch.W / ch.N : 0;
          const s = q + cPuct * ch.prior * sqrtN / (1 + ch.N);
          if (s > bestScore) { bestScore = s; best = ch; }
        }
        n = best;
      }
      // value v is from n.toPlay's point of view
      const [done, winner] = terminal(n.board);
      let v;
      if (done) v = winner === 0 ? 0 : (winner === n.toPlay ? 1 : -1);
      else v = expand(n);
      let back = -v;
      while (n) { n.N += 1; n.W += back; back = -back; n = n.parent; }
    }

    expand(rootNode);
    rootNode.N = 1;

    return {
      run(k) { for (let i = 0; i < k; i++) simulate(); },
      // Per-column visit counts and the searcher's estimated win chance for `toPlay`.
      stats() {
        const visits = new Array(COLS).fill(0);
        let total = 0, wsum = 0;
        for (const c in rootNode.children) {
          const ch = rootNode.children[c];
          visits[c] = ch.N; total += ch.N; wsum += ch.W;
        }
        const q = total ? wsum / total : 0; // in [-1, 1] for the player to move
        return { visits, total, winChance: (q + 1) / 2 };
      },
      best() {
        let bestC = -1, bestN = -1;
        for (const c in rootNode.children) if (rootNode.children[c].N > bestN) { bestN = rootNode.children[c].N; bestC = +c; }
        return bestC;
      },
    };
  }

  // Run `sims` simulations in small time-sliced chunks so the page stays responsive.
  function think(net, board, toPlay, sims, onProgress) {
    const search = createSearch(net, board, toPlay);
    return new Promise((resolve) => {
      let done = 0;
      function chunk() {
        const t = performance.now();
        while (done < sims && performance.now() - t < 12) { search.run(5); done += 5; }
        if (onProgress) onProgress(search.stats(), Math.min(done, sims) / sims);
        if (done < sims) setTimeout(chunk, 0);
        else resolve({ move: search.best(), stats: search.stats() });
      }
      chunk();
    });
  }

  root.C4 = { ROWS, COLS, emptyBoard, topRow, validMoves, drop, winningLine, terminal, makeNet, createSearch, think };
  if (typeof module !== 'undefined') module.exports = root.C4;
})(typeof window !== 'undefined' ? window : globalThis);
