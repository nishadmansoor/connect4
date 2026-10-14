// Renders the "Model results" section from results.json (written by train.py).
(function () {
  'use strict';
  const $ = (id) => document.getElementById(id);

  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  function stats(meta) {
    const items = [
      [meta.self_play_games.toLocaleString(), 'self-play games'],
      [String(meta.sims_per_move), 'MCTS simulations per move'],
      [Math.round(meta.train_minutes) + ' min', 'training time (laptop CPU)'],
      [meta.params.toLocaleString(), 'network parameters'],
    ];
    const box = $('train-stats');
    box.replaceChildren(...items.map(([v, k]) => {
      const d = el('div', 'stat');
      d.append(el('div', 'v', v), el('div', 'k', k));
      return d;
    }));
  }

  function table(rows) {
    const body = $('h2h-body');
    body.replaceChildren(...rows.map((r) => {
      const n = r.W + r.L + r.D;
      const tr = el('tr');
      tr.append(el('td', null, r.opponent), el('td', null, r.setting), el('td', 'num', `${r.W}–${r.L}–${r.D}`));
      const bar = el('div', 'wld');
      bar.title = `${r.W} wins, ${r.L} losses, ${r.D} draws`;
      const w = el('span', 'w'); w.style.width = (100 * r.W / n) + '%';
      const d = el('span', 'd'); d.style.width = (100 * r.D / n) + '%';
      bar.append(w, d);
      const td = el('td'); td.append(bar);
      tr.append(td);
      return tr;
    }));
    if (rows.length) $('games-per-row').textContent = rows[0].games;
  }

  // Factual summary: network alone vs. with search, against the strongest baseline.
  function summary(rows) {
    const opp = rows.length ? rows[rows.length - 1].opponent : null;
    const vs = rows.filter((r) => r.opponent === opp);
    const raw = vs.find((r) => r.setting === 'Network only');
    const searched = vs.filter((r) => r.setting !== 'Network only');
    if (!raw || !searched.length) return;
    const tot = searched.reduce((a, r) => ({ W: a.W + r.W, L: a.L + r.L, D: a.D + r.D }), { W: 0, L: 0, D: 0 });
    const n = (r) => r.W + r.L + r.D;
    const p = $('h2h-summary');
    p.replaceChildren(
      'Against the strongest baseline (', el('strong', null, opp.toLowerCase()), '), the network on its own goes ',
      el('strong', null, `${raw.W}–${raw.L}–${raw.D}`), ` (${Math.round(100 * raw.L / n(raw))}% losses). With tree search, across all three levels, it goes `,
      el('strong', null, `${tot.W}–${tot.L}–${tot.D}`), ` (${Math.round(100 * tot.L / n(tot))}% losses). `,
      'The network makes the search smart about where to look, and the search catches tactics the network misses. That combination is the core idea of AlphaZero.'
    );
  }

  function chart(history) {
    const W = 760, H = 290, ml = 48, mr = 92, mt = 16, mb = 46;
    const pw = W - ml - mr, ph = H - mt - mb;
    const first = history[0].iteration, last = history[history.length - 1].iteration;
    const ymax = Math.max(2, Math.ceil(Math.max(...history.map((h) => h.policy)) * 2) / 2);
    const X = (i) => ml + (last === first ? 0 : (i - first) / (last - first)) * pw;
    const Y = (v) => mt + ph * (1 - v / ymax);
    const path = (k) => 'M' + history.map((h) => `${X(h.iteration).toFixed(1)},${Y(h[k]).toFixed(1)}`).join(' L');
    let g = '';
    for (let v = 0; v <= ymax + 1e-9; v += 0.5) {
      g += `<line x1="${ml}" x2="${ml + pw}" y1="${Y(v).toFixed(1)}" y2="${Y(v).toFixed(1)}" class="grid"/>` +
           `<text x="${ml - 8}" y="${(Y(v) + 4).toFixed(1)}" text-anchor="end" class="tick">${v.toFixed(1)}</text>`;
    }
    // round tick step (1, 2, 5, 10, 20, 50, ...) giving about 4 intervals
    const raw = Math.max(1, (last - first) / 4), mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw);
    const ticks = [first];
    for (let i = Math.ceil((first + 1) / step) * step; i <= last; i += step) if (i - ticks[ticks.length - 1] >= step / 2) ticks.push(i);
    for (const i of ticks) {
      g += `<text x="${X(i).toFixed(1)}" y="${mt + ph + 18}" text-anchor="middle" class="tick">${i}</text>`;
    }
    const end = history[history.length - 1];
    // keep the two end labels from overlapping
    let yp = Y(end.policy) + 4, yv = Y(end.value) + 4;
    if (Math.abs(yp - yv) < 14) { if (yp < yv) yv = yp + 14; else yp = yv + 14; }
    $('loss-chart').innerHTML =
      `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Policy and value loss over ${history.length} training iterations">` +
      g + `<text x="${ml + pw / 2}" y="${H - 4}" text-anchor="middle" class="tick">training iteration</text>` +
      `<path d="${path('policy')}" class="line pol"/><path d="${path('value')}" class="line val"/>` +
      `<text x="${X(end.iteration) + 8}" y="${yp.toFixed(1)}" class="lbl pol-t">policy ${end.policy.toFixed(2)}</text>` +
      `<text x="${X(end.iteration) + 8}" y="${yv.toFixed(1)}" class="lbl val-t">value ${end.value.toFixed(2)}</text></svg>`;

    const checks = history.filter((h) => h.vs_tactical).map((h) => {
      const r = h.vs_tactical; return r.W / (r.W + r.L + r.D);
    });
    const minPol = Math.min(...history.map((h) => h.policy));
    let note = `Loss per training iteration. Each iteration is a batch of self-play games followed by training on the replay buffer. ` +
      `Policy loss is cross-entropy against the search's move distribution; value loss is squared error against the final game result.`;
    if (end.policy - minPol > 0.05) {
      note += ` Policy loss rises again late in training. That can happen as the replay buffer fills with games from a stronger player, whose search targets are harder to predict.`;
    }
    if (checks.length) {
      const lo = Math.round(100 * Math.min(...checks)), hi = Math.round(100 * Math.max(...checks));
      note += ` In checks every 5 iterations, Easy-level search beat the win-or-block bot in ${lo === hi ? lo : lo + '–' + hi}% of games.`;
    }
    $('curve-note').textContent = note;
  }

  fetch('results.json')
    .then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); })
    .then((res) => {
      stats(res.meta);
      table(res.head_to_head);
      summary(res.head_to_head);
      if (res.history.length) chart(res.history);
    })
    .catch((err) => {
      console.error(err);
      $('results').querySelector('.section-title').insertAdjacentText('afterend', 'Results could not be loaded.');
    });
})();
