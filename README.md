# Training AlphaZero on Connect 4

An AlphaZero-style Connect-4 AI built from four parts: a game engine, a policy-value neural
network, Monte Carlo Tree Search, and a self-play training loop. To pick a move, it searches
ahead, using the network to suggest promising moves and judge who's winning. Everything it
knows, it learned by playing against itself, and with search turned on it beats a random
player, a win-or-block bot, and a 4-move minimax search far more often than it loses.


**In the browser.** The trained weights are exported to JSON, with BatchNorm folded into the
preceding convolutions. `docs/c4.js` runs the same network and search in plain JavaScript, with
no server and no ML library. I checked it against PyTorch: the outputs match to within 1e-6,
and on test positions it picks exactly the same moves as the Python search.

## Results

Against three baseline opponents, 20 games per cell, alternating who moves first (W–L–D):

| Opponent | Network only | Easy (50 sims) | Medium (250 sims) | Hard (1000 sims) |
|---|---|---|---|---|
| Random mover | 20–0–0 | 20–0–0 | 20–0–0 | 20–0–0 |
| Win-or-block bot | 13–5–2 | 18–0–2 | 19–0–1 | 20–0–0 |
| Minimax, 4 moves deep | 10–7–3 | 17–1–2 | 14–6–0 | 8–2–10 |

"Network only" plays the policy's top move with no search. Against the minimax bot, adding search
cuts the loss rate from 35% to 15%. The site renders the current numbers and the training curve
from `docs/results.json`, so it stays up to date when the model is retrained. This table won't
update on its own.

## Bugs found along the way

Getting the pieces to *run* was the easy part. Several bugs produced reasonable-looking numbers
while quietly breaking the algorithm:

- **The search was trying to lose.** `backup_value` stored each node's value from the wrong
  player's point of view, so a move that won on the spot scored -1. With 200 simulations, the
  search gave the winning move 2 visits. After the fix, it gave it 185.
- **Move selection ignored the search.** `mcts_choose_action` passed visit-count *probabilities*
  to a sampler that expects *logits*. Even when the search put 100% on one column, it played that
  column only about 30% of the time.
- **NaN policy loss.** For a full column, the log-probability is −∞ and the target probability
  is 0, and `0 × −∞` is NaN. The fix only sums terms with a positive target.
- **A scrambled board.** `greedy_agent_action` encoded the board twice, so the network saw garbage.
- **BatchNorm never trained.** Running the search put the network in eval mode and left it there,
  so training went through BatchNorm without ever updating its statistics.

## Known limitations

- **More search isn't always better here.** Against the minimax bot, Hard (1000 simulations) draws
  half its games and does worse than Easy (50). One possibility is that both players are nearly
  deterministic, so the 20 games are really a few lines played over and over. Another is a real
  weakness in deep search trusting an imperfect value head. I haven't confirmed which yet.
- **Small evaluation.** 20 games per cell is enough to see big gaps, not small ones.
- **Not perfect play.** Connect-4 is solved (the first player wins with perfect play). This AI
  beats simple bots, but it is far from solved-level play, so a careful human can likely beat it.
- **No automated tests yet.** Correctness was checked by hand: tactical positions, Python vs JS
  agreement, and resume behavior.

## Run it yourself

Tested with Python 3.13, NumPy 2.4, and PyTorch 2.10.

```bash
pip install -r requirements.txt

python scaffold.py   # quick end-to-end demo of every step (a few seconds)
python train.py      # real self-play training (~35 min on a laptop CPU)
```


| File | What it is |
|---|---|
| `checkpoint.pt` | trained PyTorch weights |
| `docs/weights.json` | the same weights, exported for the browser |
| `docs/results.json` | stats, loss curve, and head-to-head table shown on the site |
| `train_state.pt` | optimizer, replay buffer, and history for `--resume` (large, git-ignored) |

```bash
python train.py --resume       # keep training from where you left off (needs train_state.pt)
python train.py --resume       # keep training from where you left off
python train.py --eval-only    # re-run the head-to-head on checkpoint.pt without training
python train.py --help         # games, simulations, buffer size, output paths, ...
```

`train_state.pt` (optimizer, replay buffer, history) is large, so it's git-ignored; without it,
`--resume` still continues the stats and curve from `docs/results.json`, just with an empty
replay buffer.

To preview the site locally:

```bash
python -m http.server 8000 --directory docs
python -m http.server 8000 --directory docs   # then open http://localhost:8000
```

## Project structure

```
model.py       game engine, network, MCTS, losses, and training/eval helpers (57 steps)
train.py       parallel self-play training, evaluation vs baselines, export for the web
scaffold.py    small end-to-end demo that exercises every function in model.py
solution.py    shim so scaffold.py's `from solution import ...` resolves to model.py
checkpoint.pt  trained weights
docs/          the website (GitHub Pages)
  index.html   page layout and styles
  c4.js        Connect-4 rules, network forward pass, and MCTS in JavaScript
  game.js      the playable board
  results.js   renders the results section from results.json
assets/        README images
```
