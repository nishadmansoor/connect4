# Training AlphaZero on Connect 4

**Play it:** https://nishad.xyz/connect4-alphazero/

## How to run

```bash
python scaffold.py   # quick end-to-end demo of every step (a few seconds)
python train.py      # real self-play training (~30 min on a laptop CPU)
```

`train.py` saves `checkpoint.pt` and exports two files the web demo in `docs/` loads:
`weights.json` (the network) and `results.json` (training stats, the loss curve, and a
head-to-head table against random, win-or-block, and 4-move minimax opponents). The page
renders everything from those files, so retraining updates the site automatically. The browser
runs the same network and PUCT search in plain JavaScript (`docs/c4.js`), so there's no server.

```bash
python train.py --resume       # keep training from where you left off (needs train_state.pt)
python train.py --eval-only    # re-run the head-to-head on checkpoint.pt without training
python train.py --help         # games, simulations, buffer size, output paths, ...
```

`train_state.pt` (optimizer, replay buffer, history) is large, so it's git-ignored; without it,
`--resume` still continues the stats and curve from `docs/results.json`, just with an empty
replay buffer.

To preview the site locally:

```bash
python -m http.server 8000 --directory docs
```

