"""
Train the AlphaZero Connect-4 agent with real self-play and export weights for the web demo.

Run with: python train.py
Writes:   checkpoint.pt         (PyTorch weights, for resuming / local play)
          docs/weights.json     (BatchNorm-folded weights the browser loads)

Uses the building blocks in model.py, plus the standard AlphaZero extras the small demo
skipped: Dirichlet noise at the root, a temperature schedule, a replay buffer, mirror
augmentation, and parallel self-play.
"""

import argparse
import json
import os
import time
from collections import deque
from multiprocessing import get_context

import numpy as np
import torch

from model import (
    make_empty_board, valid_moves, step_env, is_terminal, other_player, drop_piece,
    build_policy_value_net, make_mcts_node, evaluate_with_network, expand_node,
    run_one_simulation, visit_count_policy, training_step,
)

NET_KWARGS = dict(in_channels=2, hidden_channels=32, num_columns=7, num_layers=4)


def new_net():
    return build_policy_value_net(**NET_KWARGS)


# --------------------------------------------------------------------------- self-play

def search(net, board, to_play, num_simulations, c_puct, noise_alpha=None, noise_frac=0.25, rng=None):
    """MCTS from (board, to_play). Optionally mixes Dirichlet noise into the root priors."""
    root = make_mcts_node()
    root['board'] = board
    root['to_play'] = to_play
    priors, _ = evaluate_with_network(net, board, to_play)
    if noise_alpha is not None:
        legal = valid_moves(board)
        noise = rng.dirichlet([noise_alpha] * len(legal))
        priors = priors.copy()
        priors[legal] = (1 - noise_frac) * priors[legal] + noise_frac * noise
    expand_node(root, priors)
    root['visit_count'] = 1
    for _ in range(num_simulations):
        run_one_simulation(root, net, c_puct)
    return root


def self_play_game(net, num_simulations, c_puct, temp_moves, rng):
    board, to_play, history = make_empty_board(), 1, []
    while True:
        root = search(net, board, to_play, num_simulations, c_puct, noise_alpha=1.0, rng=rng)
        pi = np.asarray(visit_count_policy(root, temperature=1.0), dtype=np.float32)
        history.append((board.copy(), to_play, pi))
        if len(history) <= temp_moves:
            action = int(rng.choice(7, p=pi / pi.sum()))
        else:
            action = int(np.argmax(pi))
        board, done, winner, to_play = step_env(board, action, to_play)
        if done:
            break
    samples = []
    for b, p, pi in history:
        z = 0.0 if winner == 0 else (1.0 if p == winner else -1.0)
        samples.append({'board': b, 'to_play': p, 'policy': pi, 'value': z})
        # Connect-4 is left/right symmetric: the mirrored position is free extra data
        samples.append({'board': b[:, ::-1].copy(), 'to_play': p, 'policy': pi[::-1].copy(), 'value': z})
    return samples, winner


def _worker(args):
    state_dict, n_games, num_simulations, c_puct, temp_moves, seed = args
    torch.set_num_threads(1)
    net = new_net()
    net.load_state_dict({k: torch.from_numpy(v) for k, v in state_dict.items()})
    net.eval()
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_games):
        samples, _ = self_play_game(net, num_simulations, c_puct, temp_moves, rng)
        out.extend(samples)
    return out


# --------------------------------------------------------------------------- evaluation

def mcts_agent(net, num_simulations):
    def act(board, to_play):
        root = search(net, board, to_play, num_simulations, c_puct=1.5)
        return int(np.argmax(visit_count_policy(root, temperature=0)))
    return act


def tactical_agent(rng):
    """Baseline: win if possible, else block an immediate loss, else random."""
    def act(board, to_play):
        legal = valid_moves(board)
        for p in (to_play, other_player(to_play)):
            for c in legal:
                done, winner = is_terminal(drop_piece(board, c, p))
                if done and winner == p:
                    return c
        return int(rng.choice(legal))
    return act


def play(agent_a, agent_b, a_first):
    board, to_play = make_empty_board(), 1
    agents = {1: agent_a, 2: agent_b} if a_first else {1: agent_b, 2: agent_a}
    while True:
        board, done, winner, to_play = step_env(board, agents[to_play](board, to_play), to_play)
        if done:
            if winner == 0:
                return 0
            a_player = 1 if a_first else 2
            return 1 if winner == a_player else -1


def score(agent, opponent, n):
    results = [play(agent, opponent, a_first=(i % 2 == 0)) for i in range(n)]
    return {'W': results.count(1), 'L': results.count(-1), 'D': results.count(0)}


# --------------------------------------------------------------------------- export

def export_weights(net, path, meta=None):
    """Save weights as JSON with BatchNorm folded into the preceding 1x1 convs."""
    net.eval()

    def t(x):
        return x.detach().cpu().numpy().astype(np.float32).ravel().round(6).tolist()

    def fold(conv, bn):
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        w = conv.weight * scale[:, None, None, None]
        b = (conv.bias - bn.running_mean) * scale + bn.bias
        return {'w': t(w), 'b': t(b), 'out': conv.out_channels, 'in': conv.in_channels}

    convs = [m for m in net.backbone if isinstance(m, torch.nn.Conv2d)]
    ph, vh = net.policy_head, net.value_head
    data = {
        'backbone': [{'w': t(c.weight), 'b': t(c.bias), 'out': c.out_channels, 'in': c.in_channels} for c in convs],
        'policy_conv': fold(ph[0], ph[1]),
        'policy_fc': {'w': t(ph[4].weight), 'b': t(ph[4].bias), 'out': ph[4].out_features, 'in': ph[4].in_features},
        'value_conv': fold(vh[0], vh[1]),
        'value_fc1': {'w': t(vh[4].weight), 'b': t(vh[4].bias), 'out': vh[4].out_features, 'in': vh[4].in_features},
        'value_fc2': {'w': t(vh[6].weight), 'b': t(vh[6].bias), 'out': vh[6].out_features, 'in': vh[6].in_features},
        'meta': meta,
    }
    with open(path, 'w') as f:
        json.dump(data, f, separators=(',', ':'))


# --------------------------------------------------------------------------- main loop

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iterations', type=int, default=40)
    ap.add_argument('--games', type=int, default=32, help='self-play games per iteration')
    ap.add_argument('--sims', type=int, default=120, help='MCTS simulations per move in self-play')
    ap.add_argument('--train-steps', type=int, default=300, help='minibatches per iteration')
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--buffer', type=int, default=80000, help='replay buffer size (samples)')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--checkpoint', default='checkpoint.pt', help='where to save/resume PyTorch weights')
    ap.add_argument('--weights', default='docs/weights.json', help='where to export weights for the web demo')
    args = ap.parse_args()

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    net = new_net()
    if args.resume and os.path.exists(args.checkpoint):
        net.load_state_dict(torch.load(args.checkpoint))
        print(f'Resumed from {args.checkpoint}')
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    buffer = deque(maxlen=args.buffer)
    ctx = get_context('spawn')
    t_start, games_played = time.time(), 0

    def meta():
        return {'self_play_games': games_played, 'sims_per_move': args.sims,
                'train_minutes': round((time.time() - t_start) / 60),
                'params': sum(p.numel() for p in net.parameters()), 'arch': NET_KWARGS}

    with ctx.Pool(args.workers) as pool:
        for it in range(1, args.iterations + 1):
            t0 = time.time()
            net.eval()
            sd = {k: v.cpu().numpy() for k, v in net.state_dict().items()}  # numpy pickles without shared memory
            per = [args.games // args.workers + (1 if i < args.games % args.workers else 0) for i in range(args.workers)]
            jobs = [(sd, n, args.sims, 1.5, 10, int(rng.integers(1 << 31))) for n in per if n > 0]
            for samples in pool.map(_worker, jobs):
                buffer.extend(samples)
            games_played += args.games
            t_play = time.time() - t0

            net.train()
            buf = list(buffer)
            losses = []
            for _ in range(args.train_steps):
                idx = rng.integers(len(buf), size=args.batch_size)
                losses.append(training_step(net, optimizer, [buf[i] for i in idx], l2_weight=1e-4))
            net.eval()
            avg = {k: round(float(np.mean([l[k] for l in losses])), 3) for k in ('policy', 'value')}

            msg = f"iter {it:3d}/{args.iterations}  buffer {len(buffer):6d}  loss {avg}  play {t_play:.0f}s  train {time.time() - t0 - t_play:.0f}s"
            if it % 5 == 0 or it == args.iterations:
                eval_rng = np.random.default_rng(it)
                msg += f"\n           MCTS-50 vs tactical: {score(mcts_agent(net, 50), tactical_agent(eval_rng), 20)}"
                torch.save(net.state_dict(), args.checkpoint)
                export_weights(net, args.weights, meta())
            print(msg, flush=True)

    torch.save(net.state_dict(), args.checkpoint)
    export_weights(net, args.weights, meta())
    print(f'Saved {args.checkpoint} and {args.weights}')


if __name__ == '__main__':
    main()
