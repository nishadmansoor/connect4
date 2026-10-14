"""
Train the AlphaZero Connect-4 agent with real self-play and export weights for the web demo.

Run with: python train.py
Writes:   checkpoint.pt         (PyTorch weights)
          train_state.pt        (optimizer + replay buffer + history, for --resume; not committed)
          docs/weights.json     (BatchNorm-folded weights the browser loads)
          docs/results.json     (stats, training curve and head-to-head table the page shows)

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
    run_one_simulation, visit_count_policy, training_step, check_winner, greedy_agent_action,
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


def random_agent(rng):
    return lambda board, to_play: int(rng.choice(valid_moves(board)))


def _heuristic(b, p):
    """Score open 4-cell windows (more of your pieces = better) plus a center-column bonus."""
    o, s = 3 - p, 0
    for r in range(6):
        for c in range(7):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (-1, 1)):
                rr, cc = r + 3 * dr, c + 3 * dc
                if 0 <= rr < 6 and 0 <= cc < 7:
                    w = [b[r + k * dr, c + k * dc] for k in range(4)]
                    if o not in w:
                        s += (0, 1, 4, 16, 0)[w.count(p)]
                    if p not in w:
                        s -= (0, 1, 4, 16, 0)[w.count(o)]
    return s + 3 * (int(np.sum(b[:, 3] == p)) - int(np.sum(b[:, 3] == o)))


def _negamax(b, p, depth, alpha, beta):
    w = check_winner(b)
    if w:
        return (10000 + depth) if w == p else (-10000 - depth)
    moves = valid_moves(b)
    if not moves:
        return 0
    if depth == 0:
        return _heuristic(b, p)
    best = -1e9
    for c in sorted(moves, key=lambda c: abs(c - 3)):
        v = -_negamax(drop_piece(b, c, p), 3 - p, depth - 1, -beta, -alpha)
        best, alpha = max(best, v), max(alpha, v)
        if alpha >= beta:
            break
    return best


def minimax_agent(depth, rng):
    """Alpha-beta search `depth` moves ahead with a window-counting heuristic; random tie-breaks."""
    def act(board, to_play):
        scores = {c: -_negamax(drop_piece(board, c, to_play), 3 - to_play, depth - 1, -1e9, 1e9)
                  for c in valid_moves(board)}
        best = max(scores.values())
        return int(rng.choice([c for c, v in scores.items() if v == best]))
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
    """Play n games, alternating who moves first; returns W/L/D from `agent`'s side."""
    results = [play(agent, opponent, a_first=(i % 2 == 0)) for i in range(n)]
    return {'W': results.count(1), 'L': results.count(-1), 'D': results.count(0)}


# The table on the website: (opponent label, opponent factory, AI label, MCTS sims or None for network only)
OPPONENTS = [
    ('Random mover', random_agent),
    ('Win-or-block bot', tactical_agent),
    ('Minimax, 4 moves deep', lambda rng: minimax_agent(4, rng)),
]
AI_SETTINGS = [('Network only', None), ('Easy (50 sims)', 50), ('Medium (250 sims)', 250), ('Hard (1000 sims)', 1000)]


def _eval_job(args):
    state_dict, opp_idx, setting_idx, n_games, seed = args
    torch.set_num_threads(1)
    net = new_net()
    net.load_state_dict({k: torch.from_numpy(v) for k, v in state_dict.items()})
    net.eval()
    opp_name, make_opp = OPPONENTS[opp_idx]
    ai_name, sims = AI_SETTINGS[setting_idx]
    ai = (lambda b, p: greedy_agent_action(net, b, p)) if sims is None else mcts_agent(net, sims)
    return {'opponent': opp_name, 'setting': ai_name, 'games': n_games,
            **score(ai, make_opp(np.random.default_rng(seed)), n_games)}


def head_to_head(net, pool, n_games):
    """Every AI setting vs every baseline, run in parallel. Returns rows for docs/results.json."""
    sd = {k: v.cpu().numpy() for k, v in net.state_dict().items()}
    jobs = [(sd, o, a, n_games, 1000 * o + a) for o in range(len(OPPONENTS)) for a in range(len(AI_SETTINGS))]
    return pool.map(_eval_job, jobs, chunksize=1)


def write_results(path, meta, history, table, iteration):
    with open(path, 'w') as f:
        json.dump({'meta': meta, 'history': history, 'head_to_head': table,
                   'evaluated_at_iteration': iteration}, f, indent=1)


# --------------------------------------------------------------------------- export

def export_weights(net, path):
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
    }
    with open(path, 'w') as f:
        json.dump(data, f, separators=(',', ':'))


# --------------------------------------------------------------------------- main loop

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iterations', type=int, default=80)
    ap.add_argument('--games', type=int, default=64, help='self-play games per iteration')
    ap.add_argument('--sims', type=int, default=160, help='MCTS simulations per move in self-play')
    ap.add_argument('--train-steps', type=int, default=400, help='minibatches per iteration')
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--buffer', type=int, default=80000, help='replay buffer size (samples)')
    ap.add_argument('--eval-games', type=int, default=20, help='games per row of the head-to-head table')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument('--resume', action='store_true', help='continue from --checkpoint and --state')
    ap.add_argument('--eval-only', action='store_true', help='skip training; re-run the head-to-head on --checkpoint')
    ap.add_argument('--checkpoint', default='checkpoint.pt', help='PyTorch weights (saved / resumed)')
    ap.add_argument('--state', default='train_state.pt', help='optimizer, replay buffer and history, for --resume')
    ap.add_argument('--weights', default='docs/weights.json', help='weights exported for the web demo')
    ap.add_argument('--results', default='docs/results.json', help='stats, training curve and head-to-head for the web demo')
    args = ap.parse_args()

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    net = new_net()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    buffer = deque(maxlen=args.buffer)
    history, games_played, prev_minutes, start_iter, train_sims = [], 0, 0.0, 0, args.sims

    if args.resume or args.eval_only:
        net.load_state_dict(torch.load(args.checkpoint))
        print(f'Loaded {args.checkpoint}')
        if os.path.exists(args.state):
            st = torch.load(args.state, weights_only=False)
            optimizer.load_state_dict(st['optimizer'])
            buffer.extend(st['buffer'])
            rng.bit_generator.state = st['rng']
            history, games_played, prev_minutes, start_iter = st['history'], st['games_played'], st['minutes'], st['iteration']
            train_sims = st.get('sims', args.sims)
            print(f'Loaded {args.state}: iteration {start_iter}, {games_played} games, buffer {len(buffer)}')
        elif os.path.exists(args.results):
            # No saved training state: keep the stats and curve already published so they aren't reset
            with open(args.results) as f:
                prev = json.load(f)
            history, start_iter = prev['history'], prev['evaluated_at_iteration']
            games_played, prev_minutes = prev['meta']['self_play_games'], prev['meta']['train_minutes']
            train_sims = prev['meta']['sims_per_move']
            print(f'No {args.state}; continuing stats from {args.results} (replay buffer starts empty)')

    if not args.eval_only:
        train_sims = args.sims
    t_start = time.time()

    def train_minutes():
        return prev_minutes + (0 if args.eval_only else (time.time() - t_start) / 60)

    def meta():
        return {'self_play_games': games_played, 'sims_per_move': train_sims,
                'train_minutes': round(train_minutes(), 1),
                'params': sum(p.numel() for p in net.parameters()), 'arch': NET_KWARGS}

    def save(it):
        torch.save(net.state_dict(), args.checkpoint)
        export_weights(net, args.weights)
        torch.save({'optimizer': optimizer.state_dict(), 'buffer': list(buffer), 'rng': rng.bit_generator.state,
                    'history': history, 'games_played': games_played, 'iteration': it,
                    'minutes': train_minutes(), 'sims': train_sims}, args.state)

    ctx = get_context('spawn')
    with ctx.Pool(args.workers) as pool:
        it = start_iter
        if not args.eval_only:
            for it in range(start_iter + 1, start_iter + args.iterations + 1):
                t0 = time.time()
                net.eval()
                sd = {k: v.cpu().numpy() for k, v in net.state_dict().items()}  # numpy pickles without shared memory
                per = [args.games // args.workers + (1 if i < args.games % args.workers else 0) for i in range(args.workers)]
                jobs = [(sd, n, args.sims, 1.5, 10, int(rng.integers(1 << 31))) for n in per if n > 0]
                for samples in pool.map(_worker, jobs):
                    buffer.extend(samples)
                games_played += args.games
                t_play = time.time() - t0

                buf = list(buffer)
                losses = []
                for _ in range(args.train_steps):
                    idx = rng.integers(len(buf), size=args.batch_size)
                    losses.append(training_step(net, optimizer, [buf[i] for i in idx], l2_weight=1e-4))
                net.eval()
                entry = {'iteration': it, **{k: round(float(np.mean([l[k] for l in losses])), 3) for k in ('policy', 'value')}}

                msg = (f"iter {it:3d}  buffer {len(buffer):6d}  loss policy {entry['policy']} value {entry['value']}"
                       f"  play {t_play:.0f}s  train {time.time() - t0 - t_play:.0f}s")
                if it % 5 == 0:
                    entry['vs_tactical'] = score(mcts_agent(net, 50), tactical_agent(np.random.default_rng(it)), 20)
                    msg += f"\n           MCTS-50 vs win-or-block bot: {entry['vs_tactical']}"
                    save(it)
                history.append(entry)
                print(msg, flush=True)
            save(it)

        print(f'Running head-to-head ({args.eval_games} games per row)...', flush=True)
        table = head_to_head(net, pool, args.eval_games)
    for row in table:
        print(f"  {row['opponent']:<24} {row['setting']:<18} W {row['W']:>2}  L {row['L']:>2}  D {row['D']:>2}")
    write_results(args.results, {**meta(), 'iterations': it}, history, table, it)
    saved = [args.results] if args.eval_only else [args.checkpoint, args.state, args.weights, args.results]
    print('Saved ' + ', '.join(saved))


if __name__ == '__main__':
    main()
