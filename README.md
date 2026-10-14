# AlphaZero on Connect-4 from Scratch

Build a complete AlphaZero-style agent for Connect-4, from the bare game rules to a self-play training loop that pits a policy-value network guided by PUCT Monte Carlo Tree Search against itself. Along the way you implement the board engine, neural network, MCTS, self-play data generation, training, and evaluation against baselines.

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

## Steps

- [x] **1.** make_empty_board
- [x] **2.** column_top_row
- [x] **3.** drop_piece
- [x] **4.** column_full
- [x] **5.** valid_moves
- [x] **6.** four_in_a_row_horizontal
- [x] **7.** four_in_a_row_vertical
- [x] **8.** four_in_a_row_diagonal_down_right
- [x] **9.** four_in_a_row_diagonal_up_right
- [x] **10.** check_winner
- [x] **11.** board_is_full
- [x] **12.** is_terminal
- [x] **13.** other_player
- [x] **14.** step_env
- [x] **15.** encode_board
- [x] **16.** board_to_torch_tensor
- [x] **17.** init_conv_backbone
- [x] **18.** init_policy_head
- [x] **19.** init_value_head
- [x] **20.** build_policy_value_net
- [x] **21.** policy_value_forward
- [x] **22.** action_mask
- [x] **23.** masked_policy_logits
- [x] **24.** masked_log_softmax
- [x] **25.** sample_action_from_policy
- [x] **26.** greedy_action_from_policy
- [x] **27.** make_mcts_node
- [x] **28.** node_q_value
- [x] **29.** ucb_score
- [x] **30.** select_best_child
- [x] **31.** select_leaf
- [x] **32.** evaluate_with_network
- [x] **33.** expand_node
- [x] **34.** backup_value
- [x] **35.** run_one_simulation
- [x] **36.** run_mcts
- [x] **37.** visit_count_policy
- [x] **38.** mcts_choose_action
- [x] **39.** record_self_play_step
- [x] **40.** play_self_play_game
- [x] **41.** assign_value_targets
- [x] **42.** generate_self_play_batch
- [x] **43.** value_loss_mse
- [x] **44.** policy_loss_cross_entropy
- [x] **45.** l2_regularization_loss
- [x] **46.** combined_loss
- [x] **47.** encode_batch_states
- [x] **48.** iterate_minibatches
- [x] **49.** training_step
- [x] **50.** training_epoch
- [x] **51.** self_play_iteration
- [x] **52.** train_loop
- [x] **53.** random_policy_action
- [x] **54.** greedy_agent_action
- [x] **55.** play_one_match
- [x] **56.** match_win_rate
- [x] **57.** evaluate_against_random

---

Built on Deep-ML.
