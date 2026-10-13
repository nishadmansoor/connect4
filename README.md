# AlphaZero on Connect-4 from Scratch

Build a complete AlphaZero-style agent for Connect-4, from the bare game rules to a self-play training loop that pits a policy-value network guided by PUCT Monte Carlo Tree Search against itself. Along the way you implement the board engine, neural network, MCTS, self-play data generation, training, and evaluation against baselines.

## How to run

```bash
python scaffold.py
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
- [ ] **44.** policy_loss_cross_entropy
- [ ] **45.** l2_regularization_loss
- [ ] **46.** combined_loss
- [ ] **47.** encode_batch_states
- [ ] **48.** iterate_minibatches
- [ ] **49.** training_step
- [ ] **50.** training_epoch
- [ ] **51.** self_play_iteration
- [ ] **52.** train_loop
- [ ] **53.** random_policy_action
- [ ] **54.** greedy_agent_action
- [ ] **55.** play_one_match
- [ ] **56.** match_win_rate
- [ ] **57.** evaluate_against_random

---

Built on Deep-ML.
