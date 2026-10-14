"""
AlphaZero on Connect-4 from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - make_empty_board
import numpy as np

def make_empty_board():
    """Return a 6x7 integer numpy array of zeros representing an empty Connect-4 board."""
    # TODO: create a 6x7 integer array of zeros and return it
    return np.zeros((6,7), dtype=int)

# Step 2 - column_top_row
def column_top_row(board, column):
    """Return the lowest empty row in `column`, or -1 if the column is full."""
    # TODO: scan the column from the bottom up and return the first empty row index
    for row in range(5,-1,-1):
        if board[row,column] == 0:
            return row
    return -1

# Step 3 - drop_piece
def drop_piece(board, column, player):
    # TODO: place `player` in the lowest empty row of `column` and return the new board
    row =  column_top_row(board,column)
    #print(row)

    new_board = board.copy()
    if row == -1:
        raise ValueError("column is full")
    else:
       new_board[row,column] = player

    return new_board

# Step 4 - column_full
import numpy as np

def column_full(board, column):
    """Return True if `column` has no empty rows left."""
    # TODO: check whether the column can still accept a piece
    if column_top_row(board,column) == -1:
        return True
    else:
        return False

# Step 5 - valid_moves
def valid_moves(board):
    # TODO: return a list of column indices that still have at least one empty row
    space = []

    for col in range(7):
        if board[0,col] == 0:
            space.append(col)
    return space

# Step 6 - four_in_a_row_horizontal
def four_in_a_row_horizontal(board):
    # TODO: scan every row for four consecutive matching non-zero pieces horizontally
    for row in range(6):
        for col in range(4):
            if board[row,col] != 0 and board[row,col] == board[row,col + 1] == board[row,col + 2] == board[row,col + 3]:
                return board[row,col]
    return 0

# Step 7 - four_in_a_row_vertical
def four_in_a_row_vertical(board):
    # TODO: scan every column for four consecutive matching non-zero pieces vertically
        for col in range(7):
            for row in range(3):
                if board[row,col] != 0 and board[row,col] == board[row + 1,col] == board[row + 2,col] == board[row + 3,col]:
                    return board[row,col]
        return 0

# Step 8 - four_in_a_row_diagonal_down_right
def four_in_a_row_diagonal_down_right(board):
    # TODO: scan every down-right diagonal of the 6x7 board for four matching non-zero pieces
    for row in range(3):
        for col in range(4):
            if board[row,col] != 0 and board[row,col] == board[row + 1,col + 1] == board[row + 2,col + 2] == board[row + 3,col + 3]:
                return board[row,col]
    return 0

# Step 9 - four_in_a_row_diagonal_up_right
def four_in_a_row_diagonal_up_right(board):
    # TODO: scan every up-right diagonal for four consecutive matching non-zero pieces
    for row in range(3,6):
        for col in range(4):
            if board[row,col] != 0 and board[row,col] == board[row - 1, col + 1] == board[row - 2, col + 2] == board[row - 3, col + 3]:
                return board[row,col]
    return 0

# Step 10 - check_winner
import numpy as np

def check_winner(board):
    """Return 1 or 2 if that player has four in a row, else 0."""
    # TODO: combine the four direction scans and return the first non-zero result
    check = [
        four_in_a_row_horizontal(board),
        four_in_a_row_vertical(board),
        four_in_a_row_diagonal_down_right(board),
        four_in_a_row_diagonal_up_right(board)
    ]

    for winner in check:
        if winner != 0:
            return winner
    return 0

# Step 11 - board_is_full
def board_is_full(board):
    # TODO: return True when no column has an empty slot left
    return len(valid_moves(board)) == 0

# Step 12 - is_terminal
def is_terminal(board):
    # TODO: return (done, winner) using check_winner and board_is_full.
    winner = int(check_winner(board))

    if winner != 0:
        return True, winner
    if board_is_full(board):
        return True, 0
    return False,0

# Step 13 - other_player
def other_player(player):
    # TODO: return the opponent's player code (1 <-> 2)
    return 3 - player

# Step 14 - step_env
def step_env(board, column, player):
    # TODO: drop piece for player, then return (new_board, done, winner, next_player).
    new_board = drop_piece(board,column,player)
    done, winner = is_terminal(new_board)
    next_player = other_player(player)

    return new_board, done, winner, next_player

# Step 15 - encode_board
import numpy as np

def encode_board(board, current_player):
    """Encode a 6x7 board as a (2, 6, 7) float32 tensor from current_player's view."""
    opp = other_player(current_player)
    
    player_plane = (board == current_player).astype(np.float32)
    opponent_plane = (board == opp).astype(np.float32)
    
    return np.stack([player_plane, opponent_plane], axis=0)

# Step 16 - board_to_torch_tensor
import torch
def board_to_torch_tensor(board, current_player):
    # TODO: encode the board and return it as a float32 torch tensor of shape (1, 2, 6, 7).
    encode = encode_board(board, current_player)
    tensor = torch.tensor(encode, dtype=torch.float32)

    return tensor.unsqueeze(0)

# Step 17 - init_conv_backbone
import torch
import torch.nn as nn
def init_conv_backbone(in_channels=2, hidden_channels=16):
    # TODO: Build a small convolutional backbone preserving the 6x7 spatial shape.
    
    return nn.Sequential(
        nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(hidden_channels,hidden_channels,kernel_size=3,padding=1),
        nn.ReLU()
    )

# Step 18 - init_policy_head
import torch
import torch.nn as nn

def init_policy_head(hidden_channels=16, num_columns=7):
    """Return an nn.Module mapping (B, hidden_channels, 6, 7) -> (B, num_columns) logits."""
    # TODO: build a small policy head that projects backbone features to column logits
    return nn.Sequential(
        nn.Conv2d(hidden_channels, 2, kernel_size=1),
        nn.BatchNorm2d(2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(2 * 6 * 7, num_columns)
    )

# Step 19 - init_value_head
import torch
import torch.nn as nn

def init_value_head(hidden_channels=16):
    """Return an nn.Module mapping (B, hidden_channels, 6, 7) -> (B, 1) in (-1, 1)."""
    # TODO: build a value head that collapses backbone features to a single bounded scalar per board.
    return nn.Sequential(
        nn.Conv2d(hidden_channels, 1, kernel_size=1),
        nn.BatchNorm2d(1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(1 * 6 * 7, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Tanh()
    )

# Step 20 - build_policy_value_net
import torch
import torch.nn as nn

def build_policy_value_net(in_channels=2, hidden_channels=16, num_columns=7):
    """Compose backbone + policy head + value head into one nn.Module."""
    # TODO: build an nn.Module with backbone, policy_head, value_head attributes
    net = nn.Module()
    net.backbone = init_conv_backbone(in_channels=in_channels, hidden_channels=hidden_channels)
    net.policy_head = init_policy_head(hidden_channels=hidden_channels, num_columns=num_columns)
    net.value_head = init_value_head(hidden_channels=hidden_channels)
    
    def forward(x):
        feats = net.backbone(x)
        return net.policy_head(feats), net.value_head(feats)
    
    net.forward = forward
    return net

# Step 21 - policy_value_forward
import torch
import torch.nn as nn

def policy_value_forward(net, encoded_board):
    """Run encoded_board (B,2,6,7) through net and return (logits, value)."""
    # TODO: call the network on the encoded board and return its two outputs
    return net(encoded_board)

# Step 22 - action_mask
import numpy as np

def action_mask(board):
    # TODO: return a length-7 boolean mask, True where the column is legal
    mask = np.zeros(7, dtype=bool)
    valid = valid_moves(board)
    mask[valid] = True
    return mask

# Step 23 - masked_policy_logits
import torch

def masked_policy_logits(logits, mask):
    """Set logits at illegal columns to -inf.

    logits: torch.Tensor of shape (..., 7)
    mask:   bool array/tensor of shape (7,), True = legal
    returns: torch.Tensor of same shape as logits
    """
    # TODO: replace logits at illegal columns with negative infinity
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, dtype=torch.bool, device=logits.device)
    else:
        mask = mask.to(device=logits.device, dtype=torch.bool)
    
    return torch.where(mask, logits, float('-inf'))

# Step 24 - masked_log_softmax
import torch

def masked_log_softmax(logits, mask):
    """Log-softmax of logits with illegal columns (mask=False) forced to -inf."""
    # TODO: mask out illegal columns, then apply log-softmax over the last dim.
    masked_logits = masked_policy_logits(logits, mask)
    return torch.log_softmax(masked_logits, dim=-1)

# Step 25 - sample_action_from_policy
import torch

def sample_action_from_policy(logits, mask, temperature=1.0):
    """Sample a legal column from a tempered masked categorical policy."""
    # TODO: scale logits by temperature, mask illegal columns, sample one index
    masked_logits = masked_policy_logits(logits / temperature, mask)
    distance = torch.distributions.Categorical(logits=masked_logits)
    return distance.sample().item()

# Step 26 - greedy_action_from_policy
import torch

def greedy_action_from_policy(logits, mask):
    """Return the argmax legal column index from masked policy logits."""
    # TODO: mask out illegal columns then return the argmax as a python int
    masked_logits = masked_policy_logits(logits, mask)

    return torch.argmax(masked_logits).item()

# Step 27 - make_mcts_node
def make_mcts_node(prior=0.0, parent=None):
    # TODO: build a dict with prior, visit_count, value_sum, children, and parent fields.
    return {
        'prior': prior,
        'visit_count': 0,
        'value_sum': 0.0,
        'children': {},
        'parent': parent
    }

# Step 28 - node_q_value
def node_q_value(node):
    # TODO: return the mean value Q = value_sum / visit_count, or 0.0 if visit_count == 0
    if node['visit_count'] == 0:
        return 0.0
    return node['value_sum'] / node['visit_count']

# Step 29 - ucb_score
import math

def ucb_score(parent, child, c_puct=1.5):
    # TODO: return Q(child) + c_puct * prior * sqrt(N_parent) / (1 + N_child)
    q = node_q_value(child)
    u = c_puct * child['prior'] * math.sqrt(parent['visit_count']) / (1 + child['visit_count'])
    return q + u

# Step 30 - select_best_child
def select_best_child(node, legal_actions, c_puct=1.5):
    # TODO: return (action, child) maximizing PUCT among legal children of node.
    best_score = float('-inf')
    best_action = None
    best_child = None

    for action in legal_actions:
        if action in node['children']:
            child = node['children'][action]
            score = ucb_score(node, child, c_puct)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
    return best_action, best_child

# Step 31 - select_leaf
def select_leaf(root, c_puct):
    # TODO: walk down the MCTS tree picking the best PUCT child until a non-expanded node is reached
    node = root

    while node.get('is_expanded', False):
        legal_actions = list(node['children'].keys())
        if not legal_actions:
            break
        _, node = select_best_child(node, legal_actions, c_puct)
    return node

# Step 32 - evaluate_with_network
def evaluate_with_network(net, state, to_play):
    # TODO: run net on encoded state and return (masked priors np.ndarray (7,), value float)
    net.eval()

    with torch.no_grad():
        tensor_state = board_to_torch_tensor(state, to_play)
        
        logits, value_tensor = policy_value_forward(net, tensor_state)
        
        mask = action_mask(state)
        masked_logits = masked_policy_logits(logits.squeeze(0), mask)
        log_probs = masked_log_softmax(masked_logits, mask)
        
        priors = torch.exp(log_probs).cpu().numpy()
        
        priors = priors * mask
        if priors.sum() > 0:
            priors = priors / priors.sum()
            
        value = float(value_tensor.squeeze().item())
        
    return priors, value

# Step 33 - expand_node
def expand_node(node, priors):
    # TODO: attach a child node for every legal move with the corresponding network prior
    current_board = node['board']
    current_player = node['to_play']
    
    legal_moves = valid_moves(current_board)
    
    for move in legal_moves:
        child = make_mcts_node()
        child['board'] = drop_piece(current_board, move, current_player)
        child['to_play'] = other_player(current_player)
        child['prior'] = priors[move]
        child['parent'] = node  # Connect child to parent
        
        node['children'][move] = child
        
    node['is_expanded'] = True

# Step 34 - backup_value
def backup_value(leaf, value):
    # TODO: walk from leaf up through parents, updating visit_count and value_sum with alternating signs
    current = leaf
    current_value = value
    
    while current is not None:
        current['visit_count'] += 1
        current['value_sum'] += current_value
        
        current_value = -current_value
        current = current['parent']

# Step 35 - run_one_simulation
def run_one_simulation(root, net, c_puct):
    # TODO: run one MCTS simulation: select a leaf, evaluate, expand if non-terminal, backup.
    leaf = select_leaf(root, c_puct)
    
    board = leaf['board']
    to_play = leaf['to_play']
    
    terminal, winner = is_terminal(board)
    
    if terminal:
        leaf['is_expanded'] = False
        if winner == 0:
            value = 0.0
        elif winner == to_play:
            value = 1.0
        else:
            value = -1.0
    else:
        priors, value = evaluate_with_network(net, board, to_play)
        expand_node(leaf, priors)
        
    backup_value(leaf, value)

# Step 36 - run_mcts
def run_mcts(state, to_play, net, num_simulations, c_puct):
    # TODO: build a fresh root for (state, to_play) and run num_simulations PUCT simulations
    root = make_mcts_node()
    root['board'] = state
    root['to_play'] = to_play

    for _ in range(num_simulations):
        run_one_simulation(root, net, c_puct)

    return root

# Step 37 - visit_count_policy
def visit_count_policy(root, temperature=1.0):
    # TODO: convert root child visit counts into a length-7 probability vector over columns
    counts = [0.0] * 7
    children = root.get('children', {})
    
    for col, child in children.items():
        counts[col] = float(child['visit_count'])
        
    total_visits = sum(counts)
    
    # Fallback to uniform distribution if no visits exist
    if total_visits == 0:
        return [1.0 / 7.0] * 7
        
    # Temperature 0: Argmax (one-hot vector for the most-visited column)
    if temperature == 0:
        pi = [0.0] * 7
        max_val = max(counts)
        best_col = counts.index(max_val)
        pi[best_col] = 1.0
        return pi
        
    # Temperature > 0: Exponential scaling by 1 / temperature
    counts_temp = [c ** (1.0 / temperature) for c in counts]
    sum_temp = sum(counts_temp)
    
    return [c / sum_temp for c in counts_temp]

# Step 38 - mcts_choose_action
def mcts_choose_action(state, to_play, net, num_simulations, c_puct, temperature=1.0):
    root = run_mcts(state, to_play, net, num_simulations, c_puct)
    policy = visit_count_policy(root, temperature)
    
    if not isinstance(policy, torch.Tensor):
        policy_tensor = torch.tensor(policy, dtype=torch.float32)
    else:
        policy_tensor = policy
        
    mask = action_mask(state)
    action = sample_action_from_policy(policy_tensor, mask)

    if isinstance(policy, torch.Tensor):
        policy = policy.cpu().numpy()
    elif not hasattr(policy, 'shape'):
        policy = np.array(policy, dtype=np.float32)
        
    return int(action), policy

# Step 39 - record_self_play_step
def record_self_play_step(history, board, policy, to_play):
    # TODO: append a dict with 'board', 'policy', 'to_play' to history and return history
    step_data = {
        'board': board.copy(),
        'policy': policy.copy(),
        'to_play': to_play
    }

    history.append(step_data)
    return history

# Step 40 - play_self_play_game
def play_self_play_game(net, num_simulations, c_puct, temperature=1.0):
    board = make_empty_board()
    to_play = 1
    history = []

    while True:
        action, policy = mcts_choose_action(board, to_play, net, num_simulations, c_puct, temperature)
        policy = np.asarray(policy, dtype=np.float64)
        record_self_play_step(history, board, policy, to_play)
        board, done, winner, to_play = step_env(board, action, to_play)
        if done:
            return history, winner

# Step 41 - assign_value_targets
def assign_value_targets(history, winner):
    # TODO: return a new list of step dicts each annotated with a 'value' target in {-1.0, 0.0, 1.0}.
    new_history = []
    for step in history:
        new_step = step.copy()
        if winner == 0:
            value = 0.0
        elif step['to_play'] == winner:
            value = 1.0
        else:
            value = -1.0
        new_step['value'] = value
        new_history.append(new_step)
    return new_history

# Step 42 - generate_self_play_batch
def generate_self_play_batch(net, num_games, num_simulations, c_puct, temperature=1.0):
    # TODO: play num_games self-play games and return a flat list of labelled step dicts.
    buffer = []

    for _ in range(num_games):
        history, winner = play_self_play_game(
            net, num_simulations, c_puct, temperature
        )
        labelled_history = assign_value_targets(history, winner)
        buffer.extend(labelled_history)
    return buffer

# Step 43 - value_loss_mse
import torch

def value_loss_mse(predicted_values, target_values):
    # TODO: return the mean squared error between predicted and target values
    return torch.mean((predicted_values - target_values) ** 2)

# Step 44 - policy_loss_cross_entropy
import torch

def policy_loss_cross_entropy(predicted_log_probs, target_policy):
    """Cross-entropy between MCTS target policy and network log-probs. Returns scalar tensor."""
    # TODO: compute -sum(target * log_probs) per row, then average over the batch
    return -torch.mean(torch.sum(target_policy * predicted_log_probs, dim=-1))

# Step 45 - l2_regularization_loss
def l2_regularization_loss(net):
    # Pass torch.tensor(0.0) as the start argument so it always returns a tensor
    return sum((torch.sum(p ** 2) for p in net.parameters() if p.requires_grad), torch.tensor(0.0))

# Step 46 - combined_loss
import torch

def value_loss_mse(predicted_values, target_values):
    return torch.mean((predicted_values - target_values) ** 2)

def policy_loss_cross_entropy(predicted_log_probs, target_policy):
    return -torch.mean(torch.sum(target_policy * predicted_log_probs, dim=-1))

def l2_regularization_loss(net):
    return sum(torch.sum(p ** 2) for p in net.parameters() if p.requires_grad)

def combined_loss(predicted_log_probs, predicted_values, target_policy, target_values, net, policy_weight=1.0, value_weight=1.0, l2_weight=1e-4):
    # Compute the individual loss components using helper functions
    policy_loss = policy_loss_cross_entropy(predicted_log_probs, target_policy)
    value_loss = value_loss_mse(predicted_values, target_values)
    l2_loss = l2_regularization_loss(net)
    
    # Combine using the given weights
    total_loss = (policy_weight * policy_loss) + (value_weight * value_loss) + (l2_weight * l2_loss)
    
    # Store the unweighted components in a dictionary
    parts = {
        'policy': policy_loss,
        'value': value_loss,
        'l2': l2_loss
    }
    
    return total_loss, parts

# Step 47 - encode_batch_states
import torch

def encode_batch_states(boards, to_plays):
    # Call board_to_torch_tensor directly with board and to_play for each item, then stack
    tensors = [board_to_torch_tensor(board, to_play) for board, to_play in zip(boards, to_plays)]
    return torch.cat(tensors)

# Step 48 - iterate_minibatches
import numpy as np

def iterate_minibatches(buffer, batch_size, seed=None):
    """Yield shuffled minibatches of step dicts of size <= batch_size."""
    rng = np.random.default_rng(seed)
    
    indices = np.arange(len(buffer))
    rng.shuffle(indices)
    
    for start_idx in range(0, len(buffer), batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        yield [buffer[i] for i in batch_indices]

# Step 49 - training_step
import torch

def training_step(net, optimizer, minibatch, policy_weight=1.0, value_weight=1.0, l2_weight=1e-4):
    optimizer.zero_grad()
    
    boards = [item['board'] for item in minibatch]
    to_plays = [item['to_play'] for item in minibatch]
    target_policies = torch.tensor([item['policy'] for item in minibatch], dtype=torch.float32)
    target_values = torch.tensor([item['value'] for item in minibatch], dtype=torch.float32).unsqueeze(-1)
    
    batch_tensor = encode_batch_states(boards, to_plays)
    policy_logits, predicted_values = policy_value_forward(net, batch_tensor)
    
    masks = torch.stack([torch.tensor(action_mask(board), dtype=torch.float32) for board in boards])
    
    # Pass both policy_logits and masks to masked_log_softmax (or use masked_logits with masks)
    predicted_log_probs = masked_log_softmax(policy_logits, masks)
    
    total_loss, parts = combined_loss(
        predicted_log_probs, predicted_values, target_policies, target_values, 
        net, policy_weight=policy_weight, value_weight=value_weight, l2_weight=l2_weight
    )
    
    total_loss.backward()
    optimizer.step()
    
    return {
        'total': total_loss.item(),
        'policy': parts['policy'].item(),
        'value': parts['value'].item(),
        'l2': parts['l2'].item()
    }

# Step 50 - training_epoch
def training_epoch(net, optimizer, buffer, batch_size, policy_weight=1.0, value_weight=1.0, l2_weight=1e-4, seed=None):
    # Run one full pass over the self-play buffer and return the average of each loss component.
    metrics = []
    
    for minibatch in iterate_minibatches(buffer, batch_size, seed=seed):
        step_metrics = training_step(
            net, optimizer, minibatch, 
            policy_weight=policy_weight, 
            value_weight=value_weight, 
            l2_weight=l2_weight
        )
        metrics.append(step_metrics)
    
    # Compute the mean for each key across all minibatches
    keys = ['total', 'policy', 'value', 'l2']
    avg_metrics = {key: sum(m[key] for m in metrics) / len(metrics) for key in keys}
    
    return avg_metrics

# Step 51 - self_play_iteration
def self_play_iteration(net, optimizer, num_games, num_simulations, c_puct, batch_size, num_epochs=1, temperature=1.0):
    # 1. Generate a fresh buffer of self-play games
    buffer = generate_self_play_batch(net, num_games=num_games, num_simulations=num_simulations, c_puct=c_puct, temperature=temperature)
    
    # If the buffer is empty (e.g. num_games = 0), return zero losses for each epoch
    if len(buffer) == 0:
        zero_loss = {'total': 0.0, 'policy': 0.0, 'value': 0.0, 'l2': 0.0}
        return {
            'buffer_size': 0,
            'losses': [zero_loss] * num_epochs
        }
    
    # 2. Train the network on that buffer for num_epochs using training_epoch
    losses = []
    for _ in range(num_epochs):
        epoch_loss = training_epoch(net, optimizer, buffer, batch_size=batch_size)
        losses.append(epoch_loss)
        
    # 3. Return a dictionary containing the buffer size and the list of loss dicts
    return {
        'buffer_size': len(buffer),
        'losses': losses
    }

# Step 52 - train_loop
def train_loop(net, optimizer, num_iterations, num_games, num_simulations, c_puct, batch_size, num_epochs=1, temperature=1.0):
    # Repeat self_play_iteration for num_iterations and collect the results into a list
    history = []
    for _ in range(num_iterations):
        result = self_play_iteration(
            net, optimizer, 
            num_games=num_games, 
            num_simulations=num_simulations, 
            c_puct=c_puct, 
            batch_size=batch_size, 
            num_epochs=num_epochs, 
            temperature=temperature
        )
        history.append(result)
    return history

# Step 53 - random_policy_action
import numpy as np

def random_policy_action(state, to_play, rng=None):
    """Pick a uniformly random legal column on the given board."""
    if rng is None:
        rng = np.random.default_rng()
    
    legal_cols = valid_moves(state)
    return int(rng.choice(legal_cols))

# Step 54 - greedy_agent_action
import torch

def greedy_agent_action(net, state, to_play):
    board_tensor = board_to_torch_tensor(encode_board(state, to_play), to_play)
    
    # If board_to_torch_tensor returns shape (1, 2, 6, 7) or (2, 2, 6, 7), 
    # we want to make sure it's strictly 4D: (Batch, Channels, Height, Width)
    if board_tensor.dim() == 5:
        board_tensor = board_tensor.squeeze(0)
    if board_tensor.dim() == 3:
        board_tensor = board_tensor.unsqueeze(0)
    elif board_tensor.dim() == 4 and board_tensor.shape[0] != 1:
        # If it has 4 dimensions like (2, 2, 6, 7), take the first slice or correct it
        board_tensor = board_tensor[0:1] # keeps batch dim as 1
        
    policy_logits, _ = policy_value_forward(net, board_tensor)
    
    logits = policy_logits.squeeze(0)
    mask = torch.tensor(action_mask(state), dtype=torch.float32)
    masked_logits = logits.masked_fill(mask == 0, float('-inf'))
    
    return int(torch.argmax(masked_logits).item())

# Step 55 - play_one_match
def play_one_match(agent_one, agent_two, starting_player=1):
    board = make_empty_board()
    current_player = starting_player
    
    while True:
        # 1. Get action from the current agent
        if current_player == 1:
            action = agent_one(board, current_player)
        else:
            action = agent_two(board, current_player)
            
        # 2. Step the environment safely by indexing
        step_result = step_env(board, current_player, action)
        board = step_result[0]
        winner = step_result[1]
        is_done = step_result[2]
        
        # 3. Check if the game has ended
        if is_done:
            return winner  # 1 for agent_one, 2 for agent_two, 0 for draw
            
        # 4. Switch player for the next turn
        current_player = other_player(current_player)

# Step 56 - match_win_rate
def match_win_rate(agent_one, agent_two, num_matches, alternate_starts=True):
    stats = {'wins': 0, 'losses': 0, 'draws': 0}
    
    for i in range(num_matches):
        if alternate_starts and (i % 2 == 1):
            # Alternate starts: agent_two goes first, agent_one goes second
            # From agent_one's perspective, if agent_two wins, it's a loss for agent_one.
            winner = play_one_match(agent_two, agent_one, starting_player=1)
            # Invert winner since agents were swapped:
            # If winner == 1, agent_two won -> loss for agent_one
            # If winner == 2, agent_one won -> win for agent_one
            if winner == 2:
                stats['wins'] += 1
            elif winner == 1:
                stats['losses'] += 1
            else:
                stats['draws'] += 1
        else:
            # Normal start: agent_one goes first
            winner = play_one_match(agent_one, agent_two, starting_player=1)
            if winner == 1:
                stats['wins'] += 1
            elif winner == 2:
                stats['losses'] += 1
            else:
                stats['draws'] += 1
                
    return stats

# Step 57 - evaluate_against_random
import random
import torch

def evaluate_against_random(net, num_matches, seed=None):
    """
    Measures how often the greedy network agent beats a uniform-random baseline
    over num_matches Connect-4 games.
    """
    if seed is not None:
        random.seed(seed)
        
    stats = {'wins': 0, 'losses': 0, 'draws': 0}
    
    for i in range(num_matches):
        env = Connect4()
        
        current_player = 1 if (i % 2 == 0) else 2
        
        done = False
        while not done:
            if env.current_player == 1:
                state = env.get_state() 
                action = get_greedy_action(net, env)
            else:
                valid_actions = env.get_valid_actions()
                action = random.choice(valid_actions)
                
            env.step(action)
            done = env.is_terminal()
            
        result = env.get_outcome() 
        if result == 1:
            stats['wins'] += 1
        elif result == -1:
            stats['losses'] += 1
        else:
            stats['draws'] += 1
            
    return stats

