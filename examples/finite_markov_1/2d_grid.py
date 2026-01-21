from models.finite_markov import FiniteMarkovRewards
import numpy as np

def generate_state_grid(shape:tuple):
    n_rows, n_cols = shape

    if n_rows * n_cols == 0:
        return []

    m = []
    row = -1
    for i in range(n_rows * n_cols):
        if i % n_cols == 0:
            row += 1
            m.append([f"state{i}"])  # new row
        else:
            m[row].append(f"state{i}")

    return np.array(m)

SHAPE = (5,5)
STATE_GRID = generate_state_grid(SHAPE)
print(STATE_GRID)

# need to flatten states to pass to class, makes a new copy
flat_states = STATE_GRID.flatten().tolist()
print(flat_states)

# todo: generate all with same rng
rng = np.random.default_rng()

# row = current state
# 25x25 matrix since we have 25 states.
# consider flattened joined state matrix, then walls are index 1, 6, 8, 11, 13, 18
# just make transitions to the walls probability 0
n_rows, n_cols = SHAPE
P_matrix = rng.random(size=(n_cols*n_rows, n_cols*n_rows))

# standardize it, sum of each row must equal 1. vectorized ops
row_sums = P_matrix.sum(axis=1, keepdims=True)
P_matrix = P_matrix/row_sums


# static rewards for each state... same shape as grid
R_matrix = rng.random(size=SHAPE)
flat_rewards = R_matrix.flatten().tolist()

# 0 out probability of transitions to walls
walls = [1, 6, 8, 11, 13, 18]
P_matrix[:, walls] = 0

FMRP = FiniteMarkovRewards(flat_states, P_matrix, flat_rewards)

print(FMRP.exp_rewards(1))