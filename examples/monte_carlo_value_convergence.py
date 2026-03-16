import numpy as np

INIT_VAL_GRID = np.zeros((5,5))
print(INIT_VAL_GRID)

REWARDS = np.ones_like(INIT_VAL_GRID)
print(REWARDS)

class Sim:
    def __init__(self, init_val_grid, init_reward_grid, init_state, states_to_actions):
        self.vals = init_val_grid
        self.rewards = init_reward_grid
        self.current_state = init_state

        self.states_to_actions = states_to_actions

        self.state_action_to_state_dists = {}

    def update_value(self, state):
        # max over possible actions to exp(rewards + value of next state)
        actions = self.states_to_actions[state] #vector

        # state dists given actions
        state_dists = self.state_action_to_state_dists
        state_dist_rw = self.rewards[state]
        res_dict = {}
        for action in actions:
            # calculate expectation
            exp_rw = state_dists[action] @ self.rewards[]

