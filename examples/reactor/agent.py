import numpy as np

class ReactorAgent:
    def __init__(self, policy: np.array, actions):
        self.policy = policy # matrix, row is current state, sample from the probabilities
        self.actions = actions

        seed = np.random.randint(0, 100)
        self.rng = np.random.default_rng(seed)

    def act(self, state):
        """

        :param state: integer corresponding to what is current bin
        :return:
        """

        # get row, indices correspond to actions
        p_row = self.policy[state]

        choice = self.rng.choice(self.actions, p_row)[0]
        return choice
