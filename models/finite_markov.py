from abc import abstractmethod, ABC
import numpy as np
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import gmres


class FiniteMarkovProcess(ABC):
    # TODO: add optional initial state distribution, key value for state indices
    def __init__(self, states: list):
        self.rng = np.random.default_rng()

        self.states = states

        self.n = len(states)

        self.state_to_idx = {s: i for i, s in enumerate(states)}

    @abstractmethod
    def reset(self):
        ...

    @abstractmethod
    def step(self):
        ...


class FiniteMarkovChain(FiniteMarkovProcess):
    #TODO: add key value for state indices,
    def __init__(self, states: list, transition_matrix: np.array, initial_state=None, initial_state_dist=None):

        super().__init__(states)

        self.transition_matrix = transition_matrix
        self.initial_state_distribution = initial_state_dist
        self.initial_state = None

        if initial_state is not None:
            self.current_state = initial_state

        elif self.initial_state_distribution:
            self.current_state = self.rng.choice(self.n, 1, self.initial_state_distribution)[0] # returns np array

        else:
            self.current_state = self.rng.choice(self.n, 1)[0]


    def reset(self):
        if self.initial_state is not None:
            self.current_state = self.initial_state

        elif self.initial_state_distribution:
            self.current_state = self.rng.choice(self.n, 1, self.initial_state_distribution)[0] # returns np array

        else:
            self.current_state = self.rng.choice(self.n, 1)[0]

        return self.current_state

    def get_next_state_dist(self):
        # just index the matrix to get vector of probabilities
        return self.transition_matrix[self.current_state]

    def step(self):

        dist = self.get_next_state_dist()

        #numpy random choice based on indexed dist
        self.current_state = self.rng.choice(self.n, 1, dist)[0]

        return self.current_state

class FiniteMarkovRewards(FiniteMarkovChain):
    def __init__(self, states: list, transition_matrix: np.array, rewards: list, initial_state=None, initial_state_dist=None):
        super(FiniteMarkovRewards, self).__init__(states, transition_matrix, initial_state, initial_state_dist)

        self.rewards = rewards

    def exp_rewards(self, state=None):
        # weighted sums of states' rewards, weighted by prob
        if state is not None:
            print("TRANS. MATRIX")
            print(self.transition_matrix)
            print(f"TRANSITION DIST GIVEN state={state}")
            print(self.transition_matrix[state])
            dist = self.transition_matrix[state]

            rewards = self.rewards
            print("REWARDS")
            print(rewards)

            # @ operator usage lets go (dot product)
            return dist @ rewards

        else:

            dist = self.transition_matrix[self.current_state]

            rewards = self.rewards

            # first ever @ operator usage lets go (dot product np)
            return dist @ rewards

    # used AI for this function for now
    def value_function(self, gamma=0.9, rtol=1e-8, maxiter=None, return_dict=False):
        """
        Solve (I - gamma P) v = r for the infinite-horizon discounted value function.
        Assumes rewards are transition-based: rewards[i,j] is reward for i -> j.
        """
        if not (0< gamma < 1):
            raise ValueError("gamma must be in (0, 1)")

        P = csr_matrix(self.transition_matrix)  # sparse is fine even if dense input
        I = identity(self.n, format="csr")

        # r[i] = E[reward | S_t = i] = sum_j P[i,j] * rewards[i,j]
        r = np.array([self.exp_rewards(i) for i in range(self.n)], dtype=float)

        A = I - gamma * P
        v, info = gmres(A, r, rtol=rtol, maxiter=maxiter)

        if info != 0:
            # info > 0: no convergence within maxiter; info < 0: breakdown
            raise RuntimeError(f"GMRES failed to converge (info={info}). Try larger maxiter or different solver.")

        if return_dict:
            return {self.states[i]: v[i] for i in range(self.n)}
        return v




