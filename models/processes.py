from abc import abstractmethod, ABC
import numpy as np
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import gmres


class FiniteMarkovProcess(ABC):
    # TODO: GENERALIZE INITIAL STATE AND INITIAL STATE DISTRIBUTION TO ONE OBJECT, fix initializing/current state
    def __init__(self, states: list, trans_kernel=None, initial_state=None, initial_dist=None):
        self.rng = np.random.default_rng()
        self.states = states # keeping as list just lets us access by index
        self.n = len(states)
        self.state_to_idx = {s: i for i, s in enumerate(states)}
        self.initial_state = initial_state
        if trans_kernel is not None:
            self.trans_kernel = trans_kernel
        else:
            pass
        # only need to define initial dist if no initial state
        if self.initial_state is None:
            if initial_dist is None:
                self.initial_dist = [1/self.n for i in range(self.n)] # uniform
            else:
                self.initial_dist = initial_dist

    @abstractmethod
    def reset(self):
        ...

    @abstractmethod
    def step(self):
        ...


class FiniteMarkovChain(FiniteMarkovProcess):
    #TODO: add key value for state indices,
    def __init__(self, states: list, trans_kernel: np.array, initial_state=None, initial_state_dist=None):
        super().__init__(states, trans_kernel, initial_state, initial_state_dist)

    def reset(self):
        if self.initial_state is not None:
            self.current_state = self.initial_state

        elif self.initial_dist:
            self.current_state = self.rng.choice(self.n, 1, self.initial_dist)[0] # returns np array even for j 1
        else:
            self.current_state = self.rng.choice(self.n, 1)[0]

        return self.current_state

    def get_next_state_dist(self):
        # just index the matrix to get vector of probabilities
        return self.trans_kernel[self.current_state]

    def step(self):

        dist = self.get_next_state_dist()

        #numpy random choice based on prob vector got by indexing matrix
        self.current_state = self.rng.choice(self.n, 1, dist)[0]

        return self.current_state

class FiniteMarkovRewards(FiniteMarkovChain):
    def __init__(self, states: list, trans_kernel: np.array, rewards: np.array, initial_state=None, initial_state_dist=None):
        super().__init__(states, trans_kernel, initial_state, initial_state_dist)

        self.rewards = rewards

    def exp_rewards(self, state=None):
        # weighted sums of states' rewards, weighted by prob
        if state is not None:
            dist = self.trans_kernel[state]

            rewards = self.rewards
            print("REWARDS")
            print(rewards)

            # @ operator usage lets go (dot product)
            return dist @ rewards

        else:
            dist = self.trans_kernel[self.current_state]
            rewards = self.rewards

            # first ever @ operator usage lets go (dot product np)
            return dist @ rewards

    # used AI for this function for now
    def value_function(self, gamma=0.99, rtol=1e-8, maxiter=None, return_dict=False):
        """
        Solve (I - gamma P) v = r for the infinite-horizon discounted value function.

        Assumes rewards are state-based:
            self.rewards[i] = reward received when in state i
        """
        if not (0 <= gamma < 1):
            raise ValueError("gamma must be in [0, 1) for discounted problems.")

        # Transition matrix
        P = csr_matrix(self.transition_matrix)
        I = identity(self.n, format="csr")

        # Reward vector (state-based)
        r = np.asarray(self.rewards, dtype=float)
        if r.shape != (self.n,):
            raise ValueError(f"rewards must be length {self.n}; got shape {r.shape}")

        # Solve linear system
        A = I - gamma * P
        v, info = gmres(A, r, rtol=rtol, maxiter=maxiter)

        if info != 0:
            raise RuntimeError(
                f"GMRES failed to converge (info={info}). "
                f"Try increasing maxiter, loosening rtol, or using a different solver."
            )

        if return_dict:
            return {self.states[i]: float(v[i]) for i in range(self.n)}
        return v


class FiniteMarkovDecisionProcess(FiniteMarkovProcess):
    def __init__(self, states: list, transition_matrix: np.array, rewards: list, initial_state=None,
                 initial_state_dist=None):
        super(FiniteMarkovDecisionProcess, self).__init__(states, transition_matrix, initial_state, initial_state_dist)

        self.rewards = rewards
