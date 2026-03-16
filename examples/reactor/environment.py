import numpy as np

def clip(x, low, high):
    return min(high, max(low, x))

class ReactorEnv:
    """
    Single-class environment: holds hidden mu, simulates transition + sensor,
    discretizes observation into bins, and samples reward.
    """

    def __init__(self, *,
                 n_bins,
                 max_abs_action,
                 mu_min, mu_max,
                 mu_lo, mu_hi, mu_hot,
                 alpha, delta,
                 sigma_trans, sigma_sensor, sigma_R,
                 c=0.01, M=1000.0,
                 horizon=200,
                 seed=None,
                 start_mu=None,
                 return_bins=True):
        # --- MDP specification / discretization ---
        self.n_bins = n_bins
        self.actions = list(range(-max_abs_action, max_abs_action + 1))

        self.mu_min, self.mu_max = mu_min, mu_max
        self.mu_lo, self.mu_hi = mu_lo, mu_hi
        self.mu_hot = mu_hot

        # bins over [mu_min, mu_max); below -> -2, above -> -1
        binsz = (mu_max - mu_min) / n_bins
        self.bins = [(mu_min + i*binsz, mu_min + (i+1)*binsz) for i in range(n_bins)]

        # --- dynamics / noise ---
        self.alpha, self.delta = alpha, delta
        self.sigma_trans = sigma_trans
        self.sigma_sensor = sigma_sensor
        self.sigma_R = sigma_R
        self.c, self.M = c, M
        self.horizon = horizon
        self.return_bins = return_bins

        self.rng = np.random.default_rng(seed)

        self.start_mu = start_mu if start_mu is not None else (mu_min + 1e-3)
        self.mu = None
        self.t = 0

    # ----- discretization -----
    def get_state_from_read(self, z):
        if z < self.mu_min:
            return -2  # too cold observation
        if z >= self.mu_max:
            return -1  # meltdown observation
        # map into 0..n_bins-1
        i = int((z - self.mu_min) / ((self.mu_max - self.mu_min) / self.n_bins))
        return min(max(i, 0), self.n_bins - 1)

    # ----- reward model -----
    def exp_reward(self, mu, a):
        # gets expected reward for a given (state, action) combo
        if mu >= self.mu_max:
            return -self.M
        if mu < self.mu_lo:
            return -self.c * abs(a)
        if mu <= self.mu_hi:
            return (mu - self.mu_lo) - self.c * abs(a)  # w(mu)=mu-mu_lo
        # above productive band but below meltdown: keep same form (reasonable default)
        return (mu - self.mu_lo) - self.c * abs(a)

    def reset(self):
        # sets environment back to beginning
        self.t = 0
        self.mu = self.start_mu
        z = self.rng.normal(self.mu, self.sigma_sensor)
        s = self.get_state_from_read(z)
        return (s if self.return_bins else z), {"mu": self.mu, "z": z}

    # returns observation of next state for agent, reward, done or not, and info (not for agent)
    def step(self, a):
        # optional: validate action
        # if a not in self.actions: raise ValueError("illegal action")

        # reward from current mu_t
        mean_R = self.exp_reward(self.mu, a)
        r = self.rng.normal(mean_R, self.sigma_R)

        # transition mu_t -> mu_{t+1}
        drift = self.delta if self.mu >= self.mu_hot else 0.0
        eps = self.rng.normal(0.0, self.sigma_trans)
        self.mu = clip(self.mu - self.alpha * a + drift + eps, self.mu_min, self.mu_max)

        # new observation
        z_next = self.rng.normal(self.mu, self.sigma_sensor)
        s_next = self.get_state_from_read(z_next)

        self.t += 1
        done = (self.mu >= self.mu_max) or (self.t >= self.horizon)
        info = {"mu": self.mu, "z": z_next, "mean_R": mean_R}

        obs_next = (s_next if self.return_bins else z_next)
        return obs_next, r, done, info