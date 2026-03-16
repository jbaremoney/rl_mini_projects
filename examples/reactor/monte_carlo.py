from examples.reactor.environment import ReactorEnv
import numpy as np

STATE_ACTION_PAIRS = []

# table to keep track of how many times we've encountered state action pair
sa_ct_table = {}

# if we store the actions as [0,1,-1, ...] we just need index to id what the action is
def get_action_from_index(env, index):
    return env.actions[index]

def get_index_from_action(env, action):
    return env.actions.index(action)

"""
Learns accurate Q values

TODO: Make return calculation the nice backwards loop way
"""
def mc_control(env, num_episodes, gamma=0.9, learning_rate=.9, eps=.1, Q_value_table=None):
    if Q_value_table is None:
        # initialize mapping states->value
        Q_value_table = np.ones((env.n_bins, len(env.actions)))



    for ep in range(num_episodes):
        episode_returns = []
        state, info = env.reset()
        done = False
        traj = []
        ep_return = 0.0

        while not done:
            # epsilon-greedy action choice
            if np.random.rand() > eps:
                action_idx = np.argmax(Q_value_table[state])
            else:
                action_idx = np.random.randint(len(env.actions))

            action = get_action_from_index(env, action_idx)

            new_state, r, done, info = env.step(action)

            # store CURRENT state, action, reward
            traj.append((state, action, r))
            ep_return += r
            state = new_state

        episode_returns.append(ep_return)

        # first-visit MC update
        seen = set()
        for t in range(len(traj)):
            s, a, _ = traj[t]
            sa = (s, a)

            if sa in seen:
                continue
            seen.add(sa)

            # discounted return from time t onward
            G_obs = sum((gamma ** j) * traj[t + j][2] for j in range(len(traj) - t))

            sa_ct_table[sa] = sa_ct_table.get(sa, 0) + 1
            n = sa_ct_table[sa]

            a_idx = get_index_from_action(env, a)
            old_q = Q_value_table[s][a_idx]

            # incremental sample-average update
            Q_value_table[s][a_idx] = old_q + (G_obs - old_q) / n

    return Q_value_table, episode_returns
