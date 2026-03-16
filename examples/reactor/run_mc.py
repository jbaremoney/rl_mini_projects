import numpy as np
from examples.reactor.environment import ReactorEnv
from examples.reactor.monte_carlo import mc_control
import matplotlib.pyplot as plt

# ----------------------------
# Turn Q into greedy policy
# ----------------------------

def greedy_policy_from_q(env, Q):
    """
    Returns:
        policy_actions: array of shape (n_bins,)
        where policy_actions[s] is the chosen action for state s
    """
    best_action_indices = np.argmax(Q, axis=1)
    policy_actions = np.array([env.actions[idx] for idx in best_action_indices])
    return policy_actions


# ----------------------------
# Evaluate a learned policy
# ----------------------------

def run_policy_episode(env, policy_actions, render=False):
    """
    Runs one episode using a deterministic policy:
        action = policy_actions[state]

    Returns summary stats for the episode.
    """
    state, info = env.reset()
    done = False

    total_reward = 0.0
    trajectory = []

    while not done:
        action = policy_actions[state]
        next_state, r, done, info = env.step(action)

        trajectory.append((state, action, r, info["mu"], info["z"]))
        total_reward += r
        state = next_state

    melted_down = info["mu"] >= env.mu_max

    return {
        "total_reward": total_reward,
        "steps": len(trajectory),
        "melted_down": melted_down,
        "trajectory": trajectory,
    }


def evaluate_policy(env, policy_actions, num_episodes=100):
    returns = []
    meltdowns = 0
    lengths = []

    for _ in range(num_episodes):
        out = run_policy_episode(env, policy_actions)
        returns.append(out["total_reward"])
        lengths.append(out["steps"])
        if out["melted_down"]:
            meltdowns += 1

    return {
        "avg_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "avg_length": float(np.mean(lengths)),
        "meltdown_rate": meltdowns / num_episodes,
        "returns": returns,
        "lengths": lengths,
    }


# ----------------------------
# Main experiment
# ----------------------------

def main():
    env = ReactorEnv(
        n_bins=20,
        max_abs_action=2,
        mu_min=0.0,
        mu_max=10.0,
        mu_lo=3.0,
        mu_hi=6.0,
        mu_hot=7.5,
        alpha=0.4,
        delta=0.3,
        sigma_trans=0.15,
        sigma_sensor=0.2,
        sigma_R=0.5,
        c=0.01,
        M=1000.0,
        horizon=200,
        seed=42,
        start_mu=0.5,
        return_bins=True,
    )

    num_train_episodes = 5000

    Q, training_returns = mc_control(
        env,
        num_episodes=num_train_episodes,
        gamma=0.95,
        eps=0.1,
    )

    learned_policy = greedy_policy_from_q(env, Q)

    print("Learned greedy policy by state:")
    for s in range(env.n_bins):
        print(f"state {s:2d} -> action {learned_policy[s]}")

    eval_stats = evaluate_policy(env, learned_policy, num_episodes=200)

    print("\nEvaluation results:")
    print(f"Average return:   {eval_stats['avg_return']:.3f}")
    print(f"Return std:       {eval_stats['std_return']:.3f}")
    print(f"Average length:   {eval_stats['avg_length']:.3f}")
    print(f"Meltdown rate:    {eval_stats['meltdown_rate']:.3%}")

    # plot learning curve
    plt.figure(figsize=(10, 5))
    plt.plot(training_returns, alpha=0.4, label="Episode return")

    # moving average
    window = 100
    if len(training_returns) >= window:
        moving_avg = np.convolve(training_returns, np.ones(window) / window, mode='valid')
        plt.plot(range(window - 1, len(training_returns)), moving_avg, label=f"{window}-ep moving avg")

    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Monte Carlo Training Returns")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # plot Q heatmap
    plt.figure(figsize=(10, 6))
    plt.imshow(Q, aspect='auto')
    plt.colorbar(label="Q(s,a)")
    plt.xlabel("Action index")
    plt.ylabel("State bin")
    plt.title("Learned Q-table")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()