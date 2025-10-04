import numpy as np
import matplotlib.pyplot as plt
import random

class GridWorldMaze:
    def __init__(self, penalty=-1):
        # Grid dimensions: 3x4
        self.rows = 3
        self.cols = 4

        # Terminal states coordinates
        self.terminal_states = [(0, 3), (1, 3)]

        # Actions: 0=North, 1=East, 2=South, 3=West
        self.actions = 4
        self.action_symbols = ['↑', '→', '↓', '←']

        # Rewards
        self.rewards = np.full((self.rows, self.cols), -0.04)
        self.rewards[0, 3] = 1.0
        self.rewards[1, 3] = penalty

        # Wall positions
        self.walls = [(1, 1)]

    def is_valid_state(self, state):
        row, col = state
        return (0 <= row < self.rows and
                0 <= col < self.cols and
                (row, col) not in self.walls)

    def get_next_state(self, state, action):
        # Stochastic transitions
        prob = random.random()

        # With 80% probability, move in the desired direction
        if prob < 0.8:
            actual_action = action
        # With 10% probability, move left of the desired direction
        elif prob < 0.9:
            actual_action = (action - 1) % 4
        # With 10% probability, move right of the desired direction
        else:
            actual_action = (action + 1) % 4

        row, col = state
        if actual_action == 0:  # North
            next_state = (row - 1, col)
        elif actual_action == 1:  # East
            next_state = (row, col + 1)
        elif actual_action == 2:  # South
            next_state = (row + 1, col)
        else:  # West
            next_state = (row, col - 1)

        # Check if next state is valid
        if not self.is_valid_state(next_state):
            next_state = state

        reward = self.rewards[next_state[0], next_state[1]]
        done = next_state in self.terminal_states

        return next_state, reward, done

    def reset(self):
        while True:
            state = (random.randint(0, self.rows - 1), random.randint(0, self.cols - 1))
            if state not in self.terminal_states and state not in self.walls:
                return state

def q_learning(env, episodes=2000, alpha=0.1, gamma=0.9, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
    # Initialize Q-table
    q_table = np.zeros((env.rows, env.cols, env.actions))

    # For tracking convergence
    policy_history = []
    q_value_history = []
    rewards_history = []

    epsilon = epsilon_start

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, env.actions - 1)
            else:
                action = np.argmax(q_table[state[0], state[1]])

            # Take action
            next_state, reward, done = env.get_next_state(state, action)
            total_reward += reward

            # Q-learning update
            old_q = q_table[state[0], state[1], action]
            next_max_q = np.max(q_table[next_state[0], next_state[1]]) if not done else 0

            # Q-learning update formula
            new_q = old_q + alpha * (reward + gamma * next_max_q - old_q)
            q_table[state[0], state[1], action] = new_q

            state = next_state

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Track policy and Q-values for convergence analysis
        if episode % 10 == 0:
            current_policy = np.argmax(q_table, axis=2)
            policy_history.append(current_policy.copy())
            q_value_history.append(np.sum(np.abs(q_table)))

        rewards_history.append(total_reward)

        # Print progress occasionally
        if episode % 500 == 0:
            print(f"Episode {episode}/{episodes}, Avg Reward: {np.mean(rewards_history[-100:]):.2f}, Epsilon: {epsilon:.2f}")

    return q_table, policy_history, q_value_history, rewards_history

def visualize_maze_with_policy(env, q_table=None, title="GridWorld Maze"):
    """Visualize the maze with the current policy"""
    plt.figure(figsize=(10, 8))

    # Create a grid for the maze
    grid = np.zeros((env.rows, env.cols, 3))  # RGB channels

    # Color coding
    for i in range(env.rows):
        for j in range(env.cols):
            if (i, j) in env.walls:
                grid[i, j] = [0.3, 0.3, 0.3]  # Gray for walls
            elif (i, j) == (0, 3):
                grid[i, j] = [0.0, 0.8, 0.0]  # Green for +1 terminal
            elif (i, j) == (1, 3):
                grid[i, j] = [0.8, 0.0, 0.0]  # Red for negative terminal
            else:
                grid[i, j] = [1.0, 1.0, 1.0]  # White for normal cells

    plt.imshow(grid)

    # Add gridlines
    for i in range(env.rows + 1):
        plt.axhline(i - 0.5, color='black', linewidth=1)
    for j in range(env.cols + 1):
        plt.axvline(j - 0.5, color='black', linewidth=1)

    # Add rewards text
    for i in range(env.rows):
        for j in range(env.cols):
            if (i, j) not in env.walls:
                plt.text(j, i, f"{env.rewards[i, j]:.2f}", ha='center', va='center', color='black')

    # Add policy arrows if Q-table is provided
    if q_table is not None:
        policy = np.argmax(q_table, axis=2)
        for i in range(env.rows):
            for j in range(env.cols):
                if (i, j) not in env.terminal_states and (i, j) not in env.walls:
                    action = policy[i, j]
                    plt.text(j, i - 0.2, env.action_symbols[action],
                             ha='center', va='center', color='blue', fontsize=20)

    plt.title(title)
    plt.xticks(range(env.cols))
    plt.yticks(range(env.rows))
    plt.grid(True)

    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()

def plot_convergence(policy_history, q_value_history):
    """Plot convergence of policy and Q-values"""
    # Calculate policy changes between consecutive recorded policies
    policy_changes = []
    for i in range(len(policy_history) - 1):
        changes = np.sum(policy_history[i] != policy_history[i+1])
        policy_changes.append(changes)

    # Plot results
    plt.figure(figsize=(15, 6))

    # Plot Q-value convergence
    plt.subplot(1, 2, 1)
    episodes_q = np.arange(0, len(q_value_history) * 10, 10)
    plt.plot(episodes_q, q_value_history)
    plt.title('Q-value Convergence')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of absolute Q-values')
    plt.grid(True)

    # Plot policy changes
    plt.subplot(1, 2, 2)
    episodes_p = np.arange(10, (len(policy_changes) + 1) * 10, 10)[:len(policy_changes)]
    plt.plot(episodes_p, policy_changes)
    plt.title('Policy Changes Over Time')
    plt.xlabel('Episodes')
    plt.ylabel('Number of policy changes')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('convergence_analysis.png')
    plt.show()

    # Determine which converges first
    non_zero_changes = [i for i, x in enumerate(policy_changes) if x > 0]
    last_policy_change = non_zero_changes[-1] if non_zero_changes else 0
    last_policy_change_episode = (last_policy_change + 1) * 10

    # Calculate differences in Q-values
    q_value_diffs = [abs(q_value_history[i+1] - q_value_history[i]) for i in range(len(q_value_history)-1)]
    convergence_threshold = 0.001

    # Find when Q-values stabilize below threshold
    stable_q_indices = [i for i, x in enumerate(q_value_diffs) if x < convergence_threshold]
    q_convergence_episode = (stable_q_indices[0] + 1) * 10 if stable_q_indices else len(q_value_diffs) * 10

    print(f"\nPolicy stabilized around episode {last_policy_change_episode}")
    print(f"Q-values approximately converged around episode {q_convergence_episode}")
    print(f"{'Policy' if last_policy_change_episode < q_convergence_episode else 'Q-values'} converged first")

def plot_hyperparameter_effects():
    """Test different hyperparameters and plot their effects"""
    env = GridWorldMaze(penalty=-1)

    # Test different learning rates
    alphas = [0.01, 0.1, 0.5]
    alpha_results = []

    for alpha in alphas:
        print(f"\nTesting learning rate alpha={alpha}")
        q_table, _, _, rewards = q_learning(env, episodes=1000, alpha=alpha, gamma=0.9)
        avg_reward = np.mean(rewards[-100:])
        alpha_results.append(avg_reward)

    # Test different discount factors
    gammas = [0.5, 0.9, 0.99]
    gamma_results = []

    for gamma in gammas:
        print(f"\nTesting discount factor gamma={gamma}")
        q_table, _, _, rewards = q_learning(env, episodes=1000, alpha=0.1, gamma=gamma)
        avg_reward = np.mean(rewards[-100:])
        gamma_results.append(avg_reward)

    # Plot hyperparameter results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.bar([str(a) for a in alphas], alpha_results)
    plt.title('Effect of Learning Rate (α)')
    plt.xlabel('Learning Rate')
    plt.ylabel('Average Reward (last 100 episodes)')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.bar([str(g) for g in gammas], gamma_results)
    plt.title('Effect of Discount Factor (γ)')
    plt.xlabel('Discount Factor')
    plt.ylabel('Average Reward (last 100 episodes)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('hyperparameter_analysis.png')
    plt.show()

def compare_penalties():
    """Compare results with different penalty values"""
    # Run with default penalty (-1)
    env_default = GridWorldMaze(penalty=-1)
    print("\nRunning Q-Learning with penalty=-1")
    q_table_default, _, _, _ = q_learning(env_default)
    visualize_maze_with_policy(env_default, q_table_default, title="Policy with Penalty = -1")

    # Run with high penalty (-200)
    env_high = GridWorldMaze(penalty=-200)
    print("\nRunning Q-Learning with penalty=-200")
    q_table_high, _, _, _ = q_learning(env_high)
    visualize_maze_with_policy(env_high, q_table_high, title="Policy with Penalty = -200")

    # Compare policies
    policy_default = np.argmax(q_table_default, axis=2)
    policy_high = np.argmax(q_table_high, axis=2)

    differences = np.sum(policy_default != policy_high)
    print(f"\nNumber of states with different optimal actions: {differences}")

    # Print differences in text format
    print("\nDifferences in policies:")
    for i in range(env_default.rows):
        for j in range(env_default.cols):
            if (i, j) not in env_default.terminal_states and (i, j) not in env_default.walls:
                if policy_default[i, j] != policy_high[i, j]:
                    print(f"State ({i},{j}): {env_default.action_symbols[policy_default[i, j]]} → {env_default.action_symbols[policy_high[i, j]]}")

def main():
    # Part 1: Basic Q-learning
    env = GridWorldMaze(penalty=-1)
    print("Running Q-Learning with default parameters")
    q_table, policy_history, q_value_history, _ = q_learning(env)

    # Visualize the maze and policy
    visualize_maze_with_policy(env, q_table, "Optimal Policy")

    # Part 2: Analyze convergence
    print("\nAnalyzing convergence...")
    plot_convergence(policy_history, q_value_history)

    # Part 3: Hyperparameter analysis
    print("\nAnalyzing hyperparameter effects...")
    plot_hyperparameter_effects()

    # Part 4: Compare penalties (optional)
    print("\nComparing different penalties...")
    compare_penalties()

if __name__ == "__main__":
    main()