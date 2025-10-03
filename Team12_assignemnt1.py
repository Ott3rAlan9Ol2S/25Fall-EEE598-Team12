import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ----- Gridworld -----
ACTIONS = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}  # up, right, down, left
ARROWS = {0: "^", 1: ">", 2: "v", 3: "<"}

class Gridworld:
    def __init__(self, rows=3, cols=4, step_cost=-0.04, walls={(2,2)}, terminals={(3,4):1.0, (2,4):-1.0},
                 start=(1,1), stochastic=True, seed=0):
        self.rows, self.cols = rows, cols
        self.step_cost = step_cost
        self.walls = set(walls)
        self.terminals = dict(terminals)
        self.start = start
        self.stochastic = stochastic
        self.rng = np.random.default_rng(seed)
        self.nS = rows*cols
        self.nA = 4

    def _idx(self, r, c):
        return (r-1)*self.cols + (c-1)

    def _rc(self, idx):
        r0, c0 = divmod(idx, self.cols)
        return (r0+1, c0+1)

    def reset(self):
        return self._idx(*self.start)

    def is_terminal_rc(self, rc):
        return rc in self.terminals

    def is_wall_rc(self, rc):
        return rc in self.walls

    def step(self, s_idx, a):
        r, c = self._rc(s_idx)
        if self.is_terminal_rc((r,c)):
            # already terminal, stay with terminal reward only once
            return s_idx, self.terminals[(r,c)], True
        # intended 0.8, left 0.1, right 0.1
        if self.stochastic:
            probs = np.zeros(4)
            probs[a] = 0.8
            probs[(a-1) % 4] += 0.1  # left of intended
            probs[(a+1) % 4] += 0.1  # right of intended
            a = self.rng.choice(4, p=probs)
        dr, dc = ACTIONS[a]
        nr, nc = r+dr, c+dc
        # bounce on border or walls
        if not (1 <= nr <= self.rows and 1 <= nc <= self.cols) or self.is_wall_rc((nr,nc)):
            nr, nc = r, c
        reward = self.step_cost
        done = False
        if self.is_terminal_rc((nr,nc)):
            reward += self.terminals[(nr,nc)]
            done = True
        return self._idx(nr,nc), reward, done

# ----- Q-learning -----
def q_learning(env, episodes=10000, alpha=0.1, gamma=0.99, eps_start=1.0, eps_min=0.05, eps_decay=0.999, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.nS, env.nA))
    def eps_greedy(s, eps):
        if rng.random() < eps:
            return rng.integers(env.nA)
        maxq = Q[s].max()
        best = np.flatnonzero(np.isclose(Q[s], maxq))
        return rng.choice(best)
    eps = eps_start
    for ep in range(episodes):
        s = env.reset()
        done = False
        while not done:
            a = eps_greedy(s, eps)
            ns, r, done = env.step(s, a)
            target = r if done else r + gamma * Q[ns].max()
            Q[s, a] += alpha * (target - Q[s, a])
            s = ns
        eps = max(eps_min, eps*eps_decay)
    return Q

env = Gridworld()
Q = q_learning(env)

# ----- Greedy policy -----
V = Q.max(axis=1)
Pi = Q.argmax(axis=1)

# ----- Plot the result -----
def plot_policy_value(env, Pi, V):
    rows, cols = env.rows, env.cols
    fig, ax = plt.subplots(figsize=(6, 4))
    # grid
    for c in range(cols + 1):
        ax.plot([c, c], [0, rows], color="black", linewidth=1.5)
    for r in range(rows + 1):
        ax.plot([0, cols], [r, r], color="black", linewidth=1.5)
    # wall
    for (wr,wc) in env.walls:
        ax.add_patch(Rectangle((wc-1, wr-1), 1, 1, fill=False, color="grey", hatch="///", linewidth=2))
    # terminals
    for (tr,tc), rew in env.terminals.items():
        ax.text(tc-0.5, tr-0.5, f"{int(rew):+d}", color="red", ha="center", va="center", fontsize=16, fontweight="bold")
    # policy and values
    for r in range(1, rows+1):
        for c in range(1, cols+1):
            if (r,c) in env.walls or (r,c) in env.terminals:
                continue
            idx = env._idx(r,c)
            ax.text(c-0.85, r-0.85, f"{V[idx]:.2f}", fontsize=12, color="blue")
            ax.text(c-0.15, r-0.15, ARROWS[Pi[idx]], fontsize=30, color="green", ha="right", va="top")
    # start
    sr, sc = env.start
    ax.text(sc-0.95, sr-0.15, "START", fontsize=8, ha="left", va="top")
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.set_xticks([i+0.5 for i in range(cols)])
    ax.set_yticks([i+0.5 for i in range(rows)])
    ax.set_xticklabels([str(i) for i in range(1, cols+1)])
    ax.set_yticklabels([str(i) for i in range(1, rows+1)])
    ax.set_xlabel("Columns")
    ax.set_ylabel("Rows")
    ax.set_title("Q-learning greedy policy")
    plt.tight_layout()
    plt.show()

plot_policy_value(env, Pi, V)
