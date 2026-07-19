import numpy as np

class TabularQLearner:
    def __init__(self, state_dims, action_size, alpha=0.1, gamma=0.95, epsilon=1.0, min_epsilon=0.0):
        self.state_dims = state_dims
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        
        # Initialize Tabular structure
        self.q_table = np.zeros(self.state_dims + (self.action_size,))

    def choose_action(self, state, exploit_only=False):
        if not exploit_only and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, done):
        current_q = self.q_table[state][action]
        max_next_q = 0 if done else np.max(self.q_table[next_state])
        
        # Bellman Optimality Temporal Difference Formula Engine
        td_error = reward + self.gamma * max_next_q - current_q
        self.q_table[state][action] = current_q + self.alpha * td_error

    def decay_epsilon(self, episode, total_episodes):
        # Enforce hard exploit ceiling in the final 20% of training execution
        if episode > int(0.8 * total_episodes):
            self.epsilon = self.min_epsilon
        else:
            self.epsilon = max(self.min_epsilon, 1.0 - (episode / (0.8 * total_episodes)))