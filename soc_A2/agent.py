import numpy as np

class ProbeAgent:
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.999, min_epsilon=0.01):
        self.alpha = alpha              
        self.gamma = gamma              
        self.epsilon = epsilon          
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        self.h_bins = np.linspace(0, 1200, 50)  
        self.v_bins = np.linspace(-100, 20, 50) 
        self.num_wind = 3                       
        self.num_actions = 2                    
        
        self.q_table = np.zeros((len(self.h_bins) + 1, len(self.v_bins) + 1, self.num_wind, self.num_actions))

    def discretize_state(self, raw_state):
        h, v, wind_idx = raw_state
        h_discrete = int(np.digitize(h, self.h_bins))
        v_discrete = int(np.digitize(v, self.v_bins))
        return (h_discrete, v_discrete, int(wind_idx))

    def choose_action(self, state, evaluation=False):
        if not evaluation and np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)  
        else:
            return int(np.argmax(self.q_table[state]))  

    def learn(self, state, action, reward, next_state, done):
        q_predict = self.q_table[state][action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
            
        self.q_table[state][action] += self.alpha * (target - q_predict)

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)