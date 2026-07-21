import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from .network import DirectSchedulerNet

class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, mask, action, reward, next_state, next_mask, done):
        self.buffer.append((state, mask, action, reward, next_state, next_mask, done))

    def sample(self, batch_size: int):
        samples = random.sample(self.buffer, batch_size)
        states, masks, actions, rewards, next_states, next_masks, dones = zip(*samples)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(masks), dtype=torch.bool),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(next_masks), dtype=torch.bool),
            torch.tensor(dones, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)

class DoubleDQNAgent:
    def __init__(self, feature_dim=3, queue_size=10, lr=1e-3, gamma=0.99):
        self.gamma = gamma
        self.queue_size = queue_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DirectSchedulerNet(feature_dim, embed_dim=64, num_heads=4, queue_size=queue_size).to(self.device)
        self.target_net = DirectSchedulerNet(feature_dim, embed_dim=64, num_heads=4, queue_size=queue_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer()

    def select_action(self, state: np.ndarray, mask: np.ndarray, epsilon: float) -> int:
        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            return 0
            
        if random.random() < epsilon:
            return int(random.choice(valid_indices))

        # Convert to numpy array first to eliminate the UserWarning & handle nested sequence inputs cleanly
        state_arr = np.array(state, dtype=np.float32)
        mask_arr = np.array(mask, dtype=bool)

        state_t = torch.from_numpy(state_arr).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask_arr).unsqueeze(0).to(self.device)

        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state_t, mask_t)
            action = q_values.argmax(dim=-1).item()
        self.policy_net.train()
        return action

    def train_step(self, batch_size: int = 64) -> float:
        if len(self.replay_buffer) < batch_size:
            return 0.0

        states, masks, actions, rewards, next_states, next_masks, dones = self.replay_buffer.sample(batch_size)
        states, masks = states.to(self.device), masks.to(self.device)
        actions, rewards = actions.to(self.device), rewards.to(self.device)
        next_states, next_masks = next_states.to(self.device), next_masks.to(self.device)
        dones = dones.to(self.device)

        # Current Q-values
        q_eval = self.policy_net(states, masks).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN Step:
        # 1. Action selection using Policy Network
        with torch.no_grad():
            next_q_policy = self.policy_net(next_states, next_masks)
            best_actions = next_q_policy.argmax(dim=-1, keepdim=True)
            # 2. Action evaluation using Target Network
            next_q_target = self.target_net(next_states, next_masks).gather(1, best_actions).squeeze(1)
            # Padded sequences return -inf; sanitize -inf for finished states
            next_q_target[next_q_target == float('-inf')] = 0.0
            
            q_target = rewards + (1 - dones) * self.gamma * next_q_target

        loss = nn.MSELoss()(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())