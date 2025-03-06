import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np


# 定义经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity  # 缓冲区容量
        self.buffer = []  # 存储经验的列表
        self.position = 0  # 当前存储位置

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in batch]

    def __len__(self):
        return len(self.buffer)


# 定义 PPO 模型
class PPO(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPO, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.clip_epsilon = 0.2
        self.gae_lambda = 0.95
        self.gamma = 0.99

    def forward(self, x):
        return self.actor(x), self.critic(x)

    def compute_gae(self, values, rewards, dones, next_value):
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        return advantages

    def update(self, states, actions, rewards, dones, next_state):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        with torch.no_grad():
            values = self.critic(states).squeeze()
            next_value = self.critic(torch.FloatTensor(next_state)).item()

        advantages = self.compute_gae(values, rewards, dones, next_value)
        returns = advantages + values

        for _ in range(10):
            for idx in range(0, len(states), 64):
                batch_states = states[idx:idx + 64]
                batch_actions = actions[idx:idx + 64]
                batch_returns = returns[idx:idx + 64]
                batch_advantages = advantages[idx:idx + 64]

                action_probs, values_pred = self(batch_states)
                dist = Categorical(action_probs)
                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_probs - log_probs.detach())
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy

                critic_loss = nn.MSELoss()(values_pred.squeeze(), batch_returns)

                self.optimizer.zero_grad()
                (actor_loss + critic_loss).backward()
                self.optimizer.step()


# 使用示例
if __name__ == "__main__":
    # 初始化环境
    import gym

    env = gym.make("CartPole-v1")
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    # 初始化 PPO 模型和经验回放缓冲区
    ppo = PPO(input_dim, output_dim)
    replay_buffer = ReplayBuffer(1000)

    # 训练循环
    for episode in range(1000):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            # 获取动作
            with torch.no_grad():
                action_probs, _ = ppo(torch.FloatTensor(state))
                dist = Categorical(action_probs)
                action = dist.sample().item()

            # 与环境交互
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # 存储经验
            replay_buffer.push(state, action, reward, next_state, done)

            # 更新状态
            state = next_state

        # 从缓冲区中采样并更新模型
        if len(replay_buffer) > 64:
            batch = replay_buffer.sample(64)
            states, actions, rewards, next_states, dones = zip(*batch)
            ppo.update(states, actions, rewards, dones, next_state)

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")