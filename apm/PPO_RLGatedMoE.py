import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np


class RLGatedMoE(nn.Module):
    def __init__(self, num_experts, input_dim, hidden_dim=128):
        super().__init__()
        self.num_experts = num_experts

        # 专家池（示例使用简单MLP）
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim))
            for _ in range(num_experts)])

        # 强化学习门控网络
        self.policy_net = nn.Sequential(
            nn.Linear(self._state_dim(input_dim), 256),
            nn.ReLU(),
            nn.Linear(256, num_experts))

        # 价值网络（用于基线计算）
        self.value_net = nn.Sequential(
            nn.Linear(self._state_dim(input_dim), 256),
            nn.ReLU(),
            nn.Linear(256, 1))

        # 性能跟踪器
        self.performance_history = np.ones(num_experts)  # 初始化为1

    def _state_dim(self, input_dim):
        """构建状态向量维度：
        - 当前输入特征
        - 资源使用情况（内存、CPU）
        - 专家历史性能
        - 时间特征（可选）
        """
        return input_dim + 2 + self.num_experts

    def forward(self, x, resource_info):
        # 状态构建
        state = torch.cat([
            x,
            resource_info,
            torch.tensor(self.performance_history).to(x.device)
        ], dim=-1)

        # 生成策略
        logits = self.policy_net(state)
        value = self.value_net(state)
        return logits, value

    def select_expert(self, x, resource_info):
        logits, value = self.forward(x, resource_info)
        probs = nn.Softmax(dim=-1)(logits)
        m = Categorical(probs)
        expert_idx = m.sample()
        return expert_idx, m.log_prob(expert_idx), value

    def update_performance(self, expert_idx, reward):
        # 指数衰减更新：新性能=0.2*当前奖励 + 0.8*历史平均
        self.performance_history[expert_idx] = \
            0.2 * reward + 0.8 * self.performance_history[expert_idx]


# PPO经验回放缓存
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

    def compute_advantages(self, last_value=0):
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_value])

        # GAE计算
        deltas = rewards + self.gamma * values[1:] - values[:-1]
        advantages = []
        advantage = 0
        for delta in reversed(deltas):
            advantage = delta + self.gamma * self.gae_lambda * advantage
            advantages.insert(0, advantage)

        returns = np.array(advantages) + np.array(self.values)
        return torch.tensor(advantages), torch.tensor(returns)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()


# 训练框架
class PPOTrainer:
    def __init__(self, model, lr=3e-4, gamma=0.99,clip_epsilon=0.2,
                 ppo_epochs=4, batch_size=64, capacity=10000):
        # 初始化模型、优化器、PPO参数和缓存
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(capacity)
        self.gamma = gamma

    def compute_gae(self, rewards, values, dones):
        """计算广义优势估计(GAE)"""
        advantages = []
        last_advantage = 0
        next_value = 0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            last_advantage = delta + self.gamma * 0.95 * (1 - dones[t]) * last_advantage
            advantages.insert(0, last_advantage)
            next_value = values[t]

        return torch.tensor(advantages)
    def update(self):
        # 获取缓存数据
        states = torch.stack(self.buffer.states)
        actions = torch.stack(self.buffer.actions)
        old_log_probs = torch.stack(self.buffer.log_probs).detach()
        old_values = torch.stack(self.buffer.values).detach()

        # 计算优势
        advantages, returns = self.buffer.compute_advantages()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 多轮PPO更新
        for _ in range(self.ppo_epochs):
            # 随机打乱数据
            indices = torch.randperm(len(states))
            for start in range(0, len(states), self.batch_size):
                idx = indices[start:start + self.batch_size]
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_returns = returns[idx]
                batch_advantages = advantages[idx]

                # 前向计算
                output = self.model(batch_states, action=batch_actions)
                new_log_probs = output['log_prob']
                values = output['value'].squeeze()

                # 计算比率
                ratio = (new_log_probs - batch_old_log_probs).exp()

                # 策略损失
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon,
                                    1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 价值损失
                value_loss = nn.MSELoss()(values, batch_returns)

                # 监督损失
                supervised_loss = nn.CrossEntropyLoss()(
                    output['fused_output'], y_true[batch_indices])

                # 专家多样性正则
                entropy_loss = Categorical(output['expert_probs']).entropy().mean()

                # 总损失
                total_loss = (0.7 * policy_loss +
                              0.3 * value_loss +
                              0.5 * supervised_loss -
                              0.01 * entropy_loss)

                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

        self.buffer.clear()