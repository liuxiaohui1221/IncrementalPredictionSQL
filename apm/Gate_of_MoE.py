import torch
import torch.nn as nn
from torch.distributions import Dirichlet  # 适用于连续动作空间


class RNNGate(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts):
        super(RNNGate, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_experts)

    def forward(self, x, hidden_state=None):
        if hidden_state is None:
            hidden_state = torch.zeros(1, x.size(0), self.rnn.hidden_size)
        output, new_hidden_state = self.rnn(x, hidden_state)
        gate_scores = self.fc(output)
        return gate_scores, new_hidden_state
# 硬门控网络
class HardGate(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(HardGate, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)  # 线性变换计算专家得分
        # self.gate = nn.Sequential(
        #     nn.Linear(config['gate_input_dim'], 128),
        #     nn.LayerNorm(128),
        #     nn.GELU(),
        #     nn.Linear(128, len(self.experts))
        # )
    def forward(self, x):
        scores = self.fc(x)  # 计算专家得分
        top1_indices = torch.argmax(scores, dim=-1, keepdim=True)  # 选择Top-1专家
        weights = torch.zeros_like(scores).scatter_(-1, top1_indices, 1.0)  # 生成权重矩阵，Top-1专家权重为1
        return weights
class RLGatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, num_experts))

        self.value_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1))

    def forward(self, state):
        logits = self.policy_net(state)
        value = self.value_net(state)
        return logits, value

    def get_action(self, state):
        logits, value = self.forward(state)
        policy_dist = Dirichlet(concentration=torch.exp(logits))  # 狄利克雷分布
        action = policy_dist.sample()
        log_prob = policy_dist.log_prob(action)
        return action.detach(), log_prob, value