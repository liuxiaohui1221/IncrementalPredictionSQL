import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np


# 定义特征编码器（增强版）
class FeatureEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim=64):
        super().__init__()
        # 数值特征处理（查询耗时、内存消耗等）
        self.numeric_fc = nn.Sequential(
            nn.Linear(input_dim['numeric'], 128),
            nn.ReLU(),
            nn.Linear(128, emb_dim)
        )

        # 类别特征嵌入（表类型、指标类型）
        self.table_emb = nn.EmbeddingBag(input_dim['table_types'], emb_dim, mode='mean')
        self.metric_emb = nn.EmbeddingBag(input_dim['metric_types'], emb_dim, mode='mean')

        # 时序特征处理（LSTM捕捉窗口内序列模式）
        self.temporal_lstm = nn.LSTM(input_dim['temporal'], 64, batch_first=True)

    def forward(self, numeric_feats, table_ids, metric_ids, temporal_seq):
        # 数值特征
        num_emb = self.numeric_fc(numeric_feats)

        # 类别特征
        tab_emb = self.table_emb(table_ids)
        met_emb = self.metric_emb(metric_ids)

        # 时序特征
        lstm_out, _ = self.temporal_lstm(temporal_seq)
        temp_emb = lstm_out[:, -1, :]

        return torch.cat([num_emb, tab_emb, met_emb, temp_emb], dim=1)


# 定义专家模型（扩展版）
class RuleExpert(nn.Module):
    """基于统计规则和启发式的专家"""

    def __init__(self, config):
        super().__init__()
        self.freq_threshold = config['freq_threshold']
        self.cache_rules = nn.ParameterDict({
            'weight': nn.Parameter(torch.tensor([0.7, 0.3]))  # 可学习的规则权重
        })

    def forward(self, x, history_stats):
        # 组合多个规则
        freq_score = (history_stats['query_freq'] > self.freq_threshold).float()
        recent_score = history_stats['recent_ratio']
        return self.cache_rules['weight'][0] * freq_score + self.cache_rules['weight'][1] * recent_score


class TemporalExpert(nn.Module):
    """增强时序专家（BiLSTM + Attention）"""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.attention = nn.Linear(2 * hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.bilstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        return torch.sum(lstm_out * attn_weights, dim=1)


# 强化学习策略模块（带基线估计）
class RLPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # 策略网络
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, action_dim)
        )

        # 基线价值网络（用于减少方差）
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 1)
        )

    def forward(self, state):
        logits = self.policy_net(state)
        value = self.value_net(state)
        return Categorical(logits=logits), value

# 完整MoE+RL模型
class MoE_RL_Predictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 特征编码
        self.encoder = FeatureEncoder(config['input_dims'])

        # 专家池
        self.experts = nn.ModuleList([
            RuleExpert(config['rule']),
            TemporalExpert(config['temporal']['input_dim'], config['temporal']['hidden_dim']),
            nn.Sequential(
                nn.Linear(config['temporal']['input_dim'], 256),
                nn.GELU(),
                nn.Linear(256, 1)
            )
        ])

        # 门控网络


        # 强化学习策略
        self.rl_policy = RLPolicy(config['rl_state_dim'], config['num_actions'])

        # 资源预测器
        self.resource_predictor = nn.Sequential(
            nn.Linear(config['rl_state_dim'], 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 预测内存和CPU消耗
        )

    def forward(self, numeric_feats, table_ids, metric_ids, temporal_seq, history_stats):
        # 特征编码
        x = self.encoder(numeric_feats, table_ids, metric_ids, temporal_seq)

        # 门控输出专家权重（带温度系数的softmax）
        gate_logits = self.gate(x)
        gate_weights = torch.softmax(gate_logits / self.config['temp'], dim=-1)

        # 专家输出
        expert_outputs = []
        for expert in self.experts:
            if isinstance(expert, RuleExpert):
                out = expert(x, history_stats)
                print('RuleExpert size:',expert.__class__,out.size())
            elif isinstance(expert, TemporalExpert):
                out = expert(x.unsqueeze(1))  # 添加序列维度
            else:
                out = expert(x)  # 添加序列维度
                print('OtherExpert size:', expert.__class__, out.size())
                # if out.size()

            expert_outputs.append(out.squeeze())
        # for expert_output in expert_outputs:
        #     print('expert size:',expert_output.__class__,expert_output.size())
        print(gate_weights.size())
        # 加权融合，stack多出一个维度（dim=-1 表示在最后一个维度上堆叠。）
        combined = torch.stack(expert_outputs, dim=-1) * gate_weights.unsqueeze(1) # [batch_size, 1, num_experts]
        combined = combined.sum(dim=-1) #加的方式融合[batch_size, 1]

        # 强化学习决策  换成横向拼接方式
        # 状态信息：每个专家预测
        state = torch.cat([combined.detach(), history_stats['resource_usage']], dim=1)
        action_dist, value_est = self.rl_policy(state)
        action = action_dist.sample()

        # 资源消耗预测
        resource_pred = self.resource_predictor(state)

        return combined, action, action_dist, value_est, gate_weights, resource_pred


# 完整训练框架
class TrainingFramework:
    def __init__(self, config):
        self.config = config
        self.model = MoE_RL_Predictor(config)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config['lr'])
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config['epochs'])

    def compute_loss(self, batch):
        numeric_feats, table_ids, metric_ids, temporal_seq, history_stats, targets, rewards = batch

        # 前向传播
        preds, actions, action_dist, values, gate_weights, resource_pred = self.model(
            numeric_feats, table_ids, metric_ids, temporal_seq, history_stats)

        # ======== 损失计算 ========
        # 1. 监督损失（带L2正则）
        mse_loss = nn.MSELoss()(preds, targets)
        l2_reg = torch.tensor(0.).to(preds.device)
        for param in self.model.parameters():
            l2_reg += torch.norm(param)
        mse_loss += self.config['lambda_l2'] * l2_reg

        # 2. 门控稀疏正则
        entropy_loss = -torch.mean(torch.sum(gate_weights * torch.log(gate_weights + 1e-10), dim=1))
        gate_l2 = torch.mean(torch.norm(gate_weights, dim=1))
        gate_reg = self.config['lambda_gate'] * (entropy_loss + gate_l2)

        # 3. 强化学习损失（带基线+熵正则）
        advantage = rewards - values.squeeze().detach()
        rl_loss = -torch.mean(action_dist.log_prob(actions) * advantage)
        value_loss = nn.MSELoss()(values.squeeze(), rewards)
        entropy = torch.mean(action_dist.entropy())
        rl_total = rl_loss + value_loss - self.config['lambda_ent'] * entropy

        # 4. 资源约束惩罚
        mem_penalty = torch.relu(resource_pred[:, 0] - self.config['mem_limit'])
        cpu_penalty = torch.relu(resource_pred[:, 1] - self.config['cpu_limit'])
        resource_loss = self.config['lambda_res'] * torch.mean(mem_penalty + cpu_penalty)

        # 总损失
        total_loss = (
                self.config['alpha'] * mse_loss +
                self.config['beta'] * rl_total +
                gate_reg +
                resource_loss
        )

        return {
            'total': total_loss,
            'mse': mse_loss,
            'rl': rl_total,
            'gate': gate_reg,
            'resource': resource_loss
        }

    def train_step(self, batch):
        self.model.train()
        loss_dict = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss_dict['total'].backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
        self.optimizer.step()
        self.scheduler.step()

        return loss_dict


# 示例配置
config = {
    'input_dims': {
        'numeric': 10,
        'table_types': 20,
        'metric_types': 15,
        'temporal': 64
    },
    'rule': {'freq_threshold': 0.1},
    'temporal': {'input_dim': 256, 'hidden_dim': 128},
    'gate_input_dim': 256,  # 64*4 + 64
    'rl_state_dim': 129,
    'num_actions': 5,
    'temp': 0.5,  # 门控温度系数
    'lambda_l2': 1e-4,
    'lambda_gate': 0.1,
    'lambda_ent': 0.01,
    'lambda_res': 0.2,
    'mem_limit': 0.8,  # 内存使用率阈值
    'cpu_limit': 0.7,  # CPU使用率阈值
    'alpha': 0.5,
    'beta': 1.0,
    'lr': 3e-4,
    'grad_clip': 1.0,
    'epochs': 100
}


# 模拟训练流程
def simulate_training():
    framework = TrainingFramework(config)
    for epoch in range(config['epochs']):
        # 模拟数据生成（实际需替换为真实数据加载）
        batch = (
            torch.randn(32, 10),  # numeric_feats 数值特征
            torch.randint(0, 20, (32, 5)),  # table_ids 表ID
            torch.randint(0, 15, (32, 3)),  # metric_ids 指标ID
            torch.randn(32, 10, 64),  # temporal_seq
            {'resource_usage': torch.rand(32, 1),'query_freq':torch.rand(32, 1),'recent_ratio':0.5},
            torch.rand(32, 1),  # targets
            torch.randn(32)  # rewards
        )

        loss_dict = framework.train_step(batch)
        print(f"Epoch {epoch}:")
        print(f"  Total Loss: {loss_dict['total'].item():.4f}")
        print(f"  MSE: {loss_dict['mse'].item():.4f}  RL: {loss_dict['rl'].item():.4f}")
        print(f"  Gate Reg: {loss_dict['gate'].item():.4f}  Resource: {loss_dict['resource'].item():.4f}")


if __name__ == "__main__":
    simulate_training()