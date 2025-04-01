import torch
import torch.nn as nn

from apm.PPO_RLGatedMoE import ReplayBuffer
from apm.PPO_RLGatedMoE import *
from apm.Gate_of_MoE import *



def train_moe_rl(expert_pool, dataset, epochs=100):
    # 初始化组件:输入的查询作为状态，特征进行one-host编码为RNN(RL gate)的状态输入
    encoder = StateEncoder(input_dim=256)
    # RNN门控输入：查询状态，输出专家向量
    gating_net = RLGatingNetwork(input_dim=512, num_experts=len(expert_pool))
    ppo = PPOTrainer(gating_net)
    memory = ReplayBuffer(capacity=10000)

    for epoch in range(epochs):
        for batch in dataset:
            # 编码状态
            states = encoder(batch['features'])

            # 选择专家
            actions, log_probs, values = gating_net.get_action(states)

            # 生成预测
            expert_outputs = torch.stack([expert(batch['features']) for expert in expert_pool])
            preds = torch.sum(actions.unsqueeze(-1) * expert_outputs, dim=1)

            # 计算奖励
            rewards = calculate_reward(preds, batch['labels'], batch['resource'])

            # 存储轨迹
            memory.push(states, actions, log_probs, rewards, values)

            # PPO更新
            if len(memory) >= 2048:
                samples = memory.sample(2048)
                advantages = ppo.compute_gae(samples['rewards'], samples['values'], samples['dones'])
                ppo.update(
                    samples['states'],
                    samples['actions'],
                    samples['log_probs'],
                    samples['rewards'],
                    advantages
                )


def calculate_reward(pred_dist, true_dist, resource_usage):
    # 预测准确性奖励
    cosine_sim = torch.cosine_similarity(pred_dist, true_dist, dim=-1)
    acc_reward = 10 * cosine_sim

    # 资源惩罚项
    mem_penalty = torch.relu(resource_usage['memory'] - 0.8)  # 超过80%内存使用则惩罚
    cpu_penalty = torch.relu(resource_usage['cpu'] - 0.7)
    cost_penalty = 5 * (mem_penalty + cpu_penalty)

    return acc_reward - cost_penalty

