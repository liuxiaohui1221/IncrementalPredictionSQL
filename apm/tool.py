import torch
import torch.nn as nn
from torch.distributions import Dirichlet  # 适用于连续动作空间
def safe_action_selection(logits):
    # 确保至少选择一个专家
    logits = logits.clone()
    if torch.all(logits < 0):
        logits[0] += 1.0  # 强制选择第一个专家
    return Dirichlet(torch.exp(logits))