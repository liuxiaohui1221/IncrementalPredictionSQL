import torch
from datetime import datetime
import torch.nn as nn
from torch.distributions import Dirichlet  # 适用于连续动作空间
def safe_action_selection(logits):
    # 确保至少选择一个专家
    logits = logits.clone()
    if torch.all(logits < 0):
        logits[0] += 1.0  # 强制选择第一个专家
    return Dirichlet(torch.exp(logits))

def getISOFormatTime(time_str):
    # 定义输入时间格式
    input_format = '%d/%b/%Y:%H:%M:%S %z'
    # 定义输出时间格式（ISO格式）
    output_format = '%Y-%m-%dT%H:%M:%S%z'
    # 解析时间字符串
    dt = datetime.strptime(time_str, input_format)
    # 转换为ISO格式
    iso_time_str = dt.strftime(output_format)

    # print(iso_time_str)
    return iso_time_str


import numpy as np
def generate_normal_random_int(mean=50, std_dev=10, min_value=None, max_value=None):
    """
    生成一个符合正态分布的随机整数。

    参数:
    - mean: 正态分布的均值，默认为50。
    - std_dev: 正态分布的标准差，默认为10。
    - min_value: 随机数的最小值，可选。
    - max_value: 随机数的最大值，可选。

    返回:
    - 一个符合正态分布的随机整数。
    """
    # 生成一个正态分布的随机浮点数
    random_float = np.random.normal(mean, std_dev)

    # 将浮点数四舍五入为整数
    random_integer = round(random_float)

    # 如果指定了最小值和最大值，则裁剪到指定范围
    if min_value is not None:
        random_integer = max(random_integer, min_value)
    if max_value is not None:
        random_integer = min(random_integer, max_value)

    return random_integer


def generate_druid_metrics(fields):
    metrics = []

    # 添加 count 指标
    metrics.append({
        "type": "count",
        "name": "count"
    })

    # 为每个字段生成 sum, min, max 指标
    for field in fields:
        metrics.append({
            "type": "longSum",
            "name": f"{field}_sum",
            "fieldName": field
        })
        metrics.append({
            "type": "longMin",
            "name": f"{field}_min",
            "fieldName": field
        })
        metrics.append({
            "type": "longMax",
            "name": f"{field}_max",
            "fieldName": field
        })

    return metrics

# 示例输入
fields = [
    'dur','biz', 'fail', 'httperr', 'neterr', 'err', 'tolerated',
          'frustrated','exception','err_4xx', 'err_5xx'
    ]
# # 生成Druid任务的JSON指标格式
druid_metrics = generate_druid_metrics(fields)
# 打印输出
import json
print(json.dumps(druid_metrics, indent=4))