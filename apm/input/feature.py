import torch
import torch.nn as nn
# 输入输出处理（完善版）
class FeatureProcessor:
    def __init__(self, sql_vocab_size=100, table_count=20):
        self.sql_encoder = nn.Embedding(sql_vocab_size, 64)
        self.table_encoder = nn.Embedding(table_count, 32)
        self.table_count = table_count

    def __call__(self, logs):
        # SQL特征
        sql_ids = [log.sql_type_id for log in logs]
        sql_feat = self.sql_encoder(torch.tensor(sql_ids)).mean(dim=0)

        # 表访问频率
        table_counts = torch.bincount(
            torch.cat([log.table_ids for log in logs]),
            minlength=self.table_count).float()
        table_feat = self.table_encoder(table_counts)

        # 数值特征
        num_feat = torch.tensor([
            len(logs),  # 查询次数
            sum(log.latency for log in logs),  # 总延迟
            max(log.memory for log in logs)  # 最大内存
        ])

        return torch.cat([sql_feat, table_feat.flatten(), num_feat])