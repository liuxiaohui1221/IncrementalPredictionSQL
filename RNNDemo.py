import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


# 定义一个灵活的RNN模型
class FlexibleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, rnn_type='LSTM', bidirectional=False,
                 dropout=0.0):
        super(FlexibleRNN, self).__init__()

        # 选择RNN类型
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional,
                               dropout=dropout)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional,
                              dropout=dropout)
        elif rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional,
                              dropout=dropout)
        else:
            raise ValueError("Unsupported RNN type. Choose 'LSTM', 'GRU', or 'RNN'.")

        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden=None):
        # x: (batch_size, seq_length, input_size)
        # 通过RNN层
        rnn_out, hidden = self.rnn(x, hidden)
        print("rnn_out.shape:", rnn_out.shape, ", x.shape:", x.shape)

        # 应用dropout
        rnn_out = self.dropout(rnn_out)

        # 应用全连接层到每个时间步的输出
        # out = self.fc(rnn_out)
        print("dropout rnn_out.shape:", rnn_out.shape)

        rnn_out = self.fc(rnn_out[:, -1, :])
        print("fc rnn_out.shape:", rnn_out.shape)
        return rnn_out, hidden


# 创建模型实例
# model = FlexibleRNN(input_size, hidden_size, output_size, num_layers, rnn_type, bidirectional, dropout)

# 初始化隐藏状态
# hidden = None  # None表示让模型自动处理隐藏状态的初始化
# 预测输出
# output, new_hidden = model(x, hidden)
# print("输出:")
# print(output.shape)  # 应该是 (batch_size, seq_length, output_size)
# print(new_hidden)  # 取决于RNN类型和是否双向

# 定义训练函数
def train(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, _ = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {batch_idx} \tLoss: {loss.item()}')
    print(f'Average loss: {total_loss / len(data_loader)}')


# 定义评估函数
def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    accuracy = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            correct = (output.argmax(dim=1) == target).sum().item()
            accuracy += correct / len(target)

    print(f'Evaluation loss: {total_loss / len(data_loader)}')
    print(f'Evaluation Accuracy: {accuracy / len(data_loader)}')


# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建数据集和数据加载器
# 设置随机种子以获得可重复的结果
torch.manual_seed(0)
np.random.seed(0)


# 生成合成数据
def generate_data(num_samples, seq_length, input_size, output_size):
    # 生成随机序列数据
    data = torch.randn(num_samples, seq_length, input_size)
    # 生成随机目标数据
    targets = torch.randint(0, output_size, (num_samples,))
    print("生成随机数据shape:" + str(data.shape) + "," + str(targets.shape))
    return data, targets


# 参数定义
num_samples = 1000  # 样本数量
seq_length = 15  # 序列长度

input_size = 10  # 输入特征维度
hidden_size = 20  # 隐藏层维度
output_size = 5  # 输出特征维度
num_layers = 2  # RNN层数
rnn_type = 'LSTM'  # RNN类型
bidirectional = False  # 是否双向
dropout = 0.5  # dropout比率


# 生成数据
def loader_data():
    data, targets = generate_data(num_samples, seq_length, input_size, output_size)
    # 假设我们已经有了训练数据和测试数据
    train_data = TensorDataset(data[:800], targets[:800])  # 80% 用于训练
    test_data = TensorDataset(data[800:], targets[800:])  # 20% 用于测试
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    return train_loader, test_loader


def trainMyRNN(train_loader, test_loader, epochs=10):
    # 实例化模型
    model = FlexibleRNN(input_size, hidden_size, output_size, num_layers, rnn_type, bidirectional, dropout).to(device)
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 对于分类任务
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        train(model, train_loader, criterion, optimizer, device)
        evaluate(model, test_loader, criterion, device)
    # 保存模型
    torch.save(model.state_dict(), 'flexible_rnn_model.pth')


# 训练模型
train_loader, test_loader = loader_data()
trainMyRNN(train_loader, test_loader)
