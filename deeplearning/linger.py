import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l
max_degree = 20 # 多项式的最大阶数
n_train, n_test = 100, 100 # 训练和测试数据集大小
true_w = np.zeros(max_degree) # 分配大量的空间
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])
# features的维度:(n_train+n_test, 1)
features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features)
# mi的维度:(1, max_degree)
mi = np.arange(max_degree).reshape(1, -1)
# poly_features的维度:(n_train+n_test, max_degree)
poly_features = np.power(features, mi)
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1) # gamma(n)=(n-1)!
# labels的维度:(n_train+n_test,)
labels = np.dot(poly_features, true_w)
print(labels.shape)
labels += np.random.normal(scale=0.1, size=labels.shape)
print(labels.shape)
print(np.power([[3],[4]],[[1,2,3]]))