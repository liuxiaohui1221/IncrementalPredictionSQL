import matplotlib.pyplot as plt
import numpy as np

# 模拟数据生成
np.random.seed(42)
n_points = 100

# 多目标Q-Learning数据（帕累托前沿附近）
hit_rate_ql = np.random.uniform(70, 85, n_points)
latency_improve_ql = 60 + 0.5*hit_rate_ql + np.random.normal(0,5,n_points)
resource_cost_ql = 50 + 0.3*hit_rate_ql + np.random.normal(0,10,n_points)

# 单目标Q-Learning数据（高命中率，高资源消耗）
hit_rate_single = np.random.uniform(80, 90, n_points)
latency_improve_single = 50 + np.random.normal(0,8,n_points)
resource_cost_single = 80 + np.random.normal(0,15,n_points)

# MOPSO数据（分散分布）
hit_rate_mopso = np.random.uniform(65, 82, n_points)
latency_improve_mopso = 55 + np.random.normal(0,12,n_points)
resource_cost_mopso = 60 + np.random.normal(0,20,n_points)

# 绘制帕累托前沿
plt.figure(figsize=(10,6))
sc1 = plt.scatter(hit_rate_ql, latency_improve_ql, c=resource_cost_ql,
                cmap='coolwarm', alpha=0.7, label='Multi-Objective Q-Learning')
sc2 = plt.scatter(hit_rate_single, latency_improve_single, c=resource_cost_single,
                cmap='coolwarm', marker='^', alpha=0.5, label='Single-Objective QL')
sc3 = plt.scatter(hit_rate_mopso, latency_improve_mopso, c=resource_cost_mopso,
                cmap='coolwarm', marker='s', alpha=0.5, label='MOPSO')

# 标注帕累托前沿边界
front_x = np.linspace(70,85,50)
front_y = 95 - 0.4*front_x  # 模拟前沿曲线
plt.plot(front_x, front_y, 'r--', lw=2, label='Pareto Front')

# 图表装饰
plt.colorbar(sc1, label='Resource Cost (%)')
plt.xlabel('Hit Rate (%)', fontsize=12)
plt.ylabel('Latency Improvement (%)', fontsize=12)
plt.title('Multi-Objective Pareto Frontier Analysis', fontsize=14)
plt.legend(loc='upper right')
plt.grid(alpha=0.3)
plt.show()
