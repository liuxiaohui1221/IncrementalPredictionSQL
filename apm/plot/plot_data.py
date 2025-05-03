"""
根据xlsx文件中的指定列绘制符合论文格式的曲线图
依赖库：pandas, openpyxl, matplotlib
安装方法：pip install pandas openpyxl matplotlib
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
from matplotlib import font_manager as fm
from matplotlib import rcParams


# 设置全局字体参数
def set_font():
    """配置中英文字体参数"""
    # 英文设置
    rcParams['font.family'] = 'Times New Roman'
    rcParams['font.size'] = 5  # 统一字号

    # 中文设置
    try:
        # Windows系统宋体路径
        chinese_font = fm.FontProperties(fname='C:/Windows/Fonts/simsun.ttc')
        rcParams['font.sans-serif'] = [chinese_font.get_name()]
    except:
        print("警告：中文字体配置失败，请手动指定字体路径")


def plot_curve(file_path, x_col, y_col, output_file):
    """绘制曲线图核心函数"""
    # 读取数据
    df = pd.read_excel(file_path, engine='openpyxl')

    # 验证列存在性
    if x_col not in df.columns:
        raise ValueError(f"列'{x_col}'不存在")
    if y_col not in df.columns:
        raise ValueError(f"列'{y_col}'不存在")

    # 准备数据
    x = df[x_col].values
    y = df[y_col].values

    # 创建画布
    fig, ax = plt.subplots(figsize=(8, 6))  # 8x6英寸

    # 绘制曲线
    ax.plot(x, y,
            color='#2A5CAA',  # 标准蓝色
            linewidth=0.8,
            linestyle='-',
            marker='o',
            markersize=3,
            markeredgecolor='black',
            markerfacecolor='white')

    # 设置坐标轴
    ax.set_xlabel(x_col, fontproperties='SimSun')  # 中文用宋体
    ax.set_ylabel(y_col, fontproperties='SimSun')

    # 设置刻度
    ax.tick_params(axis='both', which='both', direction='in', labelsize=5)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    # 调整布局
    plt.tight_layout(pad=1.5)  # 增加边距

    # 保存输出
    plt.savefig(output_file, dpi=600, bbox_inches='tight')  # 高分辨率输出
    print(f"图表已保存至：{output_file}")


if __name__ == "__main__":
    # 配置命令行参数
    parser = argparse.ArgumentParser(description='学术图表生成器')
    parser.add_argument('file', type=str, default='/home/xhh/db_workspace/IncrementalPredictionSQL/apm/plot/5minute_1table_RNN_F1.xlsx', help='输入文件路径')
    parser.add_argument('--x', type=str, default="迭代",required=True, help='X轴列名')
    parser.add_argument('--y', type=str, default="F1得分",required=True, help='Y轴列名')
    parser.add_argument('--output', type=str, default='5minute_1table_RNN_F1.tif', help='输出文件名')

    args = parser.parse_args()

    # 设置字体
    set_font()

    # 执行绘图
    plot_curve(args.file, args.x, args.y, args.output)