"""
根据xlsx文件中的指定列绘制符合论文格式的曲线图（直接显示）
依赖库：pandas, openpyxl, matplotlib
安装方法：pip install pandas openpyxl matplotlib
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib import rcParams


# 设置全局字体参数
def set_font():
    """配置中英文字体参数"""
    # 英文设置
    # rcParams['font.family'] = 'Times New Roman'
    rcParams['font.size'] = 12  # 统一字号

    # 中文设置
    try:
        # Windows系统宋体路径
        chinese_font = fm.FontProperties(fname='C:/Windows/Fonts/simsun.ttc')
        rcParams['font.sans-serif'] = [chinese_font.get_name()]
        rcParams['axes.unicode_minus'] = False
    except:
        print("警告：中文字体配置失败，请手动指定字体路径")


def plot_curve(file_path, x_col, y_col, title="曲线图", x_label='', y_label=''):
    set_font()
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
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)  # 8x6英寸

    # 绘制曲线
    ax.plot(x, y,
            color='#2A5CAA',  # 标准蓝色
            # linewidth=0.8,
            linestyle='-',
            marker='o',
            markersize=4,
            markeredgecolor='black',
            markerfacecolor='white'
            )
    # 设置标题
    ax.set_title(f"{title}", fontproperties='SimSun')  # 中文用宋体
    # 设置坐标轴
    if x_label=='':
        x_label=x_col
    if y_label=='':
        y_label=y_col
    ax.set_xlabel(x_label, fontproperties='SimSun')  # 中文用宋体
    ax.set_ylabel(y_label, fontproperties='SimSun')

    # 设置刻度
    ax.tick_params(axis='both', which='both', direction='in', labelsize=5)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    # 调整布局
    # plt.tight_layout(pad=1.5)  # 增加边距
    # plt.xticks(fontsize=10.5)  # 刻度字体‌:ml-citation{ref="1,2" data="citationList"}
    # plt.yticks(fontsize=10.5)
    # 显示图表
    plt.show()
    # 保存图片
    fig.savefig(f'Figure_{title}.svg', format='svg', dpi=300, bbox_inches='tight')
    print("Saved svg:", f'Figure_{title}.svg')


def plot_multi_columns(excel_path, x_col, y_cols,
                       sheet_name=0,  # 默认读取第一个工作表
                       title="Multiple Y Columns",
                       xlabel="X Axis",
                       ylabel="Y Axis",
                       legend_names=None,
                       legend_loc='best',
                       parse_dates=None):
    """
    从Excel文件读取并绘制多Y列数据

    参数:
    excel_path : str - Excel文件路径
    x_col : str - 作为X轴的列名
    y_cols : list - 需要绘制的Y列名称列表
    sheet_name : str/int - 要读取的工作表名称或索引（默认0）
    title : str - 图表标题
    xlabel : str - X轴标签
    ylabel : str - Y轴标签
    legend_loc : str - 图例位置
    parse_dates : list - 需要解析为日期类型的列（默认None）

    返回:
    fig, ax - matplotlib的figure和axes对象
    """
    set_font()
    # 从Excel读取数据
    df = pd.read_excel(
        excel_path,
        sheet_name=sheet_name,
        parse_dates=parse_dates  # 自动解析日期列
    )

    # 创建绘图
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # 验证列是否存在
    for col in [x_col] + y_cols:
        if col not in df.columns:
            raise ValueError(f"列 '{col}' 不存在于DataFrame中")
    # 添加标签和标题
    ax.set_title(title, fontproperties='SimSun')
    ax.set_xlabel(xlabel, fontproperties='SimSun')  # 中文用宋体
    ax.set_ylabel(ylabel, fontproperties='SimSun')

    # 自动旋转日期标签
    if df[x_col].dtype == 'datetime64[ns]':
        fig.autofmt_xdate()
    # 循环绘制每个Y列
    for index, y_col in enumerate(y_cols):
        ax.plot(df[x_col], df[y_col], label=legend_names[index], marker='o', markersize=4)

    # 添加图例和网格
    ax.legend(loc=legend_loc)
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    # 显示图表
    plt.show()
    # 保存图片
    fig.savefig(f'Figure_{title}.svg', format='svg', dpi=300, bbox_inches='tight')
    print("Saved svg:",f'Figure_{title}.svg')
    return fig, ax


# 使用示例
if __name__ == "__main__":
    import numpy as np
    # 设置字体



    # 输入参数（请修改以下参数）
    file_path = "/home/xhh/db_workspace/IntentPredictionEval/apm/plot/5minute_1table_RNN_F1.xlsx"  # 替换为你的文件路径
    x_col = "迭代"  # 替换为你的X轴列名
    y_col = "F1得分"  # 替换为你的Y轴列名

    # 执行绘图
    plot_curve(file_path, x_col, y_col)