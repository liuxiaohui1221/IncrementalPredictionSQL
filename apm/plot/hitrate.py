import pandas as pd
import math

from apm.plot.plot_show import set_font, plot_curve


def average_every_rows(num_avg,input_file, output_file, x_col,y_col):
    """
    对 Excel 文件中指定列的每隔10个值计算平均，输出到新文件

    参数:
    - input_file: 输入Excel文件路径
    - output_file: 输出Excel文件路径
    - sheet_name: 输入文件的工作表名（默认'Sheet1'）
    - column: 要处理的列名或字母（如'A'或'C'）
    """
    # 读取Excel数据
    df = pd.read_excel(input_file, engine='openpyxl')
    if x_col not in df.columns:
        raise ValueError(f"列'{x_col}'不存在")
    if y_col not in df.columns:
        raise ValueError(f"列'{y_col}'不存在")

    data = df[y_col].values.tolist()
    # data = df.iloc[:, 0].tolist()  # 将列数据转为列表
    x_data = df[x_col].values.tolist()
    # 计算每组10个值的平均
    averages = []
    avg_x_data= []
    for i in range(0, len(data), num_avg):
        chunk = data[i:i + num_avg]
        if len(chunk) >= 1:  # 允许保留最后不足的分组
            avg = sum(chunk) / len(chunk)
            averages.append(avg)
            avg_x_data.append(i+1)

    # 将结果保存到新Excel
    result_df = pd.DataFrame({"命中率": averages,"迭代": avg_x_data})
    result_df.to_excel(output_file, index=False)
    print(f"处理完成！共生成 {len(averages)} 个平均值，已保存到 {output_file}")


# 每200个值取平均
num_avg=100
output_file=f"hitrate_{num_avg}_avg_output.xlsx"
average_every_rows(
    num_avg,
    input_file="HitRate_THRESHOLD_0.6_INCREMENTAL.xlsx",
    output_file=output_file,
    x_col="迭代",y_col="命中率"
)
# 设置字体
set_font()
# 输入参数（请修改以下参数）
file_path = output_file  # 替换为你的文件路径
x_col = "迭代"  # 替换为你的X轴列名
y_col = "命中率"  # 替换为你的Y轴列名
# 执行绘图
plot_curve(file_path, x_col, y_col, title='命中率曲线图(转换阈值0.6)')