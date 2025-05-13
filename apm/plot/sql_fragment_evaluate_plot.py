import os

import pandas as pd
from apm.plot.plot_show import set_font, plot_curve, plot_multi_columns


def average_every_rows(num_avg,input_file, x_col,y_col):
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
    print("excel columns:", x_col,df.columns)
    if isinstance(x_col, str) :
        if x_col not in df.columns:
            raise ValueError(f"列'{x_col}'不存在")
        if y_col not in df.columns:
            raise ValueError(f"列'{y_col}'不存在")
        x_data = df[x_col].values.tolist()
    else:
        x_data = x_col
    data = df[y_col].values.tolist()
    # data = df.iloc[:, 0].tolist()  # 将列数据转为列表

    # 计算每组10个值的平均
    averages = []
    avg_x_data= []
    for i in range(0, len(data), num_avg):
        chunk = data[i:i + num_avg]
        x_val= x_data[i]
        if len(chunk) >= 1:  # 允许保留最后不足的分组
            avg = sum(chunk) / len(chunk)
            averages.append(avg)
            avg_x_data.append(x_val)

    print(f"处理完成！共生成 {len(averages)} 个平均值")
    return avg_x_data,averages
def validate_input(data):
    if not isinstance(data, (str, list)):
        raise TypeError("参数必须为字符串或列表")
    return True

def read_avg_data(input_path, x_col, y_col_or_list, title, x_label, y_label,num_avg):
    if validate_input(y_col_or_list)==False:
        raise ValueError("y_col 参数无效")
    output_file = os.path.join("../output/", f"{title}_{x_label}_{y_label}_{num_avg}_avg_output.xlsx")
    if isinstance(y_col_or_list, str):
        # 处理字符串逻辑
        new_x, avg_y = average_every_rows(
            num_avg,
            input_file=input_path,
            x_col=x_col, y_col=y_col_or_list
        )
        # 将结果保存到新Excel
        result_df = pd.DataFrame({y_col_or_list: avg_y, x_col: new_x})
        result_df.to_excel(output_file, index=False)
    elif isinstance(y_col_or_list, list):
        # 处理列表逻辑
        multi_cols={}
        new_x=''
        for single_y in y_col_or_list:
            new_x_col, avg_y_col = average_every_rows(
                num_avg,
                input_file=input_path,
                x_col=x_col, y_col=single_y
            )
            new_x=new_x_col
            multi_cols[single_y]=avg_y_col
        # 将结果保存到新Excel
        multi_cols[x_col]=new_x
        result_df = pd.DataFrame(multi_cols)
        result_df.to_excel(output_file, index=False)
    # 设置字体
    set_font()
    return output_file
def plot_multi_y(input_path, x_col, y_cols, title, x_label, ylabel,legend_names,num_avg):
    file_path = read_avg_data(input_path, x_col, y_cols, title, x_label, legend_names,num_avg)
    # 多列绘图
    plot_multi_columns(
        excel_path=file_path,
        x_col=x_col,
        y_cols=y_cols,
        title=title,
        xlabel=x_label,
        ylabel=ylabel,
        legend_names=legend_names
    )

def plot_xy(input_path, x_col, y_col, title, x_label, y_label,num_avg):
    file_path=read_avg_data(input_path, x_col, y_col, title, x_label, y_label,num_avg)
    # 执行绘图
    plot_curve(file_path, x_col, y_col, title=title,x_label=x_label,y_label=y_label)

if __name__ == '__main__':
    # 每多少个值取平均
    num_avg=1
    # RNN-simple
    common_title="查询片段准确率曲线图"
    input_path="sql_fragment_hitrate.xlsx"
    titles=["RNN-simple F1(转换阈值0.6)","RNN-simple recall(转换阈值0.6)","RNN-simple accuracy(转换阈值0.6)"]
    # LSTM
    # common_title = "LSTM(转换阈值0.6)"
    # input_path="OutputExcelQuality_RNN_LSTM_FRAGMENT_BIT_TOP_K_1_all_windows_256_ACCURACY_THRESHOLD_0.95_INCREMENTAL.xlsx"
    # titles = ["LSTM F1得分(转换阈值0.6)", "LSTM 召回率(转换阈值0.6)", "LSTM 准确率(转换阈值0.6)"]

    x_cols=["threshold","threshold","threshold","threshold","threshold","threshold"]
    y_cols=["groupByCols","queryTime","timeRange","selCols","projCols","template"]
    x_label="阈值"
    y_label="准确率"
    legend_names=["分组片段","触发时间","时间范围","选择片段","投影片段","总体查询"]

    # 单列绘图
    # for i in range(len(x_cols)):
    #     plot_xy(input_path,x_cols[i],y_cols[i],titles[i],x_labels[i],y_labels[i],num_avg)

    # 多列绘图
    plot_multi_y(input_path, x_cols[0], y_cols, common_title, x_label, y_label, legend_names, num_avg)