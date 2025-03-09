import pandas as pd


def filter_csv_data(file_path, column_name, filter_condition):
    """
    读取CSV文件并根据条件筛选数据

    参数:
    file_path: CSV文件路径
    column_name: 要筛选的列名
    filter_condition: 筛选条件(可以是一个值或函数)

    返回:
    筛选后的DataFrame
    """
    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 根据条件筛选数据
    if callable(filter_condition):
        # 如果filter_condition是函数，直接应用
        filtered_df = df[df[column_name].apply(filter_condition)]
    else:
        # 如果filter_condition是值，直接比较
        filtered_df = df[df[column_name] == filter_condition]

    return filtered_df



# 读取CSV文件

file_path = r'E:\FootballBayesianPrediction\pythonProject\data\processed\merged_E0_common_sorted.csv'

df = pd.read_csv(file_path)

# 同时筛选三个条件
filtered_df = df[
    (df['Date'] == '24/05/2015') &
    (df['HomeTeam'] == 'Leicester') &
    (df['AwayTeam'] == 'QPR')
]

# 打印结果
print("筛选结果：")
print(filtered_df)

