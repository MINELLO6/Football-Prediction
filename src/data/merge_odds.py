import os
import pandas as pd

# 定义路径
raw_data_path = 'E:/FootballBayesianPrediction/pythonProject/data/raw'
merged_data_path = 'E:/FootballBayesianPrediction/pythonProject/data/processed/merged_E0_common_sorted.csv'

# 读取已合并的数据
df_merged = pd.read_csv(merged_data_path)
df_merged['Date'] = pd.to_datetime(df_merged['Date'], format='%d/%m/%Y', errors='coerce')

# 目标列及初始化（如果不存在则添加空列）
odds_columns = ['B365H', 'B365D', 'B365A']
for col in odds_columns:
    if col not in df_merged.columns:
        df_merged[col] = None

# 仅处理2020年9月12日及以后的数据
start_date = pd.Timestamp('2020-09-12')
df_merged = df_merged[df_merged['Date'] >= start_date]

# 遍历指定范围内的原始数据文件
for file_name in os.listdir(raw_data_path):
    if file_name.startswith('E0_') and file_name.endswith('.csv'):
        year = int(file_name[3:7])
        if 2021 <= year <= 2425:  # 仅处理指定年份范围内的文件
            file_path = os.path.join(raw_data_path, file_name)
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')

            # 仅保留2020年9月12日及以后的数据
            df = df[df['Date'] >= start_date]

            # 仅保留必要列，处理可能缺失的列
            for col in odds_columns:
                if col not in df.columns:
                    df[col] = None

            # 通过日期和球队名精确合并数据
            df = df[['Date', 'HomeTeam', 'AwayTeam'] + odds_columns]
            df_merged = df_merged.merge(df, on=['Date', 'HomeTeam', 'AwayTeam'], how='left', suffixes=('', '_odds'))

            # 更新赔率列，仅在原始列为空时填充，并避免空数组警告
            for col in odds_columns:
                if df_merged[col + '_odds'].notna().any():
                    df_merged[col] = df_merged[col].fillna(df_merged[col + '_odds']).astype(float)
                df_merged.drop(columns=[col + '_odds'], inplace=True)

# 保存合并后的数据
output_path = 'E:/FootballBayesianPrediction/pythonProject/data/processed/merged_E0_with_odds.csv'
df_merged.to_csv(output_path, index=False)
print(f'合并完成，数据已保存为: {output_path}')
