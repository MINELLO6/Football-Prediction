import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

sys.path.append("../../penaltyblog")
import penaltyblog as pb

# 读取数据
data_path = '../../data/processed/merged_E0_common_sorted.csv'
df = pd.read_csv(data_path)

# 预处理日期列
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')

# 划分训练集和测试集
train = df[(df['Date'] >= '2000-01-01') & (df['Date'] < '2020-09-12')]
test = df[(df['Date'] >= '2020-09-12') & (df['Date'] <= '2025-12-31')]

# 确保测试数据按日期从早到晚排序
test = test.sort_values(by='Date')

# 移除测试集中未在训练集中出现的球队
train_teams = set(train['HomeTeam']).union(set(train['AwayTeam']))
test_teams = set(test['HomeTeam']).union(set(test['AwayTeam']))
missing_teams = test_teams - train_teams
if missing_teams:
    test = test[~test['HomeTeam'].isin(missing_teams)]
    test = test[~test['AwayTeam'].isin(missing_teams)]

# Dixon-Coles模型拟合 (Time Weighted版本)
xi = 0.0002  # 使用之前找到的最佳 xi
clf_timeweighted = pb.models.DixonColesGoalModel(
    train['Date'],
    train['FTHG'],
    train['FTAG'],
    train['HomeTeam'],
    train['AwayTeam'],
    xi=xi
)
clf_timeweighted.fit()

# Dixon-Coles Basic模型拟合 (xi=0，没有时间衰减)
xi_basic = 0.0  # 基础版本不使用时间衰减
clf_basic = pb.models.DixonColesGoalModel(
    train['Date'],
    train['FTHG'],
    train['FTAG'],
    train['HomeTeam'],
    train['AwayTeam'],
    xi=xi_basic
)
clf_basic.fit()


# 定义 RPS 计算函数
def calculate_rps(predictions, outcome):
    return pb.metrics.rps(predictions, outcome)


# 创建用于存储结果的列表
results_timeweighted_data = []
results_basic_data = []

for idx, row in test.iterrows():
    # 实际结果转换为索引
    if row['FTR'] == 'H':
        outcome = 0
    elif row['FTR'] == 'D':
        outcome = 1
    elif row['FTR'] == 'A':
        outcome = 2
    else:
        continue

    # Dixon-Coles Time Weighted预测
    dixon_tw_pred = clf_timeweighted.predict(row['HomeTeam'], row['AwayTeam']).home_draw_away
    dixon_tw_rps_value = calculate_rps(dixon_tw_pred, outcome)

    # Dixon-Coles Basic预测
    dixon_basic_pred = clf_basic.predict(row['HomeTeam'], row['AwayTeam']).home_draw_away
    dixon_basic_rps_value = calculate_rps(dixon_basic_pred, outcome)

    # 博彩公司概率计算
    odds = [row['B365H'], row['B365D'], row['B365A']]
    if any(pd.isna(odds)) or any(o <= 0 for o in odds):
        continue
    bookmaker_pred = np.reciprocal(odds)
    bookmaker_pred /= bookmaker_pred.sum()  # 归一化为概率
    bookmaker_rps_value = calculate_rps(bookmaker_pred, outcome)

    # 计算差值
    tw_rps_diff_value = dixon_tw_rps_value - bookmaker_rps_value
    basic_rps_diff_value = dixon_basic_rps_value - bookmaker_rps_value

    # 存储数据
    results_timeweighted_data.append({
        'Date': row['Date'],
        'Model_RPS': dixon_tw_rps_value,
        'Bookmaker_RPS': bookmaker_rps_value,
        'RPS_Difference': tw_rps_diff_value
    })

    results_basic_data.append({
        'Date': row['Date'],
        'Model_RPS': dixon_basic_rps_value,
        'Bookmaker_RPS': bookmaker_rps_value,
        'RPS_Difference': basic_rps_diff_value
    })

# 转换为DataFrame
results_timeweighted_df = pd.DataFrame(results_timeweighted_data)
results_basic_df = pd.DataFrame(results_basic_data)

# 添加累积差值列
results_timeweighted_df['Cumulative_RPS_Difference'] = results_timeweighted_df['RPS_Difference'].cumsum()
results_basic_df['Cumulative_RPS_Difference'] = results_basic_df['RPS_Difference'].cumsum()

# 读取额外的两个CSV文件
bayes_basic_df = pd.read_csv('Bayes.Basic_RPS.csv')
bayes_timeweighted_df = pd.read_csv('Bayes.TimeWeighted_predictions_fixed.csv')

# 查看CSV文件的列
print("Bayes Basic CSV列:", bayes_basic_df.columns.tolist())
print("Bayes TimeWeighted CSV列:", bayes_timeweighted_df.columns.tolist())

# 确保日期列为datetime类型
bayes_basic_df['Date'] = pd.to_datetime(bayes_basic_df['Date'])
bayes_timeweighted_df['Date'] = pd.to_datetime(bayes_timeweighted_df['Date'])

# 处理缺失的RPS值 - 使用前一行的值填充
# 注意：这两个CSV按照日期从近到远排序，所以使用后向填充
bayes_basic_df['RPS'] = bayes_basic_df['RPS'].bfill()
bayes_timeweighted_df['RPS'] = bayes_timeweighted_df['RPS'].bfill()

# 创建一个从日期到Bookmaker RPS的映射，这样可以用于所有模型
bookmaker_rps_by_date = {}
for idx, row in test.iterrows():
    # 只处理有效的赔率数据
    odds = [row['B365H'], row['B365D'], row['B365A']]
    if any(pd.isna(odds)) or any(o <= 0 for o in odds):
        continue

    # 计算实际结果的索引
    if row['FTR'] == 'H':
        outcome = 0
    elif row['FTR'] == 'D':
        outcome = 1
    elif row['FTR'] == 'A':
        outcome = 2
    else:
        continue

    # 计算Bookmaker RPS
    bookmaker_pred = np.reciprocal(odds)
    bookmaker_pred /= bookmaker_pred.sum()  # 归一化为概率
    bookmaker_rps = calculate_rps(bookmaker_pred, outcome)

    # 存储到映射中
    bookmaker_rps_by_date[row['Date']] = bookmaker_rps


# 为Bayes模型找到对应日期的Bookmaker RPS
def match_bookmaker_rps(df):
    rps_diff = []
    no_match_count = 0

    for i, row in df.iterrows():
        date = row['Date']
        model_rps = row['RPS']

        # 尝试找到最接近的日期
        closest_date = None
        min_days_diff = float('inf')

        for bm_date in bookmaker_rps_by_date.keys():
            days_diff = abs((date - bm_date).days)
            if days_diff < min_days_diff:
                min_days_diff = days_diff
                closest_date = bm_date

        # 如果找到了接近的日期（比如在7天以内），则使用对应的Bookmaker RPS
        if closest_date is not None and min_days_diff <= 7:
            bookmaker_rps = bookmaker_rps_by_date[closest_date]
            rps_diff.append(model_rps - bookmaker_rps)
        else:
            # 如果找不到接近的日期，就保持模型RPS不变
            rps_diff.append(model_rps)
            no_match_count += 1

    return rps_diff, no_match_count


# 计算RPS差值
bayes_basic_rps_diff, bayes_basic_no_match = match_bookmaker_rps(bayes_basic_df)
bayes_basic_df['RPS_Difference'] = bayes_basic_rps_diff

bayes_timeweighted_rps_diff, bayes_timeweighted_no_match = match_bookmaker_rps(bayes_timeweighted_df)
bayes_timeweighted_df['RPS_Difference'] = bayes_timeweighted_rps_diff

print(f"处理的比赛场次: {len(bookmaker_rps_by_date)}")
print(f"Bayes Basic 样本数: {len(bayes_basic_df)}")
print(f"Bayes Basic 未匹配Bookmaker RPS的样本数: {bayes_basic_no_match}")
print(f"Bayes TimeWeighted 样本数: {len(bayes_timeweighted_df)}")
print(f"Bayes TimeWeighted 未匹配Bookmaker RPS的样本数: {bayes_timeweighted_no_match}")

# 计算累积差值
bayes_basic_df['Cumulative_RPS_Difference'] = bayes_basic_df['RPS_Difference'].cumsum()
bayes_timeweighted_df['Cumulative_RPS_Difference'] = bayes_timeweighted_df['RPS_Difference'].cumsum()

# 保存CSV
results_timeweighted_df.to_csv('rps_comparison_dixon_timeweighted.csv', index=False)
results_basic_df.to_csv('rps_comparison_dixon_basic.csv', index=False)
print('结果已保存至 rps_comparison_dixon_timeweighted.csv 和 rps_comparison_dixon_basic.csv')

# 创建图表比较四个模型与Bookmaker的累积RPS差值
plt.figure(figsize=(14, 8))

# 确保所有数据集按日期从早到晚排序，这样图表更直观
results_timeweighted_df = results_timeweighted_df.sort_values(by='Date')
results_basic_df = results_basic_df.sort_values(by='Date')
bayes_basic_df = bayes_basic_df.sort_values(by='Date')
bayes_timeweighted_df = bayes_timeweighted_df.sort_values(by='Date')

# 绘制Dixon-Coles Time Weighted模型
plt.plot(results_timeweighted_df['Date'], results_timeweighted_df['Cumulative_RPS_Difference'],
         label='Dixon-Coles Time Weighted (xi=0.0002)',
         color='blue', linewidth=2)

# 绘制Dixon-Coles Basic模型
plt.plot(results_basic_df['Date'], results_basic_df['Cumulative_RPS_Difference'],
         label='Dixon-Coles Basic (xi=0)',
         color='purple', linewidth=2)

# 绘制Bayes Basic模型
plt.plot(bayes_basic_df['Date'], bayes_basic_df['Cumulative_RPS_Difference'],
         label='Bayes Basic',
         color='red', linewidth=2)

# 绘制Bayes Time Weighted模型
plt.plot(bayes_timeweighted_df['Date'], bayes_timeweighted_df['Cumulative_RPS_Difference'],
         label='Bayes Time Weighted',
         color='green', linewidth=2)

# 添加水平参考线
plt.axhline(0, color='black', linestyle='--', linewidth=1)

# 设置图表标题和坐标轴标签
plt.title('Cumulative RPS Difference Relative to Bookmakers', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Cumulative RPS Difference (Model - Bookmakers)', fontsize=12)

# 调整日期格式
plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.gcf().autofmt_xdate()

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.7)

# 添加图例
plt.legend(loc='best', fontsize=12)

# 添加注释说明
plt.figtext(0.5, 0.01,
            'Note: Values below 0 indicate model outperforms bookmakers (lower RPS is better)',
            ha='center', fontsize=10, style='italic')

plt.tight_layout()

# 保存图表
plt.savefig('models_vs_bookmakers_rps_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print('分析完成并保存图表为：models_vs_bookmakers_rps_comparison.png')