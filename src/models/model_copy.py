import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../../penaltyblog")
import penaltyblog as pb

# 读取数据
data_path = "../../data/processed/merged_E1.csv"
df = pd.read_csv(data_path)

# 预处理日期列，尝试不同的日期格式
df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")

# 检查是否有未能解析的日期
if df["Date"].isna().sum() > 0:
    print("无法解析的日期行:")
    print(df[df["Date"].isna()][["Date"]])

print(df[["Date", "FTHG", "FTAG", "HomeTeam", "AwayTeam"]].head())
print("日期数据类型:", df["Date"].dtype)  # 应为 datetime64[ns]
print("日期范围:", df["Date"].min(), df["Date"].max())

# 划分训练集和测试集
train = df[(df["Date"] >= "2000-01-01") & (df["Date"] < "2020-06-11")]
test = df[(df["Date"] >= "2020-06-11") & (df["Date"] <= "2025-12-31")]

print(f"训练集样本数量: {len(train)}")
print(f"测试集样本数量: {len(test)}")

# 检查测试集中是否有未出现在训练集中的球队
train_teams = set(train["HomeTeam"]).union(set(train["AwayTeam"]))
test_teams = set(test["HomeTeam"]).union(set(test["AwayTeam"]))

missing_teams = test_teams - train_teams
if missing_teams:
    print(f"以下球队在测试集中存在，但未在训练集中出现: {missing_teams}")
    # 移除测试集中包含这些球队的比赛
    test = test[~test["HomeTeam"].isin(missing_teams)]
    test = test[~test["AwayTeam"].isin(missing_teams)]
    print(f"清理后测试集样本数量: {len(test)}")
else:
    print("所有测试集中的球队在训练集中都有出现。")


# 定义 RPS 计算函数
def calculate_rps(clf, df):
    rps = list()
    for idx, row in df.iterrows():
        if row["FTR"] == "H":
            outcome = 0
        elif row["FTR"] == "D":
            outcome = 1
        elif row["FTR"] == "A":
            outcome = 2

        predictions = clf.predict(row["HomeTeam"], row["AwayTeam"]).home_draw_away
        rps.append(pb.metrics.rps(predictions, outcome))

    return np.mean(rps)


# 定义要测试的 xi 值
xis = np.arange(0, 0.00255, 0.00005)
rps = list()


# 依次拟合不同的 xi 并计算 RPS
for xi in xis:
    print(f"正在拟合模型, xi = {xi}")

    # 重新拟合模型
    clf = pb.models.DixonColesGoalModel(
        train["Date"],
        train["FTHG"],
        train["FTAG"],
        train["HomeTeam"],
        train["AwayTeam"],
        xi=xi
    )
    clf.fit()

    # 计算 RPS
    rps_value = calculate_rps(clf, test)
    print(f"xi = {xi}, RPS = {rps_value}")
    rps.append(rps_value)

# 绘制 RPS 曲线
plt.figure(figsize=(8, 6))
plt.plot(xis, rps, marker='o', linestyle='-', color='b')
plt.xlabel("xi")
plt.ylabel("RPS")
plt.title("RPS vs xi")
plt.grid(True)

# 找到 RPS 的最低点及对应的 xi
min_rps = min(rps)
min_xi = xis[rps.index(min_rps)]

# 在图像上标注最低点
plt.annotate(f"Min RPS: {min_rps:.4f}\nxi: {min_xi:.5f}",
             xy=(min_xi, min_rps),
             xytext=(min_xi + 0.0002, min_rps + 0.0005),
             arrowprops=dict(facecolor='red', shrink=0.05),
             fontsize=10, color='red')

# 保存为图片文件
plt.savefig("rps_vs_xi.png")
print("图像已保存为 rps_vs_xi.png")

# 使用最优的 xi 重新拟合模型，以获取最优模型的参数
print(f"正在使用最优 xi = {min_xi} 拟合最终模型")

best_model = pb.models.DixonColesGoalModel(
    train["Date"],
    train["FTHG"],
    train["FTAG"],
    train["HomeTeam"],
    train["AwayTeam"],
    xi=min_xi
)
best_model.fit()

# 提取球队的攻击和防守参数
teams = best_model.teams
n_teams = len(teams)
attack_params = best_model._params[:n_teams]
defence_params = best_model._params[n_teams:n_teams*2]

# 将参数从对数空间转换回原始空间
attack_exp = attack_params
defence_exp = defence_params

# 创建一个包含球队名称和对应参数的数据框
team_params = pd.DataFrame({
    'Team': teams,
    'Attack': attack_exp,
    'Defence': defence_exp
})

# 绘制攻击和防守参数的散点图
plt.figure(figsize=(14, 8))

# 计算实际数据范围
x_min, x_max = team_params['Defence'].min(), team_params['Defence'].max()
y_min, y_max = team_params['Attack'].min(), team_params['Attack'].max()

# 添加一些额外空间
x_padding = (x_max - x_min) * 0.1
y_padding = (y_max - y_min) * 0.1

# 绘制散点图，使用稍大的点
plt.scatter(team_params['Defence'], team_params['Attack'],
            alpha=0.8, s=60, color='steelblue', edgecolor='black')

# 避免标签重叠的更好方法
from adjustText import adjust_text

texts = []
for i, team in enumerate(teams):
    # 创建文本注释但先不添加到图表中
    texts.append(plt.text(team_params['Defence'][i], team_params['Attack'][i],
                          team, fontsize=9))

# 使用adjust_text库自动调整文本位置以避免重叠
adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red', alpha=0.5))

# 设置更合适的坐标轴范围
plt.xlim(x_min - x_padding, x_max + x_padding)
plt.ylim(y_min - y_padding, y_max + y_padding)

# 坐标轴和标题
plt.xlabel('Defence', fontsize=12)
plt.ylabel('Attack', fontsize=12)
plt.title('Team Attack vs Defence Parameters', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)

# 保存图片
plt.tight_layout()  # 确保所有元素都在图中显示
plt.savefig("team_attack_defence_params_improved.png", dpi=300)
print("改进后的球队攻防参数图已保存")