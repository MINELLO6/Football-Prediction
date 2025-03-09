import os
import tempfile
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
import time
import warnings
from base_model import BaseModel

# 设置临时文件目录
temp_dir = 'E:/FootballBayesianPrediction/pythonProject/temp'
os.environ['TMPDIR'] = temp_dir
os.environ['TEMP'] = temp_dir
os.environ['TMP'] = temp_dir

# 确保临时目录存在
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# 设置临时文件夹
tempfile.tempdir = temp_dir

# 设置工作目录
work_dir = 'E:/FootballBayesianPrediction/pythonProject'
if not os.path.exists(work_dir):
    os.makedirs(work_dir)
os.chdir(work_dir)

class DixonColesModel(BaseModel):
    def __init__(self):
        self.teams = None
        self.params_ = None
        self.param_names_ = None
        self.initial_params_ = None
        self._res = None
        self.loglikelihood = None
        self.aic = None
        self.iteration_count = 0

    def _initialize_params(self, teams):
        self.teams = teams
        self.n_teams = len(teams)
        params = {}
        for team in teams:
            params[f"attack_{team}"] = 1.0  # log(1.0) = 0
            params[f"defense_{team}"] = -1.0  # log(1.0) = 0
        params["home_advantage"] = 0.25
        params["rho"] = -0.2
        self.param_names_ = list(params.keys())
        self.initial_params_ = np.array(list(params.values()))
        # print("self.initial_params_ array:")
        # print(self.initial_params_)

    def _tau(self, x, y, lambda_x, lambda_y, rho):
        if x == 0 and y == 0:
            return 1 - (lambda_x * lambda_y * rho)
        elif x == 0 and y == 1:
            return 1 + (lambda_x * rho)
        elif x == 1 and y == 0:
            return 1 + (lambda_y * rho)
        elif x == 1 and y == 1:
            return 1 - rho
        else:
            return 1.0

    def _neg_log_likelihood(self, params, data):
        ll = 0.0
        param_dict = dict(zip(self.param_names_, params))
        rho = param_dict["rho"]

        # 直接使用普通的 for 循环，去掉 tqdm
        for idx, row in data.iterrows():
            # print(f"  rho: {rho:.4f}")
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            goals_home = int(row['FTHG'])
            goals_away = int(row['FTAG'])
            weight = row["weight"]

            # print(
            #     f"DEBUG -  attack parameter for home team {home_team} is {param_dict[f'attack_{home_team}']}")

            log_lambda_home = (param_dict[f"attack_{home_team}"] +
                               param_dict[f"defense_{away_team}"] +
                               param_dict["home_advantage"])
            log_lambda_away = (param_dict[f"attack_{away_team}"] +
                               param_dict[f"defense_{home_team}"])

            lambda_home = np.exp(log_lambda_home)
            lambda_away = np.exp(log_lambda_away)

            tau = self._tau(goals_home, goals_away, lambda_home, lambda_away, rho)

            # Calculate Poisson log probabilities
            log_prob_home = poisson.logpmf(goals_home, lambda_home)
            log_prob_away = poisson.logpmf(goals_away, lambda_away)

            log_tau = np.log(tau)
            # 尝试计算 log(tau)
            # try:
            #     log_tau = np.log(tau)
            # except Exception as e:
            #     print(f"Error computing log(tau) at idx {idx}: tau = {tau}, error: {e}")
            #     raise

            combined = (log_prob_home + log_prob_away + log_tau) * weight

            if not np.isfinite(combined):
                print(f"DEBUG - idx: {idx}")
                print(f"  goals_home: {goals_home}, goals_away: {goals_away}")
                print(f"  lambda_home: {lambda_home:.4f}, lambda_away: {lambda_away:.4f}")
                print(f"  log_prob_home: {log_prob_home:.4f}")
                print(f"  log_prob_away: {log_prob_away:.4f}")
                print(f"  log_tau: {log_tau:.4f}")
                print(f"  rho: {rho:.4f}")
                print(f"  Combined: {combined}")


            ll += -combined

        return ll

    @staticmethod
    def dc_decay(xi, t):
        return np.exp(-xi * t)

    def fit(self, X, xi=0.0001):
        self.iteration_count = 0
        teams = pd.concat([X['HomeTeam'], X['AwayTeam']]).unique()
        self._initialize_params(teams)
        X['days_since'] = (X["Date"].max() - X["Date"]).dt.days
        X["weight"] = self.dc_decay(xi, X["days_since"])

        # 设置约束条件
        constraints = [
            {
                "type": "eq",
                "fun": lambda x: np.sum(x[0:2*self.n_teams:2]) - self.n_teams
            }
            # {
            #     "type": "eq",
            #     "fun": lambda x: np.sum(x[self.n_teams:2 * self.n_teams])  # 防守参数之和为0
            # }
        ]

        # 设置参数边界
        bounds = []
        bounds.extend([(-3.0, 3.0)] * self.n_teams)  # 攻击力参数边界
        bounds.extend([(-3.0, 3.0)] * self.n_teams)  # 防守力参数边界
        bounds.append((np.log(1.0), 2))  # 主场优势参数边界
        bounds.append((-0.1, 0.1))  # rho参数边界


        # 添加callback函数
        def callback(xk):
            """
            xk是当前的参数值
            """
            iteration = getattr(callback, 'iteration', 0)
            callback.iteration = iteration + 1

            # 将当前参数数组转换为字典，便于查看每个参数的意义
            current_params = dict(zip(self.param_names_, xk))

            # 计算当前的负对数似然值
            current_ll = self._neg_log_likelihood(xk, X)

            print(f"\nOptimization iteration {callback.iteration}:")
            print(f"Current negative log-likelihood: {current_ll:.4f}")

            print("Current parameters:")

            # 打印attack参数
            print("\nAttack parameters:")
            for team in self.teams:
                print(f"{team:15}: {np.exp(current_params[f'attack_{team}']):.4f}")

            # 打印defense参数
            print("\nDefense parameters:")
            for team in self.teams:
                print(f"{team:15}: {np.exp(current_params[f'defense_{team}']):.4f}")

            # 打印home_advantage和rho
            print(f"\nHome advantage: {np.exp(current_params['home_advantage']):.4f}")
            print(f"Rho: {current_params['rho']:.4f}")
            print("-" * 50)

        print("Starting optimization...")
        start_time = time.time()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            result = minimize(
                self._neg_log_likelihood,
                self.initial_params_,
                args=(X,),
                bounds=bounds,
                constraints=constraints,
                callback=callback,
                # options={'maxiter': 100,
                #          'maxfun': 100}
                options={'maxiter': 50}
            )
        end_time = time.time()

        if result.success:
            self.params_ = dict(zip(self.param_names_, result.x))
            self._res = result
            print(f"\nOptimization completed in {end_time - start_time:.2f} seconds")
            print("\nFinal Parameters:")
            print(f"home_advantage: {np.exp(self.params_['home_advantage']):.4f}")
            print(f"rho: {self.params_['rho']:.4f}")
            print("\nTeam Parameters:")
            for team in self.teams:
                attack = np.exp(self.params_[f'attack_{team}'])
                defense = np.exp(self.params_[f'defense_{team}'])
                print(f"{team:15} - Attack: {attack:.4f}, Defense: {defense:.4f}")
        else:
            raise RuntimeError(f"Optimization failed: {result.message}")

    def predict(self, X):
        # if self.params_ is None:
        #     raise ValueError("Model not fitted yet")
        #
        # predictions = []
        # for idx, row in X.iterrows():
        #     home = row['HomeTeam']
        #     away = row['AwayTeam']
        #
        #     log_lambda_home = (self.params_[f"attack_{home}"] +
        #                        self.params_[f"defense_{away}"] +
        #                        self.params_["home_advantage"])
        #     log_lambda_away = (self.params_[f"attack_{away}"] +
        #                        self.params_[f"defense_{home}"])
        #
        #     lambda_home = np.exp(log_lambda_home)
        #     lambda_away = np.exp(log_lambda_away)
        #
        #     predictions.append((lambda_home, lambda_away))
        #
        # return predictions
        pass

    def evaluate(self, X, y):
        """
        评估模型性能

        参数:
        X: DataFrame, 包含 HomeTeam 和 AwayTeam 列
        y: DataFrame, 包含实际的 FTHG 和 FTAG 列

        返回:
        dict: 包含评估指标的字典
        """
        # if self.params_ is None:
        #     raise ValueError("Model is not fitted yet")
        #
        # predictions = self.predict(X)
        # pred_home = np.array([p[0] for p in predictions])
        # pred_away = np.array([p[1] for p in predictions])
        #
        # # 计算均方误差
        # mse_home = np.mean((y['FTHG'] - pred_home) ** 2)
        # mse_away = np.mean((y['FTAG'] - pred_away) ** 2)
        #
        # # 计算平均绝对误差
        # mae_home = np.mean(np.abs(y['FTHG'] - pred_home))
        # mae_away = np.mean(np.abs(y['FTAG'] - pred_away))
        #
        # print(predictions)
        #
        # return {
        #     "MSE_Home": mse_home,
        #     "MSE_Away": mse_away,
        #     "MAE_Home": mae_home,
        #     "MAE_Away": mae_away
        # }
        pass


# 运行测试
if __name__ == "__main__":
    # 加载数据
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "../../data/processed/merged_E0_common_sorted.csv")
    try:
        data = pd.read_csv(data_path)
    except Exception as e:
        raise RuntimeError("Error reading processed data: " + str(e))

    # 检查必需的列
    required_cols = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'Date']
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Column {col} is missing from the data.")

    # 按季度筛选数据
    data['Date'] = pd.to_datetime(data['Date'], format="%d/%m/%Y", errors="coerce")
    # quarter_to_filter = 1  # 选择第一季度数据
    # filtered_data = data[data['Date_dt'].dt.quarter == quarter_to_filter].copy()
    # print(f"Filtered data: {filtered_data.shape[0]} rows for Q{quarter_to_filter}.")
    # filtered_data.drop(columns=['Date_dt'], inplace=True)
    #
    # # 创建并训练模型
    # model = DixonColesModel()
    # model.fit(filtered_data)
    #
    # # 预测示例
    # new_match = pd.DataFrame({"HomeTeam": ["Chelsea"], "AwayTeam": ["Tottenham"]})
    # predictions = model.predict(new_match)
    # print("\nPredicted expected goals for Chelsea vs Tottenham:", predictions)
    #
    # # 评估模型
    # eval_results = model.evaluate(filtered_data, filtered_data[['FTHG', 'FTAG']])
    # print("\nModel Evaluation Results:")
    # for metric, value in eval_results.items():
    #     print(f"{metric}: {value:.4f}")

    all_data = data.copy()
    # all_data.drop(columns=['Date'], inplace=True)

    # 创建并训练模型，使用所有数据进行训练
    model = DixonColesModel()
    model.fit(all_data)

    # # 预测示例
    # new_match = pd.DataFrame({"HomeTeam": ["Chelsea"], "AwayTeam": ["Tottenham"]})
    # predictions = model.predict(new_match)
    # print("\nPredicted expected goals for Chelsea vs Tottenham:", predictions)
    #
    # # 评估模型
    # eval_results = model.evaluate(all_data, all_data[['FTHG', 'FTAG']])
    # print("\nModel Evaluation Results:")
    # for metric, value in eval_results.items():
    #     print(f"{metric}: {value:.4f}")