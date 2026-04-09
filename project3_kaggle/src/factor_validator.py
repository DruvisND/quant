"""
因子检验模块（Kaggle适配版）
对IRSF因子进行检验
"""
from __future__ import annotations
import os
import sys
from typing import Dict, Optional, List, Tuple
import pandas as pd
import numpy as np
from scipy import stats

# 初始化为False，在函数内部尝试导入
HAS_STATSMODELS = False

# Kaggle平台的输出目录
KAGGLE_OUTPUT_DIR = "../output"
LOCAL_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

# 自动检测运行环境
IS_KAGGLE = os.path.exists("../input")

if IS_KAGGLE:
    OUTPUT_DIR = KAGGLE_OUTPUT_DIR
else:
    OUTPUT_DIR = LOCAL_OUTPUT_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)


def calculate_ic(
    factor: pd.Series,
    returns: pd.Series
) -> Tuple[float, float, float, float, float]:
    """
    计算因子与收益的IC相关系数

    Parameters:
    -----------
    factor : pd.Series
        因子值
    returns : pd.Series
        收益率

    Returns:
    --------
    Tuple[float, float, float, float, float]
        IC均值, IC标准差, ICIR, 胜率, p值
    """
    ic = factor.corr(returns, method="pearson")
    rank_ic = factor.corr(returns, method="spearman")

    return ic, rank_ic


def run_fama_macbeth(
    factor_df: pd.DataFrame,
    factor_col: str,
    return_col: str = "return_next",
    control_vars: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    运行Fama-MacBeth回归

    Parameters:
    -----------
    factor_df : pd.DataFrame
        因子和收益数据
    factor_col : str
        因子列名
    return_col : str
        收益列名
    control_vars : List[str], optional
        控制变量列名列表

    Returns:
    --------
    Dict[str, float]
        回归结果
    """
    results = {}
    local_has_statsmodels = False
    sm = None
    
    # 尝试导入statsmodels
    try:
        import statsmodels.api as sm
        local_has_statsmodels = True
    except ImportError:
        print("警告: statsmodels导入失败，将使用简化的回归方法")

    # 按日期分组进行横截面回归
    coefficients = []
    t_stats = []

    for date, group in factor_df.groupby("date"):
        if len(group) < 10:  # 样本量太小，跳过
            continue

        X = group[[factor_col]]
        if control_vars:
            X = X.join(group[control_vars])
        y = group[return_col]

        try:
            if local_has_statsmodels and sm is not None:
                X_sm = sm.add_constant(X)
                model = sm.OLS(y, X_sm).fit()
                coefficients.append(model.params[factor_col])
                t_stats.append(model.tvalues[factor_col])
            else:
                # 使用简化的回归方法
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                model.fit(X, y)
                coefficients.append(model.coef_[0])
                # 简化的t统计量计算
                y_pred = model.predict(X)
                residual = y - y_pred
                mse = np.mean(residual ** 2)
                var_beta = mse / np.sum((X[factor_col] - np.mean(X[factor_col])) ** 2)
                std_beta = np.sqrt(var_beta)
                t_stat = model.coef_[0] / std_beta if std_beta > 0 else 0
                t_stats.append(t_stat)
        except Exception as e:
            print(f"回归失败: {e}")
            continue

    if coefficients:
        # 计算平均系数和t统计量
        avg_coef = np.mean(coefficients)
        avg_t = np.mean(t_stats)
        std_coef = np.std(coefficients)
        t_stat = avg_coef / (std_coef / np.sqrt(len(coefficients)))
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=len(coefficients)-1))

        results = {
            "avg_coef": avg_coef,
            "avg_t": avg_t,
            "std_coef": std_coef,
            "t_stat": t_stat,
            "p_value": p_value,
            "n_periods": len(coefficients)
        }

    return results


def run_quintile_test(
    factor_df: pd.DataFrame,
    factor_col: str,
    return_col: str = "return_next",
    n_groups: int = 5
) -> Dict[str, pd.DataFrame]:
    """
    运行分组测试

    Parameters:
    -----------
    factor_df : pd.DataFrame
        因子和收益数据
    factor_col : str
        因子列名
    return_col : str
        收益列名
    n_groups : int
        分组数

    Returns:
    --------
    Dict[str, pd.DataFrame]
        分组测试结果
    """
    results = {}

    # 按日期和因子值分组
    grouped_returns = []
    for date, group in factor_df.groupby("date"):
        if len(group) < n_groups:
            continue

        # 按因子值排序并分组
        group = group.sort_values(factor_col)
        group["group"] = pd.qcut(group[factor_col], n_groups, labels=False) + 1

        # 计算每组的平均收益
        group_returns = group.groupby("group")[return_col].mean().reset_index()
        group_returns["date"] = date
        grouped_returns.append(group_returns)

    if grouped_returns:
        group_df = pd.concat(grouped_returns)
        avg_returns = group_df.groupby("group")[return_col].mean()
        std_returns = group_df.groupby("group")[return_col].std()
        t_stats = group_df.groupby("group")[return_col].apply(
            lambda x: stats.ttest_1samp(x, 0)[0]
        )
        p_values = group_df.groupby("group")[return_col].apply(
            lambda x: stats.ttest_1samp(x, 0)[1]
        )

        results["avg_returns"] = avg_returns
        results["std_returns"] = std_returns
        results["t_stats"] = t_stats
        results["p_values"] = p_values
        results["long_short"] = avg_returns.iloc[-1] - avg_returns.iloc[0]

    return results


def run_ic_analysis(
    factor_df: pd.DataFrame,
    factor_col: str,
    return_col: str = "return_next"
) -> Dict[str, Dict[str, float]]:
    """
    运行IC分析

    Parameters:
    -----------
    factor_df : pd.DataFrame
        因子和收益数据
    factor_col : str
        因子列名
    return_col : str
        收益列名

    Returns:
    --------
    Dict[str, Dict[str, float]]
        IC分析结果
    """
    ic_values = []
    rank_ic_values = []

    for date, group in factor_df.groupby("date"):
        if len(group) < 10:
            continue

        ic, rank_ic = calculate_ic(group[factor_col], group[return_col])
        ic_values.append(ic)
        rank_ic_values.append(rank_ic)

    results = {}

    if ic_values:
        results["IC"] = {
            "mean": np.mean(ic_values),
            "std": np.std(ic_values),
            "ir": np.mean(ic_values) / np.std(ic_values) if np.std(ic_values) > 0 else 0,
            "win_rate": np.mean([1 if x > 0 else 0 for x in ic_values]),
            "p_value": stats.ttest_1samp(ic_values, 0)[1]
        }

    if rank_ic_values:
        results["Rank_IC"] = {
            "mean": np.mean(rank_ic_values),
            "std": np.std(rank_ic_values),
            "ir": np.mean(rank_ic_values) / np.std(rank_ic_values) if np.std(rank_ic_values) > 0 else 0,
            "win_rate": np.mean([1 if x > 0 else 0 for x in rank_ic_values]),
            "p_value": stats.ttest_1samp(rank_ic_values, 0)[1]
        }

    return results


class FactorValidator:
    """因子检验器"""

    def __init__(self, factor_df: pd.DataFrame):
        """
        初始化因子检验器

        Parameters:
        -----------
        factor_df : pd.DataFrame
            因子和收益数据
        """
        self.factor_df = factor_df.copy()

    def run_fama_macbeth(
        self,
        factor_col: str,
        return_col: str = "return_next",
        control_vars: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        运行Fama-MacBeth回归
        """
        return run_fama_macbeth(
            self.factor_df, factor_col, return_col, control_vars
        )

    def run_quintile_test(
        self,
        factor_col: str,
        return_col: str = "return_next",
        n_groups: int = 5
    ) -> Dict[str, pd.DataFrame]:
        """
        运行分组测试
        """
        return run_quintile_test(
            self.factor_df, factor_col, return_col, n_groups
        )

    def run_ic_analysis(
        self,
        factor_col: str,
        return_col: str = "return_next"
    ) -> Dict[str, Dict[str, float]]:
        """
        运行IC分析
        """
        return run_ic_analysis(
            self.factor_df, factor_col, return_col
        )

    def run_full_validation(
        self,
        factor_col: str,
        return_col: str = "return_next",
        control_vars: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, any]:
        """
        运行完整的因子检验

        Parameters:
        -----------
        factor_col : str
            因子列名
        return_col : str
            收益列名
        control_vars : List[str], optional
            控制变量列名列表
        save_path : str, optional
            结果保存路径

        Returns:
        --------
        Dict[str, any]
            完整的检验结果
        """
        results = {}

        print("[因子检验] 运行Fama-MacBeth回归...")
        fm_results = self.run_fama_macbeth(factor_col, return_col, control_vars)
        results["fama_macbeth"] = fm_results

        print("[因子检验] 运行分组测试...")
        quintile_results = self.run_quintile_test(factor_col, return_col)
        results["quintile_test"] = quintile_results

        print("[因子检验] 运行IC分析...")
        ic_results = self.run_ic_analysis(factor_col, return_col)
        results["ic_analysis"] = ic_results

        # 保存结果
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write("# 因子检验报告\n\n")
                
                f.write("## Fama-MacBeth回归结果\n")
                if fm_results:
                    f.write(f"平均系数: {fm_results['avg_coef']:.4f}\n")
                    f.write(f"t统计量: {fm_results['t_stat']:.4f}\n")
                    f.write(f"p值: {fm_results['p_value']:.4f}\n")
                    f.write(f"观测期数: {fm_results['n_periods']}\n")
                else:
                    f.write("无结果\n")
                f.write("\n")

                f.write("## 分组测试结果\n")
                if quintile_results:
                    f.write("各分组平均收益:\n")
                    for group, ret in quintile_results["avg_returns"].items():
                        f.write(f"组 {group}: {ret:.4f}\n")
                    f.write(f"多空收益: {quintile_results['long_short']:.4f}\n")
                else:
                    f.write("无结果\n")
                f.write("\n")

                f.write("## IC分析结果\n")
                if ic_results:
                    if "Rank_IC" in ic_results:
                        f.write(f"Rank IC均值: {ic_results['Rank_IC']['mean']:.4f}\n")
                        f.write(f"Rank ICIR: {ic_results['Rank_IC']['ir']:.4f}\n")
                        f.write(f"Rank IC胜率: {ic_results['Rank_IC']['win_rate']:.4f}\n")
                        f.write(f"Rank IC p值: {ic_results['Rank_IC']['p_value']:.4f}\n")
                else:
                    f.write("无结果\n")

        return results


if __name__ == "__main__":
    print("因子检验模块测试")
    
    # 生成示例数据
    dates = pd.date_range("2020-01-01", "2024-12-31", freq="M")
    stock_codes = ["00" + str(i).zfill(4) + ".SZ" for i in range(1, 101)]
    
    data = []
    for date in dates:
        for code in stock_codes:
            data.append({
                "ts_code": code,
                "date": date,
                "IRSF_score": np.random.normal(0, 1),
                "return_next": np.random.normal(0, 0.05) + 0.01 * np.random.normal(0, 1),
                "density_ratio": np.random.uniform(0, 5)
            })
    
    df = pd.DataFrame(data)
    
    validator = FactorValidator(df)
    
    # 运行Fama-MacBeth回归
    fm_results = validator.run_fama_macbeth("IRSF_score", control_vars=["density_ratio"])
    print("Fama-MacBeth回归结果:")
    print(f"平均系数: {fm_results.get('avg_coef', 0):.4f}")
    print(f"t统计量: {fm_results.get('t_stat', 0):.4f}")
    print(f"p值: {fm_results.get('p_value', 0):.4f}")
    
    # 运行分组测试
    quintile_results = validator.run_quintile_test("IRSF_score")
    print("\n分组测试结果:")
    if "avg_returns" in quintile_results:
        print(quintile_results["avg_returns"])
        print(f"多空收益: {quintile_results.get('long_short', 0):.4f}")
    
    # 运行IC分析
    ic_results = validator.run_ic_analysis("IRSF_score")
    print("\nIC分析结果:")
    if "Rank_IC" in ic_results:
        print(f"Rank IC均值: {ic_results['Rank_IC']['mean']:.4f}")
        print(f"Rank ICIR: {ic_results['Rank_IC']['ir']:.4f}")
