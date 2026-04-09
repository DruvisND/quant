"""
回测模块（Kaggle适配版）
对IRSF因子策略进行回测
"""
from __future__ import annotations
import os
import sys
from typing import Dict, Optional, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

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


def calculate_returns(
    factor_data: pd.DataFrame,
    price_data: pd.DataFrame,
    n_long: int = 30,
    n_short: int = 30,
    weight_method: str = "equal"
) -> pd.DataFrame:
    """
    计算策略收益

    Parameters:
    -----------
    factor_data : pd.DataFrame
        因子数据，包含 date, ts_code, IRSF_score
    price_data : pd.DataFrame
        价格数据，包含 trade_date, ts_code, close
    n_long : int
        多头数量
    n_short : int
        空头数量
    weight_method : str
        权重方法: "equal", "factor_weighted"

    Returns:
    --------
    pd.DataFrame : 回测结果
    """
    price_data = price_data.copy()
    price_data["trade_date"] = pd.to_datetime(price_data["trade_date"])
    price_data = price_data.set_index(["trade_date", "ts_code"])["close"].unstack()

    factor_data = factor_data.copy()
    factor_data["date"] = pd.to_datetime(factor_data["date"])

    # 时间对齐检查
    min_factor_date = factor_data["date"].min()
    max_factor_date = factor_data["date"].max()
    min_price_date = price_data.index.min()
    max_price_date = price_data.index.max()
    
    print(f"因子数据时间范围: {min_factor_date} 到 {max_factor_date}")
    print(f"价格数据时间范围: {min_price_date} 到 {max_price_date}")
    
    # 检查因子数据是否早于价格数据
    if min_factor_date >= min_price_date:
        print("警告: 因子数据起始日期晚于或等于价格数据起始日期")
    
    # 检查因子数据是否覆盖价格数据
    if max_factor_date >= max_price_date:
        print("警告: 因子数据结束日期晚于或等于价格数据结束日期")

    # 计算调仓日期
    rebalance_dates = factor_data["date"].unique()
    rebalance_dates = sorted(rebalance_dates)

    # 计算每期的选股
    portfolio_returns = []
    for i, rebalance_date in enumerate(rebalance_dates[:-1]):
        next_rebalance_date = rebalance_dates[i+1]

        # 选择调仓日的因子数据
        current_factor = factor_data[factor_data["date"] == rebalance_date]

        # 按因子值排序，选择多头和空头
        current_factor = current_factor.sort_values("IRSF_score", ascending=False)
        long_stocks = current_factor.head(n_long)["ts_code"].tolist()
        short_stocks = current_factor.tail(n_short)["ts_code"].tolist()

        # 计算权重
        if weight_method == "equal":
            long_weights = {stock: 1/n_long for stock in long_stocks}
            short_weights = {stock: -1/n_short for stock in short_stocks}
        elif weight_method == "factor_weighted":
            # 因子加权
            long_factor_values = current_factor.head(n_long)["IRSF_score"]
            long_weights = (long_factor_values / long_factor_values.sum()).to_dict()
            
            short_factor_values = current_factor.tail(n_short)["IRSF_score"]
            short_weights = (-short_factor_values / short_factor_values.sum()).to_dict()
        else:
            long_weights = {stock: 1/n_long for stock in long_stocks}
            short_weights = {stock: -1/n_short for stock in short_stocks}

        # 计算持有期收益
        try:
            # 获取持有期的价格数据
            hold_period_prices = price_data.loc[rebalance_date:next_rebalance_date]
            if len(hold_period_prices) < 2:
                continue

            # 计算每只股票的收益
            stock_returns = hold_period_prices.pct_change().iloc[1:]

            # 计算组合收益
            portfolio_return = 0
            for stock, weight in {**long_weights, **short_weights}.items():
                if stock in stock_returns.columns:
                    stock_return = stock_returns[stock].mean()
                    portfolio_return += weight * stock_return

            portfolio_returns.append({
                "rebalance_date": rebalance_date,
                "next_rebalance_date": next_rebalance_date,
                "return": portfolio_return,
                "long_stocks": long_stocks,
                "short_stocks": short_stocks
            })
        except Exception as e:
            print(f"计算收益失败: {e}")
            continue

    return pd.DataFrame(portfolio_returns)


def calculate_metrics(returns: pd.DataFrame) -> Dict[str, float]:
    """
    计算回测指标

    Parameters:
    -----------
    returns : pd.DataFrame
        策略收益数据，包含 return 列

    Returns:
    --------
    Dict[str, float] : 回测指标
    """
    if len(returns) == 0:
        return {}

    # 计算基本指标
    total_return = (1 + returns["return"]).prod() - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    annual_vol = returns["return"].std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0

    # 计算最大回撤
    cumulative_returns = (1 + returns["return"]).cumprod()
    drawdown = (cumulative_returns / cumulative_returns.cummax() - 1)
    max_drawdown = drawdown.min()

    # 计算胜率
    win_rate = (returns["return"] > 0).mean()

    # 计算信息比率（假设无风险利率为0）
    info_ratio = sharpe_ratio

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "info_ratio": info_ratio,
        "n_trades": len(returns)
    }


def plot_equity_curve(
    returns: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """
    绘制权益曲线

    Parameters:
    -----------
    returns : pd.DataFrame
        策略收益数据
    save_path : str, optional
        保存路径
    """
    if len(returns) == 0:
        return

    cumulative_returns = (1 + returns["return"]).cumprod()

    plt.figure(figsize=(12, 6))
    plt.plot(returns["rebalance_date"], cumulative_returns, label="策略权益")
    plt.title("策略权益曲线")
    plt.xlabel("日期")
    plt.ylabel("累计收益")
    plt.grid(True)
    plt.legend()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"权益曲线已保存至: {save_path}")
    else:
        plt.show()


def plot_drawdown(
    returns: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """
    绘制回撤曲线

    Parameters:
    -----------
    returns : pd.DataFrame
        策略收益数据
    save_path : str, optional
        保存路径
    """
    if len(returns) == 0:
        return

    cumulative_returns = (1 + returns["return"]).cumprod()
    drawdown = (cumulative_returns / cumulative_returns.cummax() - 1)

    plt.figure(figsize=(12, 6))
    plt.plot(returns["rebalance_date"], drawdown, label="回撤")
    plt.fill_between(returns["rebalance_date"], drawdown, 0, alpha=0.3)
    plt.title("策略回撤曲线")
    plt.xlabel("日期")
    plt.ylabel("回撤")
    plt.grid(True)
    plt.legend()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"回撤曲线已保存至: {save_path}")
    else:
        plt.show()


class BacktestEngine:
    """回测引擎"""

    def __init__(
        self,
        initial_capital: float = 10000000,
        rebalance_frequency: str = "2W",
        commission_rate: float = 0.00015,
        stamp_tax: float = 0.001,
        slippage_bps: int = 10
    ):
        """
        初始化回测引擎

        Parameters:
        -----------
        initial_capital : float
            初始资金
        rebalance_frequency : str
            调仓频率: "1W", "2W", "1M"
        commission_rate : float
            佣金率
        stamp_tax : float
            印花税
        slippage_bps : int
            滑点（基点）
        """
        self.initial_capital = initial_capital
        self.rebalance_frequency = rebalance_frequency
        self.commission_rate = commission_rate
        self.stamp_tax = stamp_tax
        self.slippage_bps = slippage_bps / 10000

    def run(
        self,
        factor_data: pd.DataFrame,
        price_data: pd.DataFrame,
        n_long: int = 30,
        n_short: int = 30,
        weight_method: str = "equal"
    ) -> Dict[str, float]:
        """
        运行回测

        Parameters:
        -----------
        factor_data : pd.DataFrame
            因子数据
        price_data : pd.DataFrame
            价格数据
        n_long : int
            多头数量
        n_short : int
            空头数量
        weight_method : str
            权重方法

        Returns:
        --------
        Dict[str, float] : 回测指标
        """
        print("[回测] 计算策略收益...")
        returns = calculate_returns(
            factor_data, price_data, n_long, n_short, weight_method
        )

        print("[回测] 计算回测指标...")
        metrics = calculate_metrics(returns)

        # 保存结果
        results_dir = os.path.join(OUTPUT_DIR, "results")
        os.makedirs(results_dir, exist_ok=True)

        # 保存收益数据
        returns_path = os.path.join(results_dir, "returns.csv")
        returns.to_csv(returns_path, index=False, encoding="utf-8-sig")
        print(f"收益数据已保存至: {returns_path}")

        # 保存指标数据
        metrics_path = os.path.join(results_dir, "metrics.json")
        import json
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"回测指标已保存至: {metrics_path}")

        # 绘制权益曲线
        equity_path = os.path.join(results_dir, "equity_curve.png")
        plot_equity_curve(returns, equity_path)

        # 绘制回撤曲线
        drawdown_path = os.path.join(results_dir, "drawdown_curve.png")
        plot_drawdown(returns, drawdown_path)

        return metrics


class BacktestRunner:
    """回测运行器"""

    def __init__(self, config: Dict):
        """
        初始化回测运行器

        Parameters:
        -----------
        config : Dict
            回测配置
        """
        self.config = config
        self.engine = BacktestEngine(
            initial_capital=config.get("initial_capital", 10000000),
            rebalance_frequency=config.get("rebalance_frequency", "2W"),
            commission_rate=config.get("commission_rate", 0.00015),
            stamp_tax=config.get("stamp_tax", 0.001),
            slippage_bps=config.get("slippage_bps", 10)
        )

    def run(
        self,
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        n_long: int = 30,
        n_short: int = 30,
        weight_method: str = "equal"
    ) -> Dict[str, float]:
        """
        运行回测

        Parameters:
        -----------
        factor_df : pd.DataFrame
            因子数据
        price_df : pd.DataFrame
            价格数据
        n_long : int
            多头数量
        n_short : int
            空头数量
        weight_method : str
            权重方法

        Returns:
        --------
        Dict[str, float] : 回测指标
        """
        print(f"回测配置: {self.config}")
        print(f"多头数量: {n_long}, 空头数量: {n_short}")

        # 准备数据
        factor_df = factor_df.copy()
        price_df = price_df.copy()

        # 运行回测
        metrics = self.engine.run(
            factor_df, price_df, n_long, n_short, weight_method
        )

        return metrics


if __name__ == "__main__":
    print("回测模块测试")
    
    # 生成示例数据
    dates = pd.date_range("2020-01-01", "2024-12-31", freq="2W")
    stock_codes = ["00" + str(i).zfill(4) + ".SZ" for i in range(1, 101)]
    
    # 生成因子数据
    factor_data = []
    for date in dates:
        for code in stock_codes:
            factor_data.append({
                "ts_code": code,
                "date": date,
                "IRSF_score": np.random.normal(0, 1)
            })
    factor_df = pd.DataFrame(factor_data)
    
    # 生成价格数据
    price_data = []
    for date in dates:
        for code in stock_codes:
            price_data.append({
                "ts_code": code,
                "trade_date": date,
                "close": np.random.uniform(10, 50)
            })
    price_df = pd.DataFrame(price_data)
    
    # 运行回测
    runner = BacktestRunner({
        "initial_capital": 10000000,
        "rebalance_frequency": "2W",
        "commission_rate": 0.00015,
        "stamp_tax": 0.001,
        "slippage_bps": 10
    })
    
    metrics = runner.run(factor_df, price_df, n_long=30, n_short=30)
    print("回测结果:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
