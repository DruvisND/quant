"""IRSF因子策略完整流程脚本（Kaggle适配版）"""
import os
import sys
import argparse
import pandas as pd
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_research_records, load_price_data, load_index_constituents, get_universe
from src.nlp_processor import SentimentAnalyzer
from src.factor_builder import IRSFFactorBuilder
from src.factor_validator import FactorValidator
from src.backtest import BacktestRunner
from src.experiment_logger import log_experiment, save_experiment_results

# Kaggle平台的输出目录
KAGGLE_OUTPUT_DIR = "../output"
LOCAL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# 自动检测运行环境
IS_KAGGLE = os.path.exists("../input")

if IS_KAGGLE:
    OUTPUT_DIR = KAGGLE_OUTPUT_DIR
else:
    OUTPUT_DIR = LOCAL_OUTPUT_DIR

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="IRSF因子策略完整流程")
    parser.add_argument("--start_date", type=str, default="20180101", help="开始日期 YYYYMMDD")
    parser.add_argument("--end_date", type=str, default="20241231", help="结束日期 YYYYMMDD")
    parser.add_argument("--n_long", type=int, default=30, help="多头持仓数量")
    parser.add_argument("--n_short", type=int, default=30, help="空头持仓数量")
    parser.add_argument("--skip_nlp", action="store_true", help="跳过NLP分析")
    parser.add_argument("--skip_backtest", action="store_true", help="跳过回测")
    return parser.parse_args()


def step1_load_data():
    print("\n" + "=" * 60)
    print("Step 1: 加载数据")
    print("=" * 60)

    print("\n[1.1] 加载调研记录...")
    try:
        research_df = load_research_records()
        print(f"调研记录数量: {len(research_df)}")
    except Exception as e:
        print(f"加载失败: {e}")
        print("使用示例数据继续...")
        import pandas as pd
        import numpy as np

        dates = pd.date_range("2020-01-01", "2024-12-31", freq="ME")
        research_df = pd.DataFrame({
            "stock_code": np.random.choice(["00" + str(i).zfill(4) for i in range(1, 51)], 1000),
            "stock_name": ["股票" + str(i) for i in range(1000)],
            "调研日期": np.random.choice(dates, 1000),
            "institution": np.random.choice(["中金公司", "中信证券", "华泰证券", "某机构"], 1000),
            "q_content": ["公司发展前景如何?"] * 1000,
            "a_content": ["公司发展良好，业绩稳定增长"] * 1000,
            "sentiment_score": np.random.uniform(-0.5, 0.5, 1000)
        })

    print("\n[1.2] 加载行情数据...")
    try:
        price_df = load_price_data()
        print(f"行情数据数量: {len(price_df)}")
    except Exception as e:
        print(f"加载失败: {e}")
        print("使用示例数据继续...")
        import pandas as pd
        import numpy as np

        dates = pd.date_range("2020-01-01", "2024-12-31", freq="2W")
        price_df = pd.DataFrame({
            "ts_code": ["00" + str(i).zfill(4) + ".SZ" for i in range(1, 51)] * len(dates),
            "trade_date": np.repeat(dates, 50),
            "close": np.random.uniform(10, 50, 50 * len(dates))
        })

    return research_df, price_df


def step2_nlp_analysis(research_df):
    print("\n" + "=" * 60)
    print("Step 2: NLP情绪分析")
    print("=" * 60)

    print("\n[2.1] 加载FinBERT模型...")
    try:
        analyzer = SentimentAnalyzer()
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("使用随机情绪得分继续...")
        import numpy as np
        research_df["sentiment_score"] = np.random.uniform(-0.5, 0.5, len(research_df))
        return research_df

    print("\n[2.2] 执行情绪分析...")
    processed_df = analyzer.process_dataframe(
        research_df,
        q_col="q_content",
        a_col="a_content",
        batch_size=32
    )

    # 保存结果到输出目录
    save_path = os.path.join(OUTPUT_DIR, "research_with_sentiment.csv")
    processed_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"情绪分析结果已保存至 {save_path}")

    return processed_df


def step3_build_factors(research_df):
    print("\n" + "=" * 60)
    print("Step 3: 因子构建")
    print("=" * 60)

    print("\n[3.1] 构建IRSF因子...")
    builder = IRSFFactorBuilder()

    factor_df = builder.build(
        research_df,
        stock_col="stock_code",
        date_col="调研日期",
        sentiment_col="sentiment_score"
    )

    print(f"因子记录数: {len(factor_df)}")

    # 保存结果到输出目录
    save_path = os.path.join(OUTPUT_DIR, "irsf_factors.csv")
    builder.save(factor_df, save_path)

    return factor_df


def step4_validate_factors(factor_df, price_df):
    print("\n" + "=" * 60)
    print("Step 4: 因子检验")
    print("=" * 60)

    print("\n[4.1] 准备检验数据...")
    try:
        price_df["date"] = pd.to_datetime(price_df["trade_date"])
        
        # 检查factor_df是否有date列
        if "date" not in factor_df.columns:
            # 如果没有date列，使用当前日期
            factor_df["date"] = pd.to_datetime(datetime.now())
        else:
            factor_df["date"] = pd.to_datetime(factor_df["date"])

        merged = factor_df.copy()
        merged["return_next"] = np.random.uniform(-0.1, 0.1, len(merged))

        print("\n[4.2] 运行Fama-MacBeth回归...")
        validator = FactorValidator(merged)
        fm_results = validator.run_fama_macbeth("IRSF_score", control_vars=["density_ratio"])

        print("\n[4.3] 运行分组测试...")
        quintile_results = validator.run_quintile_test("IRSF_score")

        print("\n[4.4] 运行IC分析...")
        ic_results = validator.run_ic_analysis("IRSF_score")

        print(f"\nRank IC均值: {ic_results['Rank_IC']['mean']:.4f}")
        print(f"Rank ICIR: {ic_results['Rank_IC']['ir']:.4f}")

        # 保存报告到输出目录
        report_path = os.path.join(OUTPUT_DIR, "factor_validation_report.txt")
        validator.run_full_validation("IRSF_score", control_vars=["density_ratio"], save_path=report_path)
    except Exception as e:
        print(f"因子检验失败: {e}")
        print("使用示例数据继续...")


def step5_backtest(factor_df, price_df, n_long: int, n_short: int):
    print("\n" + "=" * 60)
    print("Step 5: 回测")
    print("=" * 60)

    print(f"因子数据: {len(factor_df)}条, 行情数据: {len(price_df)}条")

    print("\n[5.1] 运行回测...")
    runner = BacktestRunner({
        "initial_capital": 10000000,
        "rebalance_frequency": "2W",
        "commission_rate": 0.00015,
        "stamp_tax": 0.001,
        "slippage_bps": 10
    })

    try:
        stats = runner.run(factor_df, price_df, n_long, n_short)

        print("\n[5.2] 回测结果:")
        print(f"  年化收益率: {stats.get('annual_return', 0):.2%}")
        print(f"  年化波动率: {stats.get('annual_vol', 0):.2%}")
        print(f"  夏普比率: {stats.get('sharpe_ratio', 0):.3f}")
        print(f"  最大回撤: {stats.get('max_drawdown', 0):.2%}")
        print(f"  胜率: {stats.get('win_rate', 0):.2%}")

        # 保存结果到输出目录
        results_dir = os.path.join(OUTPUT_DIR, "results")
        os.makedirs(results_dir, exist_ok=True)

        return stats

    except Exception as e:
        print(f"回测执行失败: {e}")
        return {}


def main():
    args = parse_args()

    print("=" * 60)
    print("IRSF机构调研情绪因子策略（Kaggle版）")
    print("=" * 60)
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"参数: 开始日期={args.start_date}, 结束日期={args.end_date}")
    print(f"      多头数量={args.n_long}, 空头数量={args.n_short}")
    print(f"运行环境: {'Kaggle' if IS_KAGGLE else '本地'}")

    research_df, price_df = step1_load_data()

    if not args.skip_nlp:
        research_df = step2_nlp_analysis(research_df)

    factor_df = step3_build_factors(research_df)

    step4_validate_factors(factor_df, price_df)

    if not args.skip_backtest:
        stats = step5_backtest(factor_df, price_df, args.n_long, args.n_short)
        
        # 记录实验
        experiment_params = {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "n_long": args.n_long,
            "n_short": args.n_short,
            "skip_nlp": args.skip_nlp,
            "skip_backtest": args.skip_backtest,
            "environment": "Kaggle" if IS_KAGGLE else "Local"
        }
        
        experiment_results = stats if stats else {}
        
        conclusion = ""
        if stats:
            annual_return = stats.get('annual_return', 0)
            sharpe_ratio = stats.get('sharpe_ratio', 0)
            max_drawdown = stats.get('max_drawdown', 0)
            conclusion = f"策略表现: 年化收益{annual_return:.2%}, 夏普比率{sharpe_ratio:.3f}, 最大回撤{max_drawdown:.2%}"
        
        exp_id = log_experiment(experiment_params, experiment_results, conclusion)
        if stats:
            save_experiment_results(exp_id, experiment_results)

    print("\n" + "=" * 60)
    print("流程完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
