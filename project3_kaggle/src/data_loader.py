"""
数据加载模块（Kaggle适配版）
加载各类金融数据用于IRSF因子策略
"""
from __future__ import annotations

import os
import sqlite3
from typing import Dict, Optional
import pandas as pd
import numpy as np

# Kaggle平台的数据路径
KAGGLE_INPUT_DIR = "../input"
KAGGLE_OUTPUT_DIR = "../output"

# 本地开发时的路径
LOCAL_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# 自动检测运行环境
IS_KAGGLE = os.path.exists(KAGGLE_INPUT_DIR)

if IS_KAGGLE:
    DATA_DIR = KAGGLE_INPUT_DIR
    OUTPUT_DIR = KAGGLE_OUTPUT_DIR
else:
    DATA_DIR = LOCAL_DATA_DIR
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_research_records(
    stock_code: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    data_file: Optional[str] = None
) -> pd.DataFrame:
    """
    加载机构调研记录数据

    Parameters:
    -----------
    stock_code : str, optional
        股票代码筛选，如 "000001" 或 "000001.SZ"
    start_date : str, optional
        起始日期，格式 "YYYYMMDD" 或 "YYYY-MM-DD"
    end_date : str, optional
        结束日期，格式 "YYYYMMDD" 或 "YYYY-MM-DD"
    data_file : str, optional
        数据文件路径，默认使用默认路径

    Returns:
    --------
    pd.DataFrame
        包含列: stock_code, stock_name,调研日期, institution, q_content, a_content
    """
    if data_file is None:
        # Kaggle上使用输入文件
        if IS_KAGGLE:
            data_file = os.path.join(DATA_DIR, "research_records.csv")
        else:
            data_file = os.path.join(DATA_DIR, "research_records.csv")

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"未找到调研数据: {data_file}")

    df = pd.read_csv(data_file, parse_dates=["调研日期"])

    if stock_code:
        df = df[df["stock_code"] == stock_code]
    if start_date:
        df = df[df["调研日期"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["调研日期"] <= pd.to_datetime(end_date)]

    return df


def load_price_data(
    stock_codes: Optional[list] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    data_file: Optional[str] = None
) -> pd.DataFrame:
    """
    加载日行情数据

    Parameters:
    -----------
    stock_codes : list, optional
        股票代码列表，如 ["000001.SZ", "600000.SH"]
    start_date : str, optional
        起始日期，格式 "YYYYMMDD"
    end_date : str, optional
        结束日期，格式 "YYYYMMDD"
    data_file : str, optional
        数据文件路径，默认使用默认路径

    Returns:
    --------
    pd.DataFrame
        包含列: ts_code, trade_date, open, high, low, close, vol, amount, adj_factor
    """
    if data_file is None:
        if IS_KAGGLE:
            data_file = os.path.join(DATA_DIR, "daily_price.csv")
        else:
            data_file = os.path.join(DATA_DIR, "daily_price.csv")

    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"未找到行情数据: {data_file}\n请确保数据已上传到Kaggle输入目录"
        )

    df = pd.read_csv(data_file, parse_dates=["trade_date"])

    if stock_codes:
        df = df[df["ts_code"].isin(stock_codes)]
    if start_date:
        df = df[df["trade_date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["trade_date"] <= pd.to_datetime(end_date)]

    return df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)


def load_index_constituents(
    index_code: str = "000300.SH",
    data_file: Optional[str] = None
) -> pd.DataFrame:
    """
    加载指数成分股

    Parameters:
    -----------
    index_code : str
        指数代码，默认沪深300 "000300.SH"，中证500为 "000905.SH"
    data_file : str, optional
        数据文件路径，默认使用默认路径

    Returns:
    --------
    pd.DataFrame
        包含列: ts_code, weight, trade_date
    """
    if data_file is None:
        if IS_KAGGLE:
            data_file = os.path.join(DATA_DIR, f"index_{index_code.replace('.', '_')}_constituents.csv")
        else:
            data_file = os.path.join(DATA_DIR, f"index_{index_code.replace('.', '_')}_constituents.csv")

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"未找到成分股数据: {data_file}")

    df = pd.read_csv(data_file, parse_dates=["trade_date"])
    return df


def load_financial_fundamentals(
    stock_codes: Optional[list] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    data_file: Optional[str] = None
) -> pd.DataFrame:
    """
    加载财务基本面数据

    Parameters:
    -----------
    stock_codes : list, optional
        股票代码列表
    start_date : str, optional
        起始日期
    end_date : str, optional
        结束日期
    data_file : str, optional
        数据文件路径，默认使用默认路径

    Returns:
    --------
    pd.DataFrame
        包含列: ts_code, report_date, mcap, pe, pb, bm, roe, roa
    """
    if data_file is None:
        if IS_KAGGLE:
            data_file = os.path.join(DATA_DIR, "financial_fundamentals.csv")
        else:
            data_file = os.path.join(DATA_DIR, "financial_fundamentals.csv")

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"未找到财务数据: {data_file}")

    df = pd.read_csv(data_file, parse_dates=["report_date"])

    if stock_codes:
        df = df[df["ts_code"].isin(stock_codes)]
    if start_date:
        df = df[df["report_date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["report_date"] <= pd.to_datetime(end_date)]

    return df


def get_universe(
    include_hs300: bool = True,
    include_zz500: bool = True
) -> list:
    """
    获取选股池（沪深300 + 中证500）

    Parameters:
    -----------
    include_hs300 : bool
        是否包含沪深300
    include_zz500 : bool
        是否包含中证500

    Returns:
    --------
    list : 股票代码列表
    """
    stocks = set()

    if include_hs300:
        try:
            hs300 = load_index_constituents("000300.SH")
            stocks.update(hs300["ts_code"].tolist())
        except Exception as e:
            print(f"加载沪深300成分股失败: {e}")

    if include_zz500:
        try:
            zz500 = load_index_constituents("000905.SH")
            stocks.update(zz500["ts_code"].tolist())
        except Exception as e:
            print(f"加载中证500成分股失败: {e}")

    return sorted(list(stocks))


def merge_factor_data(
    factor_df: pd.DataFrame,
    price_df: pd.DataFrame,
    control_factors: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    合并因子数据与收益数据、控制变量

    Parameters:
    -----------
    factor_df : pd.DataFrame
        因子数据，包含 ts_code, date, factor columns
    price_df : pd.DataFrame
        行情数据，包含 ts_code, trade_date, close
    control_factors : dict, optional
        控制变量列名映射，如 {"size": "mcap", "bm": "bm"}

    Returns:
    --------
    pd.DataFrame
        合并后的数据，包含因子、收益、控制变量
    """
    price_df = price_df.copy()
    price_df["date"] = price_df["trade_date"]

    price_df["return"] = price_df.groupby("ts_code")["close"].pct_change()

    next_month = price_df.copy()
    next_month["date"] = next_month["date"] + pd.DateOffset(months=1)
    next_month["return_next"] = next_month.groupby("ts_code")["return"].shift(-1)

    merged = pd.merge(factor_df, next_month, on=["ts_code", "date"], how="inner")

    if control_factors:
        try:
            fund_df = load_financial_fundamentals()
            fund_df["date"] = fund_df["report_date"]
            merged = pd.merge(merged, fund_df, on=["ts_code", "date"], how="left")
        except Exception as e:
            print(f"加载财务数据失败: {e}")

    return merged.dropna()
