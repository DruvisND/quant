"""
因子构建模块（Kaggle适配版）
构建IRSF机构调研情绪因子
"""
from __future__ import annotations
import os
from typing import Dict, Optional, List, Tuple, Union
import pandas as pd
import numpy as np

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


def calculate_research_density(
    df: pd.DataFrame,
    stock_col: str = "stock_code",
    date_col: str = "调研日期",
    window_days: int = 90
) -> pd.DataFrame:
    """
    计算调研密度

    Parameters:
    -----------
    df : pd.DataFrame
        调研记录数据
    stock_col : str
        股票代码列名
    date_col : str
        日期列名
    window_days : int
        计算窗口，默认90天

    Returns:
    --------
    pd.DataFrame
        包含调研密度的DataFrame
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # 确保stock_col列存在
    if stock_col not in df.columns:
        raise ValueError(f"DataFrame中不存在列: {stock_col}")
    
    # 计算每只股票在每个日期的调研次数（过去window_days天）
    results = []
    for stock in df[stock_col].unique():
        stock_df = df[df[stock_col] == stock].copy()
        stock_df = stock_df.sort_values(date_col)
        
        counts = []
        for i, date in enumerate(stock_df[date_col]):
            start_date = date - pd.Timedelta(days=window_days)
            count = len(stock_df[(stock_df[date_col] >= start_date) & (stock_df[date_col] <= date)])
            counts.append(count)
        
        stock_df["research_count"] = counts
        results.append(stock_df)
    
    # 合并结果
    result_df = pd.concat(results, ignore_index=True)

    return result_df


def calculate_density_ratio(
    df: pd.DataFrame,
    stock_col: str = "stock_code",
    date_col: str = "调研日期",
    count_col: str = "research_count",
    short_window: int = 30,
    long_window: int = 180
) -> pd.DataFrame:
    """
    计算调研密度异动（短期/长期）

    Parameters:
    -----------
    df : pd.DataFrame
        包含调研计数的DataFrame
    stock_col : str
        股票代码列名
    date_col : str
        日期列名
    count_col : str
        调研计数列名
    short_window : int
        短期窗口，默认30天
    long_window : int
        长期窗口，默认180天

    Returns:
    --------
    pd.DataFrame
        包含密度比率的DataFrame
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # 确保stock_col列存在
    if stock_col not in df.columns:
        raise ValueError(f"DataFrame中不存在列: {stock_col}")
    
    # 计算短期和长期平均调研次数
    results = []
    for stock in df[stock_col].unique():
        stock_df = df[df[stock_col] == stock].copy()
        stock_df = stock_df.sort_values(date_col)
        
        # 滚动平均，使用时间窗口
        stock_df["short_avg"] = stock_df[count_col].rolling(
            window=short_window, min_periods=1, center=False
        ).mean()
        stock_df["long_avg"] = stock_df[count_col].rolling(
            window=long_window, min_periods=1, center=False
        ).mean()
        
        results.append(stock_df)
    
    # 合并结果
    result_df = pd.concat(results, ignore_index=True)

    # 计算密度比率，避免除以零
    result_df["density_ratio"] = np.where(
        result_df["long_avg"] > 0,
        result_df["short_avg"] / result_df["long_avg"],
        0
    )

    return result_df


def calculate_institution_quality(
    df: pd.DataFrame,
    institution_col: str = "institution",
    stock_col: str = "stock_code",
    date_col: str = "调研日期"
) -> Dict[str, float]:
    """
    计算机构质量得分

    Parameters:
    -----------
    df : pd.DataFrame
        调研记录数据
    institution_col : str
        机构名称列名
    stock_col : str
        股票代码列名
    date_col : str
        日期列名

    Returns:
    --------
    Dict[str, float]
        机构质量得分字典
    """
    # 机构质量评分规则：
    # 1. 调研频率（覆盖股票数）
    # 2. 调研持续性（调研次数）
    # 3. 调研影响力（调研后股价表现，此处简化处理）

    # 计算每个机构的调研股票数
    institution_stocks = df.groupby(institution_col)[stock_col].nunique()
    
    # 计算每个机构的调研总次数
    institution_counts = df.groupby(institution_col).size()
    
    # 合并计算质量得分
    quality_df = pd.DataFrame({
        "stock_coverage": institution_stocks,
        "research_count": institution_counts
    })
    
    # 标准化得分
    quality_df["stock_coverage_norm"] = (
        (quality_df["stock_coverage"] - quality_df["stock_coverage"].min()) /
        (quality_df["stock_coverage"].max() - quality_df["stock_coverage"].min())
    )
    
    quality_df["research_count_norm"] = (
        (quality_df["research_count"] - quality_df["research_count"].min()) /
        (quality_df["research_count"].max() - quality_df["research_count"].min())
    )
    
    # 加权计算最终质量得分
    quality_df["quality_score"] = (
        0.6 * quality_df["stock_coverage_norm"] +
        0.4 * quality_df["research_count_norm"]
    )
    
    return quality_df["quality_score"].to_dict()


def calculate_quality_weighted_sentiment(
    df: pd.DataFrame,
    institution_col: str = "institution",
    sentiment_col: str = "sentiment_score",
    institution_quality: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    计算机构质量加权的情绪得分

    Parameters:
    -----------
    df : pd.DataFrame
        调研记录数据
    institution_col : str
        机构名称列名
    sentiment_col : str
        情绪得分列名
    institution_quality : Dict[str, float], optional
        机构质量得分字典

    Returns:
    --------
    pd.DataFrame
        包含质量加权情绪得分的DataFrame
    """
    df = df.copy()

    # 检查institution_col是否存在
    if institution_col not in df.columns:
        # 如果不存在，使用默认质量得分0.5
        df["institution_quality"] = 0.5
        df["quality_weighted_score"] = df[sentiment_col] * 0.5
        return df

    if institution_quality is None:
        institution_quality = calculate_institution_quality(df)

    # 映射机构质量得分
    df["institution_quality"] = df[institution_col].map(institution_quality).fillna(0.5)

    # 计算加权情绪得分
    df["quality_weighted_score"] = df[sentiment_col] * df["institution_quality"]

    return df


def calculate_rolling_sentiment(
    df: pd.DataFrame,
    stock_col: str = "stock_code",
    date_col: str = "调研日期",
    sentiment_col: str = "sentiment_score",
    n: int = 5
) -> pd.DataFrame:
    """
    计算每只股票最近n次调研的滚动情绪均值

    Parameters:
    -----------
    df : pd.DataFrame
        包含调研记录和情绪得分的DataFrame
    stock_col : str
        股票代码列名
    date_col : str
        日期列名
    sentiment_col : str
        情绪得分列名
    n : int
        滚动窗口大小，默认5次调研

    Returns:
    --------
    pd.DataFrame : 包含滚动情绪均值的DataFrame
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # 检查stock_col是否存在
    if stock_col not in df.columns:
        # 如果不存在，直接计算整个数据集的滚动情绪
        df = df.sort_values([date_col])
        df[f"rolling_sentiment_{n}"] = df[sentiment_col].rolling(
            window=n, min_periods=1
        ).mean()
        return df
    
    df = df.sort_values([stock_col, date_col])

    df[f"rolling_sentiment_{n}"] = df.groupby(stock_col)[sentiment_col].transform(
        lambda x: x.rolling(window=n, min_periods=1).mean()
    )

    return df


def composite_irsf(
    df: pd.DataFrame,
    stock_col: str = "stock_code",
    date_col: str = "调研日期",
    density_ratio_col: str = "density_ratio",
    sentiment_col: str = "rolling_sentiment",
    quality_col: str = "quality_weighted_score",
    weights: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    构建综合IRSF因子

    Parameters:
    -----------
    df : pd.DataFrame
        包含各因子的DataFrame
    stock_col : str
        股票代码列名
    date_col : str
        日期列名
    density_ratio_col : str
        密度比率列名
    sentiment_col : str
        滚动情绪得分列名
    quality_col : str
        质量加权得分列名
    weights : Dict[str, float], optional
        各因子权重

    Returns:
    --------
    pd.DataFrame
        包含综合因子的DataFrame
    """
    if weights is None:
        weights = {
            "density_ratio": 0.3,
            "sentiment": 0.4,
            "quality": 0.3
        }

    # 处理缺失值
    def fill_missing_by_history(df_temp):
        df_temp = df_temp.sort_values(date_col)
        # 滚动填充，使用历史数据的均值
        df_temp[sentiment_col] = df_temp[sentiment_col].fillna(df_temp[sentiment_col].expanding().mean())
        df_temp[quality_col] = df_temp[quality_col].fillna(df_temp[quality_col].expanding().mean())
        # 如果仍然有缺失值（例如第一个数据点），则填充为0
        df_temp[sentiment_col] = df_temp[sentiment_col].fillna(0)
        df_temp[quality_col] = df_temp[quality_col].fillna(0)
        return df_temp

    # 密度比率直接裁剪，不需要标准化
    df["density_ratio_norm"] = df[density_ratio_col].clip(0, 5)
    
    # 检查stock_col是否存在
    if stock_col in df.columns:
        # 按股票和日期排序
        df = df.sort_values([stock_col, date_col])
        
        # 按股票和时间滚动标准化情绪因子
        def rolling_standardize(group):
            # 按时间排序
            group = group.sort_values(date_col)
            
            # 计算滚动均值和标准差（使用历史数据）
            rolling_mean = group[sentiment_col].expanding().mean()
            rolling_std = group[sentiment_col].expanding().std()
            
            # 标准化
            mask = rolling_std > 0
            group.loc[mask, "rolling_sentiment_norm"] = (group.loc[mask, sentiment_col] - rolling_mean[mask]) / rolling_std[mask]
            group.loc[~mask, "rolling_sentiment_norm"] = 0
            
            # 同样处理质量因子
            rolling_mean_quality = group[quality_col].expanding().mean()
            rolling_std_quality = group[quality_col].expanding().std()
            
            mask_quality = rolling_std_quality > 0
            group.loc[mask_quality, "quality_norm"] = (group.loc[mask_quality, quality_col] - rolling_mean_quality[mask_quality]) / rolling_std_quality[mask_quality]
            group.loc[~mask_quality, "quality_norm"] = 0
            
            return group
        
        # 按股票分组进行滚动标准化
        df = df.groupby(stock_col).apply(rolling_standardize).reset_index(drop=True)
        
        # 确保stock_col列仍然存在
        if stock_col not in df.columns:
            # 如果stock_col列不存在，使用默认值
            df[stock_col] = "000001"
        
        # 按股票分组处理缺失值
        df = df.groupby(stock_col).apply(fill_missing_by_history).reset_index(drop=True)
    else:
        # 如果stock_col不存在，直接对整个数据集处理
        df = df.sort_values([date_col])
        
        # 计算滚动均值和标准差（使用历史数据）
        rolling_mean = df[sentiment_col].expanding().mean()
        rolling_std = df[sentiment_col].expanding().std()
        
        # 标准化
        mask = rolling_std > 0
        df.loc[mask, "rolling_sentiment_norm"] = (df.loc[mask, sentiment_col] - rolling_mean[mask]) / rolling_std[mask]
        df.loc[~mask, "rolling_sentiment_norm"] = 0
        
        # 同样处理质量因子
        rolling_mean_quality = df[quality_col].expanding().mean()
        rolling_std_quality = df[quality_col].expanding().std()
        
        mask_quality = rolling_std_quality > 0
        df.loc[mask_quality, "quality_norm"] = (df.loc[mask_quality, quality_col] - rolling_mean_quality[mask_quality]) / rolling_std_quality[mask_quality]
        df.loc[~mask_quality, "quality_norm"] = 0
        
        # 处理缺失值
        df = fill_missing_by_history(df)

    # 计算综合因子
    df["IRSF_score"] = (
        weights.get("density_ratio", 0.3) * df["density_ratio_norm"] +
        weights.get("sentiment", 0.4) * df["rolling_sentiment_norm"] +
        weights.get("quality", 0.3) * df["quality_norm"]
    )

    return df


def normalize_factor(
    df: pd.DataFrame,
    factor_col: str = "IRSF_score",
    stock_col: str = "stock_code",
    date_col: str = "date"
) -> pd.DataFrame:
    """
    对因子进行截面标准化

    Parameters:
    -----------
    df : pd.DataFrame
        因子数据
    factor_col : str
        因子列名
    stock_col : str
        股票代码列名
    date_col : str
        日期列名

    Returns:
    --------
    pd.DataFrame
        标准化后的因子数据
    """
    df = df.copy()

    # 按日期进行截面标准化
    def normalize_cross_section(group):
        mean = group[factor_col].mean()
        std = group[factor_col].std()
        if std > 0:
            group[f"{factor_col}_norm"] = (group[factor_col] - mean) / std
        else:
            group[f"{factor_col}_norm"] = 0
        return group

    df = df.groupby(date_col).apply(normalize_cross_section).reset_index(drop=True)

    return df


class IRSFFactorBuilder:
    """IRSF因子构建器"""

    def __init__(
        self,
        window_days: int = 90,
        short_window: int = 30,
        long_window: int = 180,
        rolling_n: int = 5
    ):
        """
        初始化因子构建器

        Parameters:
        -----------
        window_days : int
            调研密度计算窗口
        short_window : int
            短期平均窗口
        long_window : int
            长期平均窗口
        rolling_n : int
            滚动情绪计算窗口
        """
        self.window_days = window_days
        self.short_window = short_window
        self.long_window = long_window
        self.rolling_n = rolling_n

    def build(
        self,
        research_df: pd.DataFrame,
        stock_col: str = "stock_code",
        date_col: str = "调研日期",
        sentiment_col: str = "sentiment_score",
        institution_col: str = "institution"
    ) -> pd.DataFrame:
        """
        构建IRSF因子

        Parameters:
        -----------
        research_df : pd.DataFrame
            调研记录数据
        stock_col : str
            股票代码列名
        date_col : str
            日期列名
        sentiment_col : str
            情绪得分列名
        institution_col : str
            机构名称列名

        Returns:
        --------
        pd.DataFrame
            因子数据
        """
        print("[因子构建] 计算调研密度...")
        df = calculate_research_density(
            research_df, stock_col, date_col, self.window_days
        )

        print("[因子构建] 计算密度比率...")
        df = calculate_density_ratio(
            df, stock_col, date_col, "research_count",
            self.short_window, self.long_window
        )

        print("[因子构建] 计算机构质量...")
        institution_quality = calculate_institution_quality(
            research_df, institution_col, stock_col, date_col
        )

        print("[因子构建] 计算质量加权情绪...")
        df = calculate_quality_weighted_sentiment(
            df, institution_col, sentiment_col, institution_quality
        )

        print("[因子构建] 计算滚动情绪...")
        df = calculate_rolling_sentiment(
            df, stock_col, date_col, sentiment_col, self.rolling_n
        )

        print("[因子构建] 构建综合因子...")
        df = composite_irsf(
            df, stock_col, date_col,
            "density_ratio", f"rolling_sentiment_{self.rolling_n}",
            "quality_weighted_score"
        )

        print("[因子构建] 标准化因子...")
        df = normalize_factor(
            df, "IRSF_score", stock_col, date_col
        )

        # 重命名列以保持一致性
        rename_dict = {}
        if stock_col in df.columns:
            rename_dict[stock_col] = "ts_code"
        if date_col in df.columns:
            rename_dict[date_col] = "date"
        
        df = df.rename(columns=rename_dict)

        # 确保ts_code列存在
        if "ts_code" not in df.columns:
            df["ts_code"] = "000001.SZ"
        else:
            # 确保ts_code格式正确（添加.SZ/.SH后缀）
            def format_ts_code(code):
                code_str = str(code)
                if len(code_str) == 6:
                    if code_str.startswith(('0', '3')):
                        return code_str + ".SZ"
                    elif code_str.startswith('6'):
                        return code_str + ".SH"
                return code_str

            df["ts_code"] = df["ts_code"].apply(format_ts_code)

        return df

    def save(
        self,
        factor_df: pd.DataFrame,
        output_path: str
    ) -> None:
        """
        保存因子数据

        Parameters:
        -----------
        factor_df : pd.DataFrame
            因子数据
        output_path : str
            输出路径
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        factor_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"因子数据已保存至: {output_path}")


if __name__ == "__main__":
    print("因子构建模块测试")
    
    # 生成示例数据
    dates = pd.date_range("2020-01-01", "2024-12-31", freq="M")
    stock_codes = ["00" + str(i).zfill(4) for i in range(1, 11)]
    
    data = []
    for code in stock_codes:
        for date in dates:
            data.append({
                "stock_code": code,
                "调研日期": date,
                "institution": f"机构{code[-2:]}",
                "sentiment_score": np.random.uniform(-1, 1)
            })
    
    df = pd.DataFrame(data)
    
    builder = IRSFFactorBuilder()
    factor_df = builder.build(df)
    
    print(f"因子数据形状: {factor_df.shape}")
    print(f"因子数据列: {list(factor_df.columns)}")
    print(f"前5行数据:\n{factor_df.head()}")
