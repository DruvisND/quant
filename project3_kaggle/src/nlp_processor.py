"""
NLP情绪分析模块（Kaggle适配版）
使用FinBERT对机构调研文本进行情绪打分
"""
from __future__ import annotations
import os
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# Kaggle平台的输出目录
KAGGLE_OUTPUT_DIR = "../output"
LOCAL_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

# 自动检测运行环境
IS_KAGGLE = os.path.exists("../input")

if IS_KAGGLE:
    OUTPUT_DIR = KAGGLE_OUTPUT_DIR
else:
    OUTPUT_DIR = LOCAL_OUTPUT_DIR

MODEL_CACHE_DIR = os.path.join(OUTPUT_DIR, "models")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

DEFAULT_MODEL = "yiyanghkust/finbert-tone"

SENTIMENT_LABELS = ["negative", "neutral", "positive"]


def load_finbert(model_name: str = DEFAULT_MODEL) -> Tuple:
    """加载FinBERT模型和分词器"""
    try:
        from transformers import BertTokenizer, BertForSequenceClassification
    except ImportError:
        raise ImportError("请安装transformers: pip install transformers torch")

    tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=MODEL_CACHE_DIR)
    model = BertForSequenceClassification.from_pretrained(model_name, cache_dir=MODEL_CACHE_DIR)

    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    return tokenizer, model


def get_sentiment_score(text: str, tokenizer, model) -> Dict[str, float]:
    """对单条文本进行情绪打分

    Parameters:
    -----------
    text : str
        待分析文本
    tokenizer : BertTokenizer
    model : BertForSequenceClassification

    Returns:
    --------
    Dict[str, float] : 包含 prob_negative, prob_neutral, prob_positive, sentiment, confidence
    """
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        return {
            "prob_negative": 0.0,
            "prob_neutral": 1.0,
            "prob_positive": 0.0,
            "sentiment": "neutral",
            "confidence": 0.0
        }

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()

    prob_neg, prob_neu, prob_pos = probs
    sentiment = SENTIMENT_LABELS[np.argmax(probs)]
    confidence = float(np.max(probs))

    return {
        "prob_negative": float(prob_neg),
        "prob_neutral": float(prob_neu),
        "prob_positive": float(prob_pos),
        "sentiment": sentiment,
        "confidence": confidence
    }


def batch_sentiment_process(
    texts: List[str],
    tokenizer,
    model,
    batch_size: int = 32,
    show_progress: bool = True
) -> List[Dict[str, float]]:
    """批量处理文本情绪分析"""
    results = []

    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="情绪分析")

    for i in iterator:
        batch_texts = texts[i:i+batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()

        for prob in probs:
            prob_neg, prob_neu, prob_pos = prob
            sentiment = SENTIMENT_LABELS[np.argmax(prob)]
            confidence = float(np.max(prob))

            results.append({
                "prob_negative": float(prob_neg),
                "prob_neutral": float(prob_neu),
                "prob_positive": float(prob_pos),
                "sentiment": sentiment,
                "confidence": confidence
            })

    return results


def process_research_records(
    df: pd.DataFrame,
    tokenizer,
    model,
    q_col: str = "q_content",
    a_col: str = "a_content",
    batch_size: int = 32,
    confidence_threshold: float = 0.5
) -> pd.DataFrame:
    """处理调研记录，添加情绪得分

    Parameters:
    -----------
    df : pd.DataFrame
        调研记录数据
    tokenizer, model
    q_col : str
        问题列名
    a_col : str
        回答列名
    batch_size : int
        批处理大小
    confidence_threshold : float
        置信度阈值，低于该值的结果将被标记

    Returns:
    --------
    pd.DataFrame : 添加了情绪得分的调研记录
    """
    df = df.copy()

    df["text_for_analysis"] = df[q_col].fillna("") + " " + df[a_col].fillna("")
    df["text_for_analysis"] = df["text_for_analysis"].str.strip()

    texts = df["text_for_analysis"].tolist()
    sentiment_results = batch_sentiment_process(texts, tokenizer, model, batch_size)

    df["sentiment_prob_neg"] = [r["prob_negative"] for r in sentiment_results]
    df["sentiment_prob_neu"] = [r["prob_neutral"] for r in sentiment_results]
    df["sentiment_prob_pos"] = [r["prob_positive"] for r in sentiment_results]
    df["sentiment"] = [r["sentiment"] for r in sentiment_results]
    df["sentiment_confidence"] = [r["confidence"] for r in sentiment_results]

    df["sentiment_score"] = (
        df["sentiment_prob_pos"] - df["sentiment_prob_neg"]
    )

    df["low_confidence"] = df["sentiment_confidence"] < confidence_threshold

    df = df.drop(columns=["text_for_analysis"], errors="ignore")

    return df


def calculate_rolling_sentiment(
    df: pd.DataFrame,
    stock_col: str = "stock_code",
    date_col: str = "调研日期",
    sentiment_col: str = "sentiment_score",
    n: int = 5
) -> pd.DataFrame:
    """计算每只股票最近n次调研的滚动情绪均值

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
    df = df.sort_values([stock_col, date_col])

    df[f"rolling_sentiment_{n}"] = df.groupby(stock_col)[sentiment_col].transform(
        lambda x: x.rolling(window=n, min_periods=1).mean()
    )

    return df


def aggregate_daily_sentiment(
    df: pd.DataFrame,
    stock_col: str = "stock_code",
    date_col: str = "调研日期",
    sentiment_col: str = "sentiment_score",
    agg_method: str = "mean"
) -> pd.DataFrame:
    """按股票和日期聚合情绪得分

    Parameters:
    -----------
    df : pd.DataFrame
    stock_col : str
    date_col : str
    sentiment_col : str
    agg_method : str
        聚合方法: "mean", "median", "max"

    Returns:
    --------
    pd.DataFrame : 聚合后的日度情绪数据
    """
    if agg_method == "mean":
        agg_func = "mean"
    elif agg_method == "median":
        agg_func = "median"
    elif agg_method == "max":
        agg_func = "max"
    else:
        agg_func = "mean"

    daily_sentiment = df.groupby([stock_col, date_col]).agg({
        sentiment_col: agg_func,
        "sentiment_confidence": "mean",
        "sentiment": lambda x: x.value_counts().index[0] if len(x) > 0 else "neutral"
    }).reset_index()

    return daily_sentiment


class SentimentAnalyzer:
    """情绪分析器封装类"""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None
    ):
        """初始化分析器

        Parameters:
        -----------
        model_name : str
            模型名称
        device : str, optional
            设备，"cuda"或"cpu"
        """
        self.model_name = model_name

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.tokenizer, self.model = load_finbert(model_name)

        if device == "cuda":
            self.model = self.model.cuda()

    def analyze(self, text: str) -> Dict[str, float]:
        """分析单条文本"""
        return get_sentiment_score(text, self.tokenizer, self.model)

    def analyze_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[Dict[str, float]]:
        """批量分析文本"""
        return batch_sentiment_process(
            texts, self.tokenizer, self.model,
            batch_size=batch_size,
            show_progress=show_progress
        )

    def process_dataframe(
        self,
        df: pd.DataFrame,
        q_col: str = "q_content",
        a_col: str = "a_content",
        batch_size: int = 32,
        confidence_threshold: float = 0.5
    ) -> pd.DataFrame:
        """处理调研记录DataFrame"""
        return process_research_records(
            df, self.tokenizer, self.model,
            q_col=q_col, a_col=a_col,
            batch_size=batch_size,
            confidence_threshold=confidence_threshold
        )


if __name__ == "__main__":
    print("NLP情绪分析模块测试")
    analyzer = SentimentAnalyzer()

    test_texts = [
        "公司未来发展前景良好，业绩预计持续增长",
        "公司面临较大经营压力，业绩可能出现下滑",
        "公司经营正常，无重大变化"
    ]

    results = analyzer.analyze_batch(test_texts)
    for text, result in zip(test_texts, results):
        print(f"文本: {text[:20]}...")
        print(f"  情绪: {result['sentiment']}, 置信度: {result['confidence']:.3f}")
        print(f"  得分: {result['prob_positive']:.3f} (正) / {result['prob_neutral']:.3f} (中) / {result['prob_negative']:.3f} (负)")
        print()
