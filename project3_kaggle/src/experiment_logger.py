"""实验记录模块（Kaggle适配版）
记录所有实验参数和结果
"""
import os
import time
from datetime import datetime
import json
from typing import Dict, Optional

# Kaggle平台的输出目录
KAGGLE_OUTPUT_DIR = "../output"
LOCAL_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

# 自动检测运行环境
IS_KAGGLE = os.path.exists("../input")

if IS_KAGGLE:
    OUTPUT_DIR = KAGGLE_OUTPUT_DIR
else:
    OUTPUT_DIR = LOCAL_OUTPUT_DIR

EXPERIMENTS_DIR = os.path.join(OUTPUT_DIR, "experiments")
EXPERIMENT_LOG_FILE = os.path.join(EXPERIMENTS_DIR, "exp_log.md")

os.makedirs(EXPERIMENTS_DIR, exist_ok=True)


def init_experiment_log():
    """初始化实验日志文件"""
    if not os.path.exists(EXPERIMENT_LOG_FILE):
        with open(EXPERIMENT_LOG_FILE, "w", encoding="utf-8") as f:
            f.write("# 实验记录日志\n\n")
            f.write("## 目录\n\n")
            f.write("- [实验记录格式](#实验记录格式)\n")
            f.write("- [实验列表](#实验列表)\n\n")
            f.write("## 实验记录格式\n\n")
            f.write("- **实验ID**: 自动生成的唯一标识\n")
            f.write("- **实验时间**: 实验开始时间\n")
            f.write("- **实验参数**: 包括因子参数、回测参数等\n")
            f.write("- **实验结果**: 包括年化收益、夏普比率、最大回撤等\n")
            f.write("- **实验结论**: 对实验结果的简要分析\n\n")
            f.write("## 实验列表\n\n")


def generate_experiment_id():
    """生成实验ID"""
    return f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def log_experiment(
    params: Dict,
    results: Dict,
    conclusion: str = ""
) -> str:
    """记录实验
    
    Parameters:
    -----------
    params : Dict
        实验参数
    results : Dict
        实验结果
    conclusion : str
        实验结论
    
    Returns:
    --------
    str : 实验ID
    """
    # 初始化日志文件
    init_experiment_log()
    
    # 生成实验ID
    experiment_id = generate_experiment_id()
    
    # 实验时间
    experiment_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 构建实验记录
    log_entry = []
    log_entry.append(f"### 实验 {experiment_id}\n")
    log_entry.append(f"**实验时间**: {experiment_time}\n\n")
    
    # 记录参数
    log_entry.append("**实验参数**:\n")
    log_entry.append("```json\n")
    log_entry.append(json.dumps(params, ensure_ascii=False, indent=2))
    log_entry.append("\n```\n\n")
    
    # 记录结果
    log_entry.append("**实验结果**:\n")
    log_entry.append("```json\n")
    log_entry.append(json.dumps(results, ensure_ascii=False, indent=2))
    log_entry.append("\n```\n\n")
    
    # 记录结论
    if conclusion:
        log_entry.append(f"**实验结论**: {conclusion}\n\n")
    else:
        log_entry.append("**实验结论**: 无\n\n")
    
    log_entry.append("---\n\n")
    
    # 写入日志文件
    with open(EXPERIMENT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write("".join(log_entry))
    
    print(f"实验记录已保存至: {EXPERIMENT_LOG_FILE}")
    print(f"实验ID: {experiment_id}")
    
    return experiment_id


def get_experiment_history() -> list:
    """获取实验历史记录"""
    if not os.path.exists(EXPERIMENT_LOG_FILE):
        return []
    
    experiments = []
    with open(EXPERIMENT_LOG_FILE, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 简单解析实验记录
    import re
    experiment_pattern = r"### 实验 (exp_\d{8}_\d{6})"
    experiments = re.findall(experiment_pattern, content)
    
    return experiments


def save_experiment_results(
    experiment_id: str,
    results: Dict,
    save_dir: Optional[str] = None
) -> str:
    """保存实验结果到单独的文件
    
    Parameters:
    -----------
    experiment_id : str
        实验ID
    results : Dict
        实验结果
    save_dir : str, optional
        保存目录，默认使用experiments目录
    
    Returns:
    --------
    str : 保存文件路径
    """
    if save_dir is None:
        save_dir = EXPERIMENTS_DIR
    
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, f"{experiment_id}_results.json")
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"实验结果已保存至: {save_path}")
    
    return save_path


if __name__ == "__main__":
    # 测试实验记录功能
    test_params = {
        "start_date": "20180101",
        "end_date": "20241231",
        "n_long": 30,
        "n_short": 30,
        "window_short": 20,
        "window_long": 120,
        "rolling_n": 5,
        "environment": "Kaggle" if IS_KAGGLE else "Local"
    }
    
    test_results = {
        "annual_return": 0.12,
        "sharpe_ratio": 0.95,
        "max_drawdown": -0.12,
        "win_rate": 0.55
    }
    
    test_conclusion = "策略表现良好，年化收益12%，夏普比率0.95，最大回撤12%"
    
    exp_id = log_experiment(test_params, test_results, test_conclusion)
    print(f"测试实验ID: {exp_id}")
    
    history = get_experiment_history()
    print(f"实验历史: {history}")
