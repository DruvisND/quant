# IRSF机构调研情绪因子策略（Kaggle版）

本项目是IRSF机构调研情绪因子策略的Kaggle适配版本，专为在Kaggle平台上运行而设计。

## 项目结构

```
project3_kaggle/
├── src/                # 核心代码模块
│   ├── data_loader.py      # 数据加载模块（Kaggle适配版）
│   ├── nlp_processor.py    # NLP情绪分析模块（Kaggle适配版）
│   ├── factor_builder.py   # 因子构建模块（Kaggle适配版）
│   ├── factor_validator.py # 因子检验模块（Kaggle适配版）
│   ├── backtest.py         # 回测模块（Kaggle适配版）
│   └── experiment_logger.py # 实验记录模块（Kaggle适配版）
├── notebooks/          # 示例Notebook
│   └── irsf_factor_strategy_kaggle.ipynb # Kaggle平台使用示例
├── run_pipeline.py     # 完整流程运行脚本（Kaggle适配版）
├── requirements.txt    # 依赖包列表
└── README.md           # 项目说明
```

## 环境要求

- Python 3.8+
- 依赖包：见 `requirements.txt`

## 在Kaggle平台上使用

### 1. 准备数据

在Kaggle平台上，需要将数据上传到 `../input` 目录。以下是需要的数据文件：

- `research_records.csv` - 机构调研记录数据
  - 列名：stock_code, stock_name, 调研日期, institution, q_content, a_content
- `daily_price.csv` - 日行情数据
  - 列名：ts_code, trade_date, open, high, low, close, vol, amount, adj_factor
- `index_000300_SH_constituents.csv` - 沪深300成分股数据（可选）
  - 列名：ts_code, weight, trade_date
- `index_000905_SH_constituents.csv` - 中证500成分股数据（可选）
  - 列名：ts_code, weight, trade_date
- `financial_fundamentals.csv` - 财务基本面数据（可选）
  - 列名：ts_code, report_date, mcap, pe, pb, bm, roe, roa

如果没有上传数据，代码会自动使用示例数据进行演示。

### 2. 安装依赖

```bash
!pip install -r ../requirements.txt
```

### 3. 运行完整流程

```bash
!python ../run_pipeline.py --start_date 20200101 --end_date 20241231 --n_long 30 --n_short 30
```

### 4. 查看结果

运行完成后，结果会保存在 `../output` 目录中，包括：

- `research_with_sentiment.csv` - 带情绪得分的调研记录
- `irsf_factors.csv` - 构建的IRSF因子数据
- `factor_validation_report.txt` - 因子检验报告
- `results/` - 回测结果目录
  - `returns.csv` - 策略收益数据
  - `metrics.json` - 回测指标
  - `equity_curve.png` - 权益曲线
  - `drawdown_curve.png` - 回撤曲线
- `experiments/` - 实验记录目录
  - `exp_log.md` - 实验日志
  - `exp_*.json` - 实验结果文件

### 5. 使用Notebook

项目提供了一个示例Notebook `notebooks/irsf_factor_strategy_kaggle.ipynb`，展示如何在Kaggle平台上运行和分析策略。

## 核心功能

1. **数据加载**：从Kaggle输入目录加载数据，支持CSV格式
2. **NLP情绪分析**：使用FinBERT模型对调研文本进行情绪打分
3. **因子构建**：构建IRSF机构调研情绪因子，包括调研密度异动、调研情绪、机构质量加权
4. **因子检验**：运行Fama-MacBeth回归、五分位分组测试、IC分析
5. **回测**：双周调仓的回测框架，计算策略绩效指标
6. **实验记录**：记录实验参数和结果，方便复现和分析

## 自定义参数

运行脚本支持以下参数：

- `--start_date`：开始日期，格式 YYYYMMDD
- `--end_date`：结束日期，格式 YYYYMMDD
- `--n_long`：多头持仓数量
- `--n_short`：空头持仓数量
- `--skip_nlp`：跳过NLP分析（使用示例数据）
- `--skip_backtest`：跳过回测

## 示例命令

```bash
# 基本运行
!python ../run_pipeline.py --start_date 20200101 --end_date 20241231 --n_long 30 --n_short 30

# 使用不同的多头/空头数量
!python ../run_pipeline.py --start_date 20200101 --end_date 20241231 --n_long 50 --n_short 50

# 跳过NLP分析（使用示例数据）
!python ../run_pipeline.py --start_date 20200101 --end_date 20241231 --n_long 30 --n_short 30 --skip_nlp
```

## 注意事项

1. **数据格式**：确保上传的数据格式正确，列名与要求一致
2. **内存限制**：Kaggle平台有内存限制，处理大规模数据时可能需要调整批处理大小
3. **计算时间**：NLP情绪分析可能需要较长时间，建议使用GPU加速
4. **输出限制**：Kaggle平台对输出文件大小有限制，建议控制输出文件大小

## 故障排除

- **数据加载失败**：检查数据文件是否上传到正确位置，列名是否正确
- **依赖安装失败**：尝试使用最新版本的依赖包
- **内存不足**：减小批处理大小，或使用更小的数据集
- **运行时间过长**：跳过NLP分析，或使用示例数据进行测试

## 许可证

MIT License
