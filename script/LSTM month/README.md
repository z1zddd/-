# 玉米价格时序预测项目

本项目围绕玉米价格时间序列预测展开，主要目标是预测 `dce_corn_close` 的未来价格与上涨/下跌方向。当前实验重点已经从“先预测价格，再判断涨跌”逐步转向“直接学习上涨/下跌标签”，并尝试通过 LSTM、Attention LSTM、多任务损失和 MDLS 方向损失提升趋势预测稳定性。

## 项目目标

- 使用历史玉米价格和相关特征预测下一期 `dce_corn_close`。
- 构造月度数据，使用过去若干个月预测下一个月走势。
- 比较价格回归、方向分类、多任务学习和 MDLS 方向损失的效果。
- 重点关注上涨和下跌识别的平衡性，而不是只追求普通准确率。

## 数据说明

原始数据文件：

```text
玉米价格原始数据.csv
```

核心目标变量：

```text
dce_corn_close
```

月度实验使用脚本 `build_monthly_corn_data.py` 从原始日频数据生成月度数据：

```text
monthly_corn_data/玉米价格月度_月末版.csv
monthly_corn_data/玉米价格月度_混合特征版.csv
```

方向标签字段：

```text
dce_corn_close_next_month_direction
```

标签定义：

```text
下月 dce_corn_close > 当月 dce_corn_close -> 1，上涨
否则 -> 0，下跌或不涨
```

注意：方向标签只作为监督学习目标，不作为输入特征。训练特征中排除了以下未来信息字段，避免未来函数：

```text
dce_corn_close_next_month
dce_corn_close_next_month_ret
dce_corn_close_next_month_direction
```

## 当前主要实验口径

月度 LSTM 方向预测实验的默认口径：

| 项目 | 设置 |
|---|---|
| 输入窗口 | `seq_len = 12`，过去 12 个月 |
| 预测步长 | `horizon = 1`，预测下一个月 |
| 训练/验证/测试 | `70% / 15% / 15%` 按时间顺序切分 |
| 月度样本 | 121 个月 |
| 窗口样本 | 109 个 |
| 测试区间 | 2025-02 至 2026-06 |
| 特征处理 | 仅用训练集拟合标准化参数 |
| 阈值选择 | 在验证集上按 balanced accuracy 选择 |

## 环境

本地已验证环境：

| 依赖 | 版本 |
|---|---|
| Python | 3.12.13 |
| PyTorch | 2.12.1+cpu |
| pandas | 3.0.3 |
| NumPy | 2.5.0 |
| scikit-learn | 1.9.0 |

本项目可以使用 CPU 运行。当前本地虚拟环境路径为：

```powershell
.\.venv_lstm_cpu\Scripts\python.exe
```

## 快速开始

生成月度数据：

```powershell
.\.venv_lstm_cpu\Scripts\python.exe build_monthly_corn_data.py
```

运行当前表现最好的 MDLS-only 方向预测实验：

```powershell
.\.venv_lstm_cpu\Scripts\python.exe run_multitask_attention_lstm_mdls_price.py `
  --mdls-loss-weight 1.0 `
  --price-loss-weight 0.0 `
  --output-dir mdls_only_attention_lstm_outputs
```

运行 BCE 方向分类版 Attention LSTM：

```powershell
.\.venv_lstm_cpu\Scripts\python.exe run_balanced_attention_lstm_trend.py
```

运行 BCE + MSE 多任务版：

```powershell
.\.venv_lstm_cpu\Scripts\python.exe run_multitask_attention_lstm_trend_price.py
```

运行 MDLS + MSE 多任务版：

```powershell
.\.venv_lstm_cpu\Scripts\python.exe run_multitask_attention_lstm_mdls_price.py
```

运行 MDLS/MSE 权重扫描：

```powershell
.\.venv_lstm_cpu\Scripts\python.exe run_mdls_mse_weight_sweep.py
```

运行月度价格回归 LSTM 最优固定版：

```powershell
.\.venv_lstm_cpu\Scripts\python.exe run_best_monthly_lstm_seq12_h1.py
```

## 主要脚本

| 脚本 | 作用 |
|---|---|
| `build_monthly_corn_data.py` | 将原始日频数据整理为月度数据，并构造下月价格、收益率和方向标签 |
| `run_monthly_lstm_dce_corn.py` | 月度价格回归 LSTM，先预测价格，再由价格变化判断方向 |
| `run_best_monthly_lstm_seq12_h1.py` | 固定当前较优参数的月度回归 LSTM |
| `run_balanced_attention_lstm_trend.py` | 直接方向分类版 Attention LSTM，损失为 `BCEWithLogitsLoss` |
| `run_multitask_attention_lstm_trend_price.py` | BCE 方向分类 + MSE 价格回归的多任务模型 |
| `run_multitask_attention_lstm_mdls_price.py` | MDLS 方向损失 + MSE 价格回归的多任务模型，也可设置为 MDLS-only |
| `run_mdls_mse_weight_sweep.py` | 批量扫描 MDLS 和 MSE 的损失权重 |
| `summarize_monthly_lstm_seq_compare.py` | 汇总不同 `seq_len` 的月度 LSTM 结果 |
| `summarize_monthly_lstm_tuning_seq12_h1.py` | 汇总 `seq_len=12, horizon=1` 调参结果 |

## 最新关键结果

测试集共 17 个窗口，其中上涨 8 个、下跌 9 个。

| 实验 | 方向准确率 | 平衡准确率 | AUC | Macro F1 | 上涨召回 | 下跌召回 | 说明 |
|---|---:|---:|---:|---:|---:|---:|---|
| 月度回归 LSTM，价格转方向 | 70.59% | 72.22% | 75.00% | - | 100.00% | - | 价格 RMSE 73.31，R2 0.1844 |
| BCE Attention LSTM | 64.71% | 63.89% | 65.28% | 63.57% | 50.00% | 77.78% | 下跌识别较好，上涨召回偏低 |
| BCE + MSE 多任务 | 47.06% | 49.31% | 69.44% | 39.53% | 87.50% | 11.11% | 明显偏向上涨 |
| MDLS + MSE，0.5/0.5 | 64.71% | 63.89% | 69.44% | 63.57% | 50.00% | 77.78% | 方向更均衡，但价格回归仍弱 |
| MDLS-only | 70.59% | 71.53% | 70.83% | 70.18% | 87.50% | 55.56% | 当前直接方向预测里综合最好 |

结论：如果目标是直接预测上涨/下跌，当前更推荐 `MDLS-only`。它没有让模型只盯着上涨，测试集上上涨召回和下跌召回相对更均衡。若需要同时预测价格，仍需要继续改进价格回归分支，因为当前多任务中的 MSE 价格头会削弱方向判断。

## MDLS 损失说明

MDLS 是本项目中使用的方向损失函数，核心思想是直接优化预测方向，而不是先拟合价格水平。

当前实现形式：

```text
MDLSLoss =
mean(class_weight * abs(true_return) / mean_abs_train_return
     * softplus(-sign(true_return) * direction_score))
```

含义：

- `sign(true_return)` 表示真实方向。
- `direction_score` 是模型输出的方向分数，经过 sigmoid 后可理解为上涨概率。
- `abs(true_return)` 让真实波动幅度更大的月份拥有更高权重。
- `class_weight` 用于平衡上涨和下跌类别，避免模型只偏向某一类。

## 输出目录

| 目录 | 内容 |
|---|---|
| `monthly_corn_data/` | 月度数据文件 |
| `monthly_lstm_best_seq12_h1/` | 当前较优的月度回归 LSTM 结果 |
| `balanced_attention_lstm_trend_outputs/` | BCE 方向分类 LSTM 结果 |
| `multitask_attention_lstm_trend_price_outputs/` | BCE + MSE 多任务结果 |
| `mdls_multitask_attention_lstm_outputs/` | MDLS + MSE 多任务结果 |
| `mdls_mse_weight_sweep_outputs/` | MDLS/MSE 权重扫描结果 |
| `mdls_only_attention_lstm_outputs/` | MDLS-only 方向预测结果 |
| `experiment_reports/` | 实验报告 Markdown 文件 |
| `github_experiment_scripts/` | 整理出的 GitHub 实验脚本 |
| `github_experiments_by_report/` | 按报告拆分整理的实验脚本 |

## 推荐阅读

完整实验报告：

```text
experiment_reports/月度LSTM方向预测与MDLS损失实验报告.md
```

历史整理报告：

```text
玉米价格时序预测项目模型与实验结果合并报告.md
玉米价格二分类趋势预测实验标准口径.md
玉米价格趋势预测模型清单.md
```

## 指标解释

- `Accuracy`：普通方向准确率，预测对的样本数占比。
- `Balanced Accuracy`：上涨召回率和下跌召回率的平均值，更适合类别不完全均衡或模型偏向某一类时使用。
- `AUC`：衡量模型把上涨样本排在下跌样本前面的能力，不依赖单一阈值。
- `Recall Up`：真实上涨月份中，被预测为上涨的比例。
- `Recall Down`：真实下跌月份中，被预测为下跌的比例。
- `Macro F1`：上涨类 F1 和下跌类 F1 的平均值，能同时反映精确率和召回率。

## 注意事项

- 测试集只有 17 个窗口，单次结果波动会比较大，不应只凭一次测试结果判断模型优劣。
- `MDLS-only` 的价格回归头没有被 MSE 监督训练，因此它输出目录里的价格回归指标不适合作为价格预测能力依据。
- 方向预测应优先看 `Balanced Accuracy`、`Recall Up`、`Recall Down`、`Macro F1` 和 `AUC`。
- 若继续调参，建议固定 `seq_len=12`、`horizon=1` 后做多随机种子复跑，并记录均值和标准差。

## 下一步建议

- 对 `MDLS-only` 做 5 到 10 个随机种子复跑，确认结果是否稳定。
- 尝试把方向任务拆成更明确的分类模型，去掉未监督的价格头。
- 扩展验证方式，例如滚动窗口回测，而不是只做一次时间切分。
- 对特征做消融实验，检查哪些特征真正贡献方向预测能力。
- 如果继续做价格预测，建议单独训练价格模型，不要让价格 MSE 过度干扰方向分类目标。
