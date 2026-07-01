# 月度 LSTM 方向预测与 MDLS 损失实验报告

## 实验概览

| 字段 | 内容 |
|---|---|
| 实验名称 | 月度玉米价格方向预测 LSTM 改进实验 |
| 实验日期 | 2026-07-01 |
| 任务类型 | 月度时间序列预测，趋势二分类，价格回归辅助 |
| 运行平台 | 本机 CPU |
| 数据目录 | `C:\时序玉米\monthly_corn_data` |
| 主要代码路径 | `C:\时序玉米\run_balanced_attention_lstm_trend.py`，`C:\时序玉米\run_multitask_attention_lstm_trend_price.py`，`C:\时序玉米\run_multitask_attention_lstm_mdls_price.py` |
| 主要输出目录 | `C:\时序玉米\balanced_attention_lstm_trend_outputs`，`C:\时序玉米\multitask_attention_lstm_trend_price_outputs`，`C:\时序玉米\mdls_multitask_attention_lstm_outputs`，`C:\时序玉米\mdls_mse_weight_sweep_outputs`，`C:\时序玉米\mdls_only_attention_lstm_outputs` |
| 报告人 | Codex |

## 实验目标与假设

目标：

- 将原先“先预测价格，再判断上涨/下跌”的回归式 LSTM，改成更贴合方向预测目标的模型。
- 比较 `BCEWithLogitsLoss`、`BCE+MSE`、`MDLS+MSE`、`MDLS-only` 对上涨/下跌平衡性的影响。
- 重点观察模型是否过度偏向上涨，以及是否能同时识别上涨和下跌。

假设：

- 直接优化方向标签会比价格回归后转方向更符合趋势预测目标。
- 价格 MSE 辅助项可能提供价格水平约束，但权重过高时可能削弱方向平衡。
- MDLS 方向损失会比 BCE 更重视真实收益幅度较大的月份，并可能改善下跌识别。

判断标准：

- 方向任务优先看 `Balanced Accuracy`、`Recall Up`、`Recall Down`、`Macro F1` 和 `AUC`。
- 不单独追求普通 accuracy，因为测试集很小，且模型可能通过偏向某一类获得表面准确率。
- 价格回归只在有 MSE 监督的实验中作为辅助指标观察。

## 数据与任务设置

| 字段 | 内容 |
|---|---|
| 原始数据 | `C:\时序玉米\玉米价格原始数据.csv` |
| 月度数据 | `C:\时序玉米\monthly_corn_data\玉米价格月度_混合特征版.csv` |
| 月度样本 | 121 个月 |
| 输入窗口 | `seq_len = 12`，过去 12 个月 |
| 预测步长 | `horizon = 1`，预测下一月 |
| 方向目标 | `dce_corn_close_next_month_direction` |
| 价格目标 | `dce_corn_close_next_month` 或回归版中的下一月 `dce_corn_close` |
| 输入特征数 | 91 |
| 样本窗口数 | 109 |
| 划分方式 | 按时间顺序划分，训练 76，验证 16，测试 17 |
| 测试月份 | 2025-02 至 2026-06 |
| 测试集类别 | 真实上涨 8，真实下跌 9 |

防未来函数设置：

- 以下未来列不作为输入特征：
  - `dce_corn_close_next_month`
  - `dce_corn_close_next_month_ret`
  - `dce_corn_close_next_month_direction`
- 这些列只作为监督目标或评估辅助使用。
- 实验口径为：月末已知当月完整月度特征后，预测下一月方向。

## 代码、环境与复现信息

| 字段 | 内容 |
|---|---|
| Python 环境 | `C:\时序玉米\.venv_lstm_cpu\Scripts\python.exe` |
| PyTorch | `2.12.1+cpu` |
| pandas | `3.0.3` |
| scikit-learn | `1.9.0` |
| numpy | `2.5.0` |
| 设备 | CPU |
| 随机种子 | 42 |
| Git commit | 未记录 |

主要运行命令：

```powershell
& 'C:\时序玉米\.venv_lstm_cpu\Scripts\python.exe' 'C:\时序玉米\run_best_monthly_lstm_seq12_h1.py'

& 'C:\时序玉米\.venv_lstm_cpu\Scripts\python.exe' 'C:\时序玉米\run_balanced_attention_lstm_trend.py' --device cpu --output-dir 'C:\时序玉米\balanced_attention_lstm_trend_outputs'

& 'C:\时序玉米\.venv_lstm_cpu\Scripts\python.exe' 'C:\时序玉米\run_multitask_attention_lstm_trend_price.py' --device cpu --output-dir 'C:\时序玉米\multitask_attention_lstm_trend_price_outputs'

& 'C:\时序玉米\.venv_lstm_cpu\Scripts\python.exe' 'C:\时序玉米\run_multitask_attention_lstm_mdls_price.py' --device cpu --output-dir 'C:\时序玉米\mdls_multitask_attention_lstm_outputs'

& 'C:\时序玉米\.venv_lstm_cpu\Scripts\python.exe' 'C:\时序玉米\run_mdls_mse_weight_sweep.py'

& 'C:\时序玉米\.venv_lstm_cpu\Scripts\python.exe' 'C:\时序玉米\run_multitask_attention_lstm_mdls_price.py' --device cpu --mdls-loss-weight 1.0 --price-loss-weight 0.0 --output-dir 'C:\时序玉米\mdls_only_attention_lstm_outputs'
```

## 实验配置

| 参数 | 值 |
|---|---|
| 主干模型 | 2 层 LSTM |
| 池化方式 | Attention Pooling |
| hidden_size | 32 |
| dropout | 0.20 |
| batch_size | 16 |
| learning_rate | 0.0005 |
| weight_decay | 0.0001 |
| epochs | 300 |
| patience | 40 |
| optimizer | AdamW |
| grad_clip | 1.0 |
| 阈值选择 | 验证集扫描阈值，范围 0.30 至 0.70，步长 0.01 |

模型版本：

| 实验 | 输出 | 损失函数 |
|---|---|---|
| 月度 LSTM 回归最佳版 | 下一月价格，再换算方向 | `MSELoss` |
| Balanced Attention LSTM 分类版 | 上涨 logit / 概率 | `BCEWithLogitsLoss(pos_weight)` |
| 多任务 BCE+MSE | 上涨 logit + 下一月价格 | `0.5 * BCEWithLogitsLoss + 0.5 * MSELoss` |
| 多任务 MDLS+MSE | 方向分数 + 下一月价格 | `0.5 * MDLSLoss + 0.5 * MSELoss` |
| MDLS-only | 方向分数 | `1.0 * MDLSLoss` |

MDLS 本次实现：

```text
MDLSLoss =
mean(class_weight * abs(true_return) / mean_abs_train_return
     * softplus(-sign(true_return) * direction_score))
```

说明：

- `direction_score` 越大，模型越倾向预测上涨。
- `sign(true_return)` 提供真实方向。
- `abs(true_return)` 让真实波动更大的月份权重更高。
- `class_weight` 用于缓解上涨/下跌类别比例差异。
- `softplus` 用作可导的方向惩罚函数。

## 训练/运行过程

- 月度数据先由原始日频数据聚合得到，使用混合特征口径。
- 训练、验证、测试按时间顺序划分，未打乱时间顺序。
- LSTM 输入窗口为过去 12 个月，测试窗口为 2025-02 至 2026-06。
- 纯分类版按验证集 `balanced_accuracy`、`macro_f1` 和召回平衡选择最佳模型。
- 多任务版按验证集总损失选择最佳模型。
- 权重扫描共运行 9 组：`MDLS/MSE = 0.9/0.1` 至 `0.1/0.9`。
- 运行中出现过一次直接 import torch 的 DLL 报错，原因是未先加载本地 DLL 路径；训练脚本中已有 `add_local_dll_dirs()` 处理，正式训练已正常完成。

## 结果汇总

- 回归 LSTM 最佳版测试方向准确率为 70.59%，平衡准确率为 72.22%，但该模型本质仍是价格回归模型。
- 纯分类 BCE 版下跌识别较强，测试 `Recall Down = 77.78%`，但 `Recall Up = 50.00%`。
- `BCE+MSE` 多任务版明显偏向上涨，测试 `Recall Up = 87.50%`，`Recall Down = 11.11%`，不符合“上涨和下跌都关注”的目标。
- `MDLS+MSE` 0.5/0.5 版恢复了较均衡的方向识别，测试 `Recall Up = 50.00%`，`Recall Down = 77.78%`。
- 权重扫描显示，MSE 权重较高时模型容易退化为几乎全预测上涨，尤其 `MSE >= 0.7` 时测试 `Recall Down = 0.00%`。
- `MDLS-only` 是当前直接方向模型中综合最好的一版，测试 `Accuracy = 70.59%`，`Balanced Accuracy = 71.53%`，`AUC = 70.83%`。

## 指标表

### 主实验对比

| 实验 | 损失函数 | 阈值 | Test Acc | Test Balanced Acc | Test AUC | Recall Up | Recall Down | Test MAE | Test R2 | 备注 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 月度 LSTM 回归最佳版 | MSE | 价格转方向 | 70.59% | 72.22% | 75.00% | 100.00% | 未直接记录 | 60.60 | 0.1844 | 先预测价格再转方向 |
| Balanced Attention LSTM | BCE | 0.43 | 64.71% | 63.89% | 65.28% | 50.00% | 77.78% | 不适用 | 不适用 | 直接趋势分类 |
| 多任务 BCE+MSE | 0.5 BCE + 0.5 MSE | 0.48 | 47.06% | 49.31% | 69.44% | 87.50% | 11.11% | 84.94 | -0.7492 | 明显偏上涨 |
| 多任务 MDLS+MSE | 0.5 MDLS + 0.5 MSE | 0.50 | 64.71% | 63.89% | 69.44% | 50.00% | 77.78% | 84.79 | -0.7339 | 下跌识别恢复 |
| MDLS-only | 1.0 MDLS | 0.50 | 70.59% | 71.53% | 70.83% | 87.50% | 55.56% | 不作为结论 | 不作为结论 | 当前推荐方向模型 |

### MDLS/MSE 权重扫描

| MDLS | MSE | Test Balanced Acc | Test AUC | Recall Up | Recall Down | Pred Up | Pred Down | Test MAE | Test R2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.9 | 0.1 | 63.89% | 69.44% | 50.00% | 77.78% | 6 | 11 | 85.92 | -0.7672 |
| 0.8 | 0.2 | 65.97% | 70.83% | 87.50% | 44.44% | 12 | 5 | 88.58 | -0.9951 |
| 0.7 | 0.3 | 60.42% | 70.83% | 87.50% | 33.33% | 13 | 4 | 91.32 | -1.1992 |
| 0.6 | 0.4 | 50.00% | 70.83% | 100.00% | 0.00% | 17 | 0 | 84.51 | -0.7137 |
| 0.5 | 0.5 | 63.89% | 69.44% | 50.00% | 77.78% | 6 | 11 | 84.79 | -0.7339 |
| 0.4 | 0.6 | 57.64% | 68.06% | 37.50% | 77.78% | 5 | 12 | 84.93 | -0.7435 |
| 0.3 | 0.7 | 50.00% | 66.67% | 100.00% | 0.00% | 17 | 0 | 85.04 | -0.7500 |
| 0.2 | 0.8 | 50.00% | 66.67% | 100.00% | 0.00% | 17 | 0 | 85.06 | -0.7512 |
| 0.1 | 0.9 | 50.00% | 66.67% | 100.00% | 0.00% | 17 | 0 | 85.03 | -0.7501 |

## 产物与路径

| 类型 | 路径 | 说明 |
|---|---|---|
| 月度混合特征数据 | `C:\时序玉米\monthly_corn_data\玉米价格月度_混合特征版.csv` | 121 个月，混合聚合特征 |
| 回归 LSTM 最佳版 | `C:\时序玉米\monthly_lstm_best_seq12_h1` | 价格回归后转方向 |
| 纯分类 BCE 版 | `C:\时序玉米\balanced_attention_lstm_trend_outputs` | BCE 分类模型 |
| BCE+MSE 多任务版 | `C:\时序玉米\multitask_attention_lstm_trend_price_outputs` | 同时输出方向和价格 |
| MDLS+MSE 多任务版 | `C:\时序玉米\mdls_multitask_attention_lstm_outputs` | 0.5 MDLS + 0.5 MSE |
| 权重扫描 | `C:\时序玉米\mdls_mse_weight_sweep_outputs` | 9 组 MDLS/MSE 权重 |
| MDLS-only | `C:\时序玉米\mdls_only_attention_lstm_outputs` | 当前推荐方向模型 |
| 权重扫描报告 | `C:\时序玉米\mdls_mse_weight_sweep_outputs\weight_sweep_report.md` | 权重扫描 Markdown 汇总 |
| 本报告 | `C:\时序玉米\experiment_reports\月度LSTM方向预测与MDLS损失实验报告.md` | 本次综合报告 |

## 结果分析

主要结论：

- 如果任务最终目标是趋势方向，而不是价格点位，直接方向模型比价格回归模型更符合目标。
- `BCE+MSE` 的表现说明，加入价格 MSE 不一定能改善方向预测；在当前数据上，它会把模型推向上涨偏置。
- `MDLS+MSE` 相比 `BCE+MSE` 明显改善了下跌召回，说明 MDLS 对方向平衡有帮助。
- `MDLS-only` 在当前测试集上取得了直接方向模型中最好的综合结果，且没有退化成全猜上涨。
- 价格回归结果整体仍不稳定，多任务模型的测试 R2 多为负数，说明价格点位拟合还不是可靠优势。

为什么会这样：

- 月度样本只有 121 个，LSTM 参数相对样本量仍偏多，测试结果对少数月份非常敏感。
- 价格 MSE 更关注价格水平误差，不直接约束上涨/下跌平衡，可能与方向目标冲突。
- MDLS 按真实收益幅度加权，更接近“方向判断错在大波动月份更严重”的业务目标。
- 测试集真实上涨 8 个、真实下跌 9 个，任意错 1 个都会显著改变百分比指标。

泛化判断：

- 当前结果支持“MDLS 是值得保留的方向损失”这一假设。
- 但测试集仅 17 个窗口，不能据此确认模型稳定有效。
- 还需要多随机种子、滚动验证或扩展数据来判断稳定性。

## 问题与风险

数据风险：

- 月度样本量很小，测试集只有 17 个窗口。
- 当前口径是月末预测下月，若未来改成月初预测，当前月完整 OHLC 和 volume 不能使用。
- 月度聚合可能平滑掉日频结构，也可能减少噪声。

训练风险：

- LSTM 在小样本上容易过拟合。
- 最佳 epoch 较早，部分实验第 1 或第 2 个 epoch 即被选中，说明验证集波动较大。
- `prob_up` 在一些 MDLS 实验中接近 0.5，阈值敏感。

评估风险：

- 方向准确率、平衡准确率、AUC 都受 17 个测试样本强烈影响。
- 目前没有滚动外推验证，也没有多 seed 稳定性统计。
- 价格指标在 `MSELoss=0.0` 的 MDLS-only 实验中不应作为价格预测结论。

复现风险：

- 本机 PyTorch 需要脚本中的 DLL 路径补丁，直接 import torch 可能报 DLL 错误。
- Git commit 未记录，报告只记录当前文件路径与输出目录。

## 结论

本轮实验表明，若目标是“下一月上涨/下跌方向”，当前最值得保留的是 **MDLS-only Attention LSTM**。它在测试集上达到 `Accuracy = 70.59%`、`Balanced Accuracy = 71.53%`、`AUC = 70.83%`，且上涨和下跌都有一定识别能力。`MDLS+MSE` 的权重扫描显示，MSE 权重过高会使模型重新偏向上涨，因此后续不建议把价格 MSE 作为主导目标。当前结论可作为下一轮实验基线，但还不能视为稳定最终模型。

## 下一步实验建议

| 优先级 | 实验 | 改动点 | 预期收益 | 风险 |
|---|---|---|---|---|
| P0 | MDLS-only 多 seed 验证 | 固定模型和损失，跑 seed 7、42、2024、3407、2026 | 判断当前 70.59% 是否稳定 | 小样本下方差可能很大 |
| P0 | 滚动时间验证 | 用 expanding window 或 rolling window 评估 | 降低单一测试窗口偶然性 | 实现复杂度增加 |
| P1 | 阈值稳定性分析 | 在验证集和测试集上分析 0.45 至 0.55 阈值敏感性 | 判断 prob_up 接近 0.5 时的可靠性 | 容易过拟合阈值 |
| P1 | MDLS-only 去价格头版本 | 删除无监督价格头，只保留方向头 | 模型更简洁，避免无效输出 | 可能损失少量共享表征 |
| P2 | 与树模型月度分类对比 | XGBoost/RandomForest 直接预测方向 | 建立非深度学习基线 | 需要重新做特征筛选 |
| P2 | 加收益率目标辅助 | 用收益率回归替代价格回归 | 与 MDLS 方向目标更一致 | 回归噪声仍可能较大 |

## 复现清单

- [x] 代码路径明确
- [x] 运行命令明确
- [x] 数据版本明确
- [x] 环境/依赖明确
- [x] 随机种子明确
- [x] 输出目录明确
- [x] 指标来源明确
- [x] 关键产物已保存
- [ ] Git commit 已记录
- [ ] 多 seed 稳定性已验证
- [ ] 滚动验证已完成
