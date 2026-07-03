# 双头 LSTM Platt 概率校准实验报告

## 实验概览

本报告对比两次基于前 79 个特征的双头 LSTM spike 分类实验。两次实验均采用 rolling backtest，目标变量为 `spike`，预测步长为 1 个月，模型输出为下一期发生 spike 的概率。

两次实验的核心区别是校准集构造方式：

- 实验 A：`rolling_backtest_top79_dual_lstm_platt_calibration.py`，使用最近 20% 历史样本作为 validation，并在该 validation 上拟合 Platt 校准头。
- 实验 B：`rolling_backtest_top79_dual_lstm_expanding_platt_calibration.py`，使用历史扩展均衡校准集，要求校准集同时包含至少 5 个 spike 和 5 个 non-spike。

实验 B 是对实验 A 的升级，主要为了解决 validation 单一类别导致 Platt 校准无法拟合的问题。

## 实验目标与假设

目标是解决 rolling validation 动态阈值在真实预测中难以落地的问题。原始 rolling 阈值虽然能提高回测 Accuracy 和 Balanced Accuracy，但未来预测时无法预先确定哪个阈值最适合下个月。

本实验的假设是：

- 双头 LSTM 原始概率存在校准不足，导致固定 0.5 阈值下误报较多。
- 使用历史验证集拟合 Platt calibration：`p = sigmoid(a * logit + b)`，可以让概率更接近真实发生率。
- 校准后使用固定阈值 0.5 或 0.6，比每月动态选择阈值更适合真实部署。

## 数据与任务设置

数据文件：

- `C:\Users\YLHP\Desktop\玉米价格月度_混合特征版.csv`

特征排名文件：

- `C:\时序玉米\original_rf_spike_feature_importance\original_rf_spike_feature_importance_ranking.csv`

建模数据：

- 使用原始随机森林重要性排名前 79 个特征。
- 其中结构化特征 57 个，新闻 PCA 特征 22 个。
- 测试窗口共 59 个，目标月份从 2021-08 到 2026-06。
- 测试标签分布：non-spike = 42，spike = 17。

任务设置：

- 任务类型：二分类。
- 目标变量：`spike`。
- 预测步长：1 个月。
- 输入序列长度：12 个月。
- 评价方式：rolling backtest。

## 代码、环境与复现信息

代码文件：

- `C:\时序玉米\rolling_backtest_top79_dual_lstm_platt_calibration.py`
- `C:\时序玉米\rolling_backtest_top79_dual_lstm_expanding_platt_calibration.py`

Python 环境：

- `C:\时序玉米\.venv_lstm_cpu\Scripts\python.exe`

运行设备：

- `device=cpu`
- 具体 CPU 型号：待补充。

输出目录：

- 实验 A：`C:\时序玉米\top79_dual_lstm_platt_calibration_outputs`
- 实验 B：`C:\时序玉米\top79_dual_lstm_expanding_platt_calibration_outputs`

## 实验配置

共同模型配置：

| 参数 | 值 |
|---|---:|
| 模型 | Dual-stream LSTM |
| structured branch hidden_dim | 64 |
| news branch hidden_dim | 64 |
| attention dim | 32 |
| dense dim | 64 |
| dropout | 0.30 |
| learning rate | 0.0005 |
| weight decay | 0.0001 |
| batch size | 16 |
| epochs | 120 |
| patience | 15 |
| seed | 42 |
| lookback | 12 |
| horizon | 1 |

实验 A 校准配置：

| 参数 | 值 |
|---|---:|
| validation 规则 | 最近 20% 历史样本 |
| Platt 正则 | 0.001 |
| calibrator max iter | 100 |

实验 B 校准配置：

| 参数 | 值 |
|---|---:|
| 校准规则 | 从最近历史向前扩展，直到包含足够正负样本 |
| calibration_min_train_samples | 24 |
| calibration_min_pos | 5 |
| calibration_min_neg | 5 |
| Platt 正则 | 0.01 |
| calibrator max iter | 100 |

## 训练/运行过程

两次实验均按 rolling backtest 执行。每个测试月只使用该月之前的数据进行训练、验证或校准，不使用测试月标签拟合模型或校准器。

实验 A 流程：

1. 对每个测试月，取测试月之前的历史样本。
2. 前面部分训练双头 LSTM，最近 20% 作为 validation。
3. 在 validation 上得到 logits。
4. 如果 validation 同时包含 0 和 1，则拟合 Platt 校准头。
5. 如果 validation 只有单一类别，则回退到原始 sigmoid 概率。
6. 使用固定阈值 0.5、0.6、0.7 评估校准概率。

实验 B 流程：

1. 对每个测试月，取测试月之前的历史样本。
2. 从最近历史开始向前扩展校准集，直到校准集至少包含 5 个 spike 和 5 个 non-spike。
3. 校准集之前的数据用于训练双头 LSTM。
4. 在扩展校准集上拟合 Platt 校准头。
5. 使用固定阈值 0.5、0.6、0.7 评估校准概率。

## 结果汇总

实验 A 的主要问题是校准集单一类别严重：

| 项目 | 数值 |
|---|---:|
| rolling 测试窗口 | 59 |
| 成功拟合 Platt 校准 | 16 |
| 回退到原始概率 | 43 |

实验 B 解决了该问题：

| 项目 | 数值 |
|---|---:|
| rolling 测试窗口 | 59 |
| 扩展均衡校准集 | 59 |
| 回退到最近 validation | 0 |
| 成功拟合 Platt 校准 | 59 |
| 校准集样本数均值 | 30.25 |
| 校准集样本数中位数 | 31 |
| 校准集样本数范围 | 10 到 46 |
| calibration_pos 最小值 | 5 |
| calibration_neg 最小值 | 5 |

## 指标表

实验 A：最近 20% validation Platt 校准

| 方法 | Accuracy | Balanced Acc | F1(W) | Precision(+) | Recall(+) | F1(+) | AUC | AP | FP/FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 原始概率 + 固定 0.5 | 0.593 | 0.714 | 0.596 | 0.415 | 1.000 | 0.586 | 0.999 | 0.997 | 24/0 |
| 原始概率 + validation 动态阈值参考 | 0.847 | 0.893 | 0.854 | 0.654 | 1.000 | 0.791 | 0.999 | 0.997 | 9/0 |
| 校准概率 + 固定 0.5 | 0.729 | 0.810 | 0.740 | 0.515 | 1.000 | 0.680 | 0.999 | 0.997 | 16/0 |
| 校准概率 + 固定 0.6 | 0.864 | 0.905 | 0.870 | 0.680 | 1.000 | 0.810 | 0.999 | 0.997 | 8/0 |
| 校准概率 + 固定 0.7 | 0.881 | 0.917 | 0.886 | 0.708 | 1.000 | 0.829 | 0.999 | 0.997 | 7/0 |

实验 B：历史扩展均衡 Platt 校准

| 方法 | Accuracy | Balanced Acc | F1(W) | Precision(+) | Recall(+) | F1(+) | AUC | AP | FP/FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 原始概率 + 固定 0.5 | 0.610 | 0.656 | 0.627 | 0.406 | 0.765 | 0.531 | 0.644 | 0.556 | 19/4 |
| 原始概率 + 校准集动态阈值参考 | 0.898 | 0.929 | 0.902 | 0.739 | 1.000 | 0.850 | 0.644 | 0.556 | 6/0 |
| 校准概率 + 固定 0.5 | 0.915 | 0.940 | 0.918 | 0.773 | 1.000 | 0.872 | 0.927 | 0.814 | 5/0 |
| 校准概率 + 固定 0.6 | 0.915 | 0.940 | 0.918 | 0.773 | 1.000 | 0.872 | 0.927 | 0.814 | 5/0 |
| 校准概率 + 固定 0.7 | 0.898 | 0.911 | 0.901 | 0.762 | 0.941 | 0.842 | 0.927 | 0.814 | 5/1 |

概率校准指标：

| 实验 | Raw Brier | Calibrated Brier | Raw ECE | Calibrated ECE | Raw Prob Mean | Calibrated Prob Mean |
|---|---:|---:|---:|---:|---:|---:|
| 实验 A | 0.263 | 0.207 | 0.411 | 0.325 | 0.699 | 0.613 |
| 实验 B | 0.235 | 0.105 | 0.250 | 0.189 | 0.533 | 0.450 |

## 产物与路径

实验 A 产物：

- `C:\时序玉米\top79_dual_lstm_platt_calibration_outputs\top79_dual_lstm_platt_summary.json`
- `C:\时序玉米\top79_dual_lstm_platt_calibration_outputs\top79_dual_lstm_platt_rolling_predictions.csv`
- `C:\时序玉米\top79_dual_lstm_platt_calibration_outputs\top79_selected_features_for_dual_lstm_platt.csv`

实验 B 产物：

- `C:\时序玉米\top79_dual_lstm_expanding_platt_calibration_outputs\top79_dual_lstm_expanding_platt_summary.json`
- `C:\时序玉米\top79_dual_lstm_expanding_platt_calibration_outputs\top79_dual_lstm_expanding_platt_rolling_predictions.csv`
- `C:\时序玉米\top79_dual_lstm_expanding_platt_calibration_outputs\top79_selected_features_for_dual_lstm_expanding_platt.csv`

整理后的实验数据包：

- `C:\时序玉米\top79_dual_lstm_expanding_platt_experiment_data.zip`

输入原始数据包：

- `C:\时序玉米\top79_dual_lstm_expanding_platt_input_raw_data.zip`

## 结果分析

实验 A 表明，Platt 校准可以降低原始概率偏高的问题。相较于原始概率固定 0.5，校准概率固定 0.5 将 FP 从 24 降到 16；固定 0.6 将 FP 降到 8，Balanced Accuracy 提升到 0.905。

但是实验 A 的校准集存在严重单一类别问题。59 个 rolling 窗口中只有 16 个真正拟合 Platt 校准，43 个窗口回退到原始概率。因此实验 A 更适合作为方法验证，不适合作为最终落地方案。

实验 B 通过历史扩展均衡校准集解决了上述问题。所有 59 个窗口均成功拟合 Platt 校准头，且固定阈值 0.5 和 0.6 均达到：

- Accuracy = 0.915
- Balanced Accuracy = 0.940
- F1(W) = 0.918
- Precision(+) = 0.773
- Recall(+) = 1.000
- FP/FN = 5/0

这说明在本回测区间内，扩展均衡校准比最近 20% validation 校准更适合作为可部署流程。

需要注意：实验 B 中 raw probability 的 AUC 为 0.644，而 calibrated probability 的 AUC 为 0.927。Platt scaling 在单个模型内是单调变换，但 rolling backtest 中每个测试月对应不同模型和不同校准参数，因此跨月份汇总 AUC 可能发生变化。

## 问题与风险

1. 固定阈值的选择风险

校准概率固定 0.5 和 0.6 在实验 B 中结果相同，均优于 0.7。若在查看测试结果后才选择 0.6 或 0.5，会存在测试集选择偏差。论文或落地时应预先规定阈值，例如使用 0.5 作为概率意义阈值，或基于业务风险预设 0.6 作为保守阈值。

2. 校准参数稳定性风险

实验 B 中 `a` 的中位数为 1.206，但均值为 8.349，最大值为 113.177，说明少数窗口校准斜率偏大。虽然使用 `calibrator_l2=0.01` 做了正则，但仍建议后续测试参数裁剪，例如限制 `a <= 10` 或 `a <= 20`。

3. 特征选择潜在泄露风险

本实验使用的前 79 个特征来自已有随机森林重要性排名文件。如果该排名是在全量数据上生成，则存在特征选择层面的测试期信息泄露风险。当前报告只评价 LSTM 训练、校准和阈值过程本身；若要完全无泄露，应在每个 rolling 窗口内重新进行特征选择。

4. 持续校准需求

Platt 参数不应一次训练后永久固定。真实部署时应在每次预测新月份前，仅使用当时可见的历史数据重新拟合校准头，并定期重新训练主模型。

5. 市场分布漂移风险

如果未来市场进入新制度或出现政策、天气、流动性等外生冲击，历史校准关系可能失效。需要持续监控 Brier Score、ECE、FP/FN 和校准参数漂移。

## 结论

实验 B 是当前更适合落地的版本。它相较实验 A 解决了单一类别 validation 导致无法校准的问题，并且在不使用测试月标签的前提下，用固定阈值 0.5/0.6 达到了较好的回测效果。

推荐将实验 B 作为当前主结果：

- 双头 LSTM
- rolling backtest
- 历史扩展均衡校准集
- Platt probability calibration
- 固定阈值 0.5 或预先设定的 0.6

该流程没有测试标签泄露，但仍需说明前 79 特征来源可能带来的特征选择泄露风险。

## 下一步实验建议

1. 做完全无泄露版本：每个 rolling 窗口内重新进行随机森林特征选择，再训练双头 LSTM 和校准器。
2. 测试校准参数裁剪：`a <= 10`、`a <= 20`，比较 Brier、ECE、Balanced Accuracy 和 FP/FN。
3. 做阈值预设敏感性分析：固定 0.5、0.6、0.7，不用测试集选择最终阈值。
4. 测试 `calibration_min_pos/min_neg = 3/5/8`，观察校准集大小和参数稳定性。
5. 增加持续学习模拟：每个月加入最新真实标签后重新训练或微调，比较只校准和重新训练两种部署策略。

## 复现清单

复现实验 A：

```powershell
& 'C:\时序玉米\.venv_lstm_cpu\Scripts\python.exe' -X utf8 'C:\时序玉米\rolling_backtest_top79_dual_lstm_platt_calibration.py' --csv 'C:\Users\YLHP\Desktop\玉米价格月度_混合特征版.csv' --feature-rank 'C:\时序玉米\original_rf_spike_feature_importance\original_rf_spike_feature_importance_ranking.csv' --out-dir 'C:\时序玉米\top79_dual_lstm_platt_calibration_outputs' --top-n 79 --lookback 12 --horizon 1 --initial-train-fraction 0.45 --min-train-samples 36 --val-fraction 0.2 --step 1 --max-tests 0 --hidden-dim 64 --attn-dim 32 --dense-dim 64 --dropout 0.3 --lr 0.0005 --epochs 120 --patience 15 --batch-size 16 --device cpu --seed 42
```

复现实验 B：

```powershell
& 'C:\时序玉米\.venv_lstm_cpu\Scripts\python.exe' -X utf8 'C:\时序玉米\rolling_backtest_top79_dual_lstm_expanding_platt_calibration.py' --csv 'C:\Users\YLHP\Desktop\玉米价格月度_混合特征版.csv' --feature-rank 'C:\时序玉米\original_rf_spike_feature_importance\original_rf_spike_feature_importance_ranking.csv' --out-dir 'C:\时序玉米\top79_dual_lstm_expanding_platt_calibration_outputs' --top-n 79 --lookback 12 --horizon 1 --initial-train-fraction 0.45 --min-train-samples 36 --calibration-min-train-samples 24 --calibration-min-pos 5 --calibration-min-neg 5 --step 1 --max-tests 0 --hidden-dim 64 --attn-dim 32 --dense-dim 64 --dropout 0.3 --lr 0.0005 --epochs 120 --patience 15 --batch-size 16 --calibrator-l2 0.01 --device cpu --seed 42
```

复现前确认：

- 原始 CSV 存在：`C:\Users\YLHP\Desktop\玉米价格月度_混合特征版.csv`
- 特征排名文件存在：`C:\时序玉米\original_rf_spike_feature_importance\original_rf_spike_feature_importance_ranking.csv`
- Python 环境存在：`C:\时序玉米\.venv_lstm_cpu\Scripts\python.exe`
