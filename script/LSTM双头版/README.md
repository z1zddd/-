# LSTM 双头版 rolling validation 阈值实验报告

## 实验概览

本报告记录 2026-07-02 运行的 LSTM 双头版玉米价格 spike 预测实验。实验使用月度玉米混合特征数据，以 `spike` 二分类标签为预测目标，预测步长为 1 个月。模型结构为双流 LSTM：结构化行情特征进入结构化分支，新闻 PCA 稠密向量进入新闻分支，并通过注意力/融合层输出下一期发生 spike 的概率。

本次报告的主结果采用 rolling backtest，并在每一个 rolling step 内只使用历史 validation set 选择分类阈值。

## 实验目标与假设

实验目标是验证：在保留随机森林特征重要性排名前 79 个特征后，论文风格的双头 LSTM 是否能够有效预测下一期 spike。

主要假设包括：

- 结构化价格/行情特征可以提供价格周期、波动和市场联动信息。
- 新闻 PCA 特征可以补充非价格信息，提高 spike 识别能力。
- rolling validation 动态阈值比固定 0.5 阈值更适合类别不平衡的 spike 预测任务。

## 数据与任务设置

数据文件为 `C:\Users\YLHP\Desktop\玉米价格月度_混合特征版_缺失和0值填补.csv`。任务目标为 `spike`，含义为下一期是否发生 spike。实验按月度时间顺序构造序列样本，使用过去 12 个月输入预测 1 个月后的 spike 标签。

特征来自随机森林重要性排名前 79 个变量，其中：

- 结构化特征：57 个
- 新闻 PCA 特征：22 个
- 总特征数：79 个
- rolling 测试样本数：59
- rolling 测试标签分布：非 spike 42 个，spike 17 个
- 测试目标月份范围：2021-08 到 2026-06

## 代码、环境与复现信息

实验脚本：

`C:\时序玉米\rolling_backtest选阈值版（保留前79个特征）.py`

核心输出目录：

`C:\时序玉米\top79_dual_lstm_rolling_val_threshold_outputs`

运行环境：

- Python 环境：`C:\时序玉米\.venv_lstm_cpu\Scripts\python.exe`
- 设备：CPU
- 随机种子：42
- 主要依赖：PyTorch、pandas、numpy、scikit-learn

复现命令：

```powershell
& 'C:\时序玉米\.venv_lstm_cpu\Scripts\python.exe' -X utf8 'C:\时序玉米\rolling_backtest选阈值版（保留前79个特征）.py' --csv 'C:\Users\YLHP\Desktop\玉米价格月度_混合特征版_缺失和0值填补.csv' --feature-rank 'C:\时序玉米\original_rf_spike_feature_importance\original_rf_spike_feature_importance_ranking.csv' --out-dir 'C:\时序玉米\top79_dual_lstm_rolling_val_threshold_outputs' --top-n 79 --lookback 12 --horizon 1 --initial-train-fraction 0.45 --min-train-samples 36 --val-fraction 0.2 --step 1 --max-tests 0 --hidden-dim 64 --attn-dim 32 --dense-dim 64 --dropout 0.3 --lr 0.0005 --epochs 120 --patience 15 --batch-size 16 --device cpu --seed 42
```

## 实验配置

模型配置：

- 模型：Dual-stream LSTM
- 结构化分支：LSTM
- 新闻分支：LSTM + scaled dot-product attention
- 融合方式：结构化隐状态与新闻注意力表示拼接后进入全连接层
- 输出：下一期 spike 概率
- 损失函数：BCEWithLogitsLoss
- 优化器：Adam

超参数：

- lookback：12
- horizon：1
- hidden_dim：64
- attn_dim：32
- dense_dim：64
- dropout：0.30
- learning rate：0.0005
- weight decay：0.0001
- batch size：16
- max epochs：120
- patience：15
- validation fraction：0.20

rolling 设置：

- 初始训练比例：0.45
- 最小训练样本：36
- rolling step：1
- 每轮均重新训练模型
- 每轮标准化和缺失值填补仅在该轮训练集上 fit
- 每轮阈值仅在该轮历史 validation set 上选择

## 训练/运行过程

实验共完成 59 个 rolling 测试点。每轮流程如下：

1. 按时间顺序取当前测试月之前的数据。
2. 将历史数据切分为 train 和 validation。
3. 仅用 train fit 缺失值填补器和标准化器。
4. 用 train 训练双头 LSTM，并用 validation loss 早停。
5. 在 validation 概率上遍历阈值，选择表现最好的阈值。
6. 将该阈值应用到当前测试月，生成 out-of-sample 预测。

训练过程统计：

- 平均最佳 epoch：41.58
- 平均实际训练 epoch：52.95

## 结果汇总

主结果应以 `rolling validation 选阈值` 为准，因为该结果没有使用未来测试标签选择阈值。

| 阈值方式 | Accuracy | Balanced Accuracy | Precision(W) | Recall(W) | F1(W) | Precision(+) | Recall(+) | F1(+) | AUC | AP | FP/FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 固定阈值 0.5 | 0.576 | 0.702 | 0.828 | 0.576 | 0.576 | 0.405 | 1.000 | 0.576 | 1.000 | 1.000 | 25/0 |
| rolling validation 选阈值 | 0.864 | 0.905 | 0.908 | 0.864 | 0.870 | 0.680 | 1.000 | 0.810 | 1.000 | 1.000 | 8/0 |

rolling validation 阈值统计：

| 统计量 | 阈值 |
|---|---:|
| mean | 0.406 |
| median | 0.510 |
| min | 0.050 |
| max | 0.930 |

## 指标表

混淆矩阵按 `[非 spike, spike]` 二分类理解。

固定阈值 0.5：

- TN = 17
- FP = 25
- FN = 0
- TP = 17

rolling validation 选阈值：

- TN = 34
- FP = 8
- FN = 0
- TP = 17

rolling validation 结果中 8 个误报月份为：2023-01、2023-02、2025-07、2025-12、2026-01、2026-02、2026-04、2026-05。没有漏报 spike 月份。

指标解释：

- Accuracy 越高越好，表示整体预测正确比例。
- Balanced Accuracy 越高越好，适合类别不平衡任务，等于两类召回率平均值。
- F1(W) 越高越好，是按类别支持数加权的 F1。
- Recall(+) 越高越好，表示真实 spike 被抓住的比例。
- AUC 越高越好，表示模型对 spike 与非 spike 的排序能力。
- AP 越高越好，表示正类 precision-recall 综合表现。

## 产物与路径

主要产物：

- 汇总指标：`C:\时序玉米\top79_dual_lstm_rolling_val_threshold_outputs\top79_dual_lstm_summary.json`
- rolling 逐月预测：`C:\时序玉米\top79_dual_lstm_rolling_val_threshold_outputs\top79_dual_lstm_rolling_predictions.csv`
- 入模特征清单：`C:\时序玉米\top79_dual_lstm_rolling_val_threshold_outputs\top79_selected_features_for_dual_lstm.csv`
- RF 前 79 特征清单：`C:\时序玉米\original_rf_spike_feature_importance\top79_features_for_lstm.csv`
- 实验脚本：`C:\时序玉米\rolling_backtest选阈值版（保留前79个特征）.py`

## 结果分析

固定 0.5 阈值下，模型没有漏报 spike，但误报较多，FP 为 25。说明模型输出概率整体偏高，直接使用 0.5 作为分类边界会把不少非 spike 月份判成 spike。

rolling validation 选阈值后，FP 从 25 降到 8，FN 仍为 0，F1(W) 从 0.576 提升到 0.870，Balanced Accuracy 从 0.702 提升到 0.905。这说明动态阈值在当前实验中有效降低了误报，同时保留了对 spike 的完整召回。

AUC 和 AP 均为 1.000，说明在这组 rolling 预测中，模型概率对正负类的排序非常强。但由于测试样本只有 59 个，且正类集中在特定时间阶段，AUC=1.000 需要结合更多样本、更多时间切分和无泄露特征筛选流程进一步验证。

## 问题与风险

当前实验最重要的风险有三点：

1. 前 79 个特征来自全量随机森林特征重要性排名。由于特征筛选过程可能使用了全时期标签，这存在特征选择泄露风险。正式论文版应在每个 rolling step 内只用历史训练数据重新筛选特征，或使用预先固定的领域特征。
2. 输入 CSV 是已填补缺失值和 0 值的版本。如果该填补过程使用了全量时间序列统计量或未来信息，则可能存在预处理泄露。虽然本脚本内部的 imputer/scaler 是每轮只在训练集 fit，但源数据的预填补仍需审计。
3. 部分早期 validation window 类别较单一，阈值可能被选到 0.05 等极低值。这符合严格 rolling 流程，但也提示阈值稳定性受验证集样本结构影响。

## 结论

在当前数据和 top79 特征设置下，双头 LSTM 对下一期 spike 的识别效果较强。以无测试标签泄露的 rolling validation 阈值作为主结果，模型达到：

- Accuracy = 0.864
- Balanced Accuracy = 0.905
- F1(W) = 0.870
- Recall(+) = 1.000
- FP/FN = 8/0

实验支持“双头 LSTM + 新闻 PCA + 结构化行情特征”对 spike 预测有较好效果的判断。但由于 top79 特征筛选和源数据预填补仍可能引入泄露，当前结果更适合作为阶段性实验结果，不应直接作为最终论文无泄露结果。

## 下一步实验建议

建议优先做一版完全嵌套的无泄露 rolling pipeline：

1. 每个 rolling step 内重新完成缺失值/0 值处理、标准化、特征筛选、模型训练和阈值选择。
2. 对比 top40、top56、top79、top95 等不同特征数量，观察性能与稳定性。
3. 做无新闻 PCA 消融实验，量化新闻嵌入对 spike 预测的边际贡献。
4. 对 8 个误报月份做逐月诊断，检查是否存在临近 spike、行情高位震荡或新闻信号提前反应。
5. 增加重复 seed 或 bootstrap，评估 AUC=1.000 的稳健性。

## 复现清单

复现本实验需要以下文件：

- 数据文件：`C:\Users\YLHP\Desktop\玉米价格月度_混合特征版_缺失和0值填补.csv`
- 特征排名文件：`C:\时序玉米\original_rf_spike_feature_importance\original_rf_spike_feature_importance_ranking.csv`
- 实验脚本：`C:\时序玉米\rolling_backtest选阈值版（保留前79个特征）.py`
- Python 环境：`C:\时序玉米\.venv_lstm_cpu\Scripts\python.exe`
- 输出目录：`C:\时序玉米\top79_dual_lstm_rolling_val_threshold_outputs`

复现时应确认：

- `--horizon` 保持为 1
- `--lookback` 保持为 12
- `--top-n` 保持为 79
- `--seed` 保持为 42
- 主结果使用 `validation_selected_threshold`
