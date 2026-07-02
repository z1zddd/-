# 1. 混合特征版本


## 处理规则

| 字段类型 | 月度聚合方式 |
|---|---|
| 期货 open | 月初第一个值 |
| 期货 high | 月内最大值 |
| 期货 low | 月内最小值 |
| 期货 close / settle | 月末最后一个值 |
| 成交量 volume | 月内求和 |
| 持仓量 open_interest | 月末值，也可以额外保留月均值 |
| 现货价 | 月均值 + 月末值 |
| 基差 / 价差 | 月均值 + 月末值 |
| CBOT 玉米/小麦 OHLC | 同国内期货 OHLC 规则 |
| 降水量 | 月累计 |
| 温度 | 月平均 |
| 进口量 | 月累计或月末填充值，看原始口径 |
| 抛储量 | 月累计 |
| 收获季节变量 | 取最大值，或取当月处于收获季的比例 |

# 2. 月末收盘版本
## 处理规则


- `dce_corn_open`：取当月第一个交易日
- `dce_corn_high`：取当月最大值
- `dce_corn_low`：取当月最小值
- `dce_corn_close`：取当月最后一个交易日
- `dce_corn_settle`：取当月最后一个交易日
- `dce_corn_volume`：按月求和
- `dce_corn_open_interest`：取月末值
- `dce_corn_ret_1d`：不要简单平均，建议重算月收益率：

```text
本月月末close / 上月月末close - 1
```
