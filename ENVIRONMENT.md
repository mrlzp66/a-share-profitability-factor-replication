# 环境与复现说明

## 项目定位

本仓库公开研究代码、聚合结果、图表和研究文档；不公开 RiceQuant 原始数据、个股面板、缓存、目标权重或原始回测结果。

## 环境划分

### 因子研究环境

- Python 3.9；
- 主要依赖：pandas、numpy、matplotlib、statsmodels；
- 需要具备合法授权的 RiceQuant / RQData 数据访问环境；
- 用于财务数据处理、因子构造、横截面检验、HAC 检验和研究图表生成。

### RQAlpha 回测环境

- 使用课程提供的 RQAlpha 兼容环境；
- 本研究实际运行于 Python 3.6 的 RQAlpha 环境；
- 主要依赖：rqalpha_plus、rqalpha、rqdatac、pandas；
- 用于目标权重驱动的开盘价调仓、佣金、印花税、最低佣金、涨跌停与成交量约束回测。

## 建议运行顺序

1. 在研究环境中依次阅读并执行 `notebooks/00` 至 `notebooks/06`；
2. 在 RQAlpha 环境中执行 `notebooks/07` 与 `notebooks/08`；
3. 运行 `notebooks/09_research_robustness_and_closure.ipynb` 完成稳健性检验；
4. 运行 `notebooks/10_final_research_report.ipynb` 汇总最终研究证据。

## 数据与复现边界

- 使用者须自行准备有合法授权的动态股票池、交易日历、行情、财务与行业分类数据；
- 不同的数据版本、交易日历、复权口径和 RQAlpha 版本可能导致数值存在差异；
- 公开版本不保证开箱即跑；其用途是展示透明研究流程、代码结构与已冻结的研究结论。

## 研究冻结原则

样本外区间已经被观察。不得依据既有样本外结果重新优化组件权重、持有期、阈值、股票池或交易规则。