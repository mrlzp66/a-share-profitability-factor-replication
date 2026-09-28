# A 股盈利能力因子透明复现与执行约束回测

> 研究状态：`candidate_not_confirmed`

本项目实现并检验一组由招商证券盈利能力评价框架启发的 A 股因子。目标不是展示历史收益，而是检验当前实现是否在严格样本外、交易约束和成本假设下具有独立的增量选股能力。

> **复现边界更正（2026-09-29）**：研报的正式最终因子名称为“综合盈利因子”，其第 7.2 节采用两层融合，四个底层因子的隐含权重为 1/6、1/6、1/6、1/2。本仓库已经冻结的历史结果采用四个底层因子各 25% 的**等权透明实现**。因此，仓库内的 `candidate_not_confirmed` 只适用于当前等权实现，不能外推为对研报原始综合盈利因子的验证或否定。详见 [复现边界说明](docs/11_COMPOSITE_PROFITABILITY_FACTOR_EQUAL_WEIGHT_IMPLEMENTATION.md)。

## 核心结论

- 高分组 Q5 在样本外 RQAlpha 回测中优于低分组 Q1；
- 强制清仓后，Q5 相对 Q1 的累计收益为 **4.1123%**；
- Q5 相对等权可投资券池的累计收益为 **-0.0359%**，未表现出稳定增量选股收益；
- 样本外 RankIC 为 **0.003686**，HAC p 值为 **0.8178**；
- 样本外 Q5-Q1 平均收益为 **0.1092%**，HAC p 值为 **0.8317**；
- 因子因此保留为研究候选，不认定为已确认的独立 Alpha。

## 研究设计

- 股票池：动态中证 800 成分股；
- 信号：月末横截面信号，持有期为后续 20 个交易日；
- 样本内：2016–2022；严格样本外：2023–2025；
- 当前实现：基于研报四个底层因子的等权透明实现——稳定去杠杆 ROE、稳定 ROIC 代理、稳定 RONOA 代理、重构 FCFFIC 各 25%；
- 横截面处理：MAD 去极值、行业与对数市值中性化、z-score 标准化；
- 执行验证：RQAlpha，包含佣金、印花税、最低佣金、涨跌停与成交量约束；
- 稳健性：HAC 标准误、逐年检验、最大回撤诊断与 20 个交易日强制清仓检验。

## 主要结果

| 证据 | 结果 |
| --- | ---: |
| 样本外 RankIC | 0.003686 |
| RankIC HAC p 值 | 0.817816 |
| 样本外 Q5-Q1 平均收益 | 0.1092% |
| Q5-Q1 HAC p 值 | 0.831677 |
| Q5 相对 Q1 收益（强制清仓后） | 4.1123% |
| Q5 相对等权可投资券池收益（强制清仓后） | -0.0359% |
| 最终研究状态 | candidate_not_confirmed |

## 目录说明

- `notebooks/`：无输出、可阅读的研究流程 Notebook；
- `lib/`：通用数据处理、统计检验和 RQAlpha 回测工具；
- `artifacts/tables/`：安全的聚合研究结果；
- `artifacts/figures/`：报告图表；
- `artifacts/metadata/`：实验规则、参数与决策元数据；
- `docs/`：研究协议与最终研究报告。

## 复现边界

公开版本不含原始市场数据、个股面板、缓存、目标权重、RQAlpha 原始结果或任何凭证。复现者需要在拥有合法授权的数据环境中自行准备数据，并遵守相应数据服务协议。

## 免责声明

本项目仅用于量化研究与工程展示，不构成投资建议、收益承诺或任何证券交易建议。

## 关键图表

### 05 integrated profitability factor effectiveness

![05 integrated profitability factor effectiveness](artifacts/figures/05_integrated_profitability_factor_effectiveness.png)

### 08 rqalpha oos three portfolio comparison

![08 rqalpha oos three portfolio comparison](artifacts/figures/08_rqalpha_oos_three_portfolio_comparison.png)

### 09 rqalpha forced exit sensitivity

![09 rqalpha forced exit sensitivity](artifacts/figures/09_rqalpha_forced_exit_sensitivity.png)
