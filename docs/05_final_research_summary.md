# 盈利能力因子研究：冻结证据汇总

## 研究范围

- 股票池：动态中证 800；
- 收益标签：信号日后首个交易日开盘至 T+21 开盘，持有 20 个交易日；
- 样本外区间：2023-01 至 2025-12；
- 因子处理：MAD 去极值、CITIC 2019 一级行业和对数市值中性化、z-score；
- 本文件仅汇总既有证据，不修改因子定义、组件、权重或样本外窗口。

## 冻结结论

- `reject_as_standalone_factor`：6 个证据记录。
- `candidate_not_confirmed`：3 个证据记录。
- `not_a_standalone_factor`：1 个证据记录。
- `report_component_proxy_not_confirmed`：1 个证据记录。
- `low_coverage_sensitivity_only`：1 个证据记录。

## 重要限制

- 重构 FCFFIC、盈利稳定性及当前等权合成对照版本均未达到独立 Alpha 的确认标准；
- 稳定 RONOA 与当前等权合成对照版本受限于较低覆盖率，应仅作为权重敏感性对照或辅助证据；
- 共同样本组件归因仅用于解释固定四个底层组件的等权实现，不可用于事后删组件或优化权重；
- 当前结果未纳入交易成本、涨跌停、停牌和组合执行约束。

## 样本外证据表

```text
                                     factor_key factor_label    research_role                        research_scope                  period  observations  mean_paired_stock_count  min_paired_stock_count  mean_ic  icir_annualized  mean_rank_ic  rank_icir_annualized  positive_rank_ic_ratio  mean_high_minus_low_return  high_minus_low_t_stat             formal_decision                    formal_decision_scope              final_research_position       source_label                                  decision_source_file
                               roe_ttm_baseline       原始 ROE       单因子或参考字段检验       historical_profitability_screen out_of_sample_2023_2025            36                 797.3056                     NaN  -0.0134          -0.4324       -0.0101               -0.3275                  0.4444                     -0.0051                -1.0820 reject_as_standalone_factor                                                   reject_as_standalone_factor        早期 ROE 变体筛选            02_profitability_research_decision_log.csv
                                  delevered_roe      去杠杆 ROE       单因子或参考字段检验       historical_profitability_screen out_of_sample_2023_2025            36                 743.6389                     NaN  -0.0125          -0.4037       -0.0087               -0.2857                  0.4444                     -0.0029                -0.6451 reject_as_standalone_factor                                                   reject_as_standalone_factor        早期 ROE 变体筛选            02_profitability_research_decision_log.csv
                        negative_roe_volatility        盈利稳定性        候选盈利稳定性组件       historical_profitability_screen out_of_sample_2023_2025            36                 724.1944                     NaN   0.0260           1.1678        0.0334                1.4212                  0.6667                      0.0062                 1.4962     candidate_not_confirmed                                                       candidate_not_confirmed        早期 ROE 变体筛选            02_profitability_research_decision_log.csv
                           stable_delevered_roe    稳定去杠杆 ROE           研报结构组件       historical_profitability_screen out_of_sample_2023_2025            36                 724.1944                     NaN   0.0071           0.2894        0.0112                0.4192                  0.5833                      0.0003                 0.0792     not_a_standalone_factor                                                       not_a_standalone_factor        早期 ROE 变体筛选            02_profitability_research_decision_log.csv
integrated_profitability_structure_proxy_strict  综合盈利因子等权合成对照版本     低覆盖率权重敏感性对照   low_coverage_structural_sensitivity out_of_sample_2023_2025            36                 268.5556                   241.0   0.0023           0.0734        0.0037                0.1233                  0.5833                      0.0011                 0.2132     candidate_not_confirmed low_coverage_structural_sensitivity_only              candidate_not_confirmed        基于研报四个底层子因子的等权合成对照版本 05_integrated_profitability_research_decision_log.csv
                           roa_vendor_reference  米筐 ROA 参考字段       单因子或参考字段检验                  main_ttm_factor_pool out_of_sample_2023_2025            36                 798.0278                     NaN  -0.0190          -0.6830       -0.0111               -0.4120                  0.4444                     -0.0069                -1.5528 reject_as_standalone_factor                                                   reject_as_standalone_factor         TTM 盈利因子主池               05_ttm_factor_research_decision_log.csv
                          roic_vendor_reference 米筐 ROIC 参考字段       单因子或参考字段检验                  main_ttm_factor_pool out_of_sample_2023_2025            36                 716.9722                     NaN  -0.0190          -0.6684       -0.0125               -0.4467                  0.4444                     -0.0063                -1.3350 reject_as_standalone_factor                                                   reject_as_standalone_factor         TTM 盈利因子主池               05_ttm_factor_research_decision_log.csv
                    fcffic_reconstructed_strict    重构 FCFFIC        候选现金流盈利组件                  main_ttm_factor_pool out_of_sample_2023_2025            36                 624.2778                     NaN   0.0116           0.5156        0.0252                1.0379                  0.6389                      0.0011                 0.3061     candidate_not_confirmed                                                       candidate_not_confirmed         TTM 盈利因子主池               05_ttm_factor_research_decision_log.csv
                       roa_reconstructed_strict       重构 ROA       单因子或参考字段检验                  main_ttm_factor_pool out_of_sample_2023_2025            36                 798.0278                     NaN  -0.0172          -0.6149       -0.0114               -0.4340                  0.4444                     -0.0056                -1.2245 reject_as_standalone_factor                                                   reject_as_standalone_factor         TTM 盈利因子主池               05_ttm_factor_research_decision_log.csv
                      roic_reconstructed_strict      重构 ROIC       单因子或参考字段检验                  main_ttm_factor_pool out_of_sample_2023_2025            36                 649.3333                     NaN  -0.0197          -0.7262       -0.0161               -0.5724                  0.3889                     -0.0076                -1.6907 reject_as_standalone_factor                                                   reject_as_standalone_factor         TTM 盈利因子主池               05_ttm_factor_research_decision_log.csv
                          stable_roic_ttm_proxy   稳定 ROIC 代理   研报稳定 ROIC 组件代理 report_component_proxy_or_sensitivity out_of_sample_2023_2025            36                 507.4167                   488.0  -0.0066          -0.3028       -0.0001               -0.0040                  0.4722                     -0.0018                -0.4680                         NaN                                      NaN report_component_proxy_not_confirmed 稳定 ROIC/RONOA 组件代理                                                   NaN
                         stable_ronoa_ttm_proxy  稳定 RONOA 代理 低覆盖率 RONOA 敏感性组件 report_component_proxy_or_sensitivity out_of_sample_2023_2025            36                 285.3333                   265.0   0.0011           0.0447        0.0017                0.0629                  0.5278                      0.0027                 0.5999                         NaN                                      NaN        low_coverage_sensitivity_only 稳定 ROIC/RONOA 组件代理                                                   NaN
```
