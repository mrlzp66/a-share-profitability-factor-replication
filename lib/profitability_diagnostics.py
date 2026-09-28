"""
A 股盈利能力研究：专用诊断工作流。

依赖：
- profitability_research_utils.py：通用缓存、中性化、IC/RankIC、保存工具；
- 本模块：盈利因子项目的综合因子、决策与组件归因流程。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import profitability_research_utils as pru


DIAGNOSTICS_VERSION = "0.1.0"


def run_integrated_profitability_test(
    project_root: str | Path,
) -> dict[str, Any]:
    """调用已存在的综合盈利因子收益检验工作流。"""
    return pru.run_integrated_profitability_factor_test(project_root)


def run_integrated_profitability_decision(
    project_root: str | Path,
) -> dict[str, Any]:
    """调用已存在的综合盈利因子样本外决策工作流。"""
    return pru.run_out_of_sample_decision_from_summary(
        project_root=project_root,
        summary_filename="05_integrated_profitability_effectiveness_summary.csv",
        output_prefix="05_integrated_profitability",
        factor_name="integrated_profitability_structure_proxy_strict",
        decision_scope="low_coverage_structural_sensitivity_only",
    )


def run_common_sample_component_attribution(
    project_root: str | Path,
    output_prefix: str = "05_integrated_profitability",
    composite_factor_name: str = (
        "integrated_profitability_structure_proxy_strict"
    ),
    min_stocks: int = 100,
) -> dict[str, Any]:
    """
    在综合因子的严格共同可用样本中，对全部组件进行归因诊断。

    这是解释性诊断，而非因子筛选或权重优化：
    - 四组件与 25% 固定权重不变；
    - 不删除任何组件；
    - 不改变样本外区间；
    - 不根据诊断结果构造新组合。
    """
    paths = pru.ProjectPaths.from_root(project_root)
    paths.ensure_directories()
    paths.ensure_cache_ignored()

    component_cache_path = (
        paths.cache_data_dir
        / "05_integrated_profitability_components.pkl"
    )
    component_meta_path = (
        paths.cache_metadata_dir
        / "05_integrated_profitability_components.json"
    )
    context_cache_path = (
        paths.cache_data_dir
        / "05_neutralization_context.pkl"
    )
    composite_cache_path = (
        paths.cache_data_dir
        / "05_integrated_profitability_structure_proxy.pkl"
    )
    return_cache_path = (
        paths.cache_data_dir
        / "05_forward_return_20d.pkl"
    )

    required_paths = [
        component_cache_path,
        component_meta_path,
        context_cache_path,
        composite_cache_path,
        return_cache_path,
    ]

    missing_paths = [
        str(path) for path in required_paths if not path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "缺少本地缓存：\n" + "\n".join(missing_paths)
        )

    component_panels = pd.read_pickle(component_cache_path)
    component_metadata = json.loads(
        component_meta_path.read_text(encoding="utf-8")
    )
    context = pd.read_pickle(context_cache_path)
    composite_payload = pd.read_pickle(composite_cache_path)
    forward_return_panel = pd.read_pickle(return_cache_path)

    expected_components = [
        "stable_delevered_roe",
        "stable_roic_proxy",
        "stable_ronoa_proxy",
        "fcffic_reconstructed",
    ]

    if set(component_panels.keys()) != set(expected_components):
        raise RuntimeError(
            "四组件缓存不完整，请重新运行 05.13.0。"
        )

    if "final_factor" not in composite_payload:
        raise RuntimeError(
            "综合因子缓存中缺少 final_factor，请重新运行 05.13.1。"
        )

    for context_key in [
        "research_mask",
        "industry_panel",
        "log_market_cap_panel",
    ]:
        if context_key not in context:
            raise RuntimeError(
                f"中性化上下文缺少 {context_key}，请重新运行 05.13.1。"
            )

    composite_panel = composite_payload["final_factor"]
    signal_index = composite_panel.index
    stock_columns = composite_panel.columns

    research_mask = (
        context["research_mask"]
        .reindex(index=signal_index, columns=stock_columns)
        .fillna(False)
        .astype(bool)
    )
    industry_panel = context["industry_panel"].reindex(
        index=signal_index,
        columns=stock_columns,
    )
    log_market_cap_panel = context["log_market_cap_panel"].reindex(
        index=signal_index,
        columns=stock_columns,
    ).astype(float)

    forward_return_panel = forward_return_panel.reindex(
        index=signal_index,
        columns=stock_columns,
    ).astype(float)

    aligned_components = {
        name: component_panels[name]
        .reindex(index=signal_index, columns=stock_columns)
        .astype(float)
        for name in expected_components
    }

    # 共同样本 = 研究券池内且全部四组件有效。
    common_sample_mask = research_mask.copy()

    for component_panel in aligned_components.values():
        common_sample_mask &= np.isfinite(component_panel)

    component_attribution_cache_path = (
        paths.cache_data_dir
        / "05_common_sample_component_attribution_panels.pkl"
    )
    component_attribution_meta_path = (
        paths.cache_metadata_dir
        / "05_common_sample_component_attribution_panels.json"
    )

    cache_config = {
        "cache_schema_version": "2026-09-28-v1",
        "purpose": "common_sample_component_attribution_diagnostic",
        "component_source_fingerprint": component_metadata.get(
            "fingerprint",
            "unknown",
        ),
        "components": expected_components,
        "common_sample_rule": (
            "research_mask_true_and_all_four_components_finite"
        ),
        "conditional_treatment": (
            "re_neutralize_each_component_within_common_sample"
        ),
        "minimum_stocks": min_stocks,
        "anti_optimization_rule": (
            "diagnostic_only_no_component_removal_or_weight_optimization"
        ),
    }

    def build_common_sample_component_panels():
        final_component_panels = {}
        audit_rows = []

        for component_name, raw_panel in aligned_components.items():
            final_panel = pd.DataFrame(
                np.nan,
                index=signal_index,
                columns=stock_columns,
                dtype=float,
            )

            for signal_date in signal_index:
                final_values, audit = pru.neutralize_cross_section(
                    raw_factor=raw_panel.loc[signal_date],
                    eligible_mask=common_sample_mask.loc[signal_date],
                    log_market_cap=log_market_cap_panel.loc[signal_date],
                    industry=industry_panel.loc[signal_date],
                    min_stocks=min_stocks,
                )

                final_panel.loc[
                    signal_date,
                    final_values.index,
                ] = final_values

                audit_rows.append(
                    {
                        "factor_name": component_name,
                        "signal_date": pd.Timestamp(signal_date),
                        "common_sample_stock_count": int(
                            common_sample_mask.loc[signal_date].sum()
                        ),
                        **audit,
                    }
                )

            final_component_panels[component_name] = final_panel

        return {
            "component_panels": final_component_panels,
            "neutralization_audit": pd.DataFrame(audit_rows),
        }

    attribution_payload, cache_hit = pru.load_or_build_pickle_cache(
        data_path=component_attribution_cache_path,
        metadata_path=component_attribution_meta_path,
        config=cache_config,
        builder=build_common_sample_component_panels,
        contains_individual_stock_values=True,
    )

    common_sample_component_panels = attribution_payload[
        "component_panels"
    ]
    neutralization_audit = attribution_payload[
        "neutralization_audit"
    ].copy()

    # 以综合因子实际达到门槛的日期作为唯一比较时间窗。
    composite_metrics = pru.evaluate_factor_panel(
        factor_panel=composite_panel,
        forward_return_panel=forward_return_panel,
        min_stocks=min_stocks,
        quantiles=5,
    )

    shared_dates = pd.DatetimeIndex(
        composite_metrics.loc[
            composite_metrics["status"] == "success",
            "signal_date",
        ]
    )

    if shared_dates.empty:
        raise RuntimeError("综合因子没有达到最低覆盖率门槛的日期。")

    first_available_date = shared_dates.min()
    last_available_date = shared_dates.max()

    periods = {
        (
            f"all_available_"
            f"{first_available_date.year}_{last_available_date.year}"
        ): (
            first_available_date.strftime("%Y-%m-%d"),
            last_available_date.strftime("%Y-%m-%d"),
        ),
        f"in_sample_available_{first_available_date.year}_2022": (
            first_available_date.strftime("%Y-%m-%d"),
            "2022-12-31",
        ),
        "out_of_sample_2023_2025": (
            "2023-01-01",
            "2025-12-31",
        ),
    }

    test_panels = {
        composite_factor_name: composite_panel,
        **{
            f"{component_name}_on_common_sample": component_panel
            for component_name, component_panel
            in common_sample_component_panels.items()
        },
    }

    monthly_metric_frames = []
    summary_frames = []

    for tested_factor_name, tested_panel in test_panels.items():
        monthly_metrics = pru.evaluate_factor_panel(
            factor_panel=tested_panel,
            forward_return_panel=forward_return_panel,
            min_stocks=min_stocks,
            quantiles=5,
        )

        monthly_metrics = monthly_metrics.loc[
            monthly_metrics["signal_date"].isin(shared_dates)
        ].copy()

        failed_dates = monthly_metrics.loc[
            monthly_metrics["status"] != "success"
        ]

        if not failed_dates.empty:
            raise RuntimeError(
                f"{tested_factor_name} 在共同样本日期中存在检验失败。"
            )

        monthly_metrics.insert(0, "factor_name", tested_factor_name)
        monthly_metric_frames.append(monthly_metrics)

        summary_frames.append(
            pru.summarize_factor_evaluation(
                factor_name=tested_factor_name,
                monthly_metrics=monthly_metrics,
                periods=periods,
                periods_per_year=12,
            )
        )

    attribution_monthly = pd.concat(
        monthly_metric_frames,
        ignore_index=True,
    )
    attribution_summary = pd.concat(
        summary_frames,
        ignore_index=True,
    )

    monthly_path = (
        paths.table_dir
        / f"{output_prefix}_common_sample_component_attribution_monthly.csv"
    )
    summary_path = (
        paths.table_dir
        / f"{output_prefix}_common_sample_component_attribution_summary.csv"
    )
    audit_path = (
        paths.table_dir
        / f"{output_prefix}_common_sample_component_neutralization_audit.csv"
    )
    metadata_path = (
        paths.metadata_dir
        / f"{output_prefix}_common_sample_component_attribution_rules.json"
    )
    figure_path = (
        paths.figure_dir
        / f"{output_prefix}_common_sample_component_attribution.png"
    )

    pru.save_public_csv(
        attribution_monthly[
            [
                "factor_name",
                "signal_date",
                "paired_stock_count",
                "ic",
                "rank_ic",
                "q1_return",
                "q2_return",
                "q3_return",
                "q4_return",
                "q5_return",
                "high_minus_low_return",
                "status",
            ]
        ],
        monthly_path,
    )
    pru.save_public_csv(attribution_summary, summary_path)
    pru.save_public_csv(neutralization_audit, audit_path)

    pru.save_public_json(
        {
            "purpose": (
                "Explain the fixed four-component integrated profitability "
                "proxy on its strict common sample; not factor selection."
            ),
            "common_sample_definition": (
                "Research-universe stock-date with all four components "
                "available simultaneously."
            ),
            "comparison_treatment": (
                "Each component is re-neutralized within the common sample "
                "to remove conditional industry and market-cap exposure."
            ),
            "fixed_component_weights": {
                "stable_delevered_roe": 0.25,
                "stable_roic_proxy": 0.25,
                "stable_ronoa_proxy": 0.25,
                "fcffic_reconstructed": 0.25,
            },
            "prohibited_follow_up": [
                "No component removal",
                "No weight optimization",
                "No sample-out-of-sample date selection",
                "No conversion of a component diagnostic into a new factor",
            ],
            "individual_stock_level_data_saved_to_artifacts": False,
        },
        metadata_path,
    )

    oos_summary = attribution_summary.loc[
        attribution_summary["period"] == "out_of_sample_2023_2025"
    ].copy()

    preferred_order = [
        composite_factor_name,
        "stable_delevered_roe_on_common_sample",
        "stable_roic_proxy_on_common_sample",
        "stable_ronoa_proxy_on_common_sample",
        "fcffic_reconstructed_on_common_sample",
    ]

    oos_summary["factor_name"] = pd.Categorical(
        oos_summary["factor_name"],
        categories=preferred_order,
        ordered=True,
    )
    oos_summary = oos_summary.sort_values("factor_name")

    label_map = {
        composite_factor_name: "Composite (fixed equal weight)",
        "stable_delevered_roe_on_common_sample": "Stable delevered ROE",
        "stable_roic_proxy_on_common_sample": "Stable ROIC proxy",
        "stable_ronoa_proxy_on_common_sample": "Stable RONOA proxy",
        "fcffic_reconstructed_on_common_sample": "Reconstructed FCFFIC",
    }

    plot_labels = [
        label_map[name]
        for name in oos_summary["factor_name"].astype(str)
    ]
    plot_colors = [
        "#4C78A8"
        if name == composite_factor_name
        else "#72B7B2"
        for name in oos_summary["factor_name"].astype(str)
    ]

    pru.configure_matplotlib_style()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    axes[0].barh(
        plot_labels,
        oos_summary["mean_rank_ic"],
        color=plot_colors,
    )
    axes[0].axvline(0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_title("Common-Sample Out-of-Sample Mean RankIC")
    axes[0].set_xlabel("Mean RankIC")

    axes[1].barh(
        plot_labels,
        oos_summary["mean_high_minus_low_return"],
        color=plot_colors,
    )
    axes[1].axvline(0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_title("Common-Sample Out-of-Sample Q5 - Q1 Return")
    axes[1].set_xlabel("Mean 20-Day Gross Return")

    fig.tight_layout()
    fig.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.show()

    print("===== 共同可用样本组件归因汇总 =====")
    print(attribution_summary.round(4).to_string(index=False))
    print()
    print(
        "归因组件缓存：",
        "命中，直接读取"
        if cache_hit
        else "未命中，已构建并保存",
    )
    print("共同可检验日期数：", len(shared_dates))
    print("已保存：", summary_path)
    print("已保存：", figure_path)

    return {
        "summary": attribution_summary,
        "monthly_metrics": attribution_monthly,
        "neutralization_audit": neutralization_audit,
        "shared_dates": shared_dates,
        "paths": {
            "monthly": monthly_path,
            "summary": summary_path,
            "audit": audit_path,
            "metadata": metadata_path,
            "figure": figure_path,
        },
    }


def _read_optional_csv(path: Path) -> pd.DataFrame | None:
    """读取可选的公开 CSV；文件不存在时返回 None，不中断汇总。"""
    return pd.read_csv(path) if path.exists() else None


def _pick_first_existing_column(
    frame: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    """从候选列名中找到第一个实际存在的列。"""
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _as_numeric_column(
    frame: pd.DataFrame,
    candidates: list[str],
) -> pd.Series:
    """读取第一个存在的数值列，不存在则返回全 NaN。"""
    column = _pick_first_existing_column(frame, candidates)

    if column is None:
        return pd.Series(np.nan, index=frame.index, dtype=float)

    return pd.to_numeric(frame[column], errors="coerce")


def run_profitability_research_synthesis(
    project_root: str | Path,
    output_prefix: str = "05_final_research",
) -> dict[str, Any]:
    """
    汇总已经冻结的盈利因子研究证据。

    仅读取 artifacts/ 中的日期级汇总表与决策日志：
    - 不读取或导出个股级因子、收益、价格；
    - 不重新执行因子构建；
    - 不基于结果修改组件、权重或时间区间。
    """
    paths = pru.ProjectPaths.from_root(project_root)
    paths.ensure_directories()

    docs_dir = paths.root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    oos_period = "out_of_sample_2023_2025"

    # 每个来源都可缺失；项目不同阶段的历史结果不影响当前汇总运行。
    source_specs = [
        {
            "source_id": "historical_profitability_variants",
            "source_label": "早期 ROE 变体筛选",
            "filename": "02_profitability_variant_comparison.csv",
            "factor_columns": ["factor_variant", "factor_name"],
            "research_scope": "historical_profitability_screen",
        },
        {
            "source_id": "ttm_main_factor_pool",
            "source_label": "TTM 盈利因子主池",
            "filename": "05_ttm_factor_effectiveness_summary.csv",
            "factor_columns": ["factor_name"],
            "research_scope": "main_ttm_factor_pool",
        },
        {
            "source_id": "stable_report_components",
            "source_label": "稳定 ROIC/RONOA 组件代理",
            "filename": "05_stable_profitability_effectiveness_summary.csv",
            "factor_columns": ["factor_name"],
            "research_scope": "report_component_proxy_or_sensitivity",
        },
        {
            "source_id": "integrated_report_structure",
            "source_label": "四组件等权综合盈利代理",
            "filename": "05_integrated_profitability_effectiveness_summary.csv",
            "factor_columns": ["factor_name"],
            "research_scope": "low_coverage_structural_sensitivity",
        },
    ]

    # 优先采用已经生成的正式研究决策；没有正式决策时，只记录证据而不擅自确认。
    decision_filenames = [
        "02_profitability_research_decision_log.csv",
        "05_ttm_factor_research_decision_log.csv",
        "05_integrated_profitability_research_decision_log.csv",
    ]

    decision_records = []

    for filename in decision_filenames:
        decision_frame = _read_optional_csv(paths.table_dir / filename)

        if decision_frame is None:
            continue

        factor_column = _pick_first_existing_column(
            decision_frame,
            ["factor_name", "factor_variant"],
        )
        decision_column = _pick_first_existing_column(
            decision_frame,
            ["decision", "research_decision"],
        )

        if factor_column is None or decision_column is None:
            continue

        scope_column = _pick_first_existing_column(
            decision_frame,
            ["decision_scope", "scope"],
        )

        for _, row in decision_frame.iterrows():
            decision_records.append(
                {
                    "factor_key": str(row[factor_column]),
                    "formal_decision": str(row[decision_column]),
                    "formal_decision_scope": (
                        str(row[scope_column])
                        if scope_column is not None
                        and pd.notna(row[scope_column])
                        else ""
                    ),
                    "decision_source_file": filename,
                }
            )

    decision_lookup = (
        pd.DataFrame(decision_records)
        .drop_duplicates(subset=["factor_key"], keep="last")
        .set_index("factor_key")
        if decision_records
        else pd.DataFrame()
    )

    evidence_frames = []
    source_inventory_rows = []

    for spec in source_specs:
        source_path = paths.table_dir / spec["filename"]
        source_frame = _read_optional_csv(source_path)

        source_inventory_rows.append(
            {
                "source_id": spec["source_id"],
                "source_label": spec["source_label"],
                "source_file": str(source_path),
                "file_found": source_frame is not None,
            }
        )

        if source_frame is None or "period" not in source_frame.columns:
            continue

        factor_column = _pick_first_existing_column(
            source_frame,
            spec["factor_columns"],
        )

        if factor_column is None:
            continue

        selected = source_frame.loc[
            source_frame["period"] == oos_period
        ].copy()

        if selected.empty:
            continue

        normalized = pd.DataFrame(
            {
                "factor_key": selected[factor_column].astype(str),
                "source_id": spec["source_id"],
                "source_label": spec["source_label"],
                "research_scope": spec["research_scope"],
                "period": selected["period"].astype(str),
                "observations": _as_numeric_column(
                    selected,
                    ["observations"],
                ),
                "mean_paired_stock_count": _as_numeric_column(
                    selected,
                    [
                        "mean_paired_stock_count",
                        "average_valid_stocks",
                        "mean_valid_stock_count",
                    ],
                ),
                "min_paired_stock_count": _as_numeric_column(
                    selected,
                    [
                        "min_paired_stock_count",
                        "min_valid_stocks",
                        "min_valid_stock_count",
                    ],
                ),
                "mean_ic": _as_numeric_column(
                    selected,
                    ["mean_ic"],
                ),
                "icir_annualized": _as_numeric_column(
                    selected,
                    ["icir_annualized"],
                ),
                "mean_rank_ic": _as_numeric_column(
                    selected,
                    ["mean_rank_ic"],
                ),
                "rank_icir_annualized": _as_numeric_column(
                    selected,
                    ["rank_icir_annualized"],
                ),
                "positive_rank_ic_ratio": _as_numeric_column(
                    selected,
                    ["positive_rank_ic_ratio"],
                ),
                "mean_high_minus_low_return": _as_numeric_column(
                    selected,
                    ["mean_high_minus_low_return"],
                ),
                "high_minus_low_t_stat": _as_numeric_column(
                    selected,
                    ["high_minus_low_t_stat"],
                ),
            }
        )

        evidence_frames.append(normalized)

    if not evidence_frames:
        raise RuntimeError(
            "未找到任何可用于汇总的样本外结果表，请检查 artifacts/tables/。"
        )

    evidence_register = pd.concat(
        evidence_frames,
        ignore_index=True,
    )

    # 防止同一因子被多份来源重复纳入最终证据表。
    evidence_register = evidence_register.drop_duplicates(
        subset=["factor_key", "source_id", "period"],
        keep="last",
    )

    factor_label_map = {
        "roe_ttm_baseline": "原始 ROE",
        "delevered_roe": "去杠杆 ROE",
        "negative_roe_volatility": "盈利稳定性",
        "stable_delevered_roe": "稳定去杠杆 ROE",
        "roa_reconstructed_strict": "重构 ROA",
        "roa_vendor_reference": "米筐 ROA 参考字段",
        "roic_reconstructed_strict": "重构 ROIC",
        "roic_vendor_reference": "米筐 ROIC 参考字段",
        "fcffic_reconstructed_strict": "重构 FCFFIC",
        "stable_roic_ttm_proxy": "稳定 ROIC 代理",
        "stable_ronoa_ttm_proxy": "稳定 RONOA 代理",
        "integrated_profitability_structure_proxy_strict": (
            "四组件等权综合盈利代理"
        ),
    }

    role_map = {
        "negative_roe_volatility": "候选盈利稳定性组件",
        "stable_delevered_roe": "研报结构组件",
        "fcffic_reconstructed_strict": "候选现金流盈利组件",
        "stable_roic_ttm_proxy": "研报稳定 ROIC 组件代理",
        "stable_ronoa_ttm_proxy": "低覆盖率 RONOA 敏感性组件",
        "integrated_profitability_structure_proxy_strict": (
            "低覆盖率研报结构复现代理"
        ),
    }

    evidence_register["factor_label"] = evidence_register[
        "factor_key"
    ].map(factor_label_map).fillna(evidence_register["factor_key"])

    evidence_register["research_role"] = evidence_register[
        "factor_key"
    ].map(role_map).fillna("单因子或参考字段检验")

    if not decision_lookup.empty:
        evidence_register["formal_decision"] = evidence_register[
            "factor_key"
        ].map(decision_lookup["formal_decision"])

        evidence_register["formal_decision_scope"] = evidence_register[
            "factor_key"
        ].map(decision_lookup["formal_decision_scope"])

        evidence_register["decision_source_file"] = evidence_register[
            "factor_key"
        ].map(decision_lookup["decision_source_file"])
    else:
        evidence_register["formal_decision"] = np.nan
        evidence_register["formal_decision_scope"] = ""
        evidence_register["decision_source_file"] = ""

    # 对没有正式决策日志的“研报组件代理”只给范围结论，不把它们升级为候选 Alpha。
    default_position_map = {
        "stable_roic_ttm_proxy": "report_component_proxy_not_confirmed",
        "stable_ronoa_ttm_proxy": "low_coverage_sensitivity_only",
    }

    def final_position(row: pd.Series) -> str:
        if pd.notna(row["formal_decision"]):
            return str(row["formal_decision"])

        return default_position_map.get(
            row["factor_key"],
            "evidence_record_without_formal_decision",
        )

    evidence_register["final_research_position"] = evidence_register.apply(
        final_position,
        axis=1,
    )

    evidence_register = evidence_register[
        [
            "factor_key",
            "factor_label",
            "research_role",
            "research_scope",
            "period",
            "observations",
            "mean_paired_stock_count",
            "min_paired_stock_count",
            "mean_ic",
            "icir_annualized",
            "mean_rank_ic",
            "rank_icir_annualized",
            "positive_rank_ic_ratio",
            "mean_high_minus_low_return",
            "high_minus_low_t_stat",
            "formal_decision",
            "formal_decision_scope",
            "final_research_position",
            "source_label",
            "decision_source_file",
        ]
    ].sort_values(
        ["research_scope", "factor_label"],
    ).reset_index(drop=True)

    source_inventory = pd.DataFrame(source_inventory_rows)

    evidence_path = (
        paths.table_dir / f"{output_prefix}_evidence_register.csv"
    )
    inventory_path = (
        paths.table_dir / f"{output_prefix}_source_inventory.csv"
    )
    metadata_path = (
        paths.metadata_dir / f"{output_prefix}_synthesis_rules.json"
    )
    figure_path = (
        paths.figure_dir / f"{output_prefix}_oos_scorecard.png"
    )
    markdown_path = docs_dir / f"{output_prefix}_summary.md"

    pru.save_public_csv(evidence_register, evidence_path)
    pru.save_public_csv(source_inventory, inventory_path)

    pru.save_public_json(
        {
            "purpose": (
                "Frozen research-evidence synthesis. "
                "No factor construction or parameter optimization is performed."
            ),
            "out_of_sample_period": oos_period,
            "included_sources": source_inventory.to_dict(orient="records"),
            "decision_policy": (
                "Existing formal decision logs are retained. "
                "Factors without a formal decision log are not upgraded "
                "to confirmed candidates by this synthesis."
            ),
            "data_export_policy": (
                "Only summary statistics, decisions, and metadata are exported; "
                "individual-stock factor values, returns, and prices are excluded."
            ),
        },
        metadata_path,
    )

    # 图只展示样本外汇总证据，不生成新组合。
    plot_frame = evidence_register.copy()

    pru.configure_matplotlib_style()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    rank_colors = [
        "#54A24B" if value > 0 else "#E45756"
        for value in plot_frame["mean_rank_ic"].fillna(0)
    ]

    axes[0].barh(
        plot_frame["factor_label"],
        plot_frame["mean_rank_ic"],
        color=rank_colors,
    )
    axes[0].axvline(0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_title("Out-of-Sample Mean RankIC")
    axes[0].set_xlabel("Mean RankIC")

    hml_colors = [
        "#54A24B" if value > 0 else "#E45756"
        for value in plot_frame["mean_high_minus_low_return"].fillna(0)
    ]

    axes[1].barh(
        plot_frame["factor_label"],
        plot_frame["mean_high_minus_low_return"],
        color=hml_colors,
    )
    axes[1].axvline(0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_title("Out-of-Sample Mean Q5 - Q1 Return")
    axes[1].set_xlabel("Mean 20-Day Gross Return")

    fig.tight_layout()
    fig.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.show()

    decision_counts = (
        evidence_register["final_research_position"]
        .fillna("no_formal_decision")
        .value_counts()
        .to_dict()
    )

    markdown_lines = [
        "# 盈利能力因子研究：冻结证据汇总",
        "",
        "## 研究范围",
        "",
        "- 股票池：动态中证 800；",
        "- 收益标签：信号日后首个交易日开盘至 T+21 开盘，持有 20 个交易日；",
        "- 样本外区间：2023-01 至 2025-12；",
        "- 因子处理：MAD 去极值、CITIC 2019 一级行业和对数市值中性化、z-score；",
        "- 本文件仅汇总既有证据，不修改因子定义、组件、权重或样本外窗口。",
        "",
        "## 冻结结论",
        "",
    ]

    for decision_name, count in decision_counts.items():
        markdown_lines.append(f"- `{decision_name}`：{count} 个证据记录。")

    markdown_lines.extend(
        [
            "",
            "## 重要限制",
            "",
            "- 重构 FCFFIC、盈利稳定性及综合盈利代理均未达到独立 Alpha 的确认标准；",
            "- 稳定 RONOA 与四组件综合盈利代理受限于较低覆盖率，应仅作为结构复现或敏感性证据；",
            "- 共同样本组件归因仅用于解释固定四组件结构，不可用于事后删组件或优化权重；",
            "- 当前结果未纳入交易成本、涨跌停、停牌和组合执行约束。",
            "",
            "## 样本外证据表",
            "",
            "```text",
            evidence_register.round(4).to_string(index=False),
            "```",
        ]
    )

    markdown_path.write_text(
        "\n".join(markdown_lines),
        encoding="utf-8",
    )

    print("===== 最终研究证据汇总 =====")
    print(evidence_register.round(4).to_string(index=False))
    print()
    print("===== 数据来源检查 =====")
    print(source_inventory.to_string(index=False))
    print()
    print("已保存：", evidence_path)
    print("已保存：", inventory_path)
    print("已保存：", metadata_path)
    print("已保存：", figure_path)
    print("已生成 README 可引用草稿：", markdown_path)

    return {
        "evidence_register": evidence_register,
        "source_inventory": source_inventory,
        "paths": {
            "evidence_register": evidence_path,
            "source_inventory": inventory_path,
            "metadata": metadata_path,
            "figure": figure_path,
            "markdown": markdown_path,
        },
    }
