"""Prepare the source-weight composite profitability factor in Python 3.9.

This script uses only frozen project caches created by the existing research
pipeline.  It does not request new fundamental data and it does not modify any
05--10 artifact.  Individual-stock panels and target weights stay under data/
or cache/ and therefore remain outside the public release.
"""

from __future__ import print_function

import hashlib
import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/home/rice/notebook/a-share-profitability-factor-replication")
LIB_DIR = PROJECT_ROOT / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

import research_statistics as rs
import strategy_core as sc


TABLE_DIR = PROJECT_ROOT / "artifacts" / "tables"
FIGURE_DIR = PROJECT_ROOT / "artifacts" / "figures"
METADATA_DIR = PROJECT_ROOT / "artifacts" / "metadata"
CACHE_DATA_DIR = PROJECT_ROOT / "cache" / "data"
CACHE_META_DIR = PROJECT_ROOT / "cache" / "metadata"
INPUT_DIR = PROJECT_ROOT / "data" / "strategy_inputs"

for directory in [
    TABLE_DIR,
    FIGURE_DIR,
    METADATA_DIR,
    CACHE_DATA_DIR,
    CACHE_META_DIR,
    INPUT_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


SOURCE_WEIGHT_COMPONENTS = {
    "stable_delevered_roe": 1.0 / 6.0,
    "stable_roic_proxy": 1.0 / 6.0,
    "stable_ronoa_proxy": 1.0 / 6.0,
    "fcffic_reconstructed": 1.0 / 2.0,
}

COMPONENT_LABELS = {
    "stable_delevered_roe": "线性剥离杠杆后稳定的 ROE",
    "stable_roic_proxy": "稳定的 ROIC（项目代理）",
    "stable_ronoa_proxy": "稳定的 RONOA（项目代理）",
    "fcffic_reconstructed": "FCFFIC（项目重构）",
}

FACTOR_NAME = "composite_profitability_factor_source_weight_proxy"
MIN_STOCKS = 100
MAD_MULTIPLIER = 3.5
OOS_START = pd.Timestamp("2023-01-01")
OOS_END = pd.Timestamp("2025-12-31")


def _json_default(value):
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _save_json(payload, path):
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _configure_plot_style():
    """Use a compatible plotting style across old and new Matplotlib releases."""
    for style in ["seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot", "default"]:
        try:
            plt.style.use(style)
            return style
        except OSError:
            continue
    return "default"


def _safe_corr(left, right):
    frame = pd.concat(
        [pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce")],
        axis=1,
    ).dropna()
    if len(frame) < 3 or frame.iloc[:, 0].std(ddof=0) == 0 or frame.iloc[:, 1].std(ddof=0) == 0:
        return np.nan
    return float(frame.iloc[:, 0].corr(frame.iloc[:, 1]))


def _industry_dispersion(values, industry):
    frame = pd.DataFrame(
        {"value": pd.to_numeric(values, errors="coerce"), "industry": industry}
    ).dropna()
    group_means = frame.groupby("industry")["value"].mean()
    if len(group_means) < 2:
        return np.nan
    return float(group_means.std(ddof=0))


def _neutralize_cross_section(raw_factor, eligible_mask, log_market_cap, industry):
    """The same post-combination treatment used by the 25% comparison version."""
    output = pd.Series(np.nan, index=raw_factor.index, dtype=float)
    raw_factor = pd.to_numeric(raw_factor, errors="coerce")
    log_market_cap = pd.to_numeric(log_market_cap, errors="coerce")
    valid = (
        eligible_mask.fillna(False).astype(bool)
        & raw_factor.notna()
        & np.isfinite(raw_factor)
        & log_market_cap.notna()
        & np.isfinite(log_market_cap)
        & industry.notna()
    )
    audit = {
        "raw_valid_count": int(valid.sum()),
        "model_input_count": 0,
        "final_valid_count": 0,
        "industry_groups": 0,
        "raw_cap_corr": np.nan,
        "final_cap_corr": np.nan,
        "raw_industry_dispersion": np.nan,
        "final_industry_dispersion": np.nan,
        "status": "not_run",
        "failure_reason": "",
    }
    if audit["raw_valid_count"] < MIN_STOCKS:
        audit.update(
            status="insufficient_coverage",
            failure_reason="valid_stock_count_below_{}".format(MIN_STOCKS),
        )
        return output, audit

    factor = raw_factor.loc[valid].copy()
    cap = log_market_cap.loc[valid].copy()
    ind = industry.loc[valid].astype(str).copy()
    median = factor.median()
    robust_sigma = 1.4826 * (factor - median).abs().median()
    if not np.isfinite(robust_sigma) or robust_sigma <= 0:
        audit.update(status="failed", failure_reason="non_positive_or_invalid_mad")
        return output, audit

    factor = factor.clip(
        lower=median - MAD_MULTIPLIER * robust_sigma,
        upper=median + MAD_MULTIPLIER * robust_sigma,
    )
    dummies = pd.get_dummies(ind, prefix="industry").astype(float)
    if dummies.shape[1] > 1:
        dummies = dummies.iloc[:, 1:]
    design = pd.concat(
        [
            pd.Series(1.0, index=factor.index, name="const"),
            cap.rename("log_market_cap"),
            dummies,
        ],
        axis=1,
    ).astype(float)
    audit["model_input_count"] = int(len(factor))
    audit["industry_groups"] = int(ind.nunique())
    if len(factor) <= design.shape[1] + 5:
        audit.update(status="failed", failure_reason="insufficient_degrees_of_freedom")
        return output, audit

    try:
        beta = np.linalg.lstsq(
            design.to_numpy(dtype=float), factor.to_numpy(dtype=float), rcond=None
        )[0]
        residual = pd.Series(
            factor.to_numpy(dtype=float) - design.to_numpy(dtype=float).dot(beta),
            index=factor.index,
            dtype=float,
        )
        residual_std = residual.std(ddof=0)
        if not np.isfinite(residual_std) or residual_std <= 0:
            audit.update(status="failed", failure_reason="invalid_residual_standard_deviation")
            return output, audit
        standardized = (residual - residual.mean()) / residual_std
        output.loc[standardized.index] = standardized
        audit.update(
            final_valid_count=int(standardized.notna().sum()),
            raw_cap_corr=_safe_corr(factor, cap),
            final_cap_corr=_safe_corr(standardized, cap),
            raw_industry_dispersion=_industry_dispersion(factor, ind),
            final_industry_dispersion=_industry_dispersion(standardized, ind),
            status="success",
        )
    except Exception as exc:
        audit.update(status="failed", failure_reason="{}: {}".format(type(exc).__name__, exc))
    return output, audit


def _factor_summary(monthly_metrics, period_name, start, end):
    selected = monthly_metrics.loc[monthly_metrics["status"].eq("success")].copy()
    if start is not None:
        selected = selected.loc[selected["signal_date"] >= pd.Timestamp(start)]
    if end is not None:
        selected = selected.loc[selected["signal_date"] <= pd.Timestamp(end)]
    rank = rs.hac_mean_test(selected["rank_ic"], max_lags=1)
    spread = rs.hac_mean_test(selected["high_minus_low_return"], max_lags=1)
    ic = rs.hac_mean_test(selected["ic"], max_lags=1)
    return {
        "period": period_name,
        "observations": int(len(selected)),
        "positive_rank_ic_ratio": float((selected["rank_ic"] > 0).mean()) if len(selected) else np.nan,
        "ic_mean": ic["mean"],
        "ic_hac_t_stat": ic["hac_t_stat"],
        "ic_hac_p_value": ic["hac_p_value"],
        "rank_ic_mean": rank["mean"],
        "rank_ic_hac_t_stat": rank["hac_t_stat"],
        "rank_ic_hac_p_value": rank["hac_p_value"],
        "high_minus_low_return_mean": spread["mean"],
        "high_minus_low_return_hac_t_stat": spread["hac_t_stat"],
        "high_minus_low_return_hac_p_value": spread["hac_p_value"],
    }


def _evaluate_factor_panel(factor_panel, forward_return_panel, min_stocks, quantiles):
    """Monthly IC, RankIC and equal-weight quantile returns without new data access."""
    forward_return_panel = forward_return_panel.reindex(
        index=factor_panel.index, columns=factor_panel.columns
    )
    records = []
    for signal_date in factor_panel.index:
        paired = pd.DataFrame(
            {
                "factor": pd.to_numeric(factor_panel.loc[signal_date], errors="coerce"),
                "forward_return": pd.to_numeric(forward_return_panel.loc[signal_date], errors="coerce"),
            }
        ).dropna()
        record = {
            "signal_date": pd.Timestamp(signal_date),
            "paired_stock_count": int(len(paired)),
            "ic": np.nan,
            "rank_ic": np.nan,
            "high_minus_low_return": np.nan,
            "status": "insufficient_coverage",
        }
        for group in range(1, quantiles + 1):
            record["q{}_return".format(group)] = np.nan
        if len(paired) < min_stocks:
            records.append(record)
            continue
        if paired["factor"].std(ddof=0) == 0:
            record["status"] = "zero_factor_variance"
            records.append(record)
            continue
        record["ic"] = float(paired["factor"].corr(paired["forward_return"], method="pearson"))
        record["rank_ic"] = float(paired["factor"].corr(paired["forward_return"], method="spearman"))
        try:
            quantile = pd.qcut(paired["factor"].rank(method="first"), q=quantiles, labels=False) + 1
            group_returns = paired.groupby(quantile)["forward_return"].mean()
            for group in range(1, quantiles + 1):
                record["q{}_return".format(group)] = group_returns.get(group, np.nan)
            record["high_minus_low_return"] = record["q{}_return".format(quantiles)] - record["q1_return"]
            record["status"] = "success"
        except Exception as exc:
            record["status"] = "quintile_failure: {}".format(type(exc).__name__)
        records.append(record)
    return pd.DataFrame(records)


def _annualized_ir(values):
    series = pd.Series(values, dtype=float).dropna()
    if len(series) < 2 or series.std(ddof=1) == 0:
        return np.nan
    return float(series.mean() / series.std(ddof=1) * np.sqrt(12.0))


def _effectiveness_summary(factor_name, monthly_metrics, periods):
    rows = []
    for period_name, start, end in periods:
        selected = monthly_metrics.loc[monthly_metrics["status"].eq("success")].copy()
        if start is not None:
            selected = selected.loc[selected["signal_date"] >= pd.Timestamp(start)]
        if end is not None:
            selected = selected.loc[selected["signal_date"] <= pd.Timestamp(end)]
        rows.append(
            {
                "factor_name": factor_name,
                "period": period_name,
                "observations": int(len(selected)),
                "mean_paired_stock_count": float(selected["paired_stock_count"].mean()) if len(selected) else np.nan,
                "min_paired_stock_count": int(selected["paired_stock_count"].min()) if len(selected) else np.nan,
                "mean_ic": float(selected["ic"].mean()) if len(selected) else np.nan,
                "icir_annualized": _annualized_ir(selected["ic"]),
                "mean_rank_ic": float(selected["rank_ic"].mean()) if len(selected) else np.nan,
                "rank_icir_annualized": _annualized_ir(selected["rank_ic"]),
                "positive_rank_ic_ratio": float((selected["rank_ic"] > 0).mean()) if len(selected) else np.nan,
                "mean_q1_return": float(selected["q1_return"].mean()) if len(selected) else np.nan,
                "mean_q5_return": float(selected["q5_return"].mean()) if len(selected) else np.nan,
                "mean_high_minus_low_return": float(selected["high_minus_low_return"].mean()) if len(selected) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _write_portable_targets(portfolio_name, target_weights):
    payload = {
        "format": "rqalpha_target_weights_portable_v1",
        "portfolio_name": portfolio_name,
        "dates": pd.DatetimeIndex(target_weights.index).strftime("%Y-%m-%d").tolist(),
        "assets": target_weights.columns.astype(str).tolist(),
        "weights": target_weights.fillna(0.0).astype(float).to_numpy().tolist(),
    }
    path = INPUT_DIR / "11_source_weight_{}_target_weights_portable.pkl".format(portfolio_name)
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=4)
    return path


print("===== 11.1 研报原始权重版本：准备因子、检验与 RQAlpha 输入 =====")
component_cache = CACHE_DATA_DIR / "05_integrated_profitability_components.pkl"
context_cache = CACHE_DATA_DIR / "05_neutralization_context.pkl"
forward_return_cache = CACHE_DATA_DIR / "05_forward_return_20d.pkl"
schedule_path = TABLE_DIR / "05_return_schedule.csv"

for required in [component_cache, context_cache, forward_return_cache, schedule_path]:
    if not required.exists():
        raise FileNotFoundError("缺少冻结上游输入：{}".format(required))

component_panels = pd.read_pickle(component_cache)
context = pd.read_pickle(context_cache)
expected = set(SOURCE_WEIGHT_COMPONENTS)
if set(component_panels) != expected:
    raise RuntimeError("四个底层子因子缓存不匹配：{}".format(sorted(component_panels)))
if set(context) < {"research_mask", "industry_panel", "log_market_cap_panel"}:
    raise RuntimeError("中性化上下文缺少必要字段：{}".format(sorted(context)))
if not np.isclose(sum(SOURCE_WEIGHT_COMPONENTS.values()), 1.0):
    raise RuntimeError("研报权重之和不等于 1。")

signal_index = pd.DatetimeIndex(component_panels["stable_delevered_roe"].index).normalize()
stock_columns = component_panels["stable_delevered_roe"].columns
for name, panel in component_panels.items():
    if not isinstance(panel, pd.DataFrame) or not panel.index.equals(component_panels["stable_delevered_roe"].index) or not panel.columns.equals(stock_columns):
        raise RuntimeError("底层子因子维度不一致：{}".format(name))

components = {
    key: panel.reindex(index=signal_index, columns=stock_columns).astype(float)
    for key, panel in component_panels.items()
}
research_mask = context["research_mask"].reindex(index=signal_index, columns=stock_columns).fillna(False).astype(bool)
industry = context["industry_panel"].reindex(index=signal_index, columns=stock_columns)
log_cap = context["log_market_cap_panel"].reindex(index=signal_index, columns=stock_columns).astype(float)

all_available = research_mask.copy()
for panel in components.values():
    all_available &= np.isfinite(panel)

profit_statement_aggregate = (
    components["stable_delevered_roe"]
    + components["stable_roic_proxy"]
    + components["stable_ronoa_proxy"]
) / 3.0
two_stage_score = 0.5 * profit_statement_aggregate + 0.5 * components["fcffic_reconstructed"]
expanded_score = sum(
    SOURCE_WEIGHT_COMPONENTS[name] * components[name]
    for name in SOURCE_WEIGHT_COMPONENTS
)
two_stage_score = two_stage_score.where(all_available)
expanded_score = expanded_score.where(all_available)

comparison_mask = np.isfinite(two_stage_score.to_numpy()) & np.isfinite(expanded_score.to_numpy())
if not comparison_mask.any() or not np.allclose(
    two_stage_score.to_numpy()[comparison_mask],
    expanded_score.to_numpy()[comparison_mask],
    rtol=1e-12,
    atol=1e-12,
):
    raise RuntimeError("两层公式与 1/6、1/6、1/6、1/2 展开公式不一致。")

final_factor = pd.DataFrame(np.nan, index=signal_index, columns=stock_columns, dtype=float)
construction_rows = []
for signal_date in signal_index:
    values, audit = _neutralize_cross_section(
        two_stage_score.loc[signal_date],
        research_mask.loc[signal_date],
        log_cap.loc[signal_date],
        industry.loc[signal_date],
    )
    final_factor.loc[signal_date, values.index] = values
    audit.update(
        signal_date=pd.Timestamp(signal_date),
        all_four_components_available_count=int(all_available.loc[signal_date].sum()),
    )
    construction_rows.append(audit)
construction_audit = pd.DataFrame(construction_rows)

# The four frozen subfactor panels and neutralization context become jointly
# usable only from 2019-08 in this project.  Earlier rows are retained in the
# audit table as unavailable history, but they are not part of the effective
# research sample and must not block construction of the frozen available
# sample.  Once availability begins, a failed month is a real pipeline error.
successful_construction = construction_audit.loc[
    construction_audit["status"].eq("success"), "signal_date"
]
if successful_construction.empty:
    raise RuntimeError("没有任何日期成功构建综合盈利因子。请检查四个子因子和中性化上下文缓存。")

first_available_signal_date = pd.Timestamp(successful_construction.min())
failed_after_available = construction_audit.loc[
    construction_audit["signal_date"].ge(first_available_signal_date)
    & construction_audit["status"].ne("success"),
    ["signal_date", "status", "failure_reason"],
]
if not failed_after_available.empty:
    raise RuntimeError(
        "可用样本开始后存在未成功构建的日期：\n{}".format(
            failed_after_available.to_string(index=False)
        )
    )

factor_cache = CACHE_DATA_DIR / "11_source_weight_composite_factor.pkl"
pd.to_pickle(
    {
        "factor_name": FACTOR_NAME,
        "pre_neutralization": two_stage_score,
        "final_factor": final_factor,
        "construction_audit": construction_audit,
    },
    factor_cache,
)

forward_returns = pd.read_pickle(forward_return_cache).reindex(index=signal_index, columns=stock_columns)
monthly_metrics = _evaluate_factor_panel(
    factor_panel=final_factor,
    forward_return_panel=forward_returns,
    min_stocks=MIN_STOCKS,
    quantiles=5,
)
monthly_metrics["signal_date"] = pd.to_datetime(monthly_metrics["signal_date"])
successful = monthly_metrics.loc[monthly_metrics["status"].eq("success")]
if successful.empty:
    raise RuntimeError("没有任何日期达到因子检验的最低覆盖门槛。")
first_date = successful["signal_date"].min()
last_date = successful["signal_date"].max()
periods = [
    ("all_available_{}_{}".format(first_date.year, last_date.year), first_date, last_date),
    ("in_sample_available_{}_2022".format(first_date.year), first_date, "2022-12-31"),
    ("out_of_sample_2023_2025", "2023-01-01", "2025-12-31"),
    ("out_of_sample_2023", "2023-01-01", "2023-12-31"),
    ("out_of_sample_2024", "2024-01-01", "2024-12-31"),
    ("out_of_sample_2025", "2025-01-01", "2025-12-31"),
]
hac_stability = pd.DataFrame([_factor_summary(monthly_metrics, *period) for period in periods])
effectiveness = _effectiveness_summary(FACTOR_NAME, monthly_metrics, periods[:3])

weights_table = pd.DataFrame(
    [
        {
            "component_key": key,
            "report_term": COMPONENT_LABELS[key],
            "source_weight": value,
            "source_weight_pct": value * 100.0,
            "equal_weight_comparator": 0.25,
        }
        for key, value in SOURCE_WEIGHT_COMPONENTS.items()
    ]
)
monthly_metrics.to_csv(TABLE_DIR / "11_source_weight_monthly_factor_metrics.csv", index=False, encoding="utf-8-sig")
effectiveness.to_csv(TABLE_DIR / "11_source_weight_factor_effectiveness_summary.csv", index=False, encoding="utf-8-sig")
hac_stability.to_csv(TABLE_DIR / "11_source_weight_hac_stability_summary.csv", index=False, encoding="utf-8-sig")
construction_audit.to_csv(TABLE_DIR / "11_source_weight_factor_construction_audit.csv", index=False, encoding="utf-8-sig")
weights_table.to_csv(TABLE_DIR / "11_source_weight_component_weights.csv", index=False, encoding="utf-8-sig")

print("绘图主题：", _configure_plot_style())
fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
axes[0].plot(successful["signal_date"], successful["ic"], label="IC", linewidth=1.1)
axes[0].plot(successful["signal_date"], successful["rank_ic"], label="RankIC", linewidth=1.1)
axes[0].axhline(0.0, color="black", linewidth=0.8)
axes[0].set_title("Source-weight composite: monthly IC and RankIC")
axes[0].legend()
quintile_nav = pd.DataFrame(
    {"Q{}".format(i): (1.0 + successful["q{}_return".format(i)]).cumprod() for i in range(1, 6)},
    index=successful["signal_date"],
)
quintile_nav.plot(ax=axes[1], linewidth=1.1)
axes[1].set_title("Source-weight composite: quintile NAV")
axes[1].set_ylabel("Normalized NAV")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "11_source_weight_factor_effectiveness.png", dpi=160, bbox_inches="tight")
plt.close(fig)

yearly = hac_stability.loc[hac_stability["period"].str.startswith("out_of_sample_20")].copy()
fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
axes[0].bar(yearly["period"], yearly["rank_ic_mean"])
axes[0].axhline(0.0, color="black", linewidth=0.8)
axes[0].set_title("Out-of-sample annual RankIC")
axes[0].tick_params(axis="x", rotation=25)
axes[1].bar(yearly["period"], yearly["high_minus_low_return_mean"])
axes[1].axhline(0.0, color="black", linewidth=0.8)
axes[1].set_title("Out-of-sample annual Q5-Q1 return")
axes[1].tick_params(axis="x", rotation=25)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "11_source_weight_oos_yearly_stability.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# Build the same OOS sleeves used by the earlier RQAlpha experiment.
schedule = pd.read_csv(schedule_path)
for column in ["signal_date", "entry_date_t_plus_1_open", "exit_date_t_plus_21_open"]:
    schedule[column] = pd.to_datetime(schedule[column]).dt.normalize()
schedule = schedule.loc[
    schedule["status"].astype(str).str.lower().eq("ok")
    & schedule["signal_date"].between(OOS_START, OOS_END)
].copy()
if len(schedule) != 36:
    raise RuntimeError("严格样本外应有 36 个信号期，当前为 {}。".format(len(schedule)))

q5_sleeves, q5_selection = sc.build_quantile_target_weights(
    score_panel=final_factor,
    schedule=schedule,
    signal_column="signal_date",
    entry_column="entry_date_t_plus_1_open",
    selection_fraction=0.20,
    min_valid_assets=MIN_STOCKS,
    higher_is_better=True,
)
q1_sleeves, q1_selection = sc.build_quantile_target_weights(
    score_panel=final_factor,
    schedule=schedule,
    signal_column="signal_date",
    entry_column="entry_date_t_plus_1_open",
    selection_fraction=0.20,
    min_valid_assets=MIN_STOCKS,
    higher_is_better=False,
)
universe_sleeves, universe_selection = sc.build_quantile_target_weights(
    score_panel=final_factor,
    schedule=schedule,
    signal_column="signal_date",
    entry_column="entry_date_t_plus_1_open",
    selection_fraction=1.0,
    min_valid_assets=MIN_STOCKS,
    higher_is_better=True,
)
for name, audit in [
    ("Q5_factor_high", q5_selection),
    ("Q1_factor_low", q1_selection),
    ("Equal_weight_eligible_universe", universe_selection),
]:
    if not audit["eligible"].all():
        raise RuntimeError("{} 出现不可交易信号期。".format(name))

try:
    import rqdatac
except ImportError:
    raise ImportError("需在含 rqdatac 的 RiceQuant Python 3.9 内核中运行 11.1。")
trading_calendar = pd.DatetimeIndex(
    rqdatac.get_trading_dates(
        schedule["entry_date_t_plus_1_open"].min().strftime("%Y-%m-%d"),
        schedule["exit_date_t_plus_21_open"].max().strftime("%Y-%m-%d"),
    )
).normalize()

portfolio_targets = {}
portfolio_audits = {}
for name, sleeves in [
    ("q5_factor_high", q5_sleeves),
    ("q1_factor_low", q1_sleeves),
    ("equal_weight_eligible_universe", universe_sleeves),
]:
    targets, sleeve_audit = sc.build_active_sleeve_event_targets(
        sleeve_weights=sleeves,
        schedule=schedule,
        entry_column="entry_date_t_plus_1_open",
        exit_column="exit_date_t_plus_21_open",
        trading_calendar=trading_calendar,
    )
    if (targets.sum(axis=1) > 1.0 + 1e-10).any():
        raise RuntimeError("{} 出现超过 100% 的目标总权重。".format(name))
    portfolio_targets[name] = targets
    portfolio_audits[name] = sleeve_audit

selection_audit = pd.concat(
    [
        q5_selection.assign(portfolio="Q5_factor_high"),
        q1_selection.assign(portfolio="Q1_factor_low"),
        universe_selection.assign(portfolio="Equal_weight_eligible_universe"),
    ],
    ignore_index=True,
)
event_audit = pd.concat(
    [audit.assign(portfolio=name) for name, audit in portfolio_audits.items()],
    ignore_index=True,
)
selection_audit.to_csv(TABLE_DIR / "11_source_weight_oos_selection_audit.csv", index=False, encoding="utf-8-sig")
event_audit.to_csv(TABLE_DIR / "11_source_weight_event_target_audit.csv", index=False, encoding="utf-8-sig")

input_paths = {name: _write_portable_targets(name, panel) for name, panel in portfolio_targets.items()}

rules = {
    "experiment": "source_weight_composite_profitability_factor_replication",
    "factor_name": FACTOR_NAME,
    "research_boundary": (
        "An extension replication run after the original equal-weight out-of-sample "
        "results were observed. It reuses the project reconstruction/proxy inputs and "
        "is not asserted to be a full raw-field Wind replication."
    ),
    "source_two_stage_formula": {
        "profit_statement_aggregate": "(stable_delevered_roe + stable_roic_proxy + stable_ronoa_proxy) / 3",
        "composite_profitability_factor": "0.5 * profit_statement_aggregate + 0.5 * fcffic_reconstructed",
    },
    "expanded_component_weights": SOURCE_WEIGHT_COMPONENTS,
    "comparison_version_component_weights": {key: 0.25 for key in SOURCE_WEIGHT_COMPONENTS},
    "component_missing_value_rule": "strict intersection; no stock-level imputation",
    "post_combination_treatment": (
        "Preserve the existing one-pass combined-factor treatment: MAD winsorization, "
        "CITIC 2019 level-1 industry and log-market-cap neutralization, then z-score. "
        "No additional processing step is introduced relative to the comparison version."
    ),
    "return_label": "T+1 open to T+21 open; 20 trading-day holding return",
    "oos_signal_period": "2023-01-01 to 2025-12-31",
    "rqalpha_execution": {
        "initial_cash": 100000000,
        "benchmark": "000906.XSHG",
        "matching_type": "current_bar",
        "minimum_commission": 5.0,
        "stock_commission_multiplier": 0.375,
        "tax_multiplier": 1.0,
        "price_limit": True,
        "volume_limit": True,
        "volume_percent": 0.05,
    },
    "public_export_excludes": ["raw market data", "cache", "individual-stock target weights", "pkl", "credentials"],
}
_save_json(rules, METADATA_DIR / "11_source_weight_replication_rules.json")

print("因子本地缓存：", factor_cache)
print(
    "有效因子样本：{} 至 {}；早期覆盖不足月份：{}。".format(
        first_available_signal_date.strftime("%Y-%m-%d"),
        pd.Timestamp(successful["signal_date"].max()).strftime("%Y-%m-%d"),
        int(construction_audit["signal_date"].lt(first_available_signal_date).sum()),
    )
)
print("RQAlpha 私有输入：")
for name, path in input_paths.items():
    print("  {}: {}".format(name, path))
print("公开因子结果：", TABLE_DIR / "11_source_weight_factor_effectiveness_summary.csv")
print("公开 HAC 结果：", TABLE_DIR / "11_source_weight_hac_stability_summary.csv")
print("下一步：切换至 Python 3.6 RQAlpha 内核，运行 scripts/11_source_weight_rqalpha.py。")
