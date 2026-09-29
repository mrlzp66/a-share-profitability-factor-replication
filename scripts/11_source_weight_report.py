"""Export public aggregate results for the source-weight replication in Python 3.9.

This script reads the protocol-4, JSON-shaped private RQAlpha result files
written by scripts/11_source_weight_rqalpha.py.  It exports only aggregate
tables, charts and rules; individual-stock weights, trades and raw outputs
remain under data/ and are not copied into artifacts/.
"""

from __future__ import print_function

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/home/rice/notebook/a-share-profitability-factor-replication")
TABLE_DIR = PROJECT_ROOT / "artifacts" / "tables"
FIGURE_DIR = PROJECT_ROOT / "artifacts" / "figures"
METADATA_DIR = PROJECT_ROOT / "artifacts" / "metadata"
BACKTEST_DIR = PROJECT_ROOT / "data" / "backtests"
for directory in [TABLE_DIR, FIGURE_DIR, METADATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

PORTFOLIOS = {
    "q5_factor_high": "Q5_factor_high",
    "q1_factor_low": "Q1_factor_low",
    "equal_weight_eligible_universe": "Equal_weight_eligible_universe",
}


def configure_plot_style():
    """Use a compatible plotting style across old and new Matplotlib releases."""
    for style in ["seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot", "default"]:
        try:
            plt.style.use(style)
            return style
        except OSError:
            continue
    return "default"


def load_result(portfolio_key, forced_exit):
    suffix = "forced_exit_20d" if forced_exit else "base"
    path = BACKTEST_DIR / "11_source_weight_{}_{}_portable_result.pkl".format(portfolio_key, suffix)
    if not path.exists():
        raise FileNotFoundError("缺少 RQAlpha 私有可移植结果：{}".format(path))
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("format") != "rqalpha_analyser_portable_v2":
        raise RuntimeError("可移植结果格式不正确：{}".format(path))
    return payload


def split_to_frame(payload):
    frame = pd.DataFrame(payload["data"], columns=payload["columns"])
    date_column = next(
        (column for column in frame.columns if column in ["__index__", "index", "date", "datetime"]),
        frame.columns[0],
    )
    dates = pd.to_datetime(frame.pop(date_column), utc=True, errors="coerce")
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_convert(None)
    frame.index = pd.DatetimeIndex(dates).normalize()
    return frame.loc[~frame.index.duplicated(keep="last")].sort_index()


def numeric_metric(summary, key):
    value = summary.get(key, np.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def nav_from_result(result, section):
    frame = split_to_frame(result[section])
    if "unit_net_value" not in frame:
        raise RuntimeError("{} 中缺少 unit_net_value。".format(section))
    return pd.to_numeric(frame["unit_net_value"], errors="coerce").dropna()


def trades_to_frame(result):
    trades = pd.DataFrame(result.get("trades", []))
    if trades.empty:
        return trades
    date_column = next(
        (column for column in ["datetime", "trading_datetime", "date"] if column in trades.columns),
        None,
    )
    if date_column is not None:
        dates = pd.to_datetime(trades[date_column], utc=True, errors="coerce")
        if getattr(dates.dt, "tz", None) is not None:
            dates = dates.dt.tz_convert(None)
        trades["trade_date"] = dates.dt.normalize()
    return trades


def execution_summary(result):
    trades = trades_to_frame(result)
    if trades.empty:
        return {
            "trade_record_count": 0,
            "trade_day_count": 0,
            "total_filled_notional": 0.0,
            "total_commission": 0.0,
            "total_tax": 0.0,
            "total_transaction_cost": 0.0,
        }
    def numeric_column(candidates):
        for column in candidates:
            if column in trades.columns:
                return pd.to_numeric(trades[column], errors="coerce").fillna(0.0)
        return pd.Series(0.0, index=trades.index)

    quantity = numeric_column(["last_quantity", "filled_quantity"])
    price = numeric_column(["last_price", "price"])
    commission = numeric_column(["commission"])
    tax = numeric_column(["tax"])
    return {
        "trade_record_count": int(len(trades)),
        "trade_day_count": int(trades.get("trade_date", pd.Series(dtype="datetime64[ns]")).nunique()),
        "total_filled_notional": float((quantity.abs() * price.abs()).sum()),
        "total_commission": float(commission.sum()),
        "total_tax": float(tax.sum()),
        "total_transaction_cost": float(commission.sum() + tax.sum()),
    }


def final_account_values(result):
    account = split_to_frame(result["stock_account"])
    last = account.iloc[-1]
    market_value = numeric_value(last, ["market_value", "total_market_value"])
    cash = numeric_value(last, ["cash", "total_cash"])
    return market_value, cash


def numeric_value(row, candidates):
    for column in candidates:
        if column in row.index:
            try:
                return float(row[column])
            except (TypeError, ValueError):
                pass
    return np.nan


def ratio_return(numerator, denominator):
    aligned = pd.concat([numerator.rename("numerator"), denominator.rename("denominator")], axis=1).dropna()
    if len(aligned) < 2:
        raise RuntimeError("净值交集少于两个交易日，不能计算相对收益。")
    relative_nav = aligned["numerator"] / aligned["denominator"]
    return relative_nav, float(relative_nav.iloc[-1] / relative_nav.iloc[0] - 1.0)


print("===== 11.3 研报原始权重版本：轻量聚合导出与结论判定 =====")
# Read one private result at a time.  The browser-hosted RiceQuant session can
# become unresponsive if all analyser payloads (especially transaction lists)
# are held in memory simultaneously.
base_nav = {}
base_benchmark_nav = {}
base_summaries = {}
base_execution = {}
for key in PORTFOLIOS:
    print("读取基础组合：", key)
    result = load_result(key, forced_exit=False)
    base_nav[key] = nav_from_result(result, "portfolio")
    base_benchmark_nav[key] = nav_from_result(result, "benchmark_portfolio")
    base_summaries[key] = result["summary"]
    base_execution[key] = execution_summary(result)
    del result

performance_rows = []
execution_rows = []
for key, display_name in PORTFOLIOS.items():
    summary = base_summaries[key]
    relative_to_benchmark, relative_to_benchmark_return = ratio_return(base_nav[key], base_benchmark_nav[key])
    performance_rows.append(
        {
            "portfolio": display_name,
            "total_return": numeric_metric(summary, "total_returns"),
            "annualized_return": numeric_metric(summary, "annualized_returns"),
            "benchmark_total_return": numeric_metric(summary, "benchmark_total_returns"),
            "benchmark_annualized_return": numeric_metric(summary, "benchmark_annualized_returns"),
            "annualized_excess_return": numeric_metric(summary, "annualized_excess_returns"),
            "sharpe_ratio": numeric_metric(summary, "sharpe"),
            "information_ratio": numeric_metric(summary, "information_ratio"),
            "max_drawdown": numeric_metric(summary, "max_drawdown"),
            "relative_total_return_to_benchmark_from_daily_nav": relative_to_benchmark_return,
        }
    )
    execution_rows.append(dict({"portfolio": display_name}, **base_execution[key]))
performance = pd.DataFrame(performance_rows)
execution = pd.DataFrame(execution_rows)

base_daily_nav = pd.concat(
    {PORTFOLIOS[key]: nav for key, nav in base_nav.items()}, axis=1
).dropna()
base_daily_nav.index.name = "date"
base_relative_rows = []
for control_key in ["q1_factor_low", "equal_weight_eligible_universe"]:
    relative_nav, relative_return = ratio_return(base_nav["q5_factor_high"], base_nav[control_key])
    annualized = (relative_nav.iloc[-1] / relative_nav.iloc[0]) ** (252.0 / len(relative_nav)) - 1.0
    base_relative_rows.append(
        {
            "comparison": "Q5_factor_high_vs_{}".format(PORTFOLIOS[control_key]),
            "relative_total_return_from_daily_nav": relative_return,
            "annualized_relative_return_from_daily_nav": float(annualized),
            "daily_observation_count": int(len(relative_nav)),
        }
    )
base_relative = pd.DataFrame(base_relative_rows)

# Forced-exit sensitivity: use the same original final target date for all portfolios.
original_end_dates = {nav.index.max() for nav in base_nav.values()}
if len(original_end_dates) != 1:
    raise RuntimeError("基础三组合的结束日期不一致。")
original_end = next(iter(original_end_dates))
forced_nav = {}
forced_final_values = {}
forced_post_execution = {}
for key in PORTFOLIOS:
    print("读取强制清仓组合：", key)
    result = load_result(key, forced_exit=True)
    forced_nav[key] = nav_from_result(result, "portfolio")
    forced_final_values[key] = final_account_values(result)
    trades = trades_to_frame(result)
    if not trades.empty and "trade_date" in trades.columns:
        trades = trades.loc[trades["trade_date"] > original_end].copy()
    else:
        trades = trades.iloc[0:0].copy()
    forced_post_execution[key] = execution_summary(
        {"trades": trades.to_dict(orient="records")}
    )
    del trades
    del result

forced_rows = []
for key, display_name in PORTFOLIOS.items():
    forced_series = forced_nav[key]
    at_original = forced_series.loc[forced_series.index <= original_end]
    if at_original.empty:
        raise RuntimeError("{} 强制清仓序列缺少原始结束日。".format(display_name))
    nav_at_original_end = at_original.iloc[-1]
    market_value, final_cash = forced_final_values[key]
    post_execution = forced_post_execution[key]
    forced_rows.append(
        {
            "portfolio": display_name,
            "original_backtest_end_date": original_end.strftime("%Y-%m-%d"),
            "forced_exit_backtest_end_date": forced_series.index.max().strftime("%Y-%m-%d"),
            "original_total_return_from_daily_nav": float(base_nav[key].iloc[-1] / base_nav[key].iloc[0] - 1.0),
            "total_return_at_original_end_from_daily_nav": float(nav_at_original_end / forced_series.iloc[0] - 1.0),
            "forced_exit_final_total_return_from_daily_nav": float(forced_series.iloc[-1] / forced_series.iloc[0] - 1.0),
            "post_exit_tail_return": float(forced_series.iloc[-1] / nav_at_original_end - 1.0),
            "final_market_value": market_value,
            "final_cash": final_cash,
            "fully_liquidated_at_forced_exit_end": bool(np.isfinite(market_value) and abs(market_value) <= 1e-6),
            "post_exit_trade_record_count": post_execution["trade_record_count"],
            "post_exit_filled_notional": post_execution["total_filled_notional"],
            "post_exit_transaction_cost": post_execution["total_transaction_cost"],
        }
    )
forced_performance = pd.DataFrame(forced_rows)
if not forced_performance["fully_liquidated_at_forced_exit_end"].all():
    raise RuntimeError("至少一个组合在 20 个追加交易日后仍有市值，不能形成正式结论。")

forced_relative_rows = []
for control_key in ["q1_factor_low", "equal_weight_eligible_universe"]:
    forced_relative_nav, forced_relative_return = ratio_return(forced_nav["q5_factor_high"], forced_nav[control_key])
    original_relative_nav, original_relative_return = ratio_return(base_nav["q5_factor_high"], base_nav[control_key])
    forced_relative_rows.append(
        {
            "comparison": "Q5_factor_high_vs_{}".format(PORTFOLIOS[control_key]),
            "relative_return_at_original_end_from_daily_nav": original_relative_return,
            "relative_return_at_forced_exit_end_from_daily_nav": forced_relative_return,
            "relative_return_change_during_exit_tail": forced_relative_return - original_relative_return,
            "forced_exit_daily_observation_count": int(len(forced_relative_nav)),
        }
    )
forced_relative = pd.DataFrame(forced_relative_rows)

hac_path = TABLE_DIR / "11_source_weight_hac_stability_summary.csv"
if not hac_path.exists():
    raise FileNotFoundError("缺少 11.1 生成的 HAC 表：{}".format(hac_path))
hac = pd.read_csv(hac_path)
oos = hac.loc[hac["period"].eq("out_of_sample_2023_2025")]
if len(oos) != 1:
    raise RuntimeError("HAC 汇总中无法唯一定位严格样本外行。")
oos = oos.iloc[0]
q5_vs_equal = forced_relative.loc[
    forced_relative["comparison"].eq("Q5_factor_high_vs_Equal_weight_eligible_universe"),
    "relative_return_at_forced_exit_end_from_daily_nav",
].iloc[0]
rankic_pass = bool(oos["rank_ic_mean"] > 0 and oos["rank_ic_hac_p_value"] < 0.05)
spread_pass = bool(oos["high_minus_low_return_mean"] > 0 and oos["high_minus_low_return_hac_p_value"] < 0.05)
incremental_pass = bool(q5_vs_equal > 0)
final_status = "independent_alpha_supported" if all([rankic_pass, spread_pass, incremental_pass]) else "candidate_not_confirmed"

scorecard = pd.DataFrame(
    [
        {"metric": "oos_rank_ic_mean", "value": oos["rank_ic_mean"]},
        {"metric": "oos_rank_ic_hac_p_value", "value": oos["rank_ic_hac_p_value"]},
        {"metric": "oos_q5_minus_q1_mean_return", "value": oos["high_minus_low_return_mean"]},
        {"metric": "oos_q5_minus_q1_hac_p_value", "value": oos["high_minus_low_return_hac_p_value"]},
        {"metric": "q5_vs_q1_relative_return_after_forced_exit", "value": forced_relative.iloc[0]["relative_return_at_forced_exit_end_from_daily_nav"]},
        {"metric": "q5_vs_equal_weight_relative_return_after_forced_exit", "value": q5_vs_equal},
        {"metric": "all_portfolios_fully_liquidated", "value": bool(forced_performance["fully_liquidated_at_forced_exit_end"].all())},
        {"metric": "final_research_decision", "value": final_status},
    ]
)
decision_log = pd.DataFrame(
    [
        {
            "evidence": "严格样本外 RankIC HAC 检验",
            "finding": "RankIC={:.6f}, HAC p={:.4f}".format(oos["rank_ic_mean"], oos["rank_ic_hac_p_value"]),
            "decision_implication": "通过" if rankic_pass else "未通过预设的方向与 5% 显著性联合标准",
        },
        {
            "evidence": "严格样本外 Q5-Q1 HAC 检验",
            "finding": "Q5-Q1={:.6f}, HAC p={:.4f}".format(oos["high_minus_low_return_mean"], oos["high_minus_low_return_hac_p_value"]),
            "decision_implication": "通过" if spread_pass else "未通过预设的方向与 5% 显著性联合标准",
        },
        {
            "evidence": "RQAlpha 强制清仓后 Q5 相对等权可投资股票池",
            "finding": "相对收益={:.4%}".format(q5_vs_equal),
            "decision_implication": "通过" if incremental_pass else "未超过可投资股票池等权基准组合",
        },
        {
            "evidence": "最终研究状态",
            "finding": final_status,
            "decision_implication": "本次为样本外结果已被观察后的权重口径扩展复现，不作为首次确认性样本外检验。",
        },
    ]
)

performance.to_csv(TABLE_DIR / "11_source_weight_oos_three_portfolio_performance.csv", index=False, encoding="utf-8-sig")
execution.to_csv(TABLE_DIR / "11_source_weight_oos_execution_summary.csv", index=False, encoding="utf-8-sig")
base_daily_nav.reset_index().to_csv(TABLE_DIR / "11_source_weight_oos_three_portfolio_daily_nav.csv", index=False, encoding="utf-8-sig")
base_relative.to_csv(TABLE_DIR / "11_source_weight_oos_q5_relative_comparison.csv", index=False, encoding="utf-8-sig")
forced_performance.to_csv(TABLE_DIR / "11_source_weight_forced_exit_sensitivity_performance.csv", index=False, encoding="utf-8-sig")
forced_relative.to_csv(TABLE_DIR / "11_source_weight_forced_exit_relative_comparison.csv", index=False, encoding="utf-8-sig")
scorecard.to_csv(TABLE_DIR / "11_source_weight_final_research_scorecard.csv", index=False, encoding="utf-8-sig")
decision_log.to_csv(TABLE_DIR / "11_source_weight_final_research_decision_log.csv", index=False, encoding="utf-8-sig")

print("绘图主题：", configure_plot_style())
fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
base_daily_nav.plot(ax=axes[0], linewidth=1.15)
axes[0].set_title("Source-weight RQAlpha: out-of-sample NAV")
axes[0].set_ylabel("Normalized NAV")
relative_q1, _ = ratio_return(base_nav["q5_factor_high"], base_nav["q1_factor_low"])
relative_universe, _ = ratio_return(base_nav["q5_factor_high"], base_nav["equal_weight_eligible_universe"])
axes[1].plot(relative_q1, label="Q5 / Q1", linewidth=1.15)
axes[1].plot(relative_universe, label="Q5 / equal-weight eligible universe", linewidth=1.15)
axes[1].axhline(1.0, color="black", linestyle="--", linewidth=0.8)
axes[1].set_title("Source-weight RQAlpha: relative NAV")
axes[1].legend()
fig.tight_layout()
fig.savefig(FIGURE_DIR / "11_source_weight_oos_three_portfolio_comparison.png", dpi=160, bbox_inches="tight")
plt.close(fig)

forced_nav_frame = pd.concat({PORTFOLIOS[key]: nav for key, nav in forced_nav.items()}, axis=1).dropna()
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
forced_nav_frame.plot(ax=axes[0], linewidth=1.1)
axes[0].axvline(original_end, linestyle="--", color="black", linewidth=0.8, label="Original end")
axes[0].set_title("Forced-exit sensitivity: NAV")
axes[0].legend(fontsize=7)
axes[1].bar(forced_performance["portfolio"], forced_performance["post_exit_tail_return"] * 100.0)
axes[1].axhline(0.0, color="black", linewidth=0.8)
axes[1].set_title("Return during forced-exit window")
axes[1].set_ylabel("Return (%)")
axes[1].tick_params(axis="x", rotation=25)
axes[2].bar(forced_performance["portfolio"], forced_performance["final_market_value"] / 1e6)
axes[2].axhline(0.0, color="black", linewidth=0.8)
axes[2].set_title("Residual market value on final date")
axes[2].set_ylabel("CNY million")
axes[2].tick_params(axis="x", rotation=25)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "11_source_weight_forced_exit_sensitivity.png", dpi=160, bbox_inches="tight")
plt.close(fig)

result_rules = {
    "relative_return_method": "direct ratio of aligned daily unit-net-value series, final ratio divided by initial ratio minus one",
    "forced_exit": "20 actual exchange trading days of zero targets appended after the original final target date",
    "liquidation_assertion": "final market value must be zero within 1e-6 for all three portfolios",
    "decision_rule": {
        "rank_ic": "strict OOS mean > 0 and HAC p < 0.05",
        "q5_minus_q1": "strict OOS mean > 0 and HAC p < 0.05",
        "incremental_selection": "forced-exit Q5 relative return vs equal-weight eligible universe > 0",
        "status_if_all_pass": "independent_alpha_supported",
        "status_otherwise": "candidate_not_confirmed",
    },
    "research_interpretation": "weight-specification extension after earlier OOS results were observed; not a first confirmatory OOS test",
}
(METADATA_DIR / "11_source_weight_result_rules.json").write_text(
    json.dumps(result_rules, ensure_ascii=False, indent=2), encoding="utf-8"
)

print("===== 11 研报原始权重版本完成 =====")
print("最终研究状态：", final_status)
print("严格样本外 RankIC HAC p：", oos["rank_ic_hac_p_value"])
print("严格样本外 Q5-Q1 HAC p：", oos["high_minus_low_return_hac_p_value"])
print("Q5 相对等权可投资股票池（强制清仓后，逐日净值重算）：", q5_vs_equal)
print("公开结果表：", TABLE_DIR / "11_source_weight_final_research_scorecard.csv")
