"""Run source-weight RQAlpha backtests in the RiceQuant Python 3.6 kernel.

Run this only after scripts/11_source_weight_prepare.py has written the three
portable target-weight files.  It intentionally creates private pkl files
under data/, never public artifacts.
"""

from __future__ import print_function

import json
import os
import pickle
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path("/home/rice/notebook/a-share-profitability-factor-replication")
LIB_DIR = PROJECT_ROOT / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

import rqalpha_strategy_utils as rau


INPUT_DIR = PROJECT_ROOT / "data" / "strategy_inputs"
BACKTEST_DIR = PROJECT_ROOT / "data" / "backtests"
PLOT_DIR = PROJECT_ROOT / "cache" / "rqalpha" / "11_source_weight"
for directory in [INPUT_DIR, BACKTEST_DIR, PLOT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# This script runs exactly one checkpointed RQAlpha job per invocation.  The
# Notebook sets SOURCE_WEIGHT_RQALPHA_TASK; running all six jobs in one browser
# cell can exhaust the hosted notebook's output buffer and make it unresponsive.
# Existing protocol-4 results are reused, so an interrupted job can be resumed
# without re-running a previously completed portfolio.
REUSE_COMPLETED_PORTABLE_RESULTS = True

TASKS = {
    "base_q5": ("q5_factor_high", False),
    "base_q1": ("q1_factor_low", False),
    "base_equal": ("equal_weight_eligible_universe", False),
    "forced_q5": ("q5_factor_high", True),
    "forced_q1": ("q1_factor_low", True),
    "forced_equal": ("equal_weight_eligible_universe", True),
}
SELECTED_TASK = os.environ.get("SOURCE_WEIGHT_RQALPHA_TASK", "").strip()

PORTFOLIOS = [
    "q5_factor_high",
    "q1_factor_low",
    "equal_weight_eligible_universe",
]

EXECUTION_SETTINGS = {
    "initial_cash": 100000000,
    "benchmark": "000906.XSHG",
    "matching_type": "current_bar",
    "min_commission": 5.0,
    "stock_commission_multiplier": 0.375,
    "tax_multiplier": 1.0,
    "price_limit": True,
    "volume_limit": True,
    "volume_percent": 0.05,
    "show_progress": True,
    "progress_interval": 100,
}


def load_portable_targets(portfolio_name):
    path = INPUT_DIR / "11_source_weight_{}_target_weights_portable.pkl".format(portfolio_name)
    if not path.exists():
        raise FileNotFoundError("缺少 Python 3.9 生成的私有目标权重：{}".format(path))
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("format") != "rqalpha_target_weights_portable_v1":
        raise RuntimeError("目标权重文件格式不正确：{}".format(path))
    targets = pd.DataFrame(
        payload["weights"],
        index=pd.to_datetime(payload["dates"]),
        columns=payload["assets"],
        dtype=float,
    )
    targets.index = pd.DatetimeIndex(targets.index).normalize()
    if targets.empty or targets.index.has_duplicates:
        raise RuntimeError("目标权重为空或日期重复：{}".format(path))
    if (targets.sum(axis=1) > 1.0 + 1e-10).any() or (targets < -1e-10).any().any():
        raise RuntimeError("目标权重违反仅多头、非杠杆约束：{}".format(path))
    return targets


def dataframe_as_split_payload(frame):
    """Avoid reset_index's duplicate-datetime-column failure on old pandas."""
    table = frame.copy()
    index_name = table.index.name
    if not index_name or index_name in table.columns:
        index_name = "__index__"
    table.index.name = index_name
    return json.loads(table.reset_index().to_json(orient="split", date_format="iso"))


def make_portable_result(portfolio_name, analyser):
    return {
        "format": "rqalpha_analyser_portable_v2",
        "portfolio_name": portfolio_name,
        "summary": json.loads(json.dumps(analyser["summary"], ensure_ascii=False, default=str)),
        "portfolio": dataframe_as_split_payload(analyser["portfolio"]),
        "benchmark_portfolio": dataframe_as_split_payload(analyser["benchmark_portfolio"]),
        "stock_account": dataframe_as_split_payload(analyser["stock_account"]),
        # Raw transaction records are private and used only for aggregate execution diagnostics.
        "trades": json.loads(analyser["trades"].to_json(orient="records", date_format="iso")),
    }


def run_or_load(portfolio_name, target_weights, forced_exit):
    suffix = "forced_exit_20d" if forced_exit else "base"
    portable_path = BACKTEST_DIR / "11_source_weight_{}_{}_portable_result.pkl".format(portfolio_name, suffix)
    if REUSE_COMPLETED_PORTABLE_RESULTS and portable_path.exists():
        with portable_path.open("rb") as handle:
            payload = pickle.load(handle)
        if payload.get("format") == "rqalpha_analyser_portable_v2":
            print("复用已完成的私有可移植结果：", portable_path)
            return payload
        raise RuntimeError("既有可移植结果格式不匹配：{}".format(portable_path))

    raw_path = BACKTEST_DIR / "11_source_weight_{}_{}_analyser.pkl".format(portfolio_name, suffix)
    plot_path = PLOT_DIR / "{}_{}".format(portfolio_name, suffix)
    print("\n===== RQAlpha：{} / {} =====".format(portfolio_name, suffix))
    print("目标日期：{} 至 {}".format(target_weights.index.min().date(), target_weights.index.max().date()))
    result = rau.run_target_weight_backtest(
        target_weights=target_weights,
        analyser_output_file=raw_path,
        plot_output_path=plot_path,
        **EXECUTION_SETTINGS
    )
    if not isinstance(result, dict) or "sys_analyser" not in result:
        raise RuntimeError("RQAlpha 未返回预期 analyser 结果：{} / {}".format(portfolio_name, suffix))
    payload = make_portable_result(portfolio_name, result["sys_analyser"])
    with portable_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=4)
    print("私有可移植结果已保存：", portable_path)
    return payload


def append_forced_exit_dates(target_weights):
    """Append exactly 20 actual exchange trading days of zero target weights."""
    original_end = pd.Timestamp(target_weights.index.max()).normalize()
    try:
        import rqdatac
    except ImportError:
        raise ImportError("强制清仓窗口需要 rqdatac 的真实交易日历，不能用自然日替代。")
    calendar = pd.DatetimeIndex(
        rqdatac.get_trading_dates(
            (original_end + timedelta(days=1)).strftime("%Y-%m-%d"),
            (original_end + timedelta(days=120)).strftime("%Y-%m-%d"),
        )
    ).normalize()
    if len(calendar) < 20:
        raise RuntimeError("无法取得足够的清仓交易日。")
    exit_dates = calendar[:20]
    zero_targets = pd.DataFrame(0.0, index=exit_dates, columns=target_weights.columns)
    forced = pd.concat([target_weights, zero_targets], axis=0)
    forced = forced.loc[~forced.index.duplicated(keep="last")].sort_index()
    if len(exit_dates) != 20 or (forced.loc[exit_dates].abs().sum(axis=1) > 1e-12).any():
        raise RuntimeError("强制清仓零权重窗口构建失败。")
    return forced, exit_dates


print("===== 11.2 研报原始权重版本：单组 RQAlpha 断点回测 =====")
if SELECTED_TASK not in TASKS:
    raise RuntimeError(
        "必须通过 Notebook 11.2.0--11.2.6 选择一个任务。可用任务：{}".format(
            ", ".join(sorted(TASKS))
        )
    )

base_targets = {name: load_portable_targets(name) for name in PORTFOLIOS}
first_dates = {panel.index.min() for panel in base_targets.values()}
last_dates = {panel.index.max() for panel in base_targets.values()}
if len(first_dates) != 1 or len(last_dates) != 1:
    raise RuntimeError("三组目标权重的交易日期不一致，停止回测。")

portfolio_name, forced_exit = TASKS[SELECTED_TASK]
if forced_exit:
    target_weights, exit_dates = append_forced_exit_dates(base_targets[portfolio_name])
    private_forced_input = INPUT_DIR / "11_source_weight_{}_forced_exit_20d_target_weights_portable.pkl".format(portfolio_name)
    with private_forced_input.open("wb") as handle:
        pickle.dump(
            {
                "format": "rqalpha_target_weights_portable_v1",
                "portfolio_name": portfolio_name,
                "dates": target_weights.index.strftime("%Y-%m-%d").tolist(),
                "assets": target_weights.columns.astype(str).tolist(),
                "weights": target_weights.fillna(0.0).to_numpy().tolist(),
            },
            handle,
            protocol=4,
        )
    forced_summary_path = BACKTEST_DIR / "11_source_weight_forced_exit_window_summary.json"
    existing_summary = []
    if forced_summary_path.exists():
        try:
            with forced_summary_path.open("r", encoding="utf-8") as handle:
                existing_summary = json.load(handle)
        except Exception:
            existing_summary = []
    summary_by_portfolio = {
        item.get("portfolio"): item
        for item in existing_summary
        if isinstance(item, dict) and item.get("portfolio")
    }
    summary_by_portfolio[portfolio_name] = {
        "portfolio": portfolio_name,
        "original_final_target_date": base_targets[portfolio_name].index.max().strftime("%Y-%m-%d"),
        "first_forced_exit_date": exit_dates.min().strftime("%Y-%m-%d"),
        "final_forced_exit_date": exit_dates.max().strftime("%Y-%m-%d"),
        "forced_exit_trading_day_count": int(len(exit_dates)),
    }
    with forced_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            [summary_by_portfolio[name] for name in PORTFOLIOS if name in summary_by_portfolio],
            handle,
            ensure_ascii=False,
            indent=2,
        )
else:
    target_weights = base_targets[portfolio_name]

run_or_load(portfolio_name, target_weights, forced_exit=forced_exit)
print("\n===== 11.2 单组完成：{} =====".format(SELECTED_TASK))
print("该组私有结果已写入 data/backtests/，不要上传 GitHub。")
print("继续依次运行其余 11.2 单元；六组均完成后，切回 Python 3.9 运行 11.3。")
