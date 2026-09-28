"""
通用策略研究核心函数。

只存放可复用的策略计算工具：
- 交易日程审计；
- 后续会加入权重构建、通用回测和绩效计算。

不包含任何特定因子、文件名、股票池或研究结论。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


STRATEGY_CORE_VERSION = "0.1.0"


def audit_execution_schedule(
    schedule: pd.DataFrame,
    signal_column: str,
    entry_column: str,
    exit_column: str,
    trading_calendar: pd.DatetimeIndex | None = None,
    expected_holding_intervals: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    审计任意策略的信号、入场、退出日程。

    参数：
    - schedule：包含信号日、入场日、退出日的 DataFrame；
    - signal_column / entry_column / exit_column：对应列名；
    - trading_calendar：可选交易日历，用于精确计算交易日间隔；
    - expected_holding_intervals：可选预期持有交易日间隔。

    返回：
    - 带有持有期、相邻期衔接关系的日期级审计表；
    - 汇总字典。
    """
    required_columns = {
        signal_column,
        entry_column,
        exit_column,
    }

    missing_columns = required_columns - set(schedule.columns)

    if missing_columns:
        raise ValueError(
            "交易日程缺少列：" + ", ".join(sorted(missing_columns))
        )

    audited = schedule.copy()

    for column in [signal_column, entry_column, exit_column]:
        audited[column] = pd.to_datetime(
            audited[column],
            errors="coerce",
        )

    if audited[[signal_column, entry_column, exit_column]].isna().any().any():
        raise ValueError("交易日程存在无法解析的日期。")

    audited = audited.sort_values(signal_column).reset_index(drop=True)

    audited["entry_before_or_equal_signal"] = (
        audited[entry_column] <= audited[signal_column]
    )
    audited["exit_before_or_equal_entry"] = (
        audited[exit_column] <= audited[entry_column]
    )

    audited["next_entry_date"] = audited[entry_column].shift(-1)
    audited["calendar_days_next_entry_minus_exit"] = (
        audited["next_entry_date"] - audited[exit_column]
    ).dt.days

    if trading_calendar is not None:
        calendar = pd.DatetimeIndex(
            pd.to_datetime(trading_calendar)
        ).sort_values().unique()

        calendar_position = pd.Series(
            np.arange(len(calendar)),
            index=calendar,
        )

        audited["entry_trading_position"] = audited[entry_column].map(
            calendar_position
        )
        audited["exit_trading_position"] = audited[exit_column].map(
            calendar_position
        )
        audited["next_entry_trading_position"] = audited[
            "next_entry_date"
        ].map(calendar_position)

        audited["holding_trading_intervals"] = (
            audited["exit_trading_position"]
            - audited["entry_trading_position"]
        )

        audited["next_entry_minus_exit_trading_intervals"] = (
            audited["next_entry_trading_position"]
            - audited["exit_trading_position"]
        )

        audited["calendar_date_not_found"] = (
            audited["entry_trading_position"].isna()
            | audited["exit_trading_position"].isna()
        )

        audited["transition_type"] = np.select(
            [
                audited["next_entry_date"].isna(),
                audited["next_entry_minus_exit_trading_intervals"] < 0,
                audited["next_entry_minus_exit_trading_intervals"] == 0,
                audited["next_entry_minus_exit_trading_intervals"] > 0,
            ],
            [
                "last_period",
                "overlap",
                "same_open_rebalance",
                "cash_gap",
            ],
            default="unknown",
        )
    else:
        audited["holding_trading_intervals"] = np.nan
        audited["next_entry_minus_exit_trading_intervals"] = np.nan
        audited["calendar_date_not_found"] = False

        audited["transition_type"] = np.select(
            [
                audited["next_entry_date"].isna(),
                audited["next_entry_date"] < audited[exit_column],
                audited["next_entry_date"] == audited[exit_column],
                audited["next_entry_date"] > audited[exit_column],
            ],
            [
                "last_period",
                "overlap",
                "same_open_rebalance",
                "cash_gap",
            ],
            default="unknown",
        )

    valid_holding = audited["holding_trading_intervals"].dropna()

    summary = {
        "signal_period_count": int(len(audited)),
        "first_signal_date": (
            audited[signal_column].min().strftime("%Y-%m-%d")
            if not audited.empty
            else None
        ),
        "last_signal_date": (
            audited[signal_column].max().strftime("%Y-%m-%d")
            if not audited.empty
            else None
        ),
        "entry_before_or_equal_signal_count": int(
            audited["entry_before_or_equal_signal"].sum()
        ),
        "exit_before_or_equal_entry_count": int(
            audited["exit_before_or_equal_entry"].sum()
        ),
        "overlap_period_count": int(
            (audited["transition_type"] == "overlap").sum()
        ),
        "same_open_rebalance_count": int(
            (audited["transition_type"] == "same_open_rebalance").sum()
        ),
        "cash_gap_period_count": int(
            (audited["transition_type"] == "cash_gap").sum()
        ),
        "calendar_date_not_found_count": int(
            audited["calendar_date_not_found"].sum()
        ),
        "mean_holding_trading_intervals": (
            float(valid_holding.mean())
            if not valid_holding.empty
            else np.nan
        ),
        "min_holding_trading_intervals": (
            int(valid_holding.min())
            if not valid_holding.empty
            else np.nan
        ),
        "max_holding_trading_intervals": (
            int(valid_holding.max())
            if not valid_holding.empty
            else np.nan
        ),
    }

    if expected_holding_intervals is not None:
        audited["matches_expected_holding_intervals"] = (
            audited["holding_trading_intervals"]
            == expected_holding_intervals
        )

        comparable_periods = audited[
            "holding_trading_intervals"
        ].notna()

        summary["expected_holding_intervals"] = int(
            expected_holding_intervals
        )
        summary["holding_interval_mismatch_count"] = int(
            (
                comparable_periods
                & ~audited["matches_expected_holding_intervals"]
            ).sum()
        )

    return audited, summary

def build_quantile_target_weights(
    score_panel,
    schedule,
    *,
    signal_column,
    entry_column,
    selection_fraction=0.20,
    min_valid_assets=100,
    higher_is_better=True,
):
    """
    根据任意截面分数面板，生成在指定交易日开仓的等权目标权重。

    返回：
    - target_weights：交易日 × 资产 的权重面板；仅适合本地缓存
    - selection_audit：不含资产代码的选股数量审计，可安全导出
    """
    if not isinstance(score_panel, pd.DataFrame) or score_panel.empty:
        raise ValueError("score_panel 必须是非空 DataFrame。")

    if not 0 < selection_fraction <= 1:
        raise ValueError("selection_fraction 必须在 (0, 1] 内。")

    if min_valid_assets < 1:
        raise ValueError("min_valid_assets 必须大于等于 1。")

    required_columns = {signal_column, entry_column}
    missing_columns = required_columns - set(schedule.columns)

    if missing_columns:
        raise ValueError(
            "schedule 缺少字段：" + ", ".join(sorted(missing_columns))
        )

    scores = score_panel.copy()
    scores.index = pd.to_datetime(scores.index).normalize()

    if scores.index.has_duplicates:
        raise ValueError("score_panel 的日期索引不能重复。")

    schedule_frame = schedule.loc[
        :, [signal_column, entry_column]
    ].copy()

    schedule_frame[signal_column] = pd.to_datetime(
        schedule_frame[signal_column]
    ).dt.normalize()

    schedule_frame[entry_column] = pd.to_datetime(
        schedule_frame[entry_column]
    ).dt.normalize()

    if schedule_frame[entry_column].duplicated().any():
        raise ValueError("schedule 中的开仓日期不能重复。")

    weight_rows = []
    weight_dates = []
    audit_rows = []

    for _, row in schedule_frame.iterrows():
        signal_date = row[signal_column]
        entry_date = row[entry_column]

        if signal_date not in scores.index:
            audit_rows.append(
                {
                    "signal_date": signal_date,
                    "entry_date": entry_date,
                    "valid_score_count": 0,
                    "selected_asset_count": 0,
                    "eligible": False,
                    "reason": "signal_date_not_in_score_panel",
                }
            )
            continue

        cross_section = pd.to_numeric(
            scores.loc[signal_date],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan).dropna()

        valid_count = len(cross_section)

        if valid_count < min_valid_assets:
            audit_rows.append(
                {
                    "signal_date": signal_date,
                    "entry_date": entry_date,
                    "valid_score_count": valid_count,
                    "selected_asset_count": 0,
                    "eligible": False,
                    "reason": "insufficient_valid_assets",
                }
            )
            continue

        selected_count = max(
            1,
            int(np.ceil(valid_count * selection_fraction)),
        )

        selected_scores = cross_section.sort_values(
            ascending=not higher_is_better,
            kind="mergesort",
        ).iloc[:selected_count]

        weights = pd.Series(
            0.0,
            index=scores.columns,
            dtype=float,
        )
        weights.loc[selected_scores.index] = 1.0 / selected_count

        weight_rows.append(weights)
        weight_dates.append(entry_date)

        audit_rows.append(
            {
                "signal_date": signal_date,
                "entry_date": entry_date,
                "valid_score_count": valid_count,
                "selected_asset_count": selected_count,
                "eligible": True,
                "reason": "selected",
            }
        )

    target_weights = pd.DataFrame(
        weight_rows,
        index=pd.DatetimeIndex(weight_dates, name=entry_column),
        columns=scores.columns,
        dtype=float,
    )

    selection_audit = pd.DataFrame(audit_rows)

    return target_weights, selection_audit
def build_active_sleeve_event_targets(
    sleeve_weights,
    schedule,
    *,
    entry_column,
    exit_column,
    trading_calendar=None,
):
    """
    将多个固定持有期的资产权重批次，合并为非杠杆事件日目标权重。

    每个有效批次在 [entry_date, exit_date) 内活跃。
    在任一事件日，将所有活跃批次等权合并；
    没有活跃批次时，目标权重为零，代表持有现金。

    返回：
    - event_target_weights：事件开盘日 × 资产 的目标权重，仅适合本地缓存
    - sleeve_audit：不含资产代码的批次重叠审计，可安全导出
    """
    if not isinstance(sleeve_weights, pd.DataFrame):
        raise ValueError("sleeve_weights 必须是 DataFrame。")

    if sleeve_weights.empty:
        raise ValueError("sleeve_weights 不能为空。")

    required_columns = {entry_column, exit_column}
    missing_columns = required_columns - set(schedule.columns)

    if missing_columns:
        raise ValueError(
            "schedule 缺少字段：" + ", ".join(sorted(missing_columns))
        )

    weights = sleeve_weights.copy().astype(float).fillna(0.0)
    weights.index = pd.to_datetime(weights.index).normalize()

    if weights.index.has_duplicates:
        raise ValueError("sleeve_weights 的日期索引不能重复。")

    schedule_frame = schedule.loc[
        :, [entry_column, exit_column]
    ].copy()

    schedule_frame[entry_column] = pd.to_datetime(
        schedule_frame[entry_column]
    ).dt.normalize()

    schedule_frame[exit_column] = pd.to_datetime(
        schedule_frame[exit_column]
    ).dt.normalize()

    if (schedule_frame[exit_column] <= schedule_frame[entry_column]).any():
        raise ValueError("每个持仓批次都必须满足 exit_date > entry_date。")

    if schedule_frame[entry_column].duplicated().any():
        raise ValueError("schedule 中的 entry_date 不能重复。")

    missing_weight_dates = sorted(
        set(schedule_frame[entry_column]) - set(weights.index)
    )

    if missing_weight_dates:
        raise ValueError(
            "部分开仓日缺少对应的权重行，例如："
            + ", ".join(
                pd.Timestamp(date).strftime("%Y-%m-%d")
                for date in missing_weight_dates[:5]
            )
        )

    if trading_calendar is not None:
        calendar = pd.DatetimeIndex(
            pd.to_datetime(trading_calendar)
        ).normalize()

        event_dates_to_check = pd.DatetimeIndex(
            pd.concat(
                [
                    schedule_frame[entry_column],
                    schedule_frame[exit_column],
                ]
            ).unique()
        )

        missing_calendar_dates = event_dates_to_check.difference(calendar)

        if len(missing_calendar_dates) > 0:
            raise ValueError(
                "部分开仓或退出日期不在交易日历中，例如："
                + ", ".join(
                    date.strftime("%Y-%m-%d")
                    for date in missing_calendar_dates[:5]
                )
            )

    event_dates = pd.DatetimeIndex(
        pd.concat(
            [
                schedule_frame[entry_column],
                schedule_frame[exit_column],
            ]
        ).unique()
    ).sort_values()

    target_rows = []
    audit_rows = []

    for event_date in event_dates:
        active_entries = schedule_frame.loc[
            (schedule_frame[entry_column] <= event_date)
            & (schedule_frame[exit_column] > event_date),
            entry_column,
        ].tolist()

        entry_count = int(
            (schedule_frame[entry_column] == event_date).sum()
        )

        exit_count = int(
            (schedule_frame[exit_column] == event_date).sum()
        )

        active_sleeve_count = len(active_entries)

        if active_sleeve_count > 0:
            target_weight = weights.loc[active_entries].mean(axis=0)
        else:
            target_weight = pd.Series(
                0.0,
                index=weights.columns,
                dtype=float,
            )

        gross_target_weight = float(target_weight.sum())
        cash_target_weight = float(1.0 - gross_target_weight)

        target_rows.append(target_weight)

        audit_rows.append(
            {
                "event_date": event_date,
                "entry_sleeve_count": entry_count,
                "exit_sleeve_count": exit_count,
                "active_sleeve_count": active_sleeve_count,
                "gross_target_weight": gross_target_weight,
                "cash_target_weight": cash_target_weight,
            }
        )

    event_target_weights = pd.DataFrame(
        target_rows,
        index=pd.DatetimeIndex(event_dates, name="event_date"),
        columns=weights.columns,
        dtype=float,
    )

    sleeve_audit = pd.DataFrame(audit_rows)

    return event_target_weights, sleeve_audit

def run_open_price_rebalance_backtest(
    event_target_weights,
    open_prices,
    close_prices,
    *,
    initial_cash,
    buy_cost_rate=0.0003,
    sell_cost_rate=0.0008,
    min_commission=5.0,
    lot_size=100,
    cash_interest_annual=0.02,
    annualization=252,
):
    """
    通用事件日开盘价调仓回测。

    - 事件日按开盘价先卖后买；
    - 非事件日不调仓，持仓按收盘价估值；
    - 仅做多、不加杠杆；
    - 现金按给定年化利率逐交易日计息；
    - 买卖成本及最低佣金按订单计算；
    - 目标权重不足 100% 的部分自然保留为现金。

    返回：
    - daily_account：逐日账户净值与估值审计
    - event_trade_summary：事件日交易汇总，不含股票代码
    """
    if initial_cash <= 0:
        raise ValueError("initial_cash 必须为正数。")

    if lot_size < 1:
        raise ValueError("lot_size 必须大于等于 1。")

    targets = event_target_weights.copy().astype(float)
    targets.index = pd.to_datetime(targets.index).normalize()
    targets.columns = targets.columns.astype(str)
        # 全程目标权重均为零的资产不会参与任何交易，
    # 不应要求其提供价格数据。
    active_asset_columns = targets.columns[
        targets.abs().sum(axis=0) > 1e-12
    ]

    targets = targets.loc[:, active_asset_columns]

    if targets.empty:
        raise ValueError("event_target_weights 中没有非零目标权重。")

    if targets.index.has_duplicates:
        raise ValueError("event_target_weights 的日期不能重复。")

    if not np.isfinite(targets.to_numpy()).all():
        raise ValueError("event_target_weights 包含非有限数值。")

    if (targets < -1e-12).any().any():
        raise ValueError("当前回测只支持非负权重的多头组合。")

    target_gross_weight = targets.sum(axis=1)

    if (target_gross_weight > 1.0 + 1e-8).any():
        raise ValueError("任一事件日的目标总权重不能超过 100%。")

    assets = targets.columns.tolist()

    open_panel = open_prices.copy()
    close_panel = close_prices.copy()

    open_panel.index = pd.to_datetime(open_panel.index).normalize()
    close_panel.index = pd.to_datetime(close_panel.index).normalize()

    open_panel.columns = open_panel.columns.astype(str)
    close_panel.columns = close_panel.columns.astype(str)

    if open_panel.index.has_duplicates or close_panel.index.has_duplicates:
        raise ValueError("价格面板不能包含重复日期。")

    missing_open_assets = set(assets) - set(open_panel.columns)
    missing_close_assets = set(assets) - set(close_panel.columns)

    if missing_open_assets or missing_close_assets:
        raise ValueError("价格面板缺少部分目标资产列。")

    open_panel = open_panel.reindex(columns=assets).sort_index()
    close_panel = close_panel.reindex(columns=assets).sort_index()

    trading_dates = open_panel.index.intersection(
        close_panel.index
    ).sort_values()

    if len(trading_dates) == 0:
        raise ValueError("开盘价与收盘价没有共同交易日期。")

    missing_event_dates = targets.index.difference(trading_dates)

    if len(missing_event_dates) > 0:
        raise ValueError(
            "部分事件日不在价格数据中，例如："
            + ", ".join(
                date.strftime("%Y-%m-%d")
                for date in missing_event_dates[:5]
            )
        )

    daily_cash_interest_rate = (
        (1.0 + cash_interest_annual) ** (1.0 / annualization) - 1.0
    )

    def order_fee(notional, cost_rate):
        if notional <= 0:
            return 0.0

        return max(notional * cost_rate, min_commission)

    def scaled_buy_orders(requested_shares, prices, available_cash):
        requested_shares = requested_shares.clip(lower=0.0)

        if requested_shares.sum() <= 0:
            empty = pd.Series(
                0.0,
                index=requested_shares.index,
                dtype=float,
            )
            return empty, 0.0, 0.0

        def calculate_cash_required(scale):
            quantity = (
                np.floor(
                    requested_shares * scale / lot_size
                ) * lot_size
            ).astype(float)

            notional = quantity * prices

            fee = sum(
                order_fee(value, buy_cost_rate)
                for value in notional.loc[quantity > 0]
            )

            return quantity, float(notional.sum() + fee), float(fee)

        full_quantity, full_cash_required, full_fee = (
            calculate_cash_required(1.0)
        )

        if full_cash_required <= available_cash + 1e-8:
            return full_quantity, full_cash_required, full_fee

        lower_scale = 0.0
        upper_scale = 1.0

        for _ in range(32):
            middle_scale = (lower_scale + upper_scale) / 2.0

            _, middle_cash_required, _ = calculate_cash_required(
                middle_scale
            )

            if middle_cash_required <= available_cash:
                lower_scale = middle_scale
            else:
                upper_scale = middle_scale

        quantity, cash_required, fee = calculate_cash_required(
            lower_scale
        )

        return quantity, cash_required, fee

    shares = pd.Series(0.0, index=assets, dtype=float)
    cash = float(initial_cash)

    last_close = pd.Series(np.nan, index=assets, dtype=float)

    daily_records = []
    event_records = []

    first_day = True

    for date in trading_dates:
        open_row = open_panel.loc[date].astype(float)
        close_row = close_panel.loc[date].astype(float)

        cash_interest_earned = 0.0

        if not first_day:
            cash_interest_earned = cash * daily_cash_interest_rate
            cash += cash_interest_earned

        first_day = False

        holding_before_open = shares > 0.0

        open_mark_prices = open_row.where(
            open_row.notna(),
            last_close,
        )

        open_mark_fallback_count = int(
            (holding_before_open & open_row.isna()).sum()
        )

        if (
            holding_before_open
            & open_mark_prices.isna()
        ).any():
            raise RuntimeError(
                f"{date.strftime('%Y-%m-%d')} 存在无法估值的持仓。"
            )

        portfolio_value_open = float(
            cash + (shares * open_mark_prices).sum()
        )

        is_event_rebalance = date in targets.index

        buy_notional = 0.0
        sell_notional = 0.0
        buy_fee = 0.0
        sell_fee = 0.0
        buy_order_count = 0
        sell_order_count = 0
        unfilled_buy_notional = 0.0
        desired_gross_weight = np.nan

        if is_event_rebalance:
            target_weight = targets.loc[date]
            desired_gross_weight = float(target_weight.sum())

            required_execution_prices = (
                (shares > 0.0)
                | (target_weight > 0.0)
            )

            if (
                open_row.loc[required_execution_prices].isna()
            ).any():
                raise RuntimeError(
                    f"{date.strftime('%Y-%m-%d')} 事件日存在缺失开盘价，"
                    "不能安全执行调仓。"
                )

            if (
                open_row.loc[required_execution_prices] <= 0
            ).any():
                raise RuntimeError(
                    f"{date.strftime('%Y-%m-%d')} 事件日存在非正开盘价。"
                )

            execution_prices = open_row.copy()

            target_values = (
                target_weight * portfolio_value_open
            )

            desired_shares = (
                np.floor(
                    target_values / execution_prices / lot_size
                ) * lot_size
            ).astype(float)

            sell_quantity = (
                shares - desired_shares
            ).clip(lower=0.0)

            sell_values = sell_quantity * execution_prices

            sell_fee = sum(
                order_fee(value, sell_cost_rate)
                for value in sell_values.loc[sell_quantity > 0]
            )

            sell_notional = float(sell_values.sum())
            sell_order_count = int((sell_quantity > 0).sum())

            cash += sell_notional - sell_fee
            shares -= sell_quantity

            requested_buy_shares = (
                desired_shares - shares
            ).clip(lower=0.0)

            buy_quantity, buy_cash_required, buy_fee = (
                scaled_buy_orders(
                    requested_buy_shares,
                    execution_prices,
                    cash,
                )
            )

            buy_values = buy_quantity * execution_prices

            buy_notional = float(buy_values.sum())
            buy_order_count = int((buy_quantity > 0).sum())

            cash -= buy_cash_required
            shares += buy_quantity

            unfilled_buy_notional = float(
                (
                    (requested_buy_shares - buy_quantity)
                    * execution_prices
                ).sum()
            )

        close_mark_prices = close_row.where(
            close_row.notna(),
            open_row,
        )

        close_mark_prices = close_mark_prices.where(
            close_mark_prices.notna(),
            last_close,
        )

        holding_after_trade = shares > 0.0

        close_mark_fallback_count = int(
            (holding_after_trade & close_row.isna()).sum()
        )

        if (
            holding_after_trade
            & close_mark_prices.isna()
        ).any():
            raise RuntimeError(
                f"{date.strftime('%Y-%m-%d')} 存在无法收盘估值的持仓。"
            )

        last_close.loc[close_mark_prices.notna()] = (
            close_mark_prices.loc[close_mark_prices.notna()]
        )

        market_value_close = float(
            (shares * close_mark_prices).sum()
        )

        nav = float(cash + market_value_close)

        if nav <= 0:
            raise RuntimeError(
                f"{date.strftime('%Y-%m-%d')} 账户净值非正。"
            )

        gross_market_exposure = market_value_close / nav

        daily_records.append(
            {
                "date": date,
                "nav": nav,
                "cash": cash,
                "market_value": market_value_close,
                "gross_market_exposure": gross_market_exposure,
                "is_event_rebalance": is_event_rebalance,
                "desired_gross_weight": desired_gross_weight,
                "buy_notional": buy_notional,
                "sell_notional": sell_notional,
                "transaction_cost": buy_fee + sell_fee,
                "buy_order_count": buy_order_count,
                "sell_order_count": sell_order_count,
                "unfilled_buy_notional": unfilled_buy_notional,
                "cash_interest_earned": cash_interest_earned,
                "open_mark_fallback_count": open_mark_fallback_count,
                "close_mark_fallback_count": close_mark_fallback_count,
            }
        )

        if is_event_rebalance:
            event_records.append(daily_records[-1].copy())

    daily_account = (
        pd.DataFrame(daily_records)
        .set_index("date")
        .sort_index()
    )

    event_trade_summary = (
        pd.DataFrame(event_records)
        .set_index("date")
        .sort_index()
    )

    return daily_account, event_trade_summary


def summarize_nav_performance(
    nav,
    *,
    initial_value=None,
    benchmark_nav=None,
    annualization=252,
    risk_free_annual=0.0,
):
    """
    对任意净值序列计算收益、波动、夏普、回撤及基准相对指标。
    """
    strategy_nav = pd.Series(nav).dropna().astype(float).sort_index()

    if len(strategy_nav) < 2:
        raise ValueError("至少需要两个净值观测值。")

    if (strategy_nav <= 0).any():
        raise ValueError("净值必须为正。")

    base_value = (
        float(initial_value)
        if initial_value is not None
        else float(strategy_nav.iloc[0])
    )

    if base_value <= 0:
        raise ValueError("initial_value 必须为正。")

    strategy_returns = strategy_nav.pct_change()
    strategy_returns.iloc[0] = strategy_nav.iloc[0] / base_value - 1.0

    strategy_returns = strategy_returns.dropna()

    period_count = len(strategy_returns)

    total_return = strategy_nav.iloc[-1] / base_value - 1.0

    annualized_return = (
        (strategy_nav.iloc[-1] / base_value)
        ** (annualization / period_count)
        - 1.0
    )

    annualized_volatility = (
        strategy_returns.std(ddof=1) * np.sqrt(annualization)
    )

    daily_risk_free = (
        (1.0 + risk_free_annual) ** (1.0 / annualization) - 1.0
    )

    excess_daily_return = strategy_returns - daily_risk_free

    if excess_daily_return.std(ddof=1) > 0:
        sharpe_ratio = (
            excess_daily_return.mean()
            / excess_daily_return.std(ddof=1)
            * np.sqrt(annualization)
        )
    else:
        sharpe_ratio = np.nan

    drawdown_values = np.r_[
        base_value,
        strategy_nav.to_numpy(),
    ]

    running_peak = np.maximum.accumulate(drawdown_values)

    max_drawdown = float(
        (drawdown_values / running_peak - 1.0).min()
    )

    result = {
        "observations": period_count,
        "initial_value": base_value,
        "final_value": float(strategy_nav.iloc[-1]),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
    }

    if benchmark_nav is not None:
        benchmark = (
            pd.Series(benchmark_nav)
            .dropna()
            .astype(float)
            .sort_index()
        )

        common_dates = strategy_nav.index.intersection(
            benchmark.index
        ).sort_values()

        if len(common_dates) < 2:
            raise ValueError("策略与基准没有足够的共同日期。")

        aligned_strategy = strategy_nav.reindex(common_dates)
        aligned_benchmark = benchmark.reindex(common_dates)

        benchmark_base = (
            float(initial_value)
            if initial_value is not None
            else float(aligned_benchmark.iloc[0])
        )

        benchmark_returns = aligned_benchmark.pct_change()
        benchmark_returns.iloc[0] = (
            aligned_benchmark.iloc[0] / benchmark_base - 1.0
        )

        benchmark_returns = benchmark_returns.dropna()

        benchmark_total_return = (
            aligned_benchmark.iloc[-1] / benchmark_base - 1.0
        )

        benchmark_annualized_return = (
            (aligned_benchmark.iloc[-1] / benchmark_base)
            ** (annualization / len(benchmark_returns))
            - 1.0
        )

        active_returns = (
            strategy_returns.reindex(common_dates).dropna()
            - benchmark_returns.reindex(common_dates).dropna()
        )

        if active_returns.std(ddof=1) > 0:
            information_ratio = (
                active_returns.mean()
                / active_returns.std(ddof=1)
                * np.sqrt(annualization)
            )
        else:
            information_ratio = np.nan

        result.update(
            {
                "benchmark_total_return": benchmark_total_return,
                "benchmark_annualized_return": (
                    benchmark_annualized_return
                ),
                "annualized_excess_return": (
                    annualized_return - benchmark_annualized_return
                ),
                "information_ratio": information_ratio,
            }
        )

    return pd.DataFrame([result])