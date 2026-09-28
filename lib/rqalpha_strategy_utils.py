from pathlib import Path

import pandas as pd
from rqalpha.api import order_target_percent
from rqalpha_plus import run_func


def _build_target_schedule(target_weights, tolerance=1e-10):
    """把日期×股票目标权重表转换为日期→目标权重字典。"""
    if not isinstance(target_weights, pd.DataFrame):
        raise TypeError("target_weights 必须是 DataFrame。")

    panel = target_weights.copy()
    panel.index = pd.to_datetime(panel.index).normalize()
    panel = panel[~panel.index.duplicated(keep="last")].sort_index()
    panel = panel.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    panel.columns = panel.columns.astype(str)
    panel = panel.loc[:, panel.abs().sum(axis=0) > tolerance]
    panel = panel.mask(panel.abs() <= tolerance, 0.0)

    if panel.empty:
        raise ValueError("目标权重表为空。")

    if (panel < -tolerance).any().any():
        raise ValueError("本回测函数仅支持非负目标权重。")

    if (panel.sum(axis=1) > 1.0 + tolerance).any():
        raise ValueError("存在单日目标总权重超过 100% 的情况。")

    schedule = {}
    for date in panel.index:
        schedule[date.date()] = {
            stock: float(weight)
            for stock, weight in panel.loc[date].items()
            if weight > tolerance
        }

    return panel, schedule


def run_target_weight_backtest(
    target_weights,
    initial_cash,
    benchmark,
    analyser_output_file,
    plot_output_path,
    matching_type="current_bar",
    min_commission=5.0,
    stock_commission_multiplier=0.375,
    tax_multiplier=1.0,
    price_limit=True,
    volume_limit=True,
    volume_percent=0.05,
    show_progress=True,
    progress_interval=20,
):
    """
    使用 RQAlpha 对任意日期×股票目标权重表执行开盘调仓回测。

    卖出非目标持仓优先；未成交委托在下一交易日重试。
    """
    panel, targets_by_date = _build_target_schedule(target_weights)

    analyser_output_file = Path(analyser_output_file)
    plot_output_path = Path(plot_output_path)
    analyser_output_file.parent.mkdir(parents=True, exist_ok=True)
    plot_output_path.mkdir(parents=True, exist_ok=True)

    def init(context):
        context.targets_by_date = targets_by_date
        context.current_target = {}
        context.order_queue = []
        context.pending_queue = []
        context.trading_day_count = 0

    def before_trading(context):
        context.trading_day_count += 1
        today = pd.Timestamp(context.now).date()

        if (
            context.trading_day_count == 1
            or context.trading_day_count % progress_interval == 0
        ):
            print(
                "[RQAlpha] 已处理第 {} 个交易日，当前日期：{}".format(
                    context.trading_day_count,
                    today,
                ),
                flush=True,
            )

        if today in context.targets_by_date:
            context.current_target = context.targets_by_date[today]
            held_stocks = list(context.portfolio.positions.keys())

            sell_first = [
                stock for stock in held_stocks
                if context.current_target.get(stock, 0.0) == 0.0
            ]
            held_targets = [
                stock for stock in held_stocks
                if stock not in sell_first
            ]
            new_targets = [
                stock for stock in context.current_target
                if stock not in held_stocks
            ]

            context.order_queue = list(
                dict.fromkeys(sell_first + held_targets + new_targets)
            )
            context.pending_queue = []

        elif context.pending_queue:
            context.order_queue = list(context.pending_queue)

    def open_auction(context, bar_dict):
        if not context.order_queue:
            return

        failed_orders = []

        for stock in context.order_queue:
            target_weight = context.current_target.get(stock, 0.0)
            order = order_target_percent(stock, target_weight)

            if order is None:
                failed_orders.append(stock)

        context.pending_queue = failed_orders
        context.order_queue = []

    def after_trading(context):
        pass

    config = {
        "base": {
            "start_date": panel.index.min().strftime("%Y-%m-%d"),
            "end_date": panel.index.max().strftime("%Y-%m-%d"),
            "frequency": "1d",
            "accounts": {"STOCK": float(initial_cash)},
        },
        "mod": {
            "sys_accounts": {
                "enabled": True,
                "dividend_reinvestment": True,
            },
            "sys_analyser": {
                "enabled": True,
                "benchmark": benchmark,
                "plot": True,
                "output_file": str(analyser_output_file),
                "plot_save_file": str(plot_output_path),
            },
            "sys_risk": {
                "enabled": True,
                "validate_is_trading": True,
            },
            "sys_simulation": {
                "enabled": True,
                "signal": True,
                "matching_type": matching_type,
                "price_limit": price_limit,
                "liquidity_limit": False,
                "volume_limit": volume_limit,
                "volume_percent": volume_percent,
                "slippage_model": "PriceRatioSlippage",
                "slippage": 0,
            },
            "sys_transaction_cost": {
                "enabled": True,
                "cn_stock_min_commission": float(min_commission),
                "stock_commission_multiplier": float(
                    stock_commission_multiplier
                ),
                "tax_multiplier": float(tax_multiplier),
            },
            "sys_progress": {
                "enabled": True,
                "show": bool(show_progress),
            },
        },
    }

    return run_func(
        config=config,
        init=init,
        before_trading=before_trading,
        open_auction=open_auction,
        after_trading=after_trading,
    )
