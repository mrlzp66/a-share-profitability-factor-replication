import numpy as np
import pandas as pd
import statsmodels.api as sm


def hac_mean_test(values, max_lags=1):
    """对一组时间序列均值进行 HAC/Newey-West 检验。"""
    series = pd.Series(values).dropna().astype(float)

    if len(series) < 2:
        raise ValueError("HAC 检验至少需要两个有效观测。")

    lags = min(int(max_lags), len(series) - 1)
    model = sm.OLS(series.to_numpy(), np.ones((len(series), 1))).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": lags},
    )

    return {
        "observations": int(len(series)),
        "mean": float(series.mean()),
        "hac_t_stat": float(model.tvalues[0]),
        "hac_p_value": float(model.pvalues[0]),
        "max_lags": int(lags),
    }


def nav_drawdown_diagnostics(nav):
    """计算净值序列的回撤路径及最大回撤事件。"""
    series = pd.Series(nav).dropna().astype(float).copy()
    series.index = pd.to_datetime(series.index)
    series = series[~series.index.duplicated(keep="last")].sort_index()

    if series.empty or (series <= 0).any():
        raise ValueError("净值序列必须非空且全部大于 0。")

    running_peak = series.cummax()
    drawdown = series.div(running_peak).sub(1.0)

    trough_date = drawdown.idxmin()
    peak_date = series.loc[:trough_date].idxmax()
    peak_nav = float(series.loc[peak_date])
    trough_nav = float(series.loc[trough_date])

    post_trough = series.loc[trough_date:]
    recovery_candidates = post_trough[post_trough >= peak_nav]
    recovery_date = (
        recovery_candidates.index[0]
        if len(recovery_candidates) > 0
        else pd.NaT
    )

    summary = {
        "peak_date": peak_date,
        "trough_date": trough_date,
        "recovery_date": recovery_date,
        "peak_nav": peak_nav,
        "trough_nav": trough_nav,
        "max_drawdown": float(drawdown.loc[trough_date]),
        "peak_to_trough_observations": int(
            series.index.get_loc(trough_date) - series.index.get_loc(peak_date)
        ),
        "trough_recovered": bool(pd.notna(recovery_date)),
        "trough_to_recovery_observations": (
            int(series.index.get_loc(recovery_date) - series.index.get_loc(trough_date))
            if pd.notna(recovery_date)
            else None
        ),
    }

    return drawdown, summary


def period_end_returns(nav, frequency="M"):
    """将逐日净值转换为指定频率的区间收益。"""
    series = pd.Series(nav).dropna().astype(float).copy()
    series.index = pd.to_datetime(series.index)
    series = series[~series.index.duplicated(keep="last")].sort_index()

    period_end_nav = series.resample(frequency).last()
    return period_end_nav.pct_change()
