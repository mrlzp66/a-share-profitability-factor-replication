"""
A 股盈利能力因子研究：可复用基础库

设计原则：
1. cache/ 只存本地个股级中间数据，不进入 GitHub；
2. artifacts/ 只存可公开的汇总表、规则文件和图；
3. 研究公式、样本范围、固定权重仍留在 Notebook 显式展示；
4. 所有缓存均使用配置指纹，研究口径变动时自动失效重建。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LIBRARY_VERSION = "0.1.0"


@dataclass(frozen=True)
class ProjectPaths:
    """项目中所有标准目录。"""

    root: Path
    table_dir: Path
    figure_dir: Path
    metadata_dir: Path
    cache_data_dir: Path
    cache_metadata_dir: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "ProjectPaths":
        root = Path(root)
        return cls(
            root=root,
            table_dir=root / "artifacts" / "tables",
            figure_dir=root / "artifacts" / "figures",
            metadata_dir=root / "artifacts" / "metadata",
            cache_data_dir=root / "cache" / "data",
            cache_metadata_dir=root / "cache" / "metadata",
        )

    def ensure_directories(self) -> None:
        for directory in [
            self.table_dir,
            self.figure_dir,
            self.metadata_dir,
            self.cache_data_dir,
            self.cache_metadata_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def ensure_cache_ignored(self) -> None:
        """确保本地个股级缓存不会被 Git 跟踪。"""
        gitignore_path = self.root / ".gitignore"
        current = (
            gitignore_path.read_text(encoding="utf-8")
            if gitignore_path.exists()
            else ""
        )

        if "cache/" not in current.splitlines():
            with gitignore_path.open("a", encoding="utf-8") as f:
                if current and not current.endswith("\n"):
                    f.write("\n")
                f.write("cache/\n")


def cache_fingerprint(config: Mapping[str, Any]) -> str:
    """根据研究口径生成稳定的 SHA-256 指纹。"""
    text = json.dumps(
        dict(config),
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_or_build_pickle_cache(
    data_path: str | Path,
    metadata_path: str | Path,
    config: Mapping[str, Any],
    builder: Callable[[], Any],
    contains_individual_stock_values: bool = True,
) -> tuple[Any, bool]:
    """
    读取与当前配置一致的本地缓存；没有命中时才执行 builder 并保存。

    返回：
        payload, cache_hit
    """
    data_path = Path(data_path)
    metadata_path = Path(metadata_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    fingerprint = cache_fingerprint(config)

    if data_path.exists() and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

            if metadata.get("fingerprint") == fingerprint:
                return pd.read_pickle(data_path), True

        except Exception as exc:
            print(f"缓存读取失败，将重建：{type(exc).__name__}: {exc}")

    payload = builder()

    temporary_data_path = data_path.with_suffix(data_path.suffix + ".tmp")
    temporary_metadata_path = metadata_path.with_suffix(
        metadata_path.suffix + ".tmp"
    )

    pd.to_pickle(payload, temporary_data_path)
    temporary_data_path.replace(data_path)

    metadata = {
        "fingerprint": fingerprint,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": dict(config),
        "contains_individual_stock_values": contains_individual_stock_values,
        "github_export_allowed": False,
    }

    temporary_metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary_metadata_path.replace(metadata_path)

    return payload, False


def save_public_csv(
    frame: pd.DataFrame,
    path: str | Path,
) -> Path:
    """保存可公开的汇总表；调用者不得传入个股级原始因子或价格。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_public_json(
    content: Mapping[str, Any],
    path: str | Path,
) -> Path:
    """保存可公开的规则和研究元数据。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(content, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path


def configure_matplotlib_style() -> None:
    """兼容不同 matplotlib 版本。"""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("default")


def safe_correlation(left: pd.Series, right: pd.Series) -> float:
    """计算两序列的 Pearson 相关系数；样本不足时返回 NaN。"""
    paired = pd.concat(
        [
            pd.to_numeric(left, errors="coerce"),
            pd.to_numeric(right, errors="coerce"),
        ],
        axis=1,
    ).dropna()

    if len(paired) < 3:
        return np.nan

    if paired.iloc[:, 0].std(ddof=0) == 0:
        return np.nan

    if paired.iloc[:, 1].std(ddof=0) == 0:
        return np.nan

    return float(paired.iloc[:, 0].corr(paired.iloc[:, 1]))


def mad_winsorize(
    values: pd.Series,
    multiplier: float = 3.5,
) -> pd.Series:
    """使用 median ± multiplier × 1.4826 × MAD 去极值。"""
    values = pd.to_numeric(values, errors="coerce").copy()
    valid = values.dropna()

    if valid.empty:
        return values

    median = valid.median()
    mad = (valid - median).abs().median()
    robust_sigma = 1.4826 * mad

    if not np.isfinite(robust_sigma) or robust_sigma <= 0:
        return values

    return values.clip(
        lower=median - multiplier * robust_sigma,
        upper=median + multiplier * robust_sigma,
    )


def industry_mean_dispersion(
    values: pd.Series,
    industries: pd.Series,
) -> tuple[float, int]:
    """返回一级行业均值的标准差和有效行业数。"""
    frame = pd.DataFrame(
        {
            "value": pd.to_numeric(values, errors="coerce"),
            "industry": industries,
        }
    ).dropna()

    if frame.empty:
        return np.nan, 0

    industry_means = frame.groupby("industry")["value"].mean()

    if len(industry_means) < 2:
        return np.nan, int(len(industry_means))

    return float(industry_means.std(ddof=0)), int(len(industry_means))


def neutralize_cross_section(
    raw_factor: pd.Series,
    eligible_mask: pd.Series,
    log_market_cap: pd.Series,
    industry: pd.Series,
    min_stocks: int = 100,
    winsor_multiplier: float = 3.5,
) -> tuple[pd.Series, dict[str, Any]]:
    """
    单期标准横截面处理：
    MAD 去极值 → 一级行业 + log 市值回归 → 残差 z-score。

    返回：
        final_factor: 与 raw_factor 同索引的 Series
        audit: 覆盖率与中性化质量审计
    """
    final_factor = pd.Series(np.nan, index=raw_factor.index, dtype=float)

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

    raw_valid_count = int(valid.sum())

    audit = {
        "raw_valid_count": raw_valid_count,
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

    if raw_valid_count < min_stocks:
        audit["status"] = "insufficient_coverage"
        audit["failure_reason"] = f"valid_stock_count_below_{min_stocks}"
        return final_factor, audit

    factor_valid = mad_winsorize(
        raw_factor.loc[valid],
        multiplier=winsor_multiplier,
    )
    cap_valid = log_market_cap.loc[valid]
    industry_valid = industry.loc[valid].astype(str)

    industry_dummy = pd.get_dummies(
        industry_valid,
        prefix="industry",
    ).astype(float)

    if industry_dummy.shape[1] > 1:
        industry_dummy = industry_dummy.iloc[:, 1:]

    design_matrix = pd.concat(
        [
            pd.Series(1.0, index=factor_valid.index, name="const"),
            cap_valid.rename("log_market_cap"),
            industry_dummy,
        ],
        axis=1,
    ).astype(float)

    audit["model_input_count"] = int(len(factor_valid))
    audit["industry_groups"] = int(industry_valid.nunique())

    if len(factor_valid) <= design_matrix.shape[1] + 5:
        audit["status"] = "failed"
        audit["failure_reason"] = "insufficient_degrees_of_freedom"
        return final_factor, audit

    try:
        beta = np.linalg.lstsq(
            design_matrix.to_numpy(dtype=float),
            factor_valid.to_numpy(dtype=float),
            rcond=None,
        )[0]

        residual = pd.Series(
            factor_valid.to_numpy(dtype=float)
            - design_matrix.to_numpy(dtype=float) @ beta,
            index=factor_valid.index,
        )

        residual_std = residual.std(ddof=0)

        if not np.isfinite(residual_std) or residual_std <= 0:
            audit["status"] = "failed"
            audit["failure_reason"] = "invalid_residual_std"
            return final_factor, audit

        final_values = (residual - residual.mean()) / residual_std
        final_factor.loc[final_values.index] = final_values

        raw_dispersion, _ = industry_mean_dispersion(
            factor_valid,
            industry_valid,
        )
        final_dispersion, _ = industry_mean_dispersion(
            final_values,
            industry_valid,
        )

        audit.update(
            {
                "final_valid_count": int(final_values.notna().sum()),
                "raw_cap_corr": safe_correlation(factor_valid, cap_valid),
                "final_cap_corr": safe_correlation(final_values, cap_valid),
                "raw_industry_dispersion": raw_dispersion,
                "final_industry_dispersion": final_dispersion,
                "status": "success",
            }
        )

    except Exception as exc:
        audit["status"] = "failed"
        audit["failure_reason"] = f"{type(exc).__name__}: {exc}"

    return final_factor, audit


def _one_sample_t_stat(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()

    if len(values) < 2:
        return np.nan

    standard_deviation = values.std(ddof=1)

    if not np.isfinite(standard_deviation) or standard_deviation == 0:
        return np.nan

    return float(values.mean() / (standard_deviation / np.sqrt(len(values))))


def _annualized_ir(values: pd.Series, periods_per_year: int = 12) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()

    if len(values) < 2:
        return np.nan

    standard_deviation = values.std(ddof=1)

    if not np.isfinite(standard_deviation) or standard_deviation == 0:
        return np.nan

    return float(values.mean() / standard_deviation * np.sqrt(periods_per_year))


def evaluate_factor_panel(
    factor_panel: pd.DataFrame,
    forward_return_panel: pd.DataFrame,
    min_stocks: int = 100,
    quantiles: int = 5,
) -> pd.DataFrame:
    """
    对月度横截面因子做 IC、RankIC 和等权五分组检验。

    返回的表仅含逐期汇总结果，不含个股级因子或收益。
    """
    forward_return_panel = forward_return_panel.reindex(
        index=factor_panel.index,
        columns=factor_panel.columns,
    )

    records = []

    for signal_date in factor_panel.index:
        paired = pd.DataFrame(
            {
                "factor": pd.to_numeric(
                    factor_panel.loc[signal_date],
                    errors="coerce",
                ),
                "forward_return": pd.to_numeric(
                    forward_return_panel.loc[signal_date],
                    errors="coerce",
                ),
            }
        ).dropna()

        record = {
            "signal_date": pd.Timestamp(signal_date),
            "paired_stock_count": int(len(paired)),
            "ic": np.nan,
            "rank_ic": np.nan,
            "q1_return": np.nan,
            "q2_return": np.nan,
            "q3_return": np.nan,
            "q4_return": np.nan,
            "q5_return": np.nan,
            "high_minus_low_return": np.nan,
            "status": "insufficient_coverage",
        }

        if len(paired) < min_stocks:
            records.append(record)
            continue

        if paired["factor"].std(ddof=0) == 0:
            record["status"] = "zero_factor_variance"
            records.append(record)
            continue

        record["ic"] = float(
            paired["factor"].corr(paired["forward_return"], method="pearson")
        )
        record["rank_ic"] = float(
            paired["factor"].corr(paired["forward_return"], method="spearman")
        )

        try:
            ranks = paired["factor"].rank(method="first")
            quintile = pd.qcut(
                ranks,
                q=quantiles,
                labels=False,
            ) + 1

            quintile_return = paired.groupby(quintile)["forward_return"].mean()

            for group_number in range(1, quantiles + 1):
                record[f"q{group_number}_return"] = quintile_return.get(
                    group_number,
                    np.nan,
                )

            record["high_minus_low_return"] = (
                record[f"q{quantiles}_return"] - record["q1_return"]
            )
            record["status"] = "success"

        except Exception as exc:
            record["status"] = f"quintile_failure: {type(exc).__name__}"

        records.append(record)

    return pd.DataFrame(records)


def summarize_factor_evaluation(
    factor_name: str,
    monthly_metrics: pd.DataFrame,
    periods: Mapping[str, tuple[str | None, str | None]],
    periods_per_year: int = 12,
) -> pd.DataFrame:
    """
    按预先指定的时间窗口汇总因子表现。

    periods 示例：
    {
        "all_available_periods": (None, None),
        "in_sample_2016_2022": ("2016-01-01", "2022-12-31"),
        "out_of_sample_2023_2025": ("2023-01-01", "2025-12-31"),
    }
    """
    metrics = monthly_metrics.copy()
    metrics["signal_date"] = pd.to_datetime(metrics["signal_date"])

    rows = []

    for period_name, (start_date, end_date) in periods.items():
        selected = metrics.copy()

        if start_date is not None:
            selected = selected[
                selected["signal_date"] >= pd.Timestamp(start_date)
            ]

        if end_date is not None:
            selected = selected[
                selected["signal_date"] <= pd.Timestamp(end_date)
            ]

        selected = selected[selected["status"] == "success"].copy()

        rows.append(
            {
                "factor_name": factor_name,
                "period": period_name,
                "observations": int(len(selected)),
                "mean_paired_stock_count": (
                    float(selected["paired_stock_count"].mean())
                    if not selected.empty
                    else np.nan
                ),
                "min_paired_stock_count": (
                    int(selected["paired_stock_count"].min())
                    if not selected.empty
                    else np.nan
                ),
                "mean_ic": (
                    float(selected["ic"].mean())
                    if not selected.empty
                    else np.nan
                ),
                "icir_annualized": _annualized_ir(
                    selected["ic"],
                    periods_per_year=periods_per_year,
                ),
                "mean_rank_ic": (
                    float(selected["rank_ic"].mean())
                    if not selected.empty
                    else np.nan
                ),
                "rank_icir_annualized": _annualized_ir(
                    selected["rank_ic"],
                    periods_per_year=periods_per_year,
                ),
                "positive_rank_ic_ratio": (
                    float((selected["rank_ic"] > 0).mean())
                    if not selected.empty
                    else np.nan
                ),
                "mean_q1_return": (
                    float(selected["q1_return"].mean())
                    if not selected.empty
                    else np.nan
                ),
                "mean_q5_return": (
                    float(selected["q5_return"].mean())
                    if not selected.empty
                    else np.nan
                ),
                "mean_high_minus_low_return": (
                    float(selected["high_minus_low_return"].mean())
                    if not selected.empty
                    else np.nan
                ),
                "high_minus_low_t_stat": _one_sample_t_stat(
                    selected["high_minus_low_return"]
                ),
            }
        )

    return pd.DataFrame(rows)


def run_integrated_profitability_factor_test(
    project_root: str | Path,
    factor_name: str = "integrated_profitability_structure_proxy_strict",
    min_stocks: int = 100,
) -> dict[str, Any]:
    """
    从本地缓存读取综合盈利因子与 20 日收益标签，完成全套检验并保存公开产物。

    不重新请求数据。
    不把个股级因子值、个股收益或价格写入 artifacts/。
    """
    paths = ProjectPaths.from_root(project_root)
    paths.ensure_directories()
    paths.ensure_cache_ignored()

    composite_cache_path = (
        paths.cache_data_dir
        / "05_integrated_profitability_structure_proxy.pkl"
    )
    forward_return_cache_path = (
        paths.cache_data_dir
        / "05_forward_return_20d.pkl"
    )

    if not composite_cache_path.exists():
        raise FileNotFoundError(
            "缺少综合因子缓存，请先运行 05.13.1。"
        )

    if not forward_return_cache_path.exists():
        raise FileNotFoundError(
            "缺少收益标签缓存，请先运行 05.13.2a。"
        )

    composite_payload = pd.read_pickle(composite_cache_path)

    if "final_factor" not in composite_payload:
        raise RuntimeError(
            "综合因子缓存中缺少 final_factor，请重新运行 05.13.1。"
        )

    factor_panel = composite_payload["final_factor"]
    forward_return_panel = pd.read_pickle(
        forward_return_cache_path
    ).reindex(
        index=factor_panel.index,
        columns=factor_panel.columns,
    )

    monthly_metrics = evaluate_factor_panel(
        factor_panel=factor_panel,
        forward_return_panel=forward_return_panel,
        min_stocks=min_stocks,
        quantiles=5,
    )

    successful_metrics = monthly_metrics.loc[
        monthly_metrics["status"] == "success"
    ].copy()

    if successful_metrics.empty:
        raise RuntimeError(
            f"没有日期满足最低 {min_stocks} 只股票的检验门槛。"
        )

    first_available_date = successful_metrics["signal_date"].min()
    last_available_date = successful_metrics["signal_date"].max()

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

    effectiveness_summary = summarize_factor_evaluation(
        factor_name=factor_name,
        monthly_metrics=monthly_metrics,
        periods=periods,
        periods_per_year=12,
    )

    test_failures = monthly_metrics.loc[
        monthly_metrics["status"] != "success"
    ].copy()

    ic_rankic_path = (
        paths.table_dir
        / "05_integrated_profitability_ic_rankic_timeseries.csv"
    )
    quintile_path = (
        paths.table_dir
        / "05_integrated_profitability_quintile_return_timeseries.csv"
    )
    summary_path = (
        paths.table_dir
        / "05_integrated_profitability_effectiveness_summary.csv"
    )
    failures_path = (
        paths.table_dir
        / "05_integrated_profitability_factor_test_failures.csv"
    )
    metadata_path = (
        paths.metadata_dir
        / "05_integrated_profitability_factor_test_rules.json"
    )
    figure_path = (
        paths.figure_dir
        / "05_integrated_profitability_factor_effectiveness.png"
    )

    # 仅保存日期级汇总数据，不保存个股级数据
    save_public_csv(
        monthly_metrics[
            [
                "signal_date",
                "paired_stock_count",
                "ic",
                "rank_ic",
                "status",
            ]
        ],
        ic_rankic_path,
    )

    save_public_csv(
        monthly_metrics[
            [
                "signal_date",
                "paired_stock_count",
                "q1_return",
                "q2_return",
                "q3_return",
                "q4_return",
                "q5_return",
                "high_minus_low_return",
                "status",
            ]
        ],
        quintile_path,
    )

    save_public_csv(effectiveness_summary, summary_path)
    save_public_csv(test_failures, failures_path)

    save_public_json(
        {
            "factor_name": factor_name,
            "fixed_component_weights": {
                "stable_delevered_roe": 0.25,
                "stable_roic_proxy": 0.25,
                "stable_ronoa_proxy": 0.25,
                "fcffic_reconstructed": 0.25,
            },
            "component_missing_value_rule": (
                "All four components must be available; "
                "no stock-level imputation is allowed."
            ),
            "post_combination_treatment": (
                "MAD winsorization, CITIC 2019 level-1 industry "
                "neutralization, log-market-cap neutralization, z-score."
            ),
            "return_label": (
                "T+1 open to T+21 open, "
                "20 trading-day holding return."
            ),
            "minimum_stocks_for_factor_test": min_stocks,
            "first_available_date": first_available_date.strftime(
                "%Y-%m-%d"
            ),
            "last_available_date": last_available_date.strftime(
                "%Y-%m-%d"
            ),
            "in_sample_period": (
                f"{first_available_date.strftime('%Y-%m-%d')} to 2022-12-31"
            ),
            "out_of_sample_period": "2023-01-01 to 2025-12-31",
            "anti_overfitting_rule": (
                "No component weights, components, or out-of-sample dates "
                "are changed after observing results."
            ),
            "individual_stock_level_data_saved_to_artifacts": False,
        },
        metadata_path,
    )

    configure_matplotlib_style()

    plot_metrics = successful_metrics.sort_values("signal_date").copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    axes[0].plot(
        plot_metrics["signal_date"],
        plot_metrics["ic"],
        label="IC",
        linewidth=1.2,
    )
    axes[0].plot(
        plot_metrics["signal_date"],
        plot_metrics["rank_ic"],
        label="RankIC",
        linewidth=1.2,
    )
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_title("Integrated Profitability: Monthly IC and RankIC")
    axes[0].set_xlabel("Signal Date")
    axes[0].set_ylabel("Correlation")
    axes[0].legend()

    nav_frame = pd.DataFrame(
        {
            "signal_date": plot_metrics["signal_date"],
            "Q1": (1 + plot_metrics["q1_return"]).cumprod(),
            "Q2": (1 + plot_metrics["q2_return"]).cumprod(),
            "Q3": (1 + plot_metrics["q3_return"]).cumprod(),
            "Q4": (1 + plot_metrics["q4_return"]).cumprod(),
            "Q5": (1 + plot_metrics["q5_return"]).cumprod(),
        }
    ).set_index("signal_date")

    for quintile_name in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        axes[1].plot(
            nav_frame.index,
            nav_frame[quintile_name],
            label=quintile_name,
            linewidth=1.2,
        )

    axes[1].set_title(
        "Integrated Profitability: Equal-Weighted Quintile NAV"
    )
    axes[1].set_xlabel("Signal Date")
    axes[1].set_ylabel("Gross Net Value")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.show()

    result = {
        "summary": effectiveness_summary,
        "monthly_metrics": monthly_metrics,
        "first_available_date": first_available_date,
        "last_available_date": last_available_date,
        "successful_periods": int(len(successful_metrics)),
        "failed_periods": int(len(test_failures)),
        "paths": {
            "ic_rankic": ic_rankic_path,
            "quintile_returns": quintile_path,
            "summary": summary_path,
            "failures": failures_path,
            "metadata": metadata_path,
            "figure": figure_path,
        },
    }

    print("===== 综合盈利因子等权透明实现有效性汇总 =====")
    print(effectiveness_summary.round(4).to_string(index=False))
    print()
    print("可检验首日：", first_available_date.strftime("%Y-%m-%d"))
    print("可检验末日：", last_available_date.strftime("%Y-%m-%d"))
    print("可检验期数：", len(successful_metrics))
    print("未通过覆盖率门槛的期数：", len(test_failures))
    print("已保存：", summary_path)
    print("已保存：", figure_path)

    return result


def run_out_of_sample_decision_from_summary(
    project_root: str | Path,
    summary_filename: str,
    output_prefix: str,
    factor_name: str,
    oos_period: str = "out_of_sample_2023_2025",
    decision_scope: str = "standard_factor_test",
    min_observations: int = 24,
    min_mean_paired_stocks: int = 100,
    min_positive_rank_ic_ratio: float = 0.55,
    min_rank_icir_annualized: float = 0.50,
    min_hml_t_stat: float = 1.96,
) -> dict[str, Any]:
    """
    基于已保存的样本外汇总表，生成统一、可复用的研究决策日志。

    判定规则：
    - 数据不足：inconclusive_insufficient_coverage；
    - RankIC 或 Q5-Q1 非正：reject_as_standalone_factor；
    - RankICIR、正 RankIC 比例、HML t 值均达阈值：confirmed_candidate；
    - 其余正向但证据不足：candidate_not_confirmed。

    该函数不会根据结果修改阈值、权重或样本区间。
    """
    paths = ProjectPaths.from_root(project_root)
    paths.ensure_directories()

    summary_path = paths.table_dir / summary_filename

    if not summary_path.exists():
        raise FileNotFoundError(
            f"找不到汇总表：{summary_path}"
        )

    summary = pd.read_csv(summary_path)

    required_columns = {
        "factor_name",
        "period",
        "observations",
        "mean_paired_stock_count",
        "mean_rank_ic",
        "rank_icir_annualized",
        "positive_rank_ic_ratio",
        "mean_high_minus_low_return",
        "high_minus_low_t_stat",
    }

    missing_columns = required_columns - set(summary.columns)

    if missing_columns:
        raise RuntimeError(
            "汇总表缺少字段：" + ", ".join(sorted(missing_columns))
        )

    selected = summary.loc[
        (summary["factor_name"] == factor_name)
        & (summary["period"] == oos_period)
    ].copy()

    if len(selected) != 1:
        raise RuntimeError(
            f"未能唯一定位 {factor_name} 在 {oos_period} 的样本外记录。"
        )

    row = selected.iloc[0]

    observations = int(row["observations"])
    mean_paired_stocks = float(row["mean_paired_stock_count"])
    mean_rank_ic = float(row["mean_rank_ic"])
    rank_icir = float(row["rank_icir_annualized"])
    positive_ratio = float(row["positive_rank_ic_ratio"])
    mean_hml = float(row["mean_high_minus_low_return"])
    hml_t_stat = float(row["high_minus_low_t_stat"])

    if (
        observations < min_observations
        or mean_paired_stocks < min_mean_paired_stocks
    ):
        decision = "inconclusive_insufficient_coverage"
        decision_reason = (
            "样本外观测数或平均配对股票数未达到预先设定的最低门槛。"
        )

    elif mean_rank_ic <= 0 or mean_hml <= 0:
        decision = "reject_as_standalone_factor"
        decision_reason = (
            "样本外平均 RankIC 或 Q5-Q1 收益为非正值。"
        )

    elif (
        rank_icir >= min_rank_icir_annualized
        and positive_ratio >= min_positive_rank_ic_ratio
        and hml_t_stat >= min_hml_t_stat
    ):
        decision = "confirmed_candidate"
        decision_reason = (
            "样本外方向、稳定性和高低组收益显著性均达到预设门槛。"
        )

    else:
        decision = "candidate_not_confirmed"
        decision_reason = (
            "样本外 RankIC 与 Q5-Q1 为正，但稳定性或显著性"
            "尚未达到预设确认门槛。"
        )

    decision_log = pd.DataFrame(
        [
            {
                "factor_name": factor_name,
                "period": oos_period,
                "observations": observations,
                "mean_paired_stock_count": mean_paired_stocks,
                "mean_rank_ic": mean_rank_ic,
                "rank_icir_annualized": rank_icir,
                "positive_rank_ic_ratio": positive_ratio,
                "mean_high_minus_low_return": mean_hml,
                "high_minus_low_t_stat": hml_t_stat,
                "decision": decision,
                "decision_scope": decision_scope,
                "decision_reason": decision_reason,
            }
        ]
    )

    decision_path = (
        paths.table_dir / f"{output_prefix}_research_decision_log.csv"
    )
    metadata_path = (
        paths.metadata_dir / f"{output_prefix}_research_decision_rules.json"
    )
    figure_path = (
        paths.figure_dir / f"{output_prefix}_out_of_sample_scorecard.png"
    )

    save_public_csv(decision_log, decision_path)

    save_public_json(
        {
            "factor_name": factor_name,
            "out_of_sample_period": oos_period,
            "decision_scope": decision_scope,
            "predefined_decision_thresholds": {
                "minimum_observations": min_observations,
                "minimum_mean_paired_stocks": min_mean_paired_stocks,
                "minimum_positive_rank_ic_ratio": min_positive_rank_ic_ratio,
                "minimum_rank_icir_annualized": min_rank_icir_annualized,
                "minimum_high_minus_low_t_stat": min_hml_t_stat,
            },
            "decision_rule": (
                "Reject if mean RankIC <= 0 or mean Q5-Q1 <= 0; "
                "confirm only when all predefined stability and "
                "significance thresholds are met; otherwise retain "
                "as candidate_not_confirmed."
            ),
            "anti_overfitting_rule": (
                "No weights, components, thresholds, or sample dates "
                "are modified after observing the result."
            ),
        },
        metadata_path,
    )

    configure_matplotlib_style()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(
        ["Mean RankIC"],
        [mean_rank_ic],
        color="#54A24B" if mean_rank_ic > 0 else "#E45756",
    )
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_title("Out-of-Sample Mean RankIC")
    axes[0].set_ylabel("RankIC")

    axes[1].bar(
        ["Q5 - Q1"],
        [mean_hml],
        color="#54A24B" if mean_hml > 0 else "#E45756",
    )
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_title("Out-of-Sample Mean Q5 - Q1 Return")
    axes[1].set_ylabel("20-Day Gross Return")

    fig.tight_layout()
    fig.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.show()

    print("===== 样本外研究决策 =====")
    print(decision_log.round(4).to_string(index=False))
    print("已保存：", decision_path)
    print("已保存：", metadata_path)
    print("已保存：", figure_path)

    return {
        "decision_log": decision_log,
        "decision_path": decision_path,
        "metadata_path": metadata_path,
        "figure_path": figure_path,
    }
