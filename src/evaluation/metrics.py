from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowMetrics:
    ic: float
    ic_std: float
    rank_ic: float
    rank_ic_std: float
    icir: float
    lag1_autocorr: float
    monthly_robustness: float
    coverage: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class DailyMetricSeries:
    daily_ic: pd.Series
    daily_rank_ic: pd.Series


def compute_daily_metric_series(values: pd.Series, panel: pd.DataFrame) -> DailyMetricSeries:
    label = panel["label_fwd_10d"].astype(float)
    return DailyMetricSeries(
        daily_ic=_daily_cross_sectional_corr(values, label),
        daily_rank_ic=_daily_cross_sectional_corr(_cs_rank(values), _cs_rank(label)),
    )


def compute_window_metrics(values: pd.Series, panel: pd.DataFrame) -> WindowMetrics:
    daily = compute_daily_metric_series(values, panel)
    daily_ic = daily.daily_ic
    daily_rank_ic = daily.daily_rank_ic
    ic = float(daily_ic.mean())
    ic_std = float(daily_ic.std(ddof=1))
    rank_ic = float(daily_rank_ic.mean())
    rank_ic_std = float(daily_rank_ic.std(ddof=1))
    icir = float(ic / ic_std) if daily_ic.count() >= 2 and ic_std != 0 else np.nan
    return WindowMetrics(
        ic=ic,
        ic_std=ic_std,
        rank_ic=rank_ic,
        rank_ic_std=rank_ic_std,
        icir=icir,
        lag1_autocorr=float(_lag1_autocorr(values).mean()),
        monthly_robustness=float(_monthly_robustness(daily_ic, ic)),
        coverage=float(values.notna().mean()),
    )


def _daily_cross_sectional_corr(a: pd.Series, b: pd.Series) -> pd.Series:
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    return df.groupby(level="datetime").apply(lambda g: _pearson(g["a"], g["b"]))


def _cs_rank(series: pd.Series) -> pd.Series:
    return series.groupby(level="datetime", group_keys=False).rank(method="average", pct=True)


def _lag1_autocorr(values: pd.Series) -> pd.Series:
    prev = values.groupby(level="instrument", group_keys=False).shift(1)
    df = pd.DataFrame({"cur": values, "prev": prev}).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    return df.groupby(level="datetime").apply(lambda g: _pearson(g["cur"], g["prev"]))


def _monthly_robustness(daily_ic: pd.Series, window_ic: float) -> float:
    if daily_ic.empty or not np.isfinite(window_ic) or window_ic == 0:
        return np.nan
    months = daily_ic.groupby(pd.PeriodIndex(daily_ic.index, freq="M")).mean()
    return float((np.sign(months) == np.sign(window_ic)).mean())


def _pearson(a: pd.Series, b: pd.Series) -> float:
    finite = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(finite) < 2 or finite["a"].std(ddof=1) == 0 or finite["b"].std(ddof=1) == 0:
        return np.nan
    return float(finite["a"].corr(finite["b"]))
