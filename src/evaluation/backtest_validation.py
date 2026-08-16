from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from math import e, isfinite, log, sqrt
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from zoo.store import FactorZoo, ZooMember


@dataclass(frozen=True)
class DSRReport:
    target_candidate: str | None
    icir_hat: float | None
    icir_0: float | None
    sigma_hat_icir: float | None
    sigma_icir_trials: float | None
    trials: int
    dsr: float | None
    undefined_reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PBOReport:
    pbo: float | None
    combinations_used: int
    zoo_members: int
    blocks: int
    undefined_reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def compute_backtest_validation_report(zoo: FactorZoo, evaluated_records: list[dict]) -> dict:
    return {
        "dsr": compute_dsr(zoo, evaluated_records).to_dict(),
        "pbo": compute_pbo(zoo).to_dict(),
    }


def compute_dsr(zoo: FactorZoo, evaluated_records: list[dict]) -> DSRReport:
    if not zoo.members:
        return DSRReport(None, None, None, None, None, _trial_count(evaluated_records), None, "empty zoo")
    target = _highest_validation_icir_member(zoo.members)
    if target is None:
        return DSRReport(None, None, None, None, None, _trial_count(evaluated_records), None, "no finite zoo validation ICIR")
    icir_hat = _validation_icir(target)
    n_trials = _trial_count(evaluated_records)
    if n_trials < 2:
        return DSRReport(target.name, icir_hat, None, None, None, n_trials, None, "fewer than two fully evaluated trials")
    finite_trial_icirs = np.asarray(
        [
            float(record["decision"]["validation"]["icir"])
            for record in evaluated_records
            if _has_full_step4_metrics(record) and isfinite(float(record["decision"]["validation"].get("icir", np.nan)))
        ],
        dtype=float,
    )
    if len(finite_trial_icirs) < 2:
        return DSRReport(target.name, icir_hat, None, None, None, n_trials, None, "fewer than two finite-ICIR trials for cross-trial standard deviation")
    sigma_icir_trials = float(np.std(finite_trial_icirs, ddof=1))
    if sigma_icir_trials == 0 or not isfinite(sigma_icir_trials):
        return DSRReport(target.name, icir_hat, None, None, sigma_icir_trials, n_trials, None, "zero or non-finite cross-trial ICIR variance")
    daily_ic = _load_series(target.validation_daily_ic_path)
    finite_daily = daily_ic.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    t = int(finite_daily.shape[0])
    if t < 2:
        return DSRReport(target.name, icir_hat, None, None, sigma_icir_trials, n_trials, None, "target validation daily IC has fewer than two finite days")
    skew = _sample_skew(finite_daily)
    excess_kurtosis = _sample_excess_kurtosis(finite_daily)
    sigma_hat_icir = sqrt(
        max(0.0, (1.0 - skew * icir_hat + ((excess_kurtosis - 1.0) / 4.0) * (icir_hat**2)) / (t - 1))
    )
    if sigma_hat_icir == 0 or not isfinite(sigma_hat_icir):
        return DSRReport(target.name, icir_hat, None, sigma_hat_icir, sigma_icir_trials, n_trials, None, "zero or non-finite target ICIR standard error")
    normal = NormalDist()
    euler_gamma = 0.5772
    icir_0 = sigma_icir_trials * (
        (1.0 - euler_gamma) * normal.inv_cdf(1.0 - 1.0 / n_trials)
        + euler_gamma * normal.inv_cdf(1.0 - 1.0 / (n_trials * e))
    )
    dsr = normal.cdf((icir_hat - icir_0) / sigma_hat_icir)
    return DSRReport(target.name, icir_hat, float(icir_0), float(sigma_hat_icir), sigma_icir_trials, n_trials, float(dsr))


def compute_pbo(zoo: FactorZoo, blocks: int = 16) -> PBOReport:
    if len(zoo.members) < 2:
        return PBOReport(None, 0, len(zoo.members), blocks, "zoo has fewer than two members")
    matrix = _training_ic_matrix(zoo.members)
    if matrix.shape[1] < 2:
        return PBOReport(None, 0, matrix.shape[1], blocks, "fewer than two members with stored training daily IC")
    t_days = matrix.shape[0]
    if t_days < blocks:
        return PBOReport(None, 0, matrix.shape[1], blocks, "fewer training days than CSCV blocks")

    base_size = t_days // blocks
    remainder = t_days % blocks
    block_sizes = [base_size + (1 if i < remainder else 0) for i in range(blocks)]
    block_labels = np.repeat(np.arange(blocks), block_sizes)

    lambdas: list[float] = []
    for is_blocks in combinations(range(blocks), blocks // 2):
        is_mask = np.isin(block_labels, is_blocks)
        is_means = matrix.iloc[is_mask].mean(axis=0)

        # In-sample best selection: exact ties break deterministically by selecting
        # the candidate appearing first in zoo order (matrix column order).
        max_is_ic = is_means.max()
        tied_candidates = is_means[is_means == max_is_ic].index.tolist()
        best_name = str(tied_candidates[0])

        oos_means = matrix.iloc[~is_mask].mean(axis=0)
        ranks = oos_means.rank(method="average", ascending=True)
        omega = float(ranks[best_name] / (matrix.shape[1] + 1))
        lambdas.append(log(omega / (1.0 - omega)))
    pbo = float(np.mean(np.asarray(lambdas) <= 0.0))
    return PBOReport(pbo, len(lambdas), matrix.shape[1], blocks)


def _highest_validation_icir_member(members: list[ZooMember]) -> ZooMember | None:
    finite_members = [member for member in members if isfinite(_validation_icir(member))]
    if not finite_members:
        return None
    return max(finite_members, key=_validation_icir)


def _validation_icir(member: ZooMember) -> float:
    try:
        return float(member.metrics["validation"]["icir"])
    except (KeyError, TypeError, ValueError):
        return np.nan


def _trial_count(records: list[dict]) -> int:
    return sum(1 for record in records if _has_full_step4_metrics(record))


def _has_full_step4_metrics(record: dict) -> bool:
    decision = record.get("decision")
    return isinstance(decision, dict) and isinstance(decision.get("train"), dict) and isinstance(decision.get("validation"), dict)


def _training_ic_matrix(members: list[ZooMember]) -> pd.DataFrame:
    series = {}
    for member in members:
        if member.train_daily_ic_path:
            series[member.name] = _load_series(member.train_daily_ic_path)
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).dropna(how="any")


def _load_series(path: str | None) -> pd.Series:
    if not path:
        return pd.Series(dtype=float)
    source = Path(path)
    frame = pd.read_parquet(source)
    return frame.iloc[:, 0]


def _sample_skew(series: pd.Series) -> float:
    values = series.to_numpy(dtype=float)
    std = np.std(values, ddof=1)
    if values.size < 3 or std == 0:
        return 0.0
    centered = values - np.mean(values)
    return float(np.mean(centered**3) / (std**3))


def _sample_excess_kurtosis(series: pd.Series) -> float:
    values = series.to_numpy(dtype=float)
    std = np.std(values, ddof=1)
    if values.size < 4 or std == 0:
        return 0.0
    centered = values - np.mean(values)
    return float(np.mean(centered**4) / (std**4) - 3.0)
