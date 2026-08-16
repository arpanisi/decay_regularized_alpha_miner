from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from evaluation.metrics import WindowMetrics


@dataclass(frozen=True)
class AlignmentResult:
    score: float
    justification: str


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    failed_criteria: tuple[str, ...]
    train: dict[str, float]
    validation: dict[str, float]
    alignment_score: float
    alignment_justification: str

    def to_dict(self) -> dict:
        return asdict(self)


def decide_gate(
    train: WindowMetrics,
    validation: WindowMetrics,
    alignment: AlignmentResult,
    thresholds: dict[str, float] | None = None,
) -> GateDecision:
    t = thresholds or {
        "min_abs_ic": 0.005,
        "min_abs_icir": 0.1,
        "min_lag1_autocorr": 0.6,
        "min_monthly_robustness": 0.7,
        "min_coverage": 0.9,
        "min_alignment_score": 0.5,
    }
    failures: list[str] = []
    for name, metrics in [("train", train), ("validation", validation)]:
        if not (abs(metrics.ic) >= t["min_abs_ic"]):
            failures.append(f"{name}:abs_ic")
        if not (np.isfinite(metrics.icir) and abs(metrics.icir) > t["min_abs_icir"]):
            failures.append(f"{name}:abs_icir")
        if not (metrics.lag1_autocorr > t["min_lag1_autocorr"]):
            failures.append(f"{name}:lag1_autocorr")
        if not (metrics.monthly_robustness > t["min_monthly_robustness"]):
            failures.append(f"{name}:monthly_robustness")
        if not (metrics.coverage > t["min_coverage"]):
            failures.append(f"{name}:coverage")
    if not (_same_nonzero_sign(train.ic, validation.ic)):
        failures.append("validation_ic_same_sign")
    if not (alignment.score >= t["min_alignment_score"]):
        failures.append("alignment_score")
    return GateDecision(
        accepted=not failures,
        failed_criteria=tuple(failures),
        train=train.to_dict(),
        validation=validation.to_dict(),
        alignment_score=float(alignment.score),
        alignment_justification=alignment.justification,
    )


def _same_nonzero_sign(a: float, b: float) -> bool:
    return np.isfinite(a) and np.isfinite(b) and a != 0 and b != 0 and np.sign(a) == np.sign(b)
