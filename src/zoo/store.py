from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ZooMember:
    name: str
    rationale: str
    expression: str
    node_count: int
    metrics: dict
    validation_values_path: str
    train_daily_ic_path: str | None = None
    validation_daily_ic_path: str | None = None


class FactorZoo:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.members: list[ZooMember] = []
        self.correlation_matrix: list[list[float]] = []

    def nearest(self, values: pd.Series, member_values: dict[str, pd.Series]) -> list[tuple[str, float]]:
        scores = []
        for member in self.members:
            corr = mean_cross_sectional_corr(values, member_values[member.name])
            scores.append((member.name, corr))
        return sorted(scores, key=lambda item: abs(item[1]), reverse=True)[:3]

    def add(self, member: ZooMember, values: pd.Series, member_values: dict[str, pd.Series]) -> float:
        n_old = len(self.members)
        if n_old == 0:
            self.correlation_matrix = [[1.0]]
        else:
            new_corrs = [mean_cross_sectional_corr(values, member_values[m.name]) for m in self.members]
            for row, corr in zip(self.correlation_matrix, new_corrs):
                row.append(float(corr))
            self.correlation_matrix.append([float(c) for c in new_corrs] + [1.0])
        self.members.append(member)
        return self.breadth()

    def breadth(self) -> float:
        n = len(self.correlation_matrix)
        if n == 0:
            return 0.0
        r = np.asarray(self.correlation_matrix, dtype=float)
        w = np.full(n, 1.0 / n)
        denom = float(w.T @ r @ w)
        return float(1.0 / denom) if denom != 0 else np.inf

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "members": [asdict(member) for member in self.members],
            "correlation_matrix": self.correlation_matrix,
            "breadth": self.breadth(),
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FactorZoo":
        zoo = cls(path)
        source = Path(path)
        if not source.exists():
            return zoo
        payload = json.loads(source.read_text(encoding="utf-8"))
        zoo.members = [
            ZooMember(
                **(
                    {
                        "train_daily_ic_path": None,
                        "validation_daily_ic_path": None,
                    }
                    | member
                )
            )
            for member in payload.get("members", [])
        ]
        zoo.correlation_matrix = payload.get("correlation_matrix", [])
        return zoo

    def load_member_values(self) -> dict[str, pd.Series]:
        values: dict[str, pd.Series] = {}
        for member in self.members:
            frame = pd.read_parquet(member.validation_values_path)
            values[member.name] = frame.iloc[:, 0]
        return values


def mean_cross_sectional_corr(a: pd.Series, b: pd.Series) -> float:
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    if df.empty:
        return np.nan
    daily = df.groupby(level="datetime").apply(_corr)
    return float(daily.mean())


def _corr(group: pd.DataFrame) -> float:
    if len(group) < 2 or group["a"].std(ddof=1) == 0 or group["b"].std(ddof=1) == 0:
        return np.nan
    return float(group["a"].corr(group["b"]))
