from __future__ import annotations

import pandas as pd

from evaluation.gate import AlignmentResult, decide_gate
from evaluation.metrics import WindowMetrics
from zoo.store import FactorZoo, ZooMember


def test_gate_rejects_any_failed_criterion() -> None:
    good = WindowMetrics(0.006, 0.02, 0.006, 0.02, 0.3, 0.7, 0.8, 0.95)
    bad = WindowMetrics(0.004, 0.02, 0.004, 0.02, 0.2, 0.7, 0.8, 0.95)
    decision = decide_gate(good, bad, AlignmentResult(1.0, "aligned"))
    assert not decision.accepted
    assert "validation:abs_ic" in decision.failed_criteria


def test_breadth_equals_n_for_identity_correlation_matrix() -> None:
    zoo = FactorZoo("unused.json")
    zoo.members = [
        ZooMember("a", "r", "e", 1, {}, "a.parquet"),
        ZooMember("b", "r", "e", 1, {}, "b.parquet"),
        ZooMember("c", "r", "e", 1, {}, "c.parquet"),
    ]
    zoo.correlation_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert zoo.breadth() == 3.0


def test_zoo_save_load_preserves_member_and_validation_values(tmp_path) -> None:
    value_path = tmp_path / "a.parquet"
    index = pd.MultiIndex.from_product(
        [pd.date_range("2021-01-01", periods=2), ["AAA", "BBB"]],
        names=["datetime", "instrument"],
    )
    pd.Series([1.0, 2.0, 3.0, 4.0], index=index, name="value").to_frame().to_parquet(value_path)
    zoo_path = tmp_path / "zoo.json"
    zoo = FactorZoo(zoo_path)
    zoo.members = [ZooMember("a", "r", "e", 1, {"accepted": True}, str(value_path))]
    zoo.correlation_matrix = [[1.0]]
    zoo.save()
    loaded = FactorZoo.load(zoo_path)
    values = loaded.load_member_values()
    assert loaded.members[0].validation_values_path == str(value_path)
    assert values["a"].tolist() == [1.0, 2.0, 3.0, 4.0]
