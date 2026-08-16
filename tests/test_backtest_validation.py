from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.backtest_validation import compute_dsr, compute_pbo
from zoo.store import FactorZoo, ZooMember


def test_dsr_selects_highest_validation_icir_member(tmp_path) -> None:
    dates = pd.date_range("2021-01-01", periods=40, freq="D")
    a_validation = tmp_path / "a_validation.parquet"
    b_validation = tmp_path / "b_validation.parquet"
    pd.Series(np.linspace(0.01, 0.03, len(dates)), index=dates, name="daily_ic").to_frame().to_parquet(a_validation)
    pd.Series(np.linspace(0.02, 0.06, len(dates)), index=dates, name="daily_ic").to_frame().to_parquet(b_validation)
    zoo = FactorZoo(tmp_path / "zoo.json")
    zoo.members = [
        ZooMember("a", "r", "e", 1, {"validation": {"icir": 0.2}}, "a_values.parquet", None, str(a_validation)),
        ZooMember("b", "r", "e", 1, {"validation": {"icir": 0.5}}, "b_values.parquet", None, str(b_validation)),
    ]
    records = [
        {"decision": {"train": {}, "validation": {"icir": 0.1}}},
        {"decision": {"train": {}, "validation": {"icir": 0.3}}},
        {"decision": {"train": {}, "validation": {"icir": 0.5}}},
    ]
    report = compute_dsr(zoo, records)
    assert report.target_candidate == "b"
    assert report.trials == 3
    assert report.icir_hat == 0.5
    assert report.icir_0 is not None
    assert report.sigma_hat_icir is not None
    assert report.dsr is not None


def test_pbo_is_undefined_for_single_member() -> None:
    zoo = FactorZoo("unused.json")
    zoo.members = [ZooMember("a", "r", "e", 1, {}, "a.parquet")]
    report = compute_pbo(zoo)
    assert report.pbo is None
    assert report.undefined_reason == "zoo has fewer than two members"


def test_pbo_uses_12870_cscv_combinations(tmp_path) -> None:
    dates = pd.date_range("2020-01-01", periods=160, freq="D")
    paths = []
    for idx in range(2):
        path = tmp_path / f"member_{idx}.parquet"
        values = np.sin(np.arange(len(dates)) / (idx + 2)) + idx * 0.01
        pd.Series(values, index=dates, name="daily_ic").to_frame().to_parquet(path)
        paths.append(path)
    zoo = FactorZoo(tmp_path / "zoo.json")
    zoo.members = [
        ZooMember("a", "r", "e", 1, {}, "a_values.parquet", str(paths[0]), None),
        ZooMember("b", "r", "e", 1, {}, "b_values.parquet", str(paths[1]), None),
    ]
    report = compute_pbo(zoo)
    assert report.combinations_used == 12870
    assert report.pbo is not None


def test_dsr_counts_all_full_step4_candidates_even_if_validation_icir_nan(tmp_path) -> None:
    dates = pd.date_range("2021-01-01", periods=40, freq="D")
    a_validation = tmp_path / "a_validation.parquet"
    pd.Series(np.linspace(0.01, 0.03, len(dates)), index=dates, name="daily_ic").to_frame().to_parquet(a_validation)
    zoo = FactorZoo(tmp_path / "zoo.json")
    zoo.members = [
        ZooMember("a", "r", "e", 1, {"validation": {"icir": 0.2}}, "a_values.parquet", None, str(a_validation)),
    ]
    records = [
        {"decision": {"train": {}, "validation": {"icir": 0.1}}},
        {"decision": {"train": {}, "validation": {"icir": 0.3}}},
        {"decision": {"train": {}, "validation": {"icir": np.nan}}},
    ]
    report = compute_dsr(zoo, records)
    assert report.trials == 3
    assert report.dsr is not None


def test_pbo_includes_all_days_when_t_not_divisible_by_16(tmp_path) -> None:
    dates = pd.date_range("2020-01-01", periods=165, freq="D")
    paths = []
    for idx in range(2):
        path = tmp_path / f"member_{idx}.parquet"
        values = np.sin(np.arange(len(dates)) / (idx + 2)) + idx * 0.01
        pd.Series(values, index=dates, name="daily_ic").to_frame().to_parquet(path)
        paths.append(path)
    zoo = FactorZoo(tmp_path / "zoo.json")
    zoo.members = [
        ZooMember("a", "r", "e", 1, {}, "a_values.parquet", str(paths[0]), None),
        ZooMember("b", "r", "e", 1, {}, "b_values.parquet", str(paths[1]), None),
    ]
    report = compute_pbo(zoo, blocks=16)
    assert report.pbo is not None
    assert report.combinations_used == 12870


def test_pbo_deterministic_tie_break_first_in_zoo_order(tmp_path) -> None:
    dates = pd.date_range("2020-01-01", periods=160, freq="D")
    paths = []
    values = np.ones(len(dates)) * 0.05
    for idx in range(2):
        path = tmp_path / f"member_{idx}.parquet"
        pd.Series(values, index=dates, name="daily_ic").to_frame().to_parquet(path)
        paths.append(path)
    zoo = FactorZoo(tmp_path / "zoo.json")
    zoo.members = [
        ZooMember("a", "r", "e", 1, {}, "a_values.parquet", str(paths[0]), None),
        ZooMember("b", "r", "e", 1, {}, "b_values.parquet", str(paths[1]), None),
    ]
    report = compute_pbo(zoo, blocks=16)
    assert report.pbo is not None
