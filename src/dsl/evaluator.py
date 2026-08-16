from __future__ import annotations

import numpy as np
import pandas as pd

from dsl.ast import Call, Expr, Field, Name, Number, ParsedExpression


def evaluate(parsed: ParsedExpression, panel: pd.DataFrame) -> pd.Series:
    env: dict[str, pd.Series] = {}
    for name, expr in parsed.assignments:
        env[name] = _eval(expr, panel, env)
    return _eval(parsed.output, panel, env).astype(float)


def _eval(expr: Expr, panel: pd.DataFrame, env: dict[str, pd.Series]) -> pd.Series:
    if isinstance(expr, Number):
        return pd.Series(expr.value, index=panel.index, dtype=float)
    if isinstance(expr, Field):
        if expr.name not in panel.columns:
            raise KeyError(f"panel missing field {expr.name}")
        return panel[expr.name].astype(float)
    if isinstance(expr, Name):
        return env[expr.value]
    args = [_eval(arg, panel, env) for arg in expr.args]
    op = expr.operator
    if op == "ADD":
        return args[0] + args[1]
    if op == "SUBTRACT":
        return args[0] - args[1]
    if op == "MULTIPLY":
        return args[0] * args[1]
    if op == "DIVIDE":
        return (args[0] / args[1].where(args[1] != 0)).replace([np.inf, -np.inf], np.nan)
    if op == "ABS":
        return args[0].abs()
    if op == "SIGN":
        return np.sign(args[0])
    if op == "LOG":
        return np.log(args[0].where(args[0] > 0))
    if op == "POW":
        return args[0].pow(_scalar_int(args[1], "POW exponent"))
    if op in {"GT", "LT"}:
        cond = args[0] > args[1] if op == "GT" else args[0] < args[1]
        return cond.astype(float).where(args[0].notna() & args[1].notna())
    if op == "IF_THEN_ELSE":
        truthy = args[0].notna() & (args[0] != 0)
        return args[1].where(truthy, args[2])
    if op.startswith("TS_") or op in {"DELTA", "DELAY"}:
        return _eval_ts(op, args)
    if op.startswith("CS_"):
        return _eval_cs(op, args)
    raise ValueError(f"unsupported operator {op}")


def _eval_ts(op: str, args: list[pd.Series]) -> pd.Series:
    by_inst = lambda s: s.groupby(level="instrument", group_keys=False)
    if op in {"TS_MEAN", "TS_SUM", "TS_MIN", "TS_MAX"}:
        window = _scalar_int(args[1], f"{op} window")
        rolling = by_inst(args[0]).rolling(window, min_periods=1)
        out = {"TS_MEAN": rolling.mean, "TS_SUM": rolling.sum, "TS_MIN": rolling.min, "TS_MAX": rolling.max}[op]()
        return out.droplevel(0).reindex(args[0].index)
    if op == "TS_STD":
        window = _scalar_int(args[1], "TS_STD window")
        return by_inst(args[0]).rolling(window, min_periods=2).std(ddof=1).droplevel(0).reindex(args[0].index)
    if op == "DELTA":
        return args[0] - by_inst(args[0]).shift(_scalar_int(args[1], "DELTA period"))
    if op == "DELAY":
        return by_inst(args[0]).shift(_scalar_int(args[1], "DELAY period"))
    if op == "TS_RANK":
        window = _scalar_int(args[1], "TS_RANK window")
        return by_inst(args[0]).rolling(window, min_periods=1).apply(_last_pct_rank, raw=False).droplevel(0).reindex(args[0].index)
    if op == "TS_CORR":
        window = _scalar_int(args[2], "TS_CORR window")
        df = pd.DataFrame({"x": args[0], "y": args[1]})
        return df.groupby(level="instrument", group_keys=False).apply(
            lambda g: _rolling_corr(g["x"], g["y"], window)
        ).reindex(args[0].index)
    if op == "TS_SINCE":
        return by_inst(args[0]).apply(_ts_since).droplevel(0).reindex(args[0].index)
    if op == "TS_COUNT":
        window = _scalar_int(args[1], "TS_COUNT window")
        truthy = args[0].notna() & (args[0] != 0)
        return by_inst(truthy.astype(float)).rolling(window, min_periods=1).sum().droplevel(0).reindex(args[0].index)
    raise ValueError(op)


def _eval_cs(op: str, args: list[pd.Series]) -> pd.Series:
    by_day = lambda s: s.groupby(level="datetime", group_keys=False)
    if op == "CS_RANK":
        return by_day(args[0]).rank(method="average", pct=True)
    if op == "CS_ZSCORE":
        return by_day(args[0]).transform(_zscore)
    if op == "CS_WINSORIZE":
        lo = _scalar_float(args[1], "CS_WINSORIZE lo")
        hi = _scalar_float(args[2], "CS_WINSORIZE hi")
        return by_day(args[0]).transform(lambda s: s.clip(s.quantile(lo), s.quantile(hi)))
    if op == "CS_BUCKET":
        n = _scalar_int(args[1], "CS_BUCKET n")
        return by_day(args[0]).transform(lambda s: _bucket(s, n))
    if op == "CS_NEUTRALIZE":
        df = pd.DataFrame({"x": args[0], "group": args[1]})
        return df.groupby(level="datetime", group_keys=False).apply(
            lambda g: g["x"] - g.groupby("group", dropna=False)["x"].transform("mean")
        ).reindex(args[0].index)
    raise ValueError(op)


def _scalar_int(series: pd.Series, label: str) -> int:
    finite = series.dropna().unique()
    if len(finite) != 1 or int(finite[0]) != finite[0]:
        raise ValueError(f"{label} must be one integer literal")
    return int(finite[0])


def _scalar_float(series: pd.Series, label: str) -> float:
    finite = series.dropna().unique()
    if len(finite) != 1:
        raise ValueError(f"{label} must be one numeric literal")
    return float(finite[0])


def _last_pct_rank(window: pd.Series) -> float:
    finite = window.dropna()
    if finite.empty or pd.isna(window.iloc[-1]):
        return np.nan
    return finite.rank(method="average").iloc[-1] / len(finite)


def _rolling_corr(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    out = []
    for end in range(len(x)):
        start = max(0, end - window + 1)
        frame = pd.DataFrame({"x": x.iloc[start : end + 1], "y": y.iloc[start : end + 1]}).dropna()
        if len(frame) < 2 or frame["x"].std(ddof=1) == 0 or frame["y"].std(ddof=1) == 0:
            out.append(np.nan)
        else:
            out.append(float(frame["x"].corr(frame["y"])))
    return pd.Series(out, index=x.index, dtype=float)


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=1)
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (series - series.mean()) / std


def _ts_since(series: pd.Series) -> pd.Series:
    out = []
    last = None
    for i, value in enumerate(series):
        if pd.notna(value) and value != 0:
            last = i
            out.append(0.0)
        elif last is None:
            out.append(np.nan)
        else:
            out.append(float(i - last))
    return pd.Series(out, index=series.index, dtype=float)


def _bucket(series: pd.Series, n: int) -> pd.Series:
    finite = series.dropna()
    out = pd.Series(np.nan, index=series.index, dtype=float)
    if finite.empty:
        return out
    ranks = finite.rank(method="first", pct=True)
    out.loc[finite.index] = np.ceil(ranks * n).clip(1, n)
    return out
