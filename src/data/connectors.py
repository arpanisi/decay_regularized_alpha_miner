from __future__ import annotations

import time
import os
from typing import Any

import pandas as pd
import requests


def fetch_crsp_dsf_v2(start_date: str, end_date: str) -> pd.DataFrame:
    try:
        import wrds
    except ImportError as exc:
        raise RuntimeError("Install the official wrds package before fetching CRSP data") from exc
    db = connect_wrds_noninteractive(wrds)
    query = """
        select dlycaldt, permno, ticker, primaryexch, shrcd,
               dlyopen, dlyhigh, dlylow, dlyclose, dlyvol, dlycap,
               dlyret, dlyreti
        from crsp.dsf_v2
        where dlycaldt between %(start)s and %(end)s
          and sharetype = 'NS'
          and securitytype = 'EQTY'
          and securitysubtype = 'COM'
          and usincflg = 'Y'
          and issuertype in ('CORP', 'REIT')
    """
    try:
        return db.raw_sql(query, params={"start": start_date, "end": end_date})
    finally:
        db.close()


def connect_wrds_noninteractive(wrds_module: Any | None = None) -> Any:
    if wrds_module is None:
        try:
            import wrds as wrds_module
        except ImportError as exc:
            raise RuntimeError("Install the official wrds package before connecting to WRDS") from exc
    username = os.environ.get("WRDS_USERID")
    password = _wrds_password_from_env()
    if not username or not password:
        raise RuntimeError("WRDS_USERID and WRDS_PGPASS are required for non-interactive WRDS access")
    db = wrds_module.Connection(
        autoconnect=False,
        wrds_username=username,
        wrds_password=password,
    )
    db._Connection__make_sa_engine_conn(raise_err=True)
    db.load_library_list()
    return db


def _wrds_password_from_env() -> str | None:
    raw = os.environ.get("WRDS_PASSWORD") or os.environ.get("WRDS_PGPASS")
    if raw and raw.count(":") >= 4:
        return raw.rsplit(":", 1)[-1]
    return raw


def fetch_sec_company_tickers(user_agent: str) -> dict[str, Any]:
    response = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers={"User-Agent": user_agent},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def fetch_companyfacts(cik: int, user_agent: str, min_interval_seconds: float = 0.11) -> dict[str, Any]:
    time.sleep(min_interval_seconds)
    response = requests.get(
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json",
        headers={"User-Agent": user_agent},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()
