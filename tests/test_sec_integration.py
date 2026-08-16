from __future__ import annotations

import os

import pytest

from config.env import load_project_env
from data.connectors import fetch_sec_company_tickers


@pytest.mark.integration
def test_sec_company_tickers_connection() -> None:
    load_project_env()
    if os.environ.get("RUN_SEC_INTEGRATION") != "1":
        pytest.skip("set RUN_SEC_INTEGRATION=1 to run live SEC connection test")
    user_agent = os.environ.get("SEC_USER_AGENT", "decay-regularized-alpha-miner contact@example.com")
    payload = fetch_sec_company_tickers(user_agent)
    assert isinstance(payload, dict)
    assert payload
    first = next(iter(payload.values()))
    assert "ticker" in first
    assert "cik_str" in first
