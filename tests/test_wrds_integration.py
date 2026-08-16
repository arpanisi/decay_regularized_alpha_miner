from __future__ import annotations

import os

import pytest

from config.env import load_project_env
from data.connectors import connect_wrds_noninteractive


@pytest.mark.integration
def test_wrds_crsp_dsf_v2_connection() -> None:
    load_project_env()
    if os.environ.get("RUN_WRDS_INTEGRATION") != "1":
        pytest.skip("set RUN_WRDS_INTEGRATION=1 to run live WRDS connection test")
    db = connect_wrds_noninteractive()
    frame = db.raw_sql(
        """
        select dlycaldt, ticker, dlyclose, dlycap, dlyret
        from crsp.dsf_v2
        where dlycaldt >= '2020-01-02'
          and dlycaldt <= '2020-01-03'
          and ticker is not null
        limit 5
        """
    )
    assert not frame.empty
    assert {"dlycaldt", "ticker", "dlyclose", "dlycap", "dlyret"}.issubset(frame.columns)
