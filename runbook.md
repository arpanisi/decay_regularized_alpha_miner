# Vast.ai Runbook

This project runs from `alphamining-ast/decay_regularized_alpha_miner`.

## Environment

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

WRDS authentication and OpenRouter credentials are loaded from `.env` by runtime files. The present `.env` variables are `WRDS_USERID`, `WRDS_PGPASS`, `OPENROUTER_API_KEY`, and `DEEPSEEK_API_KEY`. Cheap OpenRouter model defaults are stored in `config/cheap_models.yaml`; command-line model flags override those defaults.

## Full Data Build

Run this once. Later steps read local files only.

```bash
python src/runtime/build_full_data.py
```

Equivalent expanded commands:

```bash
python src/runtime/fetch_crsp.py \
  --start-date 2015-01-01 \
  --end-date 2023-12-31 \
  --universe-size 2750 \
  --output-crsp data/raw/crsp_dsf_v2.parquet \
  --output-universe data/raw/universe.txt

python src/runtime/fetch_sec_facts.py \
  --universe data/raw/universe.txt \
  --user-agent "decay-regularized-alpha-miner contact@example.com" \
  --output data/raw/sec_facts.csv

python src/runtime/build_panel.py \
  --crsp-parquet data/raw/crsp_dsf_v2.parquet \
  --facts-csv data/raw/sec_facts.csv \
  --output outputs/panel/full_panel.parquet
```

## Seed Evaluation

Run after `outputs/panel/full_panel.parquet` exists.

```bash
python src/runtime/evaluate_seeds.py \
  --panel outputs/panel/full_panel.parquet \
  --output outputs/reports/seed_metrics.json

python src/runtime/seed_zoo.py \
  --panel outputs/panel/full_panel.parquet \
  --zoo outputs/zoo/zoo.json \
  --values-dir outputs/zoo/values \
  --report outputs/reports/seed_zoo_report.json
```

## Tier 1

Prerequisite: `Full Data Build` and `Seed Evaluation` should be completed first so that `full_panel.parquet` and baseline seed metrics exist. Tier 1 uses a prebuilt panel subset matching the run budget: 50 names, one train period, one validation period.

```bash
python src/runtime/run_tier1.py
```

Equivalent expanded commands:

```bash
python src/runtime/slice_panel.py \
  --panel outputs/panel/full_panel.parquet \
  --output outputs/panel/tier1_panel.parquet \
  --start-date 2020-01-02 \
  --end-date 2020-02-07 \
  --max-instruments 50

python src/runtime/run_mining.py \
  --panel outputs/panel/tier1_panel.parquet \
  --zoo outputs/zoo/tier1_zoo.json \
  --values-dir outputs/zoo/tier1_values \
  --log outputs/reports/tier1_mining_log.json \
  --rounds 1 \
  --train-start 2020-01-02 \
  --train-end 2020-01-31 \
  --validation-start 2020-02-03 \
  --validation-end 2020-02-07
```

## Tier 2

Tier 2 is coded but not run locally by default. It uses a full-universe panel subset with train year 2020 and validation 2021 Q1.

```bash
python src/runtime/run_tier2.py
```

Equivalent expanded commands:

```bash
python src/runtime/slice_panel.py \
  --panel outputs/panel/full_panel.parquet \
  --output outputs/panel/tier2_panel.parquet \
  --start-date 2020-01-01 \
  --end-date 2021-03-31

python src/runtime/run_mining.py \
  --panel outputs/panel/tier2_panel.parquet \
  --zoo outputs/zoo/tier2_zoo.json \
  --values-dir outputs/zoo/tier2_values \
  --log outputs/reports/tier2_mining_log.json \
  --rounds 5 \
  --train-start 2020-01-01 \
  --train-end 2020-12-31 \
  --validation-start 2021-01-01 \
  --validation-end 2021-03-31
```

## Tier 3

Tier 3 is coded but not run locally by default. It is the full 50-round, 400-candidate mining run.

```bash
python src/runtime/run_tier3.py
```

Equivalent expanded command:

```bash
python src/runtime/run_mining.py \
  --panel outputs/panel/full_panel.parquet \
  --zoo outputs/zoo/zoo.json \
  --values-dir outputs/zoo/values \
  --log outputs/reports/full_mining_log.json \
  --rounds 50
```

## Verification

```bash
python -m pytest
```

Optional live SEC connection check:

```bash
RUN_SEC_INTEGRATION=1 python -m pytest tests/test_sec_integration.py
```

Optional live WRDS connection check:

```bash
RUN_WRDS_INTEGRATION=1 python -m pytest tests/test_wrds_integration.py
```
