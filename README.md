# Asymmetric Data-Quality Sensitivity

Code and data pipeline accompanying the manuscript
*Asymmetric Data-Quality Sensitivity in Predictive Models and LLM Agents*
(ICDM 2026 Applied Track).

The headline finding the experiments support: the data-quality issues
that most degrade an XGBoost-style ML pipeline are not the same as
those that most degrade an LLM agent on identical inputs. This
repository contains the corruption library, the paired ML/agent
evaluation harness, and the sweep driver used to produce the
results table on the public Olist Brazilian E-Commerce dataset.

The 150-row sweep result that the paper draws from
(`logs/sweep.parquet`) is checked in so that the per-cell summaries,
bootstrap CIs, and asymmetry counts cited in Section IV of the paper
can be reproduced without rerunning the full LLM sweep.

## Reproduction

```powershell
# 1. Clone, create venv, install
git clone https://github.com/amruthasatishkumar/AsymmetricDataQualitySensitivity.git
cd AsymmetricDataQualitySensitivity
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Configure Azure AI Foundry endpoint (Entra ID auth, no API key)
copy .env.example .env
# edit .env: set AZURE_AI_FOUNDRY_ENDPOINT and AZURE_AI_FOUNDRY_DEPLOYMENT
az login   # one-time

# 3. Download Olist and build the silver feature table
python scripts\download_olist.py
python scripts\load_olist.py
python -m consumers.features

# 4. Smoke-test the LLM connection
python scripts\foundry_smoke_test.py

# 5. Train ML baselines on clean data
python -m consumers.ml_pipeline

# 6. Run the pre-experiment gate
python -m eval.gate

# 7. Run the full paired sweep (this is the expensive step)
python -m eval.sweep
```

## Repository layout

```
.
├── data/
│   ├── raw_manifest.txt   Checksums for the Olist source CSVs.
│   ├── raw/               Olist source CSVs (gitignored, fetched by scripts/download_olist.py)
│   ├── bronze/            Cleaned parquet with declared dtypes (gitignored, regenerable)
│   └── silver/            order_features.parquet — 99,441 rows x 33 cols (gitignored, regenerable)
├── corruption/            9-family corruption library (deterministic, seeded)
├── consumers/
│   ├── ml_pipeline.py     LR + XGBoost (tabular task, tabular+text task)
│   ├── agent.py           gpt-4o-mini wrapper, JSON output contract
│   ├── features.py        raw -> silver feature engineering + label definitions
│   ├── profile.py         Builds the per-column training-set summaries used in prompts
│   ├── schema.py          Column dtype declarations
│   └── prompts/           Profiled prompt template
├── eval/
│   ├── gate.py            Pre-experiment sanity check
│   ├── baseline_agent.py  Clean-test agent baseline
│   └── sweep.py           Main paired sweep driver
├── scripts/               Olist downloader, EDA, smoke tests, split freezer
├── configs/               Frozen split, profiling outputs, EDA summary
├── logs/sweep.parquet     The 150-row paired sweep result the paper draws from
└── tests/                 Corruption-determinism tests
```

## Reproducibility commitments

- 5 seeds per condition, mean +/- std reported.
- Temporal split on `order_purchase_timestamp`, frozen to a checked-in file
  (`configs/split.json`).
- Hyperparameters tuned on clean data only, frozen across all corruption conditions.
- LLM calls: `temperature=0`, fixed seed, full request/response JSONL logging.
- Authentication: Entra ID (`DefaultAzureCredential`), no API keys committed or required.
- All corruption variants are deterministic functions of `(family, severity, seed)`.

## License

Code: MIT. Olist data: CC BY-NC-SA 4.0 — see `data/raw/LICENSE` after download.

## Citation

```bibtex
@inproceedings{satishkumar2026asymmetric,
  title     = {Asymmetric Data-Quality Sensitivity in Predictive Models and {LLM} Agents},
  author    = {Satishkumar, Amrutha and Padmanabha Suwarna, Krithika},
  booktitle = {Proceedings of the IEEE International Conference on Data Mining (ICDM), Applied Track},
  year      = {2026}
}
```
