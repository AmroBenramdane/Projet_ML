# Machine Learning Project

Binary classification on two datasets with one shared workflow (`ml_workflow.py`).

| Dataset | Notebook |
|---|---|
| [SPAM](https://archive.ics.uci.edu/dataset/94/spambase) | [spam.ipynb](spam.ipynb) |
| [Diabetes Health Indicators](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) | [dhi.ipynb](dhi.ipynb) |

## Setup

Python ≥ 3.12 (check with `python3 --version`). The pinned numpy/scipy versions do not install on 3.10 or 3.11.

```bash
python3 -m venv .venv               # Ubuntu/Debian: first `sudo apt install python3.12-venv`
source .venv/bin/activate           # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Data

The notebooks read the files in `datasets/` locally. SPAM uses `spam/spambase.data` (comma-separated, no header) and `spam/spambase.names` for feature names. Its last column is the spam label. DHI uses `dbh/diabetes_binary_health_indicators_BRFSS2015.csv` with the target column `Diabetes_binary`.

## Usage

```bash
jupyter notebook   # run spam.ipynb or dhi.ipynb
pytest tests.py
```

## Authors

_TODO_

## AI usage

The initial scaffold (function signatures of `ml_workflow.py`, test skeleton of `tests.py`, notebook structure, `requirements.txt`, `.gitignore`, this README) was generated with Claude Code (Anthropic).

Codex helped adapt the shared loader and notebook paths to read the local files and verify the import and exploration steps.

_TODO: why AI was used, how its output was reviewed, and any later use._
