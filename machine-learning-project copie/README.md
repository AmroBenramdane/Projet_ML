# Machine Learning Project

Binary classification on two datasets with one shared workflow (`src/ml_workflow.py`).

| Dataset | Notebook |
|---|---|
| [SPAM](https://archive.ics.uci.edu/dataset/94/spambase) | [notebooks/spam.ipynb](notebooks/spam.ipynb) |
| [Diabetes Health Indicators](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) | [notebooks/dhi.ipynb](notebooks/dhi.ipynb) |

## Structure

Structure of the project : 

```
data/
  spam/          Spambase files (UCI)
  dhi/           Diabetes Health Indicators CSV
docs/            assignment (description.pdf)
notebooks/       one notebook per dataset
src/             ml_workflow.py, the shared workflow
tests/           unit tests of ml_workflow.py
pytest.ini       makes src/ importable by the tests
```

## Setup

Python ≥ 3.12 (check with `python3 --version`). The pinned numpy/scipy versions do not install on 3.10 or 3.11.

```bash
python3 -m venv .venv               # Ubuntu/Debian: first `sudo apt install python3.12-venv`
source .venv/bin/activate           # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Data

The datasets are versioned in `data/`. SPAM is loaded from the UCI repository by `load_dataset`, DHI from `data/dhi/diabetes_binary_health_indicators_BRFSS2015.csv`.

## Usage

```bash
jupyter notebook   # open notebooks/spam.ipynb or notebooks/dhi.ipynb
pytest            # runs tests/
```

## Authors

- Kamel NEHLIL (Marchma - Problème pseudo à cause de Github)
- Karen SFEIR 
- Téodore AUTULY
- Amro BENRAMDANE

## AI usage

The initial scaffold (function signatures of `ml_workflow.py`, test skeleton of `tests.py`, notebook structure, `requirements.txt`, `.gitignore`, this README) was generated with Claude Code (Anthropic).

_TODO: why AI was used, how its output was reviewed, and any later use._
