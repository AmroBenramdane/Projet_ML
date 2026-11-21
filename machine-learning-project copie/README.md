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

## Model comparison (sections 4.3–6)

Run the setup and test commands from the directory containing this README and its `pytest.ini`. Execute the notebooks from the first cell. The three candidates are logistic regression, a decision tree and histogram gradient boosting. A majority-class predictor gives a baseline. SPAM selects by F1 after exact-row deduplication and keeps 40 variables via a random forest. DHI selects by average precision, compares class weights and keeps 15 variables via ANOVA. Each validation fold fits its own selector on training rows.

The original SPAM import calls UCI. If the service is unavailable, the following replacement for its import cell reads the bundled data with the same 57 feature names. This was the route used for the recorded modeling run. DHI reads its local CSV directly.

```python
path = ROOT / "data" / "spam" / "spambase.data"
names = [
    line.split(":", 1)[0]
    for line in path.with_suffix(".names").read_text().splitlines()
    if line.endswith("continuous.")
]
X = pd.read_csv(path, header=None)
X.columns = names + ["Class"]
y = X.pop("Class")
```

## Authors

- Kamel NEHLIL (Marchma - Problème pseudo à cause de Github)
- Karen SFEIR 
- Téodore AUTULY
- Amro BENRAMDANE

## AI usage

The initial scaffold (function signatures of `ml_workflow.py`, test skeleton of `tests.py`, notebook structure, `requirements.txt`, `.gitignore`, this README) was generated with Claude Code (Anthropic).

Codex completed the model training, evaluation and discussion starting at section 4.3, and filled the unfinished tests. The implementation reuses the selectors, stratified split and preprocessing defined in sections 1–4.2. The three classifiers were compared on the same training folds. The notebook conclusions were checked against the class counts, confusion matrices and scores computed on the bundled data.
