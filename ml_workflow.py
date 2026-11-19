"""Binary-classification workflow shared by spam.ipynb and dhi.ipynb.

Every step that learns from data lives in one Pipeline fitted on training data only.
plot_* functions draw on `ax` (or a new Axes) and return it.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import platform
import pandas as pd
import seaborn as sns
from importlib.metadata import version
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --- 0. Configuration ---

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
CV_FOLDS: int = 5


def get_environment_info() -> dict[str, str]:
    """Return the versions of Python and the main libraries."""

    info = {"python": platform.python_version()}

    libraries = [
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
    ]

    for library in libraries:
        info[library] = version(library)

    return info


# --- 1. Data import ---


def load_dataset(
    dataset_name: str, path: str | Path | None = None, target_column: str | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Read local SPAM or DHI data and separate the features from the binary target."""
    if dataset_name == "SPAM":
        data_path = Path(path) if path is not None else Path(__file__).parent / "datasets/spam/spambase.data"
        target_column = target_column or "spam"
        # Spambase stores its feature names separately, in column order.
        feature_names = [
            line.split(":", 1)[0].strip()
            for line in data_path.with_suffix(".names").read_text().splitlines()
            if ":" in line and not line.lstrip().startswith("|")
        ]
        frame = pd.read_csv(data_path, header=None)
        frame.columns = [*feature_names, target_column]
    elif dataset_name == "DHI":
        if path is None:
            raise ValueError("DHI requires a CSV path")
        frame = pd.read_csv(path)
        target_column = target_column or "Diabetes_binary"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    y = frame.pop(target_column)
    return frame, y


# --- 2. Exploration ---


def summarize_dataset(X: pd.DataFrame) -> pd.DataFrame:
    """Per-feature dtype, missing values and basic statistics."""
    
    summary = X.describe().T
    summary["dtype"] = X.dtypes
    summary["missing"] = X.isna().sum()

    return summary


def get_class_distribution(y: pd.Series) -> pd.DataFrame:
    """Count and proportion of each class."""

    counts = y.value_counts().sort_index()

    return pd.DataFrame({
        "count": counts,
        "proportion": counts / counts.sum()
    })


def plot_class_distribution(y: pd.Series, ax: Axes | None = None) -> Axes:
    """Bar chart of the class counts."""
    
    if ax is None:
        fig, ax = plt.subplots()

    distribution = get_class_distribution(y)

    ax.bar(distribution.index.astype(str), distribution["count"])

    ax.set_title("Class distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of observations")

    return ax


def plot_correlation_matrix(X: pd.DataFrame, ax: Axes | None = None) -> Axes:
    """Heat map of the feature correlations."""

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(X.corr(), cmap="coolwarm", ax=ax)

    return ax


# --- 3. Cleaning ---


def clean_dataset(
    X: pd.DataFrame, y: pd.Series, drop_duplicates: bool = True
) -> tuple[pd.DataFrame, pd.Series]:
    """Stateless cleaning (types, missing targets, duplicates); NaN features stay."""
    raise NotImplementedError


# --- 4. Splitting ---


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    stratify: bool = True,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return X_train, X_test, y_train, y_test."""
    raise NotImplementedError


# --- 5. Pre-processing ---


def build_imputer(strategy: Literal["median", "mean"] = "median") -> SimpleImputer:
    """Unfitted imputer for missing values."""
    raise NotImplementedError


def build_scaler() -> StandardScaler:
    """Unfitted scaler to zero mean and unit variance."""
    raise NotImplementedError


def build_pca(
    n_components: int | float = 0.95, random_state: int = RANDOM_STATE
) -> PCA:
    """Unfitted PCA; a float n_components is the fraction of variance kept."""
    raise NotImplementedError


def plot_pca_explained_variance(X_train: pd.DataFrame, ax: Axes | None = None) -> Axes:
    """Cumulative explained variance of the training set."""
    raise NotImplementedError


# --- 6. Feature selection ---


def build_feature_selector(
    method: Literal["kbest", "rfe", "model"] = "kbest",
    n_features: int = 10,
    random_state: int = RANDOM_STATE,
) -> BaseEstimator:
    """Unfitted selector keeping n_features."""
    raise NotImplementedError


def compute_feature_scores(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: Literal["mutual_info", "f_classif"] = "mutual_info",
    random_state: int = RANDOM_STATE,
) -> pd.Series:
    """Univariate score of each training feature, sorted descending."""
    raise NotImplementedError


def get_selected_features(
    pipeline: Pipeline, feature_names: Sequence[str]
) -> list[str]:
    """Names of the features kept by a fitted pipeline."""
    raise NotImplementedError


def plot_feature_scores(
    scores: pd.Series, top_n: int | None = None, ax: Axes | None = None
) -> Axes:
    """Horizontal bar chart of the top_n feature scores."""
    raise NotImplementedError


# --- 7. Models and training ---


def get_models(
    class_weight: str | dict | None = None, random_state: int = RANDOM_STATE
) -> dict[str, BaseEstimator]:
    """Unfitted candidate classifiers, keyed by name."""
    raise NotImplementedError


def get_param_grids() -> dict[str, dict[str, list[Any]]]:
    """Hyper-parameter grid per model, keys prefixed with ``model__``."""
    raise NotImplementedError


def build_pipeline(
    model: BaseEstimator,
    *,
    impute_strategy: Literal["median", "mean"] = "median",
    selector: BaseEstimator | None = None,
    pca_components: int | float | None = None,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Chain imputer -> scaler -> [selector] -> [pca] -> model."""
    raise NotImplementedError


def train_model(
    pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series
) -> Pipeline:
    """Return a fitted copy of `pipeline`."""
    raise NotImplementedError


def build_cv_splitter(
    n_splits: int = CV_FOLDS, random_state: int = RANDOM_STATE
) -> StratifiedKFold:
    """Seeded stratified K-fold, so every model sees the same folds."""
    raise NotImplementedError


def cross_validate_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: str | Sequence[str],
    cv: StratifiedKFold,
) -> pd.DataFrame:
    """Cross-validation scores, one row per fold."""
    raise NotImplementedError


def tune_hyperparameters(
    pipeline: Pipeline,
    param_grid: Mapping[str, list[Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: str,
    cv: StratifiedKFold,
) -> GridSearchCV:
    """Grid search refitted on the whole training set."""
    raise NotImplementedError


# --- 8. Evaluation ---


def compute_metrics(
    y_true: ArrayLike, y_pred: ArrayLike, y_score: ArrayLike | None = None
) -> dict[str, float]:
    """Classification metrics; roc_auc and average_precision need y_score."""
    raise NotImplementedError


def evaluate_model(
    model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """Metrics of a fitted model on the test set."""
    raise NotImplementedError


def compare_models(
    models: Mapping[str, Pipeline], X_test: pd.DataFrame, y_test: pd.Series
) -> pd.DataFrame:
    """Test-set metrics, one row per model."""
    raise NotImplementedError


def plot_confusion_matrix(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    normalize: bool = False,
    ax: Axes | None = None,
) -> Axes:
    """Confusion matrix on the test set."""
    raise NotImplementedError


def plot_roc_curves(
    models: Mapping[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    ax: Axes | None = None,
) -> Axes:
    """ROC curve of each model on the test set."""
    raise NotImplementedError


def plot_precision_recall_curves(
    models: Mapping[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    ax: Axes | None = None,
) -> Axes:
    """Precision-recall curve of each model on the test set."""
    raise NotImplementedError


def plot_model_comparison(
    results: pd.DataFrame, metric: str, ax: Axes | None = None
) -> Axes:
    """Bar chart of one metric across models."""
    raise NotImplementedError


# --- 9. Workflow ---


def run_workflow(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    scoring: str,
    class_weight: str | dict | None = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    """Split, tune, train and evaluate every model; return results and models."""
    raise NotImplementedError
