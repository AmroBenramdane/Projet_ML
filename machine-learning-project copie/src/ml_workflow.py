"""Binary-classification workflow shared by notebooks/spam.ipynb and notebooks/dhi.ipynb.

Every step that learns from data lives in one Pipeline fitted on training data only.
plot_* functions draw on `ax` (or a new Axes) and return it.
"""

from collections.abc import Mapping, Sequence
from importlib.metadata import version
from pathlib import Path
import platform
from typing import Any, Literal

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    f_classif,
    mutual_info_classif,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

RANDOM_STATE = 42

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


def load_dataset(dataset_name: str, path: str | Path | None = None, target_column: str | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Load SPAM from UCI or DHI from a CSV file."""

    if dataset_name == "SPAM":
        spambase = fetch_ucirepo(id=94)

        X = spambase.data.features
        y = spambase.data.targets.iloc[:, 0] # to make y a pd.Series

    elif dataset_name == "DHI":
        df = pd.read_csv(path)

        X = df.drop(columns=target_column)
        y = df[target_column]

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return X, y


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


def describe_feature_types(X: pd.DataFrame, max_ordinal_values: int = 15) -> pd.DataFrame:
    """Kind (binary / ordinal / continuous), range, share of zeros and skewness of each feature."""

    n_unique = X.nunique()
    is_integer = X.apply(lambda col: bool(np.all(np.mod(col.dropna(), 1) == 0)))

    kind = np.where(
        n_unique <= 2,
        "binary",
        np.where(is_integer & (n_unique <= max_ordinal_values), "ordinal", "continuous"),
    )

    return pd.DataFrame({
        "kind": kind,
        "n_unique": n_unique,
        "min": X.min(),
        "max": X.max(),
        "zeros_%": (X == 0).mean() * 100,
        "skewness": X.skew(),
    })


def count_duplicates(X: pd.DataFrame, y: pd.Series) -> dict[str, int]:
    """Exact duplicate rows, and rows sharing the same features but not the same label."""

    same_features = X.duplicated(keep=False)
    same_features_and_label = pd.concat([X, y], axis=1).duplicated(keep=False)

    return {
        "duplicated_rows": int(pd.concat([X, y], axis=1).duplicated().sum()),
        "conflicting_rows": int((same_features & ~same_features_and_label).sum()),
    }


def get_correlated_pairs(X: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """Feature pairs whose absolute correlation is at least threshold, sorted descending."""

    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = upper.stack().rename("abs_corr").reset_index()
    pairs.columns = ["feature_1", "feature_2", "abs_corr"]

    return (
        pairs[pairs["abs_corr"] >= threshold]
        .sort_values("abs_corr", ascending=False)
        .reset_index(drop=True)
    )


def plot_target_correlation(
    X: pd.DataFrame, y: pd.Series, top_n: int | None = None, ax: Axes | None = None
) -> Axes:
    """Horizontal bar chart of the correlation of each feature with the target."""

    if ax is None:
        _, ax = plt.subplots()

    corr = X.corrwith(y.astype(float))
    ordered = corr.reindex(corr.abs().sort_values(ascending=False).index)
    if top_n is not None:
        ordered = ordered.head(top_n)

    ordered = ordered.iloc[::-1]  # strongest at the top
    colors = ["tab:red" if value < 0 else "tab:blue" for value in ordered]
    ax.barh(ordered.index.astype(str), ordered.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Correlation with the target")
    ax.set_title("Feature / target correlation")

    return ax


def plot_feature_by_class(
    X: pd.DataFrame, y: pd.Series, feature: str, ax: Axes | None = None
) -> Axes:
    """Distribution of one feature for each class (box plot)."""

    if ax is None:
        _, ax = plt.subplots()

    data = pd.DataFrame({feature: X[feature], "class": y.astype(str)})
    sns.boxplot(data=data, x="class", y=feature, ax=ax, showfliers=False)
    ax.set_title(feature)
    ax.set_xlabel("Class")

    return ax


# --- 3. Cleaning ---

def clean_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    drop_duplicates: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
   

    # Check that X and y are aligned
    if not X.index.equals(y.index):
        raise ValueError("X and y must have the same index.")

    X = X.copy()
    y = y.copy()

    # Remove observations with missing target values
    valid_rows = y.notna()
    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    # Remove exact duplicates (features AND target)
    if drop_duplicates:
        combined = pd.concat(
            [X.reset_index(drop=True), y.reset_index(drop=True)],
            axis=1,
        )

        keep_rows = ~combined.duplicated().to_numpy()

        X = X.iloc[keep_rows]
        y = y.iloc[keep_rows]

    return X, y



# --- 4. Splitting ---


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    stratify: bool = True,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training and testing sets. Stratification preserves class proportions in both sets.
    """

    # Check that X and y are aligned
    if not X.index.equals(y.index):
        raise ValueError("X and y must have the same index.")

    # Check that the target contains no missing values
    if y.isna().any():
        raise ValueError(
            "Missing target values must be removed before splitting."
        )

    # Validate test_size
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    # Stratification preserves class proportions
    stratification = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratification,
        shuffle=True,
    )

    return X_train, X_test, y_train, y_test



# --- 5. Pre-processing ---


def build_imputer(strategy: Literal["median", "mean"] = "median") -> SimpleImputer:
    """Unfitted imputer for missing values."""
    if strategy not in ("median", "mean"):
        raise ValueError(f"Unsupported imputation strategy: {strategy}")
    return SimpleImputer(strategy=strategy)


def build_scaler() -> StandardScaler:
    """Unfitted scaler to zero mean and unit variance."""
    return StandardScaler()


def build_pca(
    n_components: int | float = 0.95, random_state: int = RANDOM_STATE
) -> PCA:
    """Unfitted PCA; a float n_components is the fraction of variance kept."""
    return PCA(n_components=n_components, random_state=random_state)


def plot_pca_explained_variance(X_train: pd.DataFrame, ax: Axes | None = None) -> Axes:
    """Cumulative explained variance of the training set."""
    if ax is None:
        _, ax = plt.subplots()

    X_clean = SimpleImputer(strategy="median").fit_transform(X_train)
    X_scaled = StandardScaler().fit_transform(X_clean)
    pca = PCA().fit(X_scaled)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    components = np.arange(1, len(cum_var) + 1)

    ax.plot(components, cum_var, marker="o", markersize=3)
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("PCA Cumulative Explained Variance")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)

    return ax


# --- 6. Feature selection ---


def build_feature_selector(
    method: Literal["kbest", "rfe", "model"] = "kbest",
    n_features: int = 10,
    random_state: int = RANDOM_STATE,
) -> BaseEstimator:
    """Unfitted selector keeping n_features."""
    k = max(1, int(n_features))

    if method == "kbest":
        return SelectKBest(score_func=f_classif, k=k)
    if method == "rfe":
        estimator = LogisticRegression(max_iter=1000, random_state=random_state)
        return RFE(estimator=estimator, n_features_to_select=k, step=1)
    if method == "model":
        estimator = RandomForestClassifier(
            n_estimators=100, random_state=random_state, n_jobs=-1
        )
        return SelectFromModel(estimator=estimator, max_features=k, threshold=-np.inf)

    raise ValueError(f"Unsupported feature selection method: {method}")


def compute_feature_scores(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: Literal["mutual_info", "f_classif"] = "mutual_info",
    random_state: int = RANDOM_STATE,
) -> pd.Series:
    """Univariate score of each training feature, sorted descending."""
    X_clean = SimpleImputer(strategy="median").fit_transform(X_train)
    feature_names = (
        list(X_train.columns)
        if isinstance(X_train, pd.DataFrame)
        else [f"f_{i}" for i in range(X_clean.shape[1])]
    )

    if method == "mutual_info":
        scores = mutual_info_classif(X_clean, y_train, random_state=random_state)
    elif method == "f_classif":
        scores, _ = f_classif(X_clean, y_train)
        scores = np.nan_to_num(scores, nan=0.0)
    else:
        raise ValueError(f"Unsupported feature scoring method: {method}")

    return pd.Series(scores, index=feature_names, name="score").sort_values(
        ascending=False
    )


def get_selected_features(
    pipeline: Pipeline, feature_names: Sequence[str]
) -> list[str]:
    """Names of the features kept by a fitted pipeline."""
    names = list(feature_names)
    current_mask = np.ones(len(names), dtype=bool)

    for _, step in pipeline.steps:
        if hasattr(step, "get_support"):
            support = step.get_support()
            if isinstance(support, np.ndarray) and support.dtype == bool:
                if len(support) == len(current_mask):
                    current_mask = current_mask & support
                else:
                    kept_indices = np.flatnonzero(current_mask)[support]
                    new_mask = np.zeros(len(names), dtype=bool)
                    new_mask[kept_indices] = True
                    current_mask = new_mask

    return [name for name, keep in zip(names, current_mask) if keep]


def plot_feature_scores(
    scores: pd.Series, top_n: int | None = None, ax: Axes | None = None
) -> Axes:
    """Horizontal bar chart of the top_n feature scores."""
    if ax is None:
        _, ax = plt.subplots()

    ordered = scores.sort_values(ascending=False)
    if top_n is not None:
        ordered = ordered.head(max(1, int(top_n)))

    # Inverser pour avoir le score le plus élevé en haut du graphique horizontal
    ordered_plot = ordered.iloc[::-1]
    ax.barh(ordered_plot.index.astype(str), ordered_plot.values)
    ax.set_xlabel(ordered.name or "Score")
    ax.set_ylabel("Feature")
    ax.set_title(
        f"Top {len(ordered)} Feature Scores" if top_n is not None else "Feature Scores"
    )
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)

    return ax


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
