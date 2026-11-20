"""Binary-classification workflow shared by spam.ipynb and dhi.ipynb.

Every step that learns from data lives in one Pipeline fitted on training data only.
plot_* functions draw on `ax` (or a new Axes) and return it.
"""

import platform
from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from numpy.typing import ArrayLike
from sklearn import metrics
from sklearn.base import BaseEstimator, clone
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV, StratifiedGroupKFold, StratifiedKFold, cross_validate, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits
from ucimlrepo import DatasetNotFoundError, fetch_ucirepo

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
    dataset_name: str,
    path: str | Path | None = None,
    target_column: str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Read a local dataset. For SPAM, omitting the path tries UCI first."""

    if dataset_name == "SPAM":
        if path is None:
            try:
                spambase = fetch_ucirepo(id=94)
            except (ConnectionError, DatasetNotFoundError):
                path = Path(__file__).parent / "datasets/spam/spambase.data"
            else:
                return spambase.data.features, spambase.data.targets.iloc[:, 0]
        path = Path(path)
        names = [
            line.split(":", 1)[0]
            for line in path.with_suffix(".names").read_text().splitlines()
            if line.endswith("continuous.")
        ]
        X = pd.read_csv(path, header=None)
        X.columns = [*names, "Class"]
        y = X.pop("Class")

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


def plot_class_distribution(
    y: pd.Series, ax: Axes | None = None, *, labels: Mapping[Any, str] | None = None
) -> Axes:
    """Show each class as a percentage of the labeled observations."""
    if ax is None:
        _, ax = plt.subplots()

    percentages = 100 * get_class_distribution(y)["proportion"]
    names = [labels.get(value, str(value)) if labels else str(value) for value in percentages.index]
    bars = ax.bar(names, percentages)
    ax.bar_label(bars, fmt="%.1f%%", padding=2)
    ax.set(ylim=(0, 100), title="Class distribution", xlabel="Class", ylabel="Observations (%)")
    return ax


def plot_feature_histogram(
    values: pd.Series, ax: Axes | None = None, *, log_scale: bool = False
) -> Axes:
    """Plot feature counts with equal-width bins on the selected scale."""
    if ax is None:
        _, ax = plt.subplots()
    sns.histplot(values, bins=30, log_scale=log_scale, ax=ax)
    ax.set_ylabel("Number of observations")
    return ax


def plot_correlation_matrix(X: pd.DataFrame, ax: Axes | None = None) -> Axes:
    """Heat map of the feature correlations."""

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(X.corr(), cmap="coolwarm", vmin=-1, vmax=1, center=0, ax=ax)

    return ax


# --- 3. Cleaning ---


def clean_dataset(
    X: pd.DataFrame, y: pd.Series, drop_duplicates: bool = False
) -> tuple[pd.DataFrame, pd.Series]:
    """Keep labeled binary observations in input order, preserving feature NaNs.

    When requested, keep only the first occurrence of each (features, target) row.
    """
    if not X.index.equals(y.index):
        raise ValueError("X and y must have the same index in the same order")

    labeled = y.notna()
    X_clean = X.loc[labeled].copy()
    y_clean = y.loc[labeled].copy()
    if not y_clean.isin([0, 1]).all():
        raise ValueError("Expected binary target values 0 or 1")

    if drop_duplicates:
        rows = pd.concat([X_clean, y_clean], axis=1, ignore_index=True)
        keep = ~rows.duplicated()
        X_clean = X_clean.loc[keep]
        y_clean = y_clean.loc[keep]

    return X_clean, y_clean


# --- 4. Splitting ---


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    stratify: bool = True,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return X_train, X_test, y_train, y_test."""
    return train_test_split(
        X, y, test_size=test_size, stratify=y if stratify else None, random_state=random_state
    )


def profile_groups(X: pd.DataFrame) -> np.ndarray:
    """Give identical feature rows a common group, independently of the target."""
    return X.groupby(list(X.columns), sort=False, dropna=False).ngroup().to_numpy()


def split_by_profile(
    X: pd.DataFrame, y: pd.Series, random_state: int = RANDOM_STATE
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Reserve one of five approximately stratified folds of distinct profiles."""
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    train, test = next(splitter.split(X, y, groups=profile_groups(X)))
    return X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test]


def training_folds(
    X: pd.DataFrame, y: pd.Series, group_profiles: bool = False, random_state: int = RANDOM_STATE
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Materialize reusable validation splits in training-row coordinates."""
    if group_profiles:
        cv = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=random_state)
        return list(cv.split(X, y, groups=profile_groups(X)))
    return list(build_cv_splitter(random_state=random_state).split(X, y))


# --- 5. Pre-processing ---


def build_imputer(strategy: Literal["median", "mean"] = "median") -> SimpleImputer:
    """Unfitted imputer for missing values."""
    if strategy not in ("median", "mean"):
        raise ValueError("Expected median or mean imputation")
    return SimpleImputer(strategy=strategy, keep_empty_features=True)


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
    steps = Pipeline([("imputer", build_imputer()), ("scaler", build_scaler()), ("pca", PCA())])
    steps.fit(X_train)
    cumulative = steps["pca"].explained_variance_ratio_.cumsum()
    ax = ax if ax is not None else plt.subplots()[1]
    ax.plot(np.arange(1, len(cumulative) + 1), cumulative)
    retained = int(np.searchsorted(cumulative, .95) + 1)
    ax.axhline(.95, color="grey", linestyle="--")
    ax.axvline(retained, color="grey", linestyle=":", label=f"95% at {retained} components")
    ax.legend()
    ax.set(xlabel="Number of components", ylabel="Cumulative explained variance", ylim=(0, 1.02))
    return ax


# --- 6. Feature selection ---


def build_feature_selector(
    method: Literal["kbest", "rfe", "model"] = "kbest",
    n_features: int = 10,
    random_state: int = RANDOM_STATE,
) -> BaseEstimator:
    """Unfitted selector keeping n_features."""
    if method == "kbest":
        return SelectKBest(f_classif, k=n_features)
    estimator = LogisticRegression(max_iter=500, random_state=random_state)
    if method == "rfe":
        return RFE(estimator, n_features_to_select=n_features)
    if method == "model":
        return SelectFromModel(estimator, max_features=n_features, threshold=-np.inf)
    raise ValueError(f"Unknown feature selection method: {method}")


def compute_feature_scores(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: Literal["mutual_info", "f_classif"] = "mutual_info",
    random_state: int = RANDOM_STATE,
) -> pd.Series:
    """Univariate score of each training feature, sorted descending."""
    values = build_imputer().fit_transform(X_train)
    if method == "f_classif":
        scores, _ = f_classif(values, y_train)
    elif method == "mutual_info":
        scores = mutual_info_classif(values, y_train, random_state=random_state)
    else:
        raise ValueError(f"Unknown feature score: {method}")
    return pd.Series(scores, index=X_train.columns, name=method).sort_values(ascending=False)


def get_selected_features(
    pipeline: Pipeline, feature_names: Sequence[str]
) -> list[str]:
    """Names of the features kept by a fitted pipeline."""
    selector = pipeline.named_steps.get("selector")
    if selector is None or selector == "passthrough":
        return list(feature_names)
    return list(np.asarray(feature_names)[selector.get_support()])


def plot_feature_scores(
    scores: pd.Series, top_n: int | None = None, ax: Axes | None = None
) -> Axes:
    """Horizontal bar chart of the top_n feature scores."""
    ax = ax if ax is not None else plt.subplots()[1]
    shown = scores.sort_values(ascending=False)
    if top_n is not None:
        shown = shown.head(top_n)
    shown.sort_values().plot.barh(ax=ax)
    ax.set_xlabel(scores.name or "Feature score")
    return ax


# --- 7. Models and training ---


def get_models(
    class_weight: str | dict | None = None, random_state: int = RANDOM_STATE
) -> dict[str, BaseEstimator]:
    """Unfitted candidate classifiers, keyed by name."""
    return {
        "Majority": DummyClassifier(strategy="prior"),
        "Logistic": LogisticRegression(max_iter=500, class_weight=class_weight, random_state=random_state),
        "Tree": DecisionTreeClassifier(
            min_samples_leaf=10, class_weight=class_weight, random_state=random_state,
        ),
        "Boosting": HistGradientBoostingClassifier(
            max_iter=80, early_stopping=False, class_weight=class_weight, random_state=random_state,
        ),
    }


def get_param_grids() -> dict[str, dict[str, list[Any]]]:
    """Hyper-parameter grid per model, keys prefixed with ``model__``."""
    return {
        "Majority": {},
        "Logistic": {"model__C": [0.1, 1.0]},
        "Tree": {"model__max_depth": [3, 8]},
        "Boosting": {"model__max_leaf_nodes": [7, 31]},
    }


def build_pipeline(
    model: BaseEstimator,
    *,
    impute_strategy: Literal["median", "mean"] = "median",
    selector: BaseEstimator | Literal["passthrough"] | None = None,
    pca_components: int | float | None = None,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Chain imputer -> scaler -> [selector] -> [pca] -> model."""
    steps = [("imputer", build_imputer(impute_strategy)), ("scaler", build_scaler())]
    if selector is not None:
        steps.append(("selector", selector))
    if pca_components is not None:
        steps.append(("pca", build_pca(pca_components, random_state)))
    return Pipeline([*steps, ("model", model)])


def train_model(
    pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series
) -> Pipeline:
    """Return a fitted copy of `pipeline`."""
    return clone(pipeline).fit(X_train, y_train)


def build_cv_splitter(
    n_splits: int = CV_FOLDS, random_state: int = RANDOM_STATE
) -> StratifiedKFold:
    """Seeded stratified K-fold, so every model sees the same folds."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def cross_validate_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: str | Sequence[str] | Mapping[str, Any],
    cv: StratifiedKFold | list,
) -> pd.DataFrame:
    """Cross-validation scores, one row per fold."""
    with threadpool_limits(limits=2):
        scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, error_score="raise")
    return pd.DataFrame({k.removeprefix("test_"): v for k, v in scores.items() if k.startswith("test_")})


def tune_hyperparameters(
    pipeline: Pipeline,
    param_grid: dict | list[dict],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: str | Callable,
    cv: StratifiedKFold | list,
) -> GridSearchCV:
    """Grid search refitted on the whole training set."""
    return GridSearchCV(pipeline, param_grid, cv=cv, scoring=scoring, error_score="raise").fit(X_train, y_train)


# --- 8. Evaluation ---


def compute_metrics(
    y_true: ArrayLike, y_pred: ArrayLike, y_score: ArrayLike | None = None
) -> dict[str, float]:
    """Classification metrics; roc_auc and average_precision need y_score."""
    result = {
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "balanced_accuracy": metrics.balanced_accuracy_score(y_true, y_pred),
        "precision": metrics.precision_score(y_true, y_pred, zero_division=0),
        "recall": metrics.recall_score(y_true, y_pred, zero_division=0),
        "f1": metrics.f1_score(y_true, y_pred, zero_division=0),
        "f0.5": metrics.fbeta_score(y_true, y_pred, beta=.5, zero_division=0),
    }
    if y_score is not None:
        result["roc_auc"] = metrics.roc_auc_score(y_true, y_score)
        result["average_precision"] = metrics.average_precision_score(y_true, y_score)
    return result


def evaluate_model(
    model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """Metrics of a fitted model on the test set."""
    prediction = model.predict(X_test)
    result = compute_metrics(y_test, prediction, model.predict_proba(X_test)[:, 1])
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, prediction, labels=[0, 1]).ravel()
    result.update(TN=tn, FP=fp, FN=fn, TP=tp)
    return result


def compare_models(
    models: Mapping[str, Pipeline], X_test: pd.DataFrame, y_test: pd.Series
) -> pd.DataFrame:
    """Test-set metrics, one row per model."""
    return pd.DataFrame.from_dict(
        {name: evaluate_model(model, X_test, y_test) for name, model in models.items()}, orient="index"
    )


def plot_confusion_matrix(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    normalize: bool = False,
    ax: Axes | None = None,
) -> Axes:
    """Confusion matrix on the test set."""
    display = metrics.ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, labels=[0, 1], normalize="true" if normalize else None,
        ax=ax, colorbar=False, values_format=".1%" if normalize else "d", cmap="Blues",
    )
    return display.ax_


def plot_roc_curves(
    models: Mapping[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    ax: Axes | None = None,
) -> Axes:
    """ROC curve of each model on the test set."""
    ax = ax if ax is not None else plt.subplots()[1]
    for name, model in models.items():
        metrics.RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    return ax


def plot_precision_recall_curves(
    models: Mapping[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    ax: Axes | None = None,
) -> Axes:
    """Precision-recall curve of each model on the test set."""
    ax = ax if ax is not None else plt.subplots()[1]
    for name, model in models.items():
        metrics.PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    return ax


def plot_model_comparison(
    results: pd.DataFrame, metric: str, ax: Axes | None = None
) -> Axes:
    """Bar chart of one metric across models."""
    ax = ax if ax is not None else plt.subplots()[1]
    results[metric].plot.bar(ax=ax, rot=0, ylim=(0, 1), ylabel=metric)
    return ax


# --- 9. Workflow ---


def run_workflow(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    scoring: str,
    class_weight: str | dict | None = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    group_profiles: bool = False,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    """Split, tune, train and evaluate every model; return results and models."""
    X, y = clean_dataset(X, y)
    if group_profiles:
        if test_size != TEST_SIZE:
            raise ValueError("The grouped holdout uses one of five folds")
        split = split_by_profile(X, y, random_state)
    else:
        split = split_dataset(X, y, test_size, random_state=random_state)
    X_train, X_test, y_train, y_test = split
    _, models = fit_candidates(
        X_train, y_train, scoring=scoring, class_weight=class_weight,
        random_state=random_state, group_profiles=group_profiles,
    )
    return compare_models(models, X_test, y_test), models


def fit_candidates(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    scoring: str,
    class_weight: str | dict | None = None,
    group_profiles: bool = False,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    """Compare candidates on the same training folds and refit each family's best pipeline."""
    folds = training_folds(X_train, y_train, group_profiles, random_state)
    scorer = scoring
    if scoring == "f0.5":
        scorer = metrics.make_scorer(metrics.fbeta_score, beta=0.5, zero_division=0)
    grids = get_param_grids()
    rows, models = [], {}
    for name, model in get_models(class_weight, random_state).items():
        grid = grids[name]
        if name == "Logistic":
            # Vary one representation at a time, keeping the model grid identical.
            pipeline = build_pipeline(model, selector="passthrough", pca_components=.95)
            k = max(1, X_train.shape[1] // 2)
            selectors = ["passthrough", build_feature_selector(n_features=k)]
            grid = [
                {**grid, "pca": ["passthrough"], "selector": selectors},
                {**grid, "pca": [build_pca(.95, random_state)]},
            ]
        else:
            pipeline = build_pipeline(model)
        with threadpool_limits(limits=2):
            search = tune_hyperparameters(pipeline, grid, X_train, y_train, scorer, folds)
        models[name] = search.best_estimator_
        results = search.cv_results_
        for params, mean, std in zip(results["params"], results["mean_test_score"], results["std_test_score"]):
            params = params.copy()
            pca = params.pop("pca", None)
            selector = params.pop("selector", None)
            representation = "All features"
            if isinstance(pca, PCA):
                representation = "PCA 95%"
            elif isinstance(selector, SelectKBest):
                representation = f"Top {selector.k}"
            rows.append({"model": name, "features": representation, "parameters": str(params),
                         "CV mean": mean, "CV SD": std})
    return pd.DataFrame(rows).sort_values("CV mean", ascending=False, kind="stable"), models
