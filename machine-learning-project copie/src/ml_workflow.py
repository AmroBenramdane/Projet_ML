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
from sklearn import metrics
from sklearn.base import BaseEstimator, clone
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    f_classif,
    mutual_info_classif,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
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
    return {
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
        "Logistic": {"model__C": [0.1, 1.0]},
        "Tree": {"model__max_depth": [3, 8]},
        "Boosting": {"model__max_leaf_nodes": [7, 31]},
    }


def build_pipeline(
    model: BaseEstimator,
    *,
    impute_strategy: Literal["median", "mean"] = "median",
    selector: BaseEstimator | None = None,
    pca_components: int | float | None = None,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Chain imputer -> scaler -> [selector] -> [pca] -> model."""
    steps = [("imputer", build_imputer(impute_strategy)), ("scaler", build_scaler())]
    if selector is not None:
        steps.append(("selector", clone(selector)))
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
    pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series,
    scoring: str | Sequence[str], cv: StratifiedKFold,
) -> pd.DataFrame:
    """Score an unfitted pipeline on each validation fold."""
    measures = [scoring] if isinstance(scoring, str) else list(scoring)
    results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=measures, error_score="raise")
    return pd.DataFrame({name: results[f"test_{name}"] for name in measures})


def tune_hyperparameters(
    pipeline: Pipeline, param_grid: Mapping[str, list[Any]],
    X_train: pd.DataFrame, y_train: pd.Series, scoring: str, cv: StratifiedKFold,
) -> GridSearchCV:
    """Tune by the primary score and record precision and recall at the default threshold."""
    measures = {
        "primary": scoring,
        "precision": metrics.make_scorer(metrics.precision_score, zero_division=0),
        "recall": "recall",
    }
    search = GridSearchCV(pipeline, param_grid, cv=cv, scoring=measures, refit="primary",
                          error_score="raise")
    return search.fit(X_train, y_train)


def fit_candidates(
    pipelines: Mapping[str, Pipeline], X_train: pd.DataFrame, y_train: pd.Series,
    *, scoring: str, cv: StratifiedKFold, compare_weights: bool = False,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    """Use identical folds and return the validation table and each family's refitted winner."""
    grids = get_param_grids()
    rows = []
    fitted = {}
    for name, pipeline in pipelines.items():
        grid = grids[name]
        if compare_weights:
            grid = {**grid, "model__class_weight": [None, "balanced"]}
        search = tune_hyperparameters(pipeline, grid, X_train, y_train, scoring, cv)
        fitted[name] = search.best_estimator_
        scores = search.cv_results_
        for i, params in enumerate(scores["params"]):
            settings = ", ".join(f"{key.removeprefix('model__')}={value}" for key, value in params.items())
            rows.append({
                "model": name, "settings": settings, "mean_score": scores["mean_test_primary"][i],
                "fold_sd": scores["std_test_primary"][i], "precision": scores["mean_test_precision"][i],
                "recall": scores["mean_test_recall"][i],
            })
    table = pd.DataFrame(rows).sort_values("mean_score", ascending=False, kind="stable")
    return table.reset_index(drop=True), fitted


def majority_baseline(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Always predict the training majority; use training prevalence as the constant score."""
    from sklearn.dummy import DummyClassifier

    return Pipeline([("model", DummyClassifier(strategy="prior"))]).fit(X_train, y_train)


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
    }
    if y_score is not None:
        result["roc_auc"] = metrics.roc_auc_score(y_true, y_score)
        result["average_precision"] = metrics.average_precision_score(y_true, y_score)
    return result


def choose_precision_threshold(y_true: ArrayLike, scores: ArrayLike, min_precision: float) -> float:
    """Maximize recall at the required empirical precision, breaking ties by higher threshold."""
    if not 0 < min_precision <= 1:
        raise ValueError("min_precision must be in (0, 1]")
    if not np.array_equal(np.unique(y_true), [0, 1]):
        raise ValueError("Threshold selection requires both binary classes")
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, scores)
    # The final PR point corresponds to predicting no positives and has no threshold.
    feasible = np.flatnonzero(precision[:-1] >= min_precision)
    if feasible.size == 0:
        raise ValueError("No observed threshold meets the precision target")
    best_recall = recall[feasible].max()
    best = feasible[recall[feasible] == best_recall][-1]
    return float(thresholds[best])


def select_blocking_threshold(
    model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, cv: StratifiedKFold,
    min_precision: float = 0.98,
) -> tuple[float, dict[str, float]]:
    """Estimate the threshold on held-out training folds for a fixed model configuration."""
    score = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold = choose_precision_threshold(y_train, score, min_precision)
    return threshold, compute_metrics(y_train, score >= threshold)


def evaluate_model(
    model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, *, threshold: float | None = None
) -> dict[str, float]:
    """Evaluate the default classifier decision or a threshold fixed on training data."""
    scores = model.predict_proba(X_test)[:, 1]
    prediction = model.predict(X_test) if threshold is None else scores >= threshold
    result = compute_metrics(y_test, prediction, scores)
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
    X: pd.DataFrame, y: pd.Series, *, scoring: str,
    class_weight: str | dict | None = None, test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE, drop_duplicates: bool = False,
    selector: BaseEstimator | None = None, pca_components: int | float | None = None,
    impute_strategy: Literal["median", "mean"] = "median", compare_weights: bool = False,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    """Clean, split, tune three candidate pipelines, then evaluate them and the baseline."""
    X, y = clean_dataset(X, y, drop_duplicates=drop_duplicates)
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size, random_state=random_state)
    pipelines = {
        name: build_pipeline(model, selector=selector, pca_components=pca_components,
                             impute_strategy=impute_strategy, random_state=random_state)
        for name, model in get_models(class_weight, random_state).items()
    }
    _, models = fit_candidates(pipelines, X_train, y_train, scoring=scoring,
                               cv=build_cv_splitter(random_state=random_state),
                               compare_weights=compare_weights)
    models["Majority"] = majority_baseline(X_train, y_train)
    return compare_models(models, X_test, y_test), models
