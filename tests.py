"""Tests of the common workflow. Run from the repository root: pytest tests.py."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import ml_workflow as mlw


@pytest.fixture
def sample():
    rng = np.random.default_rng(7)
    target = pd.Series(np.tile([0, 0, 0, 1], 40), name="label")
    features = pd.DataFrame({
        "signal": 2 * target + rng.normal(0, .45, len(target)),
        "noise": rng.normal(size=len(target)),
        "secondary": rng.normal(size=len(target)),
    })
    features.loc[7, "signal"] = np.nan
    return features, target


def test_import_spam_uses_headerless_local_file(tmp_path):
    path = tmp_path / "spambase.data"
    path.write_text("12,0\n15,1\n")
    path.with_suffix(".names").write_text("| Format: comment\nsize: continuous.\n")
    X, y = mlw.load_dataset("SPAM", path)
    assert X.to_dict("list") == {"size": [12, 15]}
    assert y.tolist() == [0, 1]


def test_import_dhi_separates_target_and_reports_missing_column(tmp_path):
    path = tmp_path / "example.csv"
    path.write_text("label,size\n1,20\n0,12\n")
    X, y = mlw.load_dataset("DHI", path, "label")
    assert X.to_dict("list") == {"size": [20, 12]}
    assert y.tolist() == [1, 0]
    with pytest.raises(KeyError):
        mlw.load_dataset("DHI", path, "missing")


def test_exploration_counts(sample):
    X, y = sample
    assert mlw.summarize_dataset(X).loc["signal", "missing"] == 1
    table = mlw.get_class_distribution(y)
    assert table["count"].tolist() == [120, 40]
    assert table["proportion"].tolist() == [.75, .25]


class TestCleanDataset:
    @pytest.mark.parametrize("drop_duplicates, expected", [(False, [0, 1, 2]), (True, [0, 2])])
    def test_keeps_labeled_rows_and_optionally_removes_exact_repeats(self, drop_duplicates, expected):
        index = [4, 4, 7, 9]
        X = pd.DataFrame({"feature": [None, None, None, 5]}, index=index)
        y = pd.Series([0, 0, 1, None], index=index, name="target")
        original_X, original_y = X.copy(), y.copy()

        X_clean, y_clean = mlw.clean_dataset(X, y, drop_duplicates=drop_duplicates)

        pd.testing.assert_frame_equal(X_clean, original_X.iloc[expected])
        pd.testing.assert_series_equal(y_clean, original_y.iloc[expected])
        pd.testing.assert_frame_equal(X, original_X)
        pd.testing.assert_series_equal(y, original_y)

    def test_keeps_repeats_by_default(self):
        X = pd.DataFrame({"feature": [5, 5]})
        y = pd.Series([1, 1])
        X_clean, y_clean = mlw.clean_dataset(X, y)
        pd.testing.assert_frame_equal(X_clean, X)
        pd.testing.assert_series_equal(y_clean, y)

    def test_rejects_misaligned_index(self):
        X = pd.DataFrame({"feature": [1, 2]})
        y = pd.Series([0, 1], index=[1, 0])
        with pytest.raises(ValueError, match="index"):
            mlw.clean_dataset(X, y)

    def test_rejects_unexpected_target(self):
        with pytest.raises(ValueError, match="binary"):
            mlw.clean_dataset(pd.DataFrame({"feature": [3]}), pd.Series([2]))


def test_stratified_split_is_disjoint_and_reproducible(sample):
    X, y = sample
    train, test, y_train, y_test = mlw.split_dataset(X, y)
    assert (len(train), len(test)) == (128, 32)
    assert train.index.intersection(test.index).empty
    assert train.index.equals(y_train.index) and test.index.equals(y_test.index)
    assert y_train.mean() == y_test.mean() == .25
    assert train.equals(mlw.split_dataset(X, y)[0])
    assert not train.equals(mlw.split_dataset(X, y, random_state=9)[0])
    with pytest.raises(ValueError):
        mlw.split_dataset(X, y, test_size=1.5)


def test_equal_profiles_never_cross_grouped_holdout(sample):
    X, y = sample
    X = pd.concat([X, X], ignore_index=True)
    y = pd.concat([y, 1 - y], ignore_index=True)
    train, test, _, _ = mlw.split_by_profile(X, y)
    groups = mlw.profile_groups(X)
    assert set(groups[train.index]).isdisjoint(groups[test.index])
    assert len(train) + len(test) == len(X)
    for fold_train, fold_valid in mlw.training_folds(X, y, group_profiles=True):
        assert set(groups[fold_train]).isdisjoint(groups[fold_valid])


@pytest.mark.parametrize("strategy, expected", [("median", 2.), ("mean", 4.)])
def test_imputation_uses_only_training_statistics(strategy, expected):
    imputer = mlw.build_imputer(strategy).fit([[1], [2], [9], [np.nan]])
    assert imputer.transform([[np.nan], [500]])[0, 0] == expected


def test_pca_scaler_and_pipeline_fit_boundaries(sample):
    X, y = sample
    train, test, y_train, _ = mlw.split_dataset(X, y)
    pipeline = mlw.build_pipeline(LogisticRegression(), pca_components=.95)
    fitted = mlw.train_model(pipeline, train, y_train)
    assert list(fitted.named_steps) == ["imputer", "scaler", "pca", "model"]
    assert not hasattr(pipeline["scaler"], "mean_")
    np.testing.assert_allclose(fitted["imputer"].statistics_, train.median())
    scaled = fitted[:2].transform(train)
    np.testing.assert_allclose(np.mean(scaled, axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(np.std(scaled, axis=0), 1)
    assert fitted["pca"].explained_variance_ratio_.sum() >= .95
    assert mlw.build_pca(1).fit_transform(scaled).shape == (len(train), 1)
    assert fitted.predict(test).shape == (len(test),)


@pytest.mark.parametrize("method", ["kbest", "rfe", "model"])
def test_selector_keeps_informative_column(sample, method):
    X, y = sample
    pipeline = mlw.build_pipeline(LogisticRegression(), selector=mlw.build_feature_selector(method, 1))
    pipeline.fit(X, y)
    assert mlw.get_selected_features(pipeline, X.columns) == ["signal"]


@pytest.mark.parametrize("method", ["mutual_info", "f_classif"])
def test_scores_rank_signal_first(sample, method):
    X, y = sample
    scores = mlw.compute_feature_scores(X, y, method)
    assert scores.index[0] == "signal"
    assert scores.is_monotonic_decreasing
    assert set(scores.index) == set(X.columns)


def test_metrics_and_baseline():
    scores = mlw.compute_metrics([0, 0, 1, 1], [0, 1, 1, 1], [.1, .6, .7, .9])
    assert scores["precision"] == pytest.approx(2/3)
    assert scores["recall"] == 1
    assert scores["balanced_accuracy"] == .75
    assert scores["roc_auc"] == scores["average_precision"] == 1
    baseline = mlw.compute_metrics([0, 0, 0, 1], [0] * 4)
    assert baseline["accuracy"] == .75 and baseline["balanced_accuracy"] == .5
    assert "roc_auc" not in baseline


def test_grid_and_folds(sample):
    X, y = sample
    model = mlw.get_models(class_weight="balanced")["Logistic"]
    assert model.class_weight == "balanced"
    pipeline = mlw.build_pipeline(model)
    cv = mlw.build_cv_splitter(3)
    scores = mlw.cross_validate_model(pipeline, X, y, ["roc_auc", "recall"], cv)
    assert scores.shape == (3, 2)
    grid = mlw.get_param_grids()["Logistic"]
    fitted = mlw.tune_hyperparameters(pipeline, grid, X, y, "roc_auc", cv)
    assert fitted.best_params_["model__C"] in grid["model__C"]


def test_workflow_and_plots(sample):
    X, y = sample
    scores, models = mlw.run_workflow(X, y, scoring="average_precision")
    assert set(scores.index) == set(models) == {"Majority", "Logistic", "Tree", "Boosting"}
    assert scores.loc["Logistic", "average_precision"] > scores.loc["Majority", "average_precision"]
    _, test, _, target = mlw.split_dataset(X, y)
    fig, axes = plt.subplots(2, 2)
    try:
        assert mlw.plot_confusion_matrix(models["Logistic"], test, target, ax=axes[0, 0]) is axes[0, 0]
        assert mlw.plot_roc_curves(models, test, target, ax=axes[0, 1]) is axes[0, 1]
        assert mlw.plot_precision_recall_curves(models, test, target, ax=axes[1, 0]) is axes[1, 0]
        assert mlw.plot_model_comparison(scores, "recall", ax=axes[1, 1]) is axes[1, 1]
    finally:
        plt.close(fig)
