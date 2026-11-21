"""Unit tests for ml_workflow.py. Run: pytest"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.pipeline import Pipeline

import ml_workflow as mlw  # noqa: F401

matplotlib.use("Agg")  # headless

# --- Fixtures ---


@pytest.fixture
def toy_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Toy data with known NaNs, one duplicate row, one informative and one noise
    feature."""
    # Rows 0 and 2 are exact duplicates, row 3 has no label, feature2 has a NaN (row 4).
    # feature1 follows the label (informative), feature2 does not (noise).
    X = pd.DataFrame({
        "feature1": [0.1, 0.9, 0.1, 0.5, 1.0, 0.0],
        "feature2": [5.0, 0.0, 5.0, 2.0, None, 4.0],
    })
    y = pd.Series([0, 1, 0, None, 1, 0], name="target")
    return X, y


@pytest.fixture
def toy_csv(tmp_path: Path, toy_dataset: tuple[pd.DataFrame, pd.Series]) -> Path:
    """toy_dataset written to a temporary CSV, target included."""
    X, y = toy_dataset
    path = tmp_path / "sample.csv"
    pd.concat([X, y], axis=1).to_csv(path, index=False)
    return path


@pytest.fixture
def toy_split(
    toy_dataset: tuple[pd.DataFrame, pd.Series],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Hand-made train/test split, independent of split_dataset."""
    X, y = toy_dataset
    train, test = [0, 1, 4, 5], [2]  # row 3 has no label
    return X.loc[train], X.loc[test], y.loc[train], y.loc[test]


@pytest.fixture
def modeling_sample():
    rng = np.random.default_rng(14)
    y = pd.Series(np.tile([0, 0, 1], 36), name="target")
    X = pd.DataFrame({"signal": y + rng.normal(0, .2, len(y)), "noise": rng.normal(size=len(y))})
    X.loc[::13, "signal"] = np.nan
    return X, y


@pytest.fixture
def fitted_models(modeling_sample):
    X_train, _, y_train, _ = mlw.split_dataset(*modeling_sample)
    return {
        name: mlw.train_model(mlw.build_pipeline(model), X_train, y_train)
        for name, model in mlw.get_models().items()
    }


# --- 1. Data import ---


class TestLoadDataset:
    def test_returns_features_and_target_of_same_length(self, toy_csv):
        X, y = mlw.load_dataset("DHI", toy_csv, "target")
        assert len(X) == len(y) == 6
        assert X.index.equals(y.index)

    def test_target_column_is_removed_from_features(self, toy_csv):
        X, y = mlw.load_dataset("DHI", toy_csv, "target")
        assert list(X.columns) == ["feature1", "feature2"]
        assert y.name == "target"

    def test_missing_file_raises_file_not_found_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            mlw.load_dataset("DHI", tmp_path / "absent.csv", "target")

    def test_unknown_target_column_raises_key_error(self, toy_csv):
        with pytest.raises(KeyError):
            mlw.load_dataset("DHI", toy_csv, "unknown")


# --- 2. Exploration ---


class TestGetClassDistribution:
    def test_counts_match_known_class_sizes(self, toy_dataset):
        _, y = toy_dataset
        assert mlw.get_class_distribution(y)["count"].to_dict() == {0: 3, 1: 2}

    def test_proportions_sum_to_one(self, toy_dataset):
        _, y = toy_dataset
        assert mlw.get_class_distribution(y)["proportion"].sum() == pytest.approx(1)


# --- 3. Cleaning ---

class TestCleanDataset:

    def test_removes_exact_duplicate_rows(self, toy_dataset):
        """Exact duplicates in both X and y should be removed."""

        X, y = toy_dataset

        X_clean, y_clean = mlw.clean_dataset(
            X, y, drop_duplicates=True
        )

        # One duplicate and one missing target are removed
        assert len(X_clean) == 4
        assert len(y_clean) == 4

        # No exact duplicate should remain
        combined = pd.concat(
            [X_clean.reset_index(drop=True),
             y_clean.reset_index(drop=True)],
            axis=1
        )

        assert not combined.duplicated().any()

    def test_keeps_duplicates_when_disabled(self, toy_dataset):
        """Duplicates should be preserved when disabled."""

        X, y = toy_dataset

        X_clean, y_clean = mlw.clean_dataset(
            X, y, drop_duplicates=False
        )

        # Only the observation with missing target is removed
        assert len(X_clean) == 5
        assert len(y_clean) == 5

        # The duplicate observations are still present
        assert X_clean.loc[0].equals(X_clean.loc[2])
        assert y_clean.loc[0] == y_clean.loc[2]

    def test_features_and_target_stay_aligned(self, toy_dataset):
        """Features and targets must retain their original correspondence."""

        X, y = toy_dataset

        X_clean, y_clean = mlw.clean_dataset(X, y)

        # Original indices must be preserved
        assert X_clean.index.equals(y_clean.index)

        # Rows with missing targets must be removed
        assert X_clean.index.tolist() == [0, 1, 2, 4, 5]

        # Targets must correspond to their original observations
        assert_series_equal(
            y_clean,
            y.loc[[0, 1, 2, 4, 5]]
        )

    def test_missing_feature_values_are_left_in_place(self, toy_dataset):
        """Missing feature values should not be removed or imputed."""

        X, y = toy_dataset

        X_clean, y_clean = mlw.clean_dataset(X, y)

        # The missing feature value must still exist
        assert X_clean["feature2"].isna().sum() == 1

        # No observation should be removed because of missing features
        assert len(X_clean) == 5

    def test_inputs_are_not_modified(self, toy_dataset):
        """The original X and y must remain unchanged."""

        X, y = toy_dataset

        X_original = X.copy(deep=True)
        y_original = y.copy(deep=True)

        mlw.clean_dataset(X, y, drop_duplicates=True)

        assert_frame_equal(X, X_original)
        assert_series_equal(y, y_original)

    def test_removes_missing_target_values(self, toy_dataset):
        """Observations with missing targets must be removed."""

        X, y = toy_dataset

        X_clean, y_clean = mlw.clean_dataset(X, y)

        assert not y_clean.isna().any()
        assert len(X_clean) == 5
        assert len(y_clean) == 5

    def test_keeps_same_features_with_different_targets(self):
        """Same features with different targets are not exact duplicates."""

        X = pd.DataFrame({
            "feature1": [1, 1],
            "feature2": [10, 10],
        })

        y = pd.Series([0, 1], name="target")

        X_clean, y_clean = mlw.clean_dataset(
            X, y, drop_duplicates=True
        )

        assert len(X_clean) == 2
        assert len(y_clean) == 2
        assert y_clean.tolist() == [0, 1]

    def test_raises_error_when_indices_are_different(self):
        """X and y must have identical indices."""

        X = pd.DataFrame({
            "feature1": [1, 2],
        }, index=[0, 1])

        y = pd.Series([0, 1], index=[1, 2])

        with pytest.raises(ValueError, match="same index"):
            mlw.clean_dataset(X, y)




# --- 4. Splitting ---


class TestSplitDataset:
    def test_subset_sizes_match_test_size(self, modeling_sample):
        train, test, _, _ = mlw.split_dataset(*modeling_sample, test_size=.25)
        assert (len(train), len(test)) == (81, 27)

    def test_train_and_test_do_not_overlap(self, modeling_sample):
        train, test, _, _ = mlw.split_dataset(*modeling_sample)
        assert train.index.intersection(test.index).empty

    def test_stratification_preserves_class_proportions(self, modeling_sample):
        _, _, train, test = mlw.split_dataset(*modeling_sample, test_size=.25)
        assert train.mean() == test.mean() == pytest.approx(1 / 3)

    def test_same_seed_gives_same_split(self, modeling_sample):
        first = mlw.split_dataset(*modeling_sample)[0]
        assert_frame_equal(first, mlw.split_dataset(*modeling_sample)[0])

    def test_different_seeds_give_different_splits(self, modeling_sample):
        first = mlw.split_dataset(*modeling_sample, random_state=42)[1]
        other = mlw.split_dataset(*modeling_sample, random_state=43)[1]
        assert set(first.index) != set(other.index)

    def test_invalid_test_size_raises_value_error(self, modeling_sample):
        with pytest.raises(ValueError, match="test_size"):
            mlw.split_dataset(*modeling_sample, test_size=1.1)


# --- 5. Pre-processing ---


class TestPreprocessingSteps:
    def test_imputer_fills_missing_values_with_training_median(self, toy_split):
        X_train, _, _, _ = toy_split
        X_imputed = mlw.build_imputer("median").fit_transform(X_train)

        # feature2 in train: [5, 0, NaN, 4] -> median 4
        assert not np.isnan(X_imputed).any()
        assert X_imputed[2, 1] == 4.0

    def test_imputer_mean_strategy_uses_training_mean(self, toy_split):
        X_train, _, _, _ = toy_split
        X_imputed = mlw.build_imputer("mean").fit_transform(X_train)

        # feature2 in train: [5, 0, NaN, 4] -> mean 3 (different from the median)
        assert X_imputed[2, 1] == 3.0

    def test_imputer_unknown_strategy_raises_value_error(self):
        with pytest.raises(ValueError):
            mlw.build_imputer("mode")

    def test_scaler_gives_zero_mean_and_unit_variance(self, toy_split):
        X_train, _, _, _ = toy_split
        X_imputed = mlw.build_imputer().fit_transform(X_train)
        X_scaled = mlw.build_scaler().fit_transform(X_imputed)

        assert np.allclose(X_scaled.mean(axis=0), 0)
        assert np.allclose(X_scaled.std(axis=0), 1)

    def test_pca_with_integer_keeps_that_many_components(self, toy_split):
        X_train, _, _, _ = toy_split
        X_imputed = mlw.build_imputer().fit_transform(X_train)

        assert mlw.build_pca(n_components=1).fit_transform(X_imputed).shape == (4, 1)

    def test_pca_with_fraction_reaches_requested_variance(self, toy_split):
        X_train, _, _, _ = toy_split
        X_imputed = mlw.build_imputer().fit_transform(X_train)
        X_scaled = mlw.build_scaler().fit_transform(X_imputed)
        pca = mlw.build_pca(n_components=0.9).fit(X_scaled)

        assert pca.explained_variance_ratio_.sum() >= 0.9


# --- 6. Feature selection ---


class TestBuildFeatureSelector:
    def test_keeps_requested_number_of_features(self, toy_split):
        X_train, _, y_train, _ = toy_split
        X_imputed = mlw.build_imputer().fit_transform(X_train)
        selector = mlw.build_feature_selector("kbest", n_features=1)

        assert selector.fit_transform(X_imputed, y_train).shape == (4, 1)

    def test_unknown_method_raises_value_error(self):
        with pytest.raises(ValueError):
            mlw.build_feature_selector("unknown")


class TestComputeFeatureScores:
    def test_returns_one_score_per_feature_sorted_descending(self, toy_split):
        X_train, _, y_train, _ = toy_split
        scores = mlw.compute_feature_scores(X_train, y_train)

        assert set(scores.index) == set(X_train.columns)
        assert scores.is_monotonic_decreasing

    def test_informative_feature_ranks_above_noise_feature(self, toy_split):
        X_train, _, y_train, _ = toy_split
        scores = mlw.compute_feature_scores(X_train, y_train, method="f_classif")

        assert list(scores.index) == ["feature1", "feature2"]


# --- 7. Models and training ---


class TestGetModels:
    def test_returns_three_unfitted_classifiers(self):
        models = mlw.get_models()
        assert set(models) == {"Logistic", "Tree", "Boosting"}
        assert all(not hasattr(model, "classes_") for model in models.values())

    def test_class_weight_is_passed_to_supporting_models(self):
        assert all(model.class_weight == "balanced" for model in mlw.get_models("balanced").values())

    def test_param_grids_match_the_models(self):
        grids = mlw.get_param_grids()
        for name, model in mlw.get_models().items():
            assert grids[name].keys() <= mlw.build_pipeline(model).get_params().keys()


class TestBuildPipeline:
    def test_steps_follow_the_fixed_order(self):
        selector = mlw.build_feature_selector("kbest", n_features=1)
        pipe = mlw.build_pipeline(mlw.get_models()["Logistic"], selector=selector, pca_components=1)
        assert list(pipe.named_steps) == ["imputer", "scaler", "selector", "pca", "model"]
        assert pipe["selector"] is not selector

    def test_optional_steps_are_left_out_by_default(self):
        pipe = mlw.build_pipeline(mlw.get_models()["Tree"])
        assert list(pipe.named_steps) == ["imputer", "scaler", "model"]

    def test_pipeline_accepts_missing_values(self, toy_split):
        train, test, target, _ = toy_split
        pipe = mlw.build_pipeline(mlw.get_models()["Logistic"])
        pipe.fit(train, target)
        assert pipe.predict(test).shape == (len(test),)

    def test_preprocessing_is_learned_from_training_data_only(self, toy_split):
        train, test, target, _ = toy_split
        pipe = mlw.build_pipeline(mlw.get_models()["Logistic"]).fit(train, target)
        pipe.predict(test + 1000)
        np.testing.assert_allclose(pipe["imputer"].statistics_, train.median())
        np.testing.assert_allclose(pipe["scaler"].mean_, train.fillna(train.median()).mean())


class TestTrainModel:
    def test_returns_fitted_pipeline(self, toy_split):
        train, _, target, _ = toy_split
        fitted = mlw.train_model(mlw.build_pipeline(mlw.get_models()["Tree"]), train, target)
        assert isinstance(fitted, Pipeline)
        np.testing.assert_array_equal(fitted.classes_, [0, 1])

    def test_input_pipeline_is_left_unfitted(self, toy_split):
        train, _, target, _ = toy_split
        pipe = mlw.build_pipeline(mlw.get_models()["Logistic"])
        mlw.train_model(pipe, train, target)
        assert not hasattr(pipe["imputer"], "statistics_")

    def test_same_seed_gives_same_predictions(self, toy_split):
        train, test, target, _ = toy_split
        pipe = mlw.build_pipeline(mlw.get_models()["Tree"])
        first = mlw.train_model(pipe, train, target)
        second = mlw.train_model(pipe, train, target)
        np.testing.assert_array_equal(first.predict(test), second.predict(test))


class TestCrossValidation:
    def test_splitter_yields_same_folds_on_every_call(self, modeling_sample):
        X, y = modeling_sample
        folds = mlw.build_cv_splitter(3)
        for (train, valid), (_, again) in zip(folds.split(X, y), folds.split(X, y)):
            np.testing.assert_array_equal(valid, again)
            assert not set(train).intersection(valid)

    def test_cross_validation_returns_one_row_per_fold(self, modeling_sample):
        X, y = modeling_sample
        pipe = mlw.build_pipeline(mlw.get_models()["Logistic"])
        scores = mlw.cross_validate_model(pipe, X, y, ["recall", "f1"], mlw.build_cv_splitter(3))
        assert list(scores.columns) == ["recall", "f1"]
        assert scores.shape == (3, 2)

    def test_tuning_selects_parameters_from_the_grid(self, modeling_sample):
        X, y = modeling_sample
        pipe = mlw.build_pipeline(mlw.get_models()["Logistic"])
        grid = mlw.get_param_grids()["Logistic"]
        search = mlw.tune_hyperparameters(pipe, grid, X, y, "f1", mlw.build_cv_splitter(3))
        assert search.best_params_["model__C"] in grid["model__C"]
        np.testing.assert_allclose(search.best_estimator_["imputer"].statistics_, X.median())


class TestComputeMetrics:
    def test_perfect_predictions_score_one(self):
        scores = mlw.compute_metrics([0, 1, 0, 1], [0, 1, 0, 1], [.2, .8, .1, .9])
        assert all(value == pytest.approx(1) for value in scores.values())

    def test_values_match_hand_computed_example(self):
        scores = mlw.compute_metrics([0, 0, 1, 1], [0, 1, 1, 1])
        assert scores["precision"] == pytest.approx(2 / 3)
        assert scores["recall"] == 1
        assert scores["f1"] == pytest.approx(.8)

    def test_score_metrics_are_present_only_with_scores(self):
        y = [0, 1, 1]
        assert "average_precision" not in mlw.compute_metrics(y, y)
        assert "roc_auc" in mlw.compute_metrics(y, y, [.1, .8, .9])

    def test_majority_class_predictor_has_chance_balanced_accuracy(self):
        y = [0, 0, 0, 1]
        assert mlw.compute_metrics(y, [0] * 4)["balanced_accuracy"] == .5


class TestCompareModels:
    def test_one_row_per_model_and_one_column_per_metric(self, fitted_models, modeling_sample):
        _, X, _, y = mlw.split_dataset(*modeling_sample)
        results = mlw.compare_models(fitted_models, X, y)
        assert set(results.index) == set(fitted_models)
        assert (results[["TP", "FP", "TN", "FN"]].sum(axis=1) == len(y)).all()
        assert results["average_precision"].between(0, 1).all()


class TestPlots:
    def test_plot_functions_return_axes(self, fitted_models, modeling_sample):
        from matplotlib.axes import Axes
        from matplotlib import pyplot as plt
        _, X, _, y = mlw.split_dataset(*modeling_sample)
        ax = mlw.plot_roc_curves(fitted_models, X, y)
        assert isinstance(ax, Axes)
        assert len(ax.lines) == len(fitted_models)
        plt.close(ax.figure)

    def test_plot_functions_draw_on_given_axes(self, fitted_models, modeling_sample):
        from matplotlib import pyplot as plt
        _, X, _, y = mlw.split_dataset(*modeling_sample)
        fig, axes = plt.subplots(1, 3)
        try:
            assert mlw.plot_precision_recall_curves(fitted_models, X, y, ax=axes[0]) is axes[0]
            assert mlw.plot_confusion_matrix(fitted_models["Tree"], X, y, ax=axes[1]) is axes[1]
            scores = mlw.compare_models(fitted_models, X, y)
            assert mlw.plot_model_comparison(scores, "f1", ax=axes[2]) is axes[2]
        finally:
            plt.close(fig)


class TestRunWorkflow:
    def test_end_to_end_with_student_selector(self, modeling_sample):
        X, y = modeling_sample
        scores, fitted = mlw.run_workflow(
            X, y, scoring="average_precision",
            selector=mlw.build_feature_selector("kbest", n_features=1), compare_weights=True,
        )
        assert set(scores.index) == {"Logistic", "Tree", "Boosting", "Majority"}
        assert scores.loc["Logistic", "average_precision"] > scores.loc["Majority", "average_precision"]
        assert mlw.get_selected_features(fitted["Logistic"], X.columns) == ["signal"]
        assert scores.loc["Majority", "recall"] == 0

    def test_same_seed_gives_identical_results(self, modeling_sample):
        first, _ = mlw.run_workflow(*modeling_sample, scoring="f1")
        second, _ = mlw.run_workflow(*modeling_sample, scoring="f1")
        assert_frame_equal(first, second)


def test_threshold_maximizes_recall_under_precision_constraint():
    labels = [1, 0, 1, 0, 1]
    scores = [.99, .97, .88, .5, .1]
    assert mlw.choose_precision_threshold(labels, scores, .98) == .99
    assert mlw.choose_precision_threshold(labels, scores, .6) == .1
    with pytest.raises(ValueError, match="No observed threshold"):
        mlw.choose_precision_threshold([0, 1], [.9, .9], .98)


def test_threshold_evaluation_counts_tied_scores_as_positive(modeling_sample):
    X, y = modeling_sample
    model = mlw.majority_baseline(X, y)
    threshold = model.predict_proba(X)[0, 1]
    result = mlw.evaluate_model(model, X, y, threshold=threshold)
    assert result["TP"] == int(y.sum())
    assert result["FP"] == len(y) - int(y.sum())
