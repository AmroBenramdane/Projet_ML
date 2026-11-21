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

not_implemented = pytest.mark.skip(reason="Test not implemented yet")


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
    raise NotImplementedError


@pytest.fixture
def toy_split(
    toy_dataset: tuple[pd.DataFrame, pd.Series],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Hand-made train/test split, independent of split_dataset."""
    X, y = toy_dataset
    train, test = [0, 1, 4, 5], [2]  # row 3 has no label
    return X.loc[train], X.loc[test], y.loc[train], y.loc[test]


@pytest.fixture
def fitted_models(
    toy_split: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
) -> dict[str, Pipeline]:
    """Two small pipelines fitted on toy_split."""
    raise NotImplementedError


# --- 1. Data import ---


class TestLoadDataset:
    @not_implemented
    def test_returns_features_and_target_of_same_length(self, toy_csv): ...

    @not_implemented
    def test_target_column_is_removed_from_features(self, toy_csv): ...

    @not_implemented
    def test_missing_file_raises_file_not_found_error(self, tmp_path): ...

    @not_implemented
    def test_unknown_target_column_raises_key_error(self, toy_csv): ...


# --- 2. Exploration ---


class TestGetClassDistribution:
    @not_implemented
    def test_counts_match_known_class_sizes(self, toy_dataset): ...

    @not_implemented
    def test_proportions_sum_to_one(self, toy_dataset): ...




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
    @not_implemented
    def test_subset_sizes_match_test_size(self, toy_dataset): ...

    @not_implemented
    def test_train_and_test_do_not_overlap(self, toy_dataset): ...

    @not_implemented
    def test_stratification_preserves_class_proportions(self, toy_dataset): ...

    @not_implemented
    def test_same_seed_gives_same_split(self, toy_dataset): ...

    @not_implemented
    def test_different_seeds_give_different_splits(self, toy_dataset): ...

    @not_implemented
    def test_invalid_test_size_raises_value_error(self, toy_dataset): ...


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
    @not_implemented
    def test_returns_unfitted_classifiers_keyed_by_name(self): ...

    @not_implemented
    def test_class_weight_is_passed_to_supporting_models(self): ...

    @not_implemented
    def test_param_grids_match_the_models(self): ...


class TestBuildPipeline:
    @not_implemented
    def test_steps_follow_the_fixed_order(self): ...

    @not_implemented
    def test_optional_steps_are_left_out_by_default(self): ...

    @not_implemented
    def test_pipeline_accepts_missing_values(self, toy_split): ...

    @not_implemented
    def test_preprocessing_is_learned_from_training_data_only(self, toy_split): ...


class TestTrainModel:
    @not_implemented
    def test_returns_fitted_pipeline(self, toy_split): ...

    @not_implemented
    def test_input_pipeline_is_left_unfitted(self, toy_split): ...

    @not_implemented
    def test_same_seed_gives_same_predictions(self, toy_split): ...


class TestCrossValidation:
    @not_implemented
    def test_splitter_yields_same_folds_on_every_call(self, toy_split): ...

    @not_implemented
    def test_cross_validation_returns_one_row_per_fold(self, toy_split): ...

    @not_implemented
    def test_tuning_selects_parameters_from_the_grid(self, toy_split): ...


# --- 8. Evaluation ---


class TestComputeMetrics:
    @not_implemented
    def test_perfect_predictions_score_one(self): ...

    @not_implemented
    def test_values_match_hand_computed_example(self): ...

    @not_implemented
    def test_score_metrics_are_present_only_with_scores(self): ...

    @not_implemented
    def test_majority_class_predictor_has_chance_balanced_accuracy(self): ...


class TestCompareModels:
    @not_implemented
    def test_one_row_per_model_and_one_column_per_metric(
        self, fitted_models, toy_split
    ): ...


class TestPlots:
    @not_implemented
    def test_plot_functions_return_axes(self, fitted_models, toy_split): ...

    @not_implemented
    def test_plot_functions_draw_on_given_axes(self, fitted_models, toy_split): ...


# --- 9. Workflow ---


class TestRunWorkflow:
    @not_implemented
    def test_runs_end_to_end_on_toy_dataset(self, toy_dataset): ...

    @not_implemented
    def test_same_seed_gives_identical_results(self, toy_dataset): ...
