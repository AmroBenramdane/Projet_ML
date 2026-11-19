"""Unit tests for ml_workflow.py. Run: pytest tests.py"""

from pathlib import Path

import matplotlib
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

import ml_workflow as mlw  # noqa: F401

matplotlib.use("Agg")  # headless

not_implemented = pytest.mark.skip(reason="Test not implemented yet")


# --- Fixtures ---


@pytest.fixture
def toy_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Toy data with known NaNs, one duplicate row, one informative and one noise
    feature."""
    raise NotImplementedError


@pytest.fixture
def toy_csv(tmp_path: Path, toy_dataset: tuple[pd.DataFrame, pd.Series]) -> Path:
    """toy_dataset written to a temporary CSV, target included."""
    raise NotImplementedError


@pytest.fixture
def toy_split(
    toy_dataset: tuple[pd.DataFrame, pd.Series],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Hand-made train/test split, independent of split_dataset."""
    raise NotImplementedError


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
    @not_implemented
    def test_imputer_fills_missing_values_with_training_median(self, toy_split): ...

    @not_implemented
    def test_imputer_mean_strategy_uses_training_mean(self, toy_split): ...

    @not_implemented
    def test_imputer_unknown_strategy_raises_value_error(self): ...

    @not_implemented
    def test_scaler_gives_zero_mean_and_unit_variance(self, toy_split): ...

    @not_implemented
    def test_pca_with_integer_keeps_that_many_components(self, toy_split): ...

    @not_implemented
    def test_pca_with_fraction_reaches_requested_variance(self, toy_split): ...


# --- 6. Feature selection ---


class TestBuildFeatureSelector:
    @not_implemented
    def test_keeps_requested_number_of_features(self, toy_split): ...

    @not_implemented
    def test_unknown_method_raises_value_error(self): ...


class TestComputeFeatureScores:
    @not_implemented
    def test_returns_one_score_per_feature_sorted_descending(self, toy_split): ...

    @not_implemented
    def test_informative_feature_ranks_above_noise_feature(self, toy_split): ...


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
