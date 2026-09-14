import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.features.build_gridded_sequences import (
    CHANNELS,
    apply_channel_standardizer,
    build_gridded_sequences,
    fit_channel_standardizer,
)
from src.features.build_tabular_dataset import (
    FEATURE_COLUMNS,
    build_tabular_dataset,
)
from src.features.temporal_features import chronological_split_indices
from src.models.train_convlstm import positive_class_weight as conv_class_weight
from src.models.calibrate import calibrate_fusion
from src.models.prediction_contracts import grid_prediction_keys
from src.models.predict import predict_probabilities
from src.models.train_fusion import train_fusion
from src.models.train_xgboost import positive_class_weight as xgb_class_weight
from src.models.train_xgboost import train_xgboost


def test_tabular_builder_freezes_order_and_rejects_leakage():
    record = {name: float(index + 1) for index, name in enumerate(FEATURE_COLUMNS)}
    result = build_tabular_dataset([record])
    assert tuple(result.columns) == FEATURE_COLUMNS
    leaking = pd.DataFrame([{**record, "observed_rainfall_mm": 100.0}])
    with pytest.raises(ValueError, match="Post-issue"):
        build_tabular_dataset(leaking)


def test_tensor_builder_and_training_only_standardization():
    shape = (2, 3, 4, 5)
    dataset = xr.Dataset(
        {
            name: (
                ("init_time", "lead_day", "latitude", "longitude"),
                np.full(shape, index + 1.0),
            )
            for index, name in enumerate(CHANNELS)
        }
    )
    tensor = build_gridded_sequences(dataset)
    assert tensor.shape == (*shape, len(CHANNELS))
    statistics = fit_channel_standardizer(tensor)
    standardized = apply_channel_standardizer(tensor, statistics)
    assert standardized.shape == tensor.shape
    assert np.allclose(standardized, 0.0)


def test_tensor_builder_rejects_observation_variables():
    dataset = xr.Dataset(
        {
            "observed_rainfall_mm": (
                ("init_time", "lead_day", "latitude", "longitude"),
                np.zeros((1, 1, 1, 1)),
            )
        }
    )
    with pytest.raises(ValueError, match="Post-issue"):
        build_gridded_sequences(dataset)


def test_chronological_split_preserves_order():
    times = np.arange("2019-07-01", "2019-07-11", dtype="datetime64[D]")[::-1]
    splits = chronological_split_indices(times)
    ordered = {
        name: times[index].astype("datetime64[D]") for name, index in splits.items()
    }
    assert ordered["train"].max() < ordered["validation"].min()
    assert ordered["validation"].max() < ordered["test"].min()


def test_training_fold_class_weights():
    labels = np.array([0, 0, 0, 1])
    assert xgb_class_weight(labels) == 3.0
    assert conv_class_weight(labels.reshape(2, 2)) == 3.0


def test_tensorflow_convlstm_output_contract():
    pytest.importorskip("tensorflow")
    from src.models.convlstm import build_convlstm

    model = build_convlstm(
        2,
        sequence_length=3,
        grid_shape=(8, 8),
        hidden_channels=(2, 3),
    )
    assert model.input_shape == (None, 3, 8, 8, 2)
    assert model.output_shape == (None, 3, 8, 8, 1)
    probabilities = model.predict(
        np.zeros((2, 3, 8, 8, 2), dtype=np.float32), verbose=0
    )
    assert probabilities.shape == (2, 3, 8, 8, 1)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_xgboost_calibration_fusion_and_prediction_smoke():
    pytest.importorskip("xgboost")
    rng = np.random.default_rng(42)
    features = rng.normal(size=(128, 4))
    labels = np.tile([0, 1], 64)
    xgb_model = train_xgboost(
        features[:32],
        labels[:32],
        {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "n_estimators": 4,
            "max_depth": 2,
            "random_state": 42,
            "n_jobs": 1,
        },
    )
    xgb_probability = xgb_model.predict_proba(features)[:, 1]
    convlstm_probability = np.clip(
        0.25 + 0.5 * labels + rng.normal(0, 0.05, labels.size), 0, 1
    )
    keys = grid_prediction_keys([f"run-{i}" for i in range(32)], [1, 2], [["a", "b"]])
    xgb_rows = keys.assign(xgboost_probability=xgb_probability)
    spatial_rows = keys.assign(spatial_probability=convlstm_probability)
    label_rows = keys.assign(bust_label=labels)
    fusion_model = train_fusion(xgb_rows.iloc[32:64], spatial_rows.iloc[32:64], label_rows.iloc[32:64])
    calibrated_fusion = calibrate_fusion(
        fusion_model, xgb_rows.iloc[64:96], spatial_rows.iloc[64:96], label_rows.iloc[64:96]
    )

    class ConvLSTMStub:
        def predict(self, grids, verbose=0):
            return convlstm_probability[96:].reshape(8, 2, 1, 2, 1)

    output = predict_probabilities(
        xgb_model,
        ConvLSTMStub(),
        calibrated_fusion,
        features[96:],
        np.zeros((8, 2, 1, 2, 1), dtype=np.float32),
        tabular_keys=keys.iloc[96:],
        grid_keys=keys.iloc[96:],
    )
    assert set(output) == {
        "run_id", "lead_day", "grid_id",
        "xgboost_probability",
        "convlstm_probability",
        "model_disagreement",
        "overall_bust_probability",
    }
    assert len(output) == 32
    assert output["overall_bust_probability"].between(0, 1).all()
