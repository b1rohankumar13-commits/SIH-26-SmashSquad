import numpy as np
import pytest

from src.models.prediction_contracts import fusion_records, grid_prediction_keys
from src.models.train_fusion import train_fusion
from src.models.predict import predict_probabilities


def test_training_stages_purge_overlapping_verification_windows():
    from src.features.temporal_features import training_stage_indices

    times = np.arange("2020-01-01", "2020-06-01", dtype="datetime64[D]")
    ends = times + np.timedelta64(10, "D")
    cuts = ["2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01"]
    result = training_stage_indices(times, ends, train_end=cuts[0], tuning_end=cuts[1],
                                     fusion_end=cuts[2], calibration_end=cuts[3])
    for name, cut in zip(["train", "tuning", "fusion", "calibration"], cuts):
        assert np.all(ends[result[name]] < np.datetime64(cut))
    assert len(result["purged"]) == 40
    joined = np.concatenate(list(result.values()))
    assert len(np.unique(joined)) == len(times)


def test_fusion_reorders_by_identity_not_position():
    keys = grid_prediction_keys(["run"], [1, 2], [["a", "b"]])
    xgb = keys.assign(xgboost_probability=[0.1, 0.2, 0.3, 0.4])
    spatial = keys.assign(spatial_probability=[0.6, 0.7, 0.8, 0.9])
    joined = fusion_records(xgb, spatial.iloc[::-1])
    np.testing.assert_allclose(joined.spatial_probability, [0.6, 0.7, 0.8, 0.9])
    np.testing.assert_allclose(joined.model_disagreement, 0.5)
    labels = keys.assign(bust_label=[0, 1, 0, 1])
    a = train_fusion(xgb, spatial, labels)
    b = train_fusion(xgb, spatial.iloc[::-1], labels.iloc[::-1])
    np.testing.assert_allclose(a.coef_, b.coef_)


@pytest.mark.parametrize("problem", ["missing", "different", "duplicate", "nan"])
def test_fusion_rejects_bad_records(problem):
    keys = grid_prediction_keys(["run"], [1, 2], [["a", "b"]])
    xgb = keys.assign(xgboost_probability=0.3)
    spatial = keys.assign(spatial_probability=0.4)
    if problem == "missing":
        spatial = spatial.iloc[:-1]
    elif problem == "different":
        spatial.loc[0, "grid_id"] = "other"
    elif problem == "duplicate":
        spatial.loc[1, ["lead_day", "grid_id"]] = [1, "a"]
    else:
        spatial.loc[0, "spatial_probability"] = np.nan
    with pytest.raises(ValueError):
        fusion_records(xgb, spatial)


def test_inference_rejects_uncalibrated_stacker():
    with pytest.raises(ValueError, match="calibrated fusion"):
        predict_probabilities(None, None, object(), None, None,
                              tabular_keys=None, grid_keys=None)


def test_masked_loss_excludes_unknown_cells():
    tf = pytest.importorskip("tensorflow")
    from src.models.train_convlstm import PositiveWeightedBinaryCrossentropy

    target = tf.constant([[[[[0.0], [1.0], [0.0]]]]])
    prediction = tf.constant([[[[[0.25], [0.75], [0.999]]]]])
    mask = tf.constant([[[[1.0, 1.0, 0.0]]]])
    loss = PositiveWeightedBinaryCrossentropy(2.0)
    actual = loss(target, prediction, sample_weight=mask).numpy()
    assert actual == pytest.approx(-1.5 * np.log(0.75), rel=1e-5)


def test_spatial_training_mask_and_serialization(tmp_path):
    tf = pytest.importorskip("tensorflow")
    from src.models.train_convlstm import train_convlstm

    rng = np.random.default_rng(7)
    inputs = rng.normal(size=(2, 2, 3, 3, 2)).astype("float32")
    labels = (inputs[..., :1] > 0).astype("float32")
    labels[:, :, 0, 0] = np.nan
    config = dict(sequence_length=2, hidden_channels=[2, 2], kernel_size=3,
                  learning_rate=0.001, batch_size=2, epochs=1, verbose=0,
                  mixed_precision="off")
    model, history = train_convlstm(inputs, labels, inputs * 0.9, labels, config)
    assert np.isfinite(history.history["loss"]).all()
    assert np.isfinite(history.history["val_brier_score"]).all()
    prediction = model.predict(inputs, verbose=0)
    assert prediction.shape == labels.shape
    path = tmp_path / "spatial.keras"
    model.save(path)
    restored = tf.keras.models.load_model(path)
    np.testing.assert_allclose(restored.predict(inputs, verbose=0), prediction, atol=1e-6)
    with pytest.raises(ValueError, match="shape"):
        train_convlstm(inputs, np.zeros((2, 2)), inputs, labels, config)
