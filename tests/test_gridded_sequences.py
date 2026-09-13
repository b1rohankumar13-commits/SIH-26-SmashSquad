import numpy as np
import pytest
import xarray as xr

from src.features.build_gridded_sequences import (
    CHANNEL_NAMES,
    ChannelStandardizer,
    build_gridded_sequences,
)

RUNS = ["2021-07-01", "2021-07-02"]
LEADS = np.arange(1, 10)
LATS = np.array([10.0, 11.0, 12.0, 13.0])
LONS = np.array([70.0, 71.0, 72.0, 73.0, 74.0])
MEMBERS = [0, 1, 2]


def _surface(seed):
    rng = np.random.default_rng(seed)
    return xr.DataArray(
        rng.normal(size=(len(RUNS), len(LEADS), len(MEMBERS), LATS.size, LONS.size)),
        dims=("run", "lead", "member", "latitude", "longitude"),
        coords={"run": RUNS, "lead": LEADS, "member": MEMBERS,
                "latitude": LATS, "longitude": LONS},
    )


def _leveled(seed, levels=(850, 500)):
    rng = np.random.default_rng(seed)
    return xr.DataArray(
        rng.normal(size=(len(RUNS), len(LEADS), len(MEMBERS), len(levels), LATS.size, LONS.size)),
        dims=("run", "lead", "member", "level", "latitude", "longitude"),
        coords={"run": RUNS, "lead": LEADS, "member": MEMBERS, "level": list(levels),
                "latitude": LATS, "longitude": LONS},
    )


def _dataset():
    return xr.Dataset({
        "total_precipitation": _surface(1),
        "mean_sea_level_pressure": _surface(2),
        "u_wind": _leveled(3),
        "v_wind": _leveled(4),
        "specific_humidity": _leveled(5),
        "geopotential_height": _leveled(6),
    })


def test_build_produces_twelve_named_channels_channels_last():
    sequences = build_gridded_sequences(_dataset(), sequence_length=9)
    assert sequences.dims == ("run", "lead", "latitude", "longitude", "channel")
    assert sequences.shape == (2, 9, 4, 5, 12)
    assert list(sequences.coords["channel"].values) == list(CHANNEL_NAMES)


def test_mean_and_spread_channels_match_manual_reduction():
    dataset = _dataset()
    sequences = build_gridded_sequences(dataset)
    rain = dataset["total_precipitation"]
    expected_mean = rain.mean("member").transpose("run", "lead", "latitude", "longitude").values
    expected_spread = rain.std("member", ddof=1).transpose("run", "lead", "latitude", "longitude").values
    np.testing.assert_allclose(sequences.sel(channel="rainfall_mean_mm").values, expected_mean)
    np.testing.assert_allclose(sequences.sel(channel="rainfall_spread_mm").values, expected_spread)


def test_level_selection_picks_the_requested_level():
    dataset = _dataset()
    sequences = build_gridded_sequences(dataset)
    expected = dataset["u_wind"].sel(level=850).mean("member").transpose(
        "run", "lead", "latitude", "longitude").values
    np.testing.assert_allclose(sequences.sel(channel="u850_mean_ms").values, expected)


def test_standardizer_is_fit_on_training_runs_and_yields_finite_zero_mean():
    sequences = build_gridded_sequences(_dataset())
    standardizer = ChannelStandardizer.fit(sequences, train_runs=[RUNS[0]])
    out = standardizer.transform(sequences)
    assert out.shape == (2, 9, 4, 5, 12) and out.dtype == np.float32
    assert np.isfinite(out).all()
    train = out[0]
    np.testing.assert_allclose(train.reshape(-1, 12).mean(axis=0), np.zeros(12), atol=1e-6)
    np.testing.assert_allclose(train.reshape(-1, 12).std(axis=0), np.ones(12), atol=1e-6)


def test_missing_cells_are_imputed_to_training_mean_zero():
    dataset = _dataset()
    dataset["total_precipitation"][0, 0, :, 0, 0] = np.nan
    sequences = build_gridded_sequences(dataset)
    standardizer = ChannelStandardizer.fit(sequences, train_runs=RUNS)
    out = standardizer.transform(sequences)
    assert np.isfinite(out).all()
    rain_mean_idx = list(CHANNEL_NAMES).index("rainfall_mean_mm")
    assert out[0, 0, 0, 0, rain_mean_idx] == 0.0


def test_missing_ensemble_dimension_is_rejected():
    dataset = _dataset()
    dataset["total_precipitation"] = dataset["total_precipitation"].isel(member=0, drop=True)
    with pytest.raises(ValueError, match="ensemble dimension"):
        build_gridded_sequences(dataset)


def test_sequence_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="sequence_length"):
        build_gridded_sequences(_dataset(), sequence_length=10)


def test_missing_variable_is_rejected():
    dataset = _dataset().drop_vars("geopotential_height")
    with pytest.raises(ValueError, match="geopotential_height"):
        build_gridded_sequences(dataset)
