import numpy as np
import xarray as xr

from src.preprocessing.align_30_day_rainfall import (
    _target_grid,
    conservative_regrid,
)


def test_approved_target_grid_uses_domain_edges():
    latitude, longitude = _target_grid()
    assert latitude.size == 66
    assert longitude.size == 70
    assert np.isclose(latitude[0], 5.25)
    assert np.isclose(latitude[-1], 37.75)
    assert np.isclose(longitude[0], 65.25)
    assert np.isclose(longitude[-1], 99.75)


def test_conservative_regrid_preserves_constant_rainfall_and_leading_dims():
    source = xr.DataArray(
        np.full((2, 4, 4), 12.5),
        dims=("member", "latitude", "longitude"),
        coords={
            "member": ["cf", "pf01"],
            "latitude": [5.125, 5.375, 5.625, 5.875],
            "longitude": [65.125, 65.375, 65.625, 65.875],
        },
    )
    result = conservative_regrid(
        source,
        np.array([5.25, 5.75]),
        np.array([65.25, 65.75]),
    )
    assert result.dims == ("member", "latitude", "longitude")
    assert result.shape == (2, 2, 2)
    assert np.allclose(result.values, 12.5)
