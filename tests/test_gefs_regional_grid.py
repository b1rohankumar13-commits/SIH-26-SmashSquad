import numpy as np

from src.preprocessing.crop_gefs_region import _target_coordinates


def test_approved_domain_cell_centres_at_one_degree():
    latitudes, longitudes = _target_coordinates(38.0, 5.0, 65.0, 100.0, 1.0)
    assert len(latitudes) == 33
    assert len(longitudes) == 35
    assert np.isclose(latitudes[0], 37.5)
    assert np.isclose(latitudes[-1], 5.5)
    assert np.isclose(longitudes[0], 65.5)
    assert np.isclose(longitudes[-1], 99.5)
