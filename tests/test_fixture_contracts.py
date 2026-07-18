from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nilmtk.tests.generate_data import RANDOM_SEED, create_random_df, power_data

DATA = Path(__file__).resolve().parents[1] / "data"
HDF_FIXTURES = (
    "co_test.h5",
    "energy.h5",
    "energy_complex.h5",
    "random.h5",
)


@pytest.mark.parametrize("filename", HDF_FIXTURES)
def test_hdf_fixture_is_readable_by_current_pandas(filename):
    with pd.HDFStore(DATA / filename, mode="r") as store:
        assert isinstance(store.root._v_attrs.metadata, dict)
        keys = store.keys()
        assert keys

        for key in keys:
            sample = store.select(key, start=0, stop=1)
            assert len(sample) == 1
            assert isinstance(sample.columns, pd.MultiIndex)

        building = store.get_node("/building1")._v_attrs.metadata
        assert isinstance(building, dict)
        assert building["elec_meters"]


def test_seeded_random_fixture_matches_the_generator():
    random = np.random.default_rng(RANDOM_SEED)
    with pd.HDFStore(DATA / "random.h5", mode="r") as store:
        for meter in range(1, 6):
            actual = store[f"/building1/elec/meter{meter}"]
            pd.testing.assert_frame_equal(actual, create_random_df(random))


@pytest.mark.parametrize(
    ("filename", "simple"),
    (("energy.h5", True), ("energy_complex.h5", False)),
)
def test_energy_fixtures_match_the_generator(filename, simple):
    with pd.HDFStore(DATA / filename, mode="r") as store:
        expected = power_data(simple=simple)
        for meter in range(1, 4):
            actual = store[f"/building1/elec/meter{meter}"]
            pd.testing.assert_frame_equal(actual, expected)
