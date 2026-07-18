import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from nilmtk.feature_detectors.cluster import _transform_data, cluster


def test_seed_controls_cluster_subsampling():
    readings = pd.Series(np.arange(11, 3_011, dtype=float))

    first = _transform_data(readings, random_state=13)
    np.random.seed(99)
    second = _transform_data(readings, random_state=13)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (2_000, 1)


def test_cluster_uses_only_distinct_states():
    readings = pd.Series([0.0] * 30 + [200.0] * 30)

    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        states = cluster(
            readings,
            exact_num_clusters=2,
            random_state=13,
        )

    np.testing.assert_array_equal(states, [0, 200])
