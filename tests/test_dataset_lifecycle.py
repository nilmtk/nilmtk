from pathlib import Path

import nilmtk
from nilmtk import DataSet

ENERGY_FIXTURE = Path(__file__).resolve().parents[1] / "data" / "energy.h5"


def test_closing_one_dataset_keeps_the_shared_stats_cache_available():
    with DataSet(str(ENERGY_FIXTURE)) as first:
        assert len(first.buildings[1].elec.meters) == 3

    assert nilmtk.STATS_CACHE.store.is_open

    with DataSet(str(ENERGY_FIXTURE)) as second:
        assert second.buildings[1].elec.total_energy()["active"] > 0

    assert nilmtk.STATS_CACHE.store.is_open
