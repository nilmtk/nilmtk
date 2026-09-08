"""Conversion contract tests using synthetic readings, without dataset downloads."""

from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from nilm_metadata import get_appliance_types

from nilmtk import DataSet
from nilmtk.measurement import LEVEL_NAMES

converter = import_module("nilmtk.dataset_converters.mored.convert_mored")


def metadata():
    return {
        house: yaml.safe_load(
            (converter.METADATA_PATH / f"building{house}.yaml").read_text()
        )
        for house in range(1, 11)
    }


def make_premises(root, houses):
    for house in houses:
        for meter, values in metadata()[house]["elec_meters"].items():
            # Construct the house independently of metadata to catch cross-house paths.
            path = root / f"Premises_{house}" / Path(values["data_location"]).name
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "timestamp": [
                        "06/07/2020 01:00:05",
                        "06/07/2020 01:00:00",
                        "06/07/2020 01:00:00",
                    ],
                    "Vrms": [231.0, 230.0, 999.0],
                    "real_power": [house * 100 + meter, house * 100 + meter, -999],
                }
            ).to_csv(path, index=False)


def test_metadata_covers_ten_premises_and_only_their_own_files():
    vocabulary = get_appliance_types()
    for house, building in metadata().items():
        assert building["instance"] == house
        instances = {}
        for meter in building["elec_meters"].values():
            assert Path(meter["data_location"]).parent.name == f"Premises_{house}"
            assert meter["device_model"] == "emonPi"
        for appliance in building["appliances"]:
            assert appliance["type"] in vocabulary
            instances.setdefault(appliance["type"], []).append(appliance["instance"])
            assert set(appliance["meters"]) <= building["elec_meters"].keys()
        for values in instances.values():
            assert sorted(values) == list(range(1, len(values) + 1))
    ninth = metadata()[9]
    assert set(ninth["elec_meters"]) == {2, 3, 4, 5, 6}
    assert not any(m.get("site_meter") for m in ninth["elec_meters"].values())


def test_full_conversion_loads_in_nilmtk_from_another_directory(tmp_path, monkeypatch):
    output_format = "HDF"
    source = tmp_path / "raw"
    make_premises(source, range(1, 11))
    monkeypatch.chdir(tmp_path)
    output = tmp_path / ("converted.h5" if output_format == "HDF" else "converted")
    converter.convert_mored(source, output, format=output_format)
    with DataSet(str(output), format=output_format) as dataset:
        assert set(dataset.buildings) == set(range(1, 11))
        assert dataset.metadata["number_of_buildings"] == 10
        assert dataset.metadata["timezone"] == converter.TIMEZONE
        for house, building in metadata().items():
            for meter in building["elec_meters"]:
                frame = next(dataset.store.load(f"/building{house}/elec/meter{meter}"))
                assert list(frame.columns) == [("voltage", ""), ("power", "active")]
                assert list(frame.columns.names) == LEVEL_NAMES
                assert frame.index.is_unique and frame.index.is_monotonic_increasing
                assert len(frame) == 2
                assert frame[("power", "active")].tolist() == [house * 100 + meter] * 2
                assert frame[("voltage", "")].tolist() == [230.0, 231.0]
                if output_format == "HDF":
                    assert all(dtype == np.float32 for dtype in frame.dtypes)
                    assert str(frame.index.tz) == converter.TIMEZONE
        assert dataset.buildings[9].elec.mains() is None


def test_subset_metadata_contains_only_converted_premises(tmp_path):
    source = tmp_path / "raw"
    make_premises(source, [9])
    output = tmp_path / "subset.h5"
    converter.convert_mored(source, output)
    with DataSet(str(output)) as dataset:
        assert set(dataset.buildings) == {9}
        assert dataset.metadata["number_of_buildings"] == 1


def test_named_columns_mixed_dates_missing_values_and_duplicates(tmp_path):
    path = tmp_path / "meter.csv"
    path.write_text(
        "real_power,timestamp,Vrms\n"
        "20,06-Jul-2020 01:00:05.125,231\n"
        "10,06/07/2020 01:00:00,230\n"
        "999,06/07/2020 01:00:00,999\n"
        "30,06/07/2020 01:00:10,\n"
        ",06/07/2020 01:00:15,\n"
        "40,,235\n"
    )
    frame = converter._read_meter_csv(path)
    assert frame[("power", "active")].tolist() == [10, 20, 30]
    assert frame[("voltage", "")].iloc[0] == 230
    assert np.isnan(frame[("voltage", "")].iloc[-1])
    assert frame.index[0] == pd.Timestamp("2020-07-06 01:00:00", tz="UTC")
    assert frame.index[1].microsecond == 125000


def test_source_timezone_is_explicit_and_preserves_legacy_utc_default(tmp_path):
    path = tmp_path / "meter.csv"
    path.write_text("timestamp,Vrms,real_power\n06/07/2020 01:00:00,230,10\n")
    legacy = converter._read_meter_csv(path)
    local = converter._read_meter_csv(path, source_timezone="Africa/Casablanca")
    assert legacy.index[0] - local.index[0] == pd.Timedelta(hours=1)


@pytest.mark.parametrize(
    "content",
    [
        "timestamp,Vrms,real_power\nnot-a-date,230,20\n",
        "timestamp,Vrms,real_power\n\n",
        "timestamp,voltage,power\n06/07/2020 01:00:00,230,20\n",
    ],
)
def test_invalid_or_empty_csv_is_rejected(tmp_path, content):
    path = tmp_path / "meter.csv"
    path.write_text(content)
    with pytest.raises(ValueError):
        converter._read_meter_csv(path)


@pytest.mark.parametrize("case", ["missing", "empty", "unknown", "incomplete"])
def test_bad_input_does_not_overwrite_destination(tmp_path, case):
    source = tmp_path / "raw"
    if case != "missing":
        source.mkdir()
    if case in ("unknown", "incomplete"):
        (source / ("Premises_11" if case == "unknown" else "Premises_1")).mkdir()
    output = tmp_path / "keep.h5"
    output.write_bytes(b"existing output")
    with pytest.raises((ValueError, FileNotFoundError)):
        converter.convert_mored(source, output)
    assert output.read_bytes() == b"existing output"


def test_output_closes_after_csv_failure(tmp_path, monkeypatch):
    source = tmp_path / "raw"
    make_premises(source, [8])
    (source / "Premises_8" / "Mains.csv").write_text("bad,header\n1,2\n")
    opened = []
    original = converter.get_datastore

    def capture(*args, **kwargs):
        store = original(*args, **kwargs)
        opened.append(store)
        return store

    monkeypatch.setattr(converter, "get_datastore", capture)
    with pytest.raises(ValueError):
        converter.convert_mored(source, tmp_path / "partial.h5")
    assert len(opened) == 1
    assert not opened[0].store.is_open


def test_unsupported_output_preserves_destination(tmp_path):
    source = tmp_path / "raw"
    make_premises(source, [8])
    output = tmp_path / "keep.h5"
    output.write_bytes(b"existing output")
    with pytest.raises(ValueError, match="Only HDF"):
        converter.convert_mored(source, output, format="CSV")
    assert output.read_bytes() == b"existing output"


def test_public_api_and_cli_convert_a_house(tmp_path):
    from nilmtk import dataset_converters
    from nilmtk.dataset_converters import cli

    assert dataset_converters.convert_mored is converter.convert_mored
    source = tmp_path / "raw"
    make_premises(source, [8])
    output = tmp_path / "mored.h5"
    assert cli.main(["mored", str(source), str(output)]) == 0
    with DataSet(str(output)) as result:
        assert set(result.buildings) == {8}


def test_cli_rejects_unsupported_csv_before_writing(tmp_path, capsys):
    from nilmtk.dataset_converters import cli

    output = tmp_path / "keep.h5"
    output.write_bytes(b"existing output")
    with pytest.raises(SystemExit) as error:
        cli.main(["mored", str(tmp_path), str(output), "--format", "CSV"])
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
    assert output.read_bytes() == b"existing output"
