"""End-to-end GeLaP tests; all meter readings are synthetic."""

import io
import tarfile
from importlib import import_module

import numpy as np
import pandas as pd
import pytest
import yaml
from nilm_metadata import get_appliance_types

from nilmtk import DataSet
from nilmtk.measurement import LEVEL_NAMES

converter = import_module("nilmtk.dataset_converters.gelap.convert_gelap")


def meter_csv(meter, house=1):
    if meter == 11:
        return (
            "time,power1,power2,power3\n"
            f"1582098565000,{house * 100},2,3\n"
            f"1582098564123,{house * 100},2,3\n"
            "1582098564123,999,999,999\n"
        ).encode()
    return (
        "time_request,time_reply,power\n"
        f"1582098564000,1582098565000,{house * 100 + meter}\n"
        f"1582098564000,1582098564123,{house * 100 + meter}\n"
        "1582098564000,1582098564123,999\n"
    ).encode()


def make_house(source, house=1):
    directory = source / f"hh-{house:02d}"
    directory.mkdir(parents=True)
    for name, meter in converter.METER_FILES.items():
        (directory / name).write_bytes(meter_csv(meter, house))
    return directory


def make_archive(path, *, prefix="./", omit=None, extra=None):
    with tarfile.open(path, "w:xz") as archive:
        for name, meter in converter.METER_FILES.items():
            if name == omit:
                continue
            content = meter_csv(meter)
            member = tarfile.TarInfo(prefix + name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
        if extra:
            archive.addfile(extra, io.BytesIO(b""))


def load_metadata(house):
    return yaml.safe_load(
        (converter.METADATA_PATH / f"building{house}.yaml").read_text()
    )


def test_all_metadata_uses_valid_types_unique_instances_and_explicit_mains():
    vocabulary = get_appliance_types()
    for house in range(1, 21):
        building = load_metadata(house)
        assert building["instance"] == house
        assert building["original_name"] == f"hh-{house:02d}"
        assert set(building["elec_meters"]) == set(range(1, 12))
        assert building["elec_meters"][11]["site_meter"]
        assert all(
            building["elec_meters"][m]["submeter_of"] == 11 for m in range(1, 11)
        )
        instances = {}
        for appliance in building["appliances"]:
            assert appliance["type"] in vocabulary
            instances.setdefault(appliance["type"], []).append(appliance["instance"])
            assert set(appliance["meters"]) <= building["elec_meters"].keys()
        for values in instances.values():
            assert sorted(values) == list(range(1, len(values) + 1))


@pytest.mark.parametrize(
    ("house", "meter", "original", "expected"),
    [
        (16, 1, "garage door opener", "unknown"),
        (20, 1, "coffee machine", "coffee maker"),
        (20, 2, "electric kettle", "kettle"),
        (15, 10, "flat iron", "clothes iron"),
        (2, 4, "dryer", "tumble dryer"),
        (3, 2, "bread slicer", "small cooking appliance"),
        (11, 6, "juicer", "small cooking appliance"),
    ],
)
def test_appliance_mappings_match_published_labels(house, meter, original, expected):
    appliance = next(
        a for a in load_metadata(house)["appliances"] if a["meters"] == [meter]
    )
    assert appliance["original_name"] == original
    assert appliance["type"] == expected


def test_twenty_house_conversion_loads_in_nilmtk_from_another_directory(
    tmp_path, monkeypatch
):
    output_format = "HDF"
    source = tmp_path / "raw"
    for house in range(1, 21):
        make_house(source, house)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / ("gelap.h5" if output_format == "HDF" else "gelap")
    converter.convert_gelap(source, output, False, format=output_format)
    with DataSet(str(output), format=output_format) as dataset:
        assert set(dataset.buildings) == set(range(1, 21))
        assert dataset.metadata["number_of_buildings"] == 20
        for house, building in dataset.buildings.items():
            assert building.elec.mains().instance() == 11
            assert (
                building.elec[1].upstream_meter().identifier
                == building.elec[11].identifier
            )
            for meter in range(1, 12):
                frame = next(dataset.store.load(f"/building{house}/elec/meter{meter}"))
                assert frame.index.is_unique and frame.index.is_monotonic_increasing
                assert list(frame.columns) == [("power", "active")]
                assert list(frame.columns.names) == LEVEL_NAMES
                expected = house * 100 + (5 if meter == 11 else meter)
                assert frame.iloc[:, 0].tolist() == [expected, expected]
                assert frame.index[0] == pd.Timestamp(
                    1582098564123, unit="ms", tz="UTC"
                )
                if output_format == "HDF":
                    assert frame.dtypes.iloc[0] == np.float32
                    assert str(frame.index.tz) == "Europe/Berlin"


@pytest.mark.parametrize("prefix", ["./", "hh-01/"])
def test_archive_conversion_writes_no_extracted_files_and_only_subset_metadata(
    tmp_path, prefix
):
    source = tmp_path / "raw"
    source.mkdir()
    archive = source / "hh-01.tar.xz"
    make_archive(archive, prefix=prefix)
    output = tmp_path / "gelap.h5"
    converter.convert_gelap(source, output)
    assert list(source.iterdir()) == [archive]
    with DataSet(str(output)) as dataset:
        assert set(dataset.buildings) == {1}
        assert dataset.metadata["number_of_buildings"] == 1
        assert dataset.store["/building1/elec/meter1"].iloc[0, 0] == 101


def test_phase_sum_needs_all_three_phases_and_ignores_extra_columns():
    frame = converter._read_site_meter_csv(
        io.StringIO(
            "power3,time,unrelated,power1,power2\n"
            "3,1582098564123,999,1,2\n"
            "3,1582098565000,999,1,\n"
            ",1582098566000,999,,\n"
            "0,1582098567000,999,0,0\n"
            "3,1582098568000,999,inf,2\n"
            "3,1582098569000,999,-10,2\n"
        )
    )
    assert frame.iloc[:, 0].tolist() == [6, 0, -5]
    assert frame.index[0] == pd.Timestamp(1582098564123, unit="ms", tz="UTC")


def test_appliances_use_named_reply_time_and_handle_dst_in_utc():
    frame = converter._read_elec_csv(
        io.StringIO(
            "power,time_reply,time_request\n"
            "20,1603589400123,0\n"
            "10,1603585800123,0\n"
            "999,1603585800123,0\n"
        )
    )
    assert frame.iloc[:, 0].tolist() == [10, 20]
    assert frame.index.is_unique
    assert frame.index[1] - frame.index[0] == pd.Timedelta(hours=1)
    assert frame.index[0].microsecond == 123000


@pytest.mark.parametrize("name", ["../escape.csv", "/escape.csv", "..\\escape.csv"])
def test_archive_rejects_unsafe_paths(tmp_path, name):
    source = tmp_path / "raw"
    source.mkdir()
    make_archive(source / "hh-01.tar.xz", extra=tarfile.TarInfo(name))
    with pytest.raises(ValueError, match="Unsafe"):
        converter.convert_gelap(source, tmp_path / "partial.h5")
    assert not (tmp_path / "escape.csv").exists()


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE])
def test_archive_rejects_links_and_special_files(tmp_path, kind):
    source = tmp_path / "raw"
    source.mkdir()
    member = tarfile.TarInfo("link")
    member.type = kind
    member.linkname = "../outside"
    make_archive(source / "hh-01.tar.xz", extra=member)
    with pytest.raises(ValueError, match="Unsupported"):
        converter.convert_gelap(source, tmp_path / "partial.h5")


@pytest.mark.parametrize("case", ["missing", "duplicate"])
def test_archive_must_contain_each_meter_exactly_once(tmp_path, case):
    source = tmp_path / "raw"
    source.mkdir()
    make_archive(
        source / "hh-01.tar.xz",
        omit="label_010.csv" if case == "missing" else None,
        extra=tarfile.TarInfo("./label_001.csv") if case == "duplicate" else None,
    )
    with pytest.raises((ValueError, FileNotFoundError), match="Missing|Duplicate"):
        converter.convert_gelap(source, tmp_path / "partial.h5")


def test_extracted_data_take_precedence_over_archive(tmp_path):
    make_house(tmp_path)
    (tmp_path / "hh-01.tar.xz").write_bytes(b"invalid archive")
    assert converter._find_houses(tmp_path, True)[1] == tmp_path / "hh-01"


@pytest.mark.parametrize(
    "case", ["missing", "empty", "unknown", "incomplete", "archives-disabled"]
)
def test_bad_input_preserves_destination(tmp_path, case):
    source = tmp_path / "raw"
    if case != "missing":
        source.mkdir()
    if case in ("unknown", "incomplete"):
        (source / ("hh-21" if case == "unknown" else "hh-01")).mkdir()
    if case == "archives-disabled":
        make_archive(source / "hh-01.tar.xz")
    output = tmp_path / "keep.h5"
    output.write_bytes(b"existing output")
    with pytest.raises((ValueError, FileNotFoundError)):
        converter.convert_gelap(source, output, uncompress_tars=False)
    assert output.read_bytes() == b"existing output"


def test_output_closes_after_failure(tmp_path, monkeypatch):
    source = tmp_path / "raw"
    directory = make_house(source)
    (directory / "smartmeter.csv").write_text("bad,header\n1,2\n")
    opened = []
    original = converter.get_datastore

    def capture(*args, **kwargs):
        store = original(*args, **kwargs)
        opened.append(store)
        return store

    monkeypatch.setattr(converter, "get_datastore", capture)
    with pytest.raises(ValueError):
        converter.convert_gelap(source, tmp_path / "partial.h5")
    assert len(opened) == 1
    assert not opened[0].store.is_open


def test_unsupported_output_preserves_destination(tmp_path):
    source = tmp_path / "raw"
    make_house(source)
    output = tmp_path / "keep.h5"
    output.write_bytes(b"existing output")
    with pytest.raises(ValueError, match="Only HDF"):
        converter.convert_gelap(source, output, format="CSV")
    assert output.read_bytes() == b"existing output"


@pytest.mark.parametrize(
    "content",
    [
        "time_request,time_reply,power\n",
        "time_request,time_reply,power\n1,not-a-time,2\n",
        "time_request,time_reply,power\n1,1582098564123,\n",
    ],
)
def test_invalid_or_empty_appliance_csv_is_rejected(content):
    with pytest.raises(ValueError):
        converter._read_elec_csv(io.StringIO(content))


def test_public_api_and_cli_convert_a_house(tmp_path):
    from nilmtk import dataset_converters
    from nilmtk.dataset_converters import cli

    assert dataset_converters.convert_gelap is converter.convert_gelap
    source = tmp_path / "raw"
    make_house(source)
    output = tmp_path / "gelap.h5"
    assert cli.main(["gelap", str(source), str(output)]) == 0
    with DataSet(str(output)) as result:
        assert set(result.buildings) == {1}


def test_cli_rejects_unsupported_csv_before_writing(tmp_path, capsys):
    from nilmtk.dataset_converters import cli

    output = tmp_path / "keep.h5"
    output.write_bytes(b"existing output")
    with pytest.raises(SystemExit) as error:
        cli.main(["gelap", str(tmp_path), str(output), "--format", "CSV"])
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
    assert output.read_bytes() == b"existing output"
