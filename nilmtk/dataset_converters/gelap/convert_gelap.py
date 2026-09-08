"""Convert the German Labeled Dataset for Power Consumption (GeLaP)."""

import io
import re
import tarfile
from pathlib import Path, PurePosixPath
from shutil import copyfile
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
from nilm_metadata import save_yaml_to_datastore

from nilmtk.datastore import Key
from nilmtk.measurement import LEVEL_NAMES
from nilmtk.utils import get_datastore

METADATA_PATH = Path(__file__).resolve().parent / "metadata"
NUM_HOUSES = 20
NUM_ELECS_PER_HOUSE = 10
SITE_METER_NUMBER = 11
TIMEZONE = "Europe/Berlin"
METER_FILES = {
    "smartmeter.csv": SITE_METER_NUMBER,
    **{f"label_{meter:03}.csv": meter for meter in range(1, NUM_ELECS_PER_HOUSE + 1)},
}


def convert_gelap(gelap_path, output_filename, uncompress_tars=True, format="HDF"):
    """Convert downloaded GeLaP households to a NILMTK datastore.

    Parameters
    ----------
    gelap_path : str or pathlib.Path
        Root directory containing ``hh-01.tar.xz`` archives and/or extracted
        ``hh-01`` directories. Any subset of the 20 households is supported.
    output_filename : str or pathlib.Path
        Destination HDF5 file.
    uncompress_tars : bool, default True
        Allow reading compressed household archives. The original positional
        flag is retained, but archives are now read directly without extracting
        files to disk. Set False to require already extracted household folders.
        When both representations exist, the extracted folder takes precedence.
    format : {'HDF'}, default 'HDF'
        Output datastore format. Only HDF5 is supported.

    Notes
    -----
    Appliance meters 1-10 use ``time_reply`` in Unix milliseconds. Meter 11 is
    the sum of all three active-power phases, in watts, at the original sample
    times. A row missing any phase is excluded, never treated as zero demand.
    Timestamps are sorted, converted from UTC to Europe/Berlin, and deduplicated
    by retaining the first valid reading. Data are not resampled or interpolated.
    Each CSV is loaded separately. A failed conversion closes the datastore but
    may leave a partial output, which must not be used as a complete dataset.
    """
    sources = _find_houses(Path(gelap_path).expanduser(), uncompress_tars)
    output_path = Path(output_filename).expanduser()
    for source in sources.values():
        paths = [source / name for name in METER_FILES] if source.is_dir() else [source]
        if any(path.resolve() == output_path.resolve() for path in paths):
            raise ValueError("Output must not replace a source file")
    if format != "HDF":
        raise ValueError("Only HDF output is supported for this dataset")

    store = get_datastore(str(output_path), format, mode="w")
    try:
        for house, source in sources.items():
            print(f"Converting GeLaP household {house}: {source.name}")
            if source.is_dir():
                for name, meter in METER_FILES.items():
                    _write_meter(source / name, house, meter, store)
            else:
                _convert_archive(source, house, store)
        _save_metadata(store, sources)
    finally:
        store.close()
    print(f"Done converting GeLaP to {format}!")


def _find_houses(input_path, allow_archives):
    if not input_path.is_dir():
        raise FileNotFoundError(f"GeLaP directory does not exist: {input_path}")
    sources = {}
    for path in sorted(input_path.iterdir()):
        match = re.fullmatch(r"hh-([0-9]{2})(\.tar\.xz)?", path.name)
        if not match:
            continue
        is_directory = match[2] is None and path.is_dir()
        is_archive = match[2] is not None and path.is_file() and allow_archives
        if not (is_directory or is_archive):
            continue
        house = int(match[1])
        if not 1 <= house <= NUM_HOUSES:
            raise ValueError(f"No GeLaP metadata for household {house}")
        if is_directory or house not in sources:
            sources[house] = path
    if not sources:
        raise ValueError(f"No GeLaP households found in {input_path}")
    for source in sources.values():
        if source.is_dir():
            for name in METER_FILES:
                if not (source / name).is_file():
                    raise FileNotFoundError(
                        f"Missing GeLaP meter file: {source / name}"
                    )
    return dict(sorted(sources.items()))


class _ArchiveReader(io.RawIOBase):
    """Expose tar stream members as non-seekable binary input to pandas."""

    def __init__(self, member):
        self.member = member

    def readable(self):
        return True

    def readinto(self, buffer):
        data = self.member.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def _convert_archive(source, house, store):
    seen = set()
    # Sequential decompression avoids repeated seeks through multi-GB tar files.
    # Never extract archive paths, links, or devices onto the user's filesystem.
    with tarfile.open(source, mode="r|xz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or "\\" in member.name:
                raise ValueError(f"Unsafe GeLaP archive member: {member.name}")
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"Unsupported GeLaP archive member: {member.name}")
            parts = path.parts
            if parts and parts[0] == f"hh-{house:02}":
                parts = parts[1:]
            if len(parts) != 1 or parts[0] not in METER_FILES:
                continue
            name = parts[0]
            if name in seen:
                raise ValueError(f"Duplicate GeLaP archive member: {member.name}")
            seen.add(name)
            with (
                archive.extractfile(member) as handle,
                io.BufferedReader(_ArchiveReader(handle)) as reader,
            ):
                _write_meter(reader, house, METER_FILES[name], store)
    missing = set(METER_FILES) - seen
    if missing:
        raise FileNotFoundError(f"Missing meter files in {source}: {sorted(missing)}")


def _write_meter(source, house, meter, store):
    if meter == SITE_METER_NUMBER:
        frame = _read_site_meter_csv(source)
    else:
        frame = _read_elec_csv(source)
    store.put(str(Key(building=house, meter=meter)), frame)


def _prepare_df(frame):
    timestamps = pd.to_datetime(frame.index, unit="ms", utc=True)
    frame = frame.astype(np.float32)
    frame.index = pd.DatetimeIndex(timestamps, name="timestamp").tz_convert(TIMEZONE)
    frame = frame.loc[frame.index.notna()]
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    frame = frame.loc[~frame.index.duplicated(keep="first")].sort_index()
    if frame.empty:
        raise ValueError("No valid GeLaP readings in meter CSV")
    frame.columns = pd.MultiIndex.from_tuples([("power", "active")], names=LEVEL_NAMES)
    return frame


def _read_site_meter_csv(source):
    frame = pd.read_csv(
        source, usecols=["time", "power1", "power2", "power3"], index_col="time"
    )
    frame = frame[["power1", "power2", "power3"]].astype(np.float64)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return _prepare_df(frame.sum(axis=1, min_count=3).to_frame())


def _read_elec_csv(source):
    frame = pd.read_csv(source, usecols=["time_reply", "power"], index_col="time_reply")
    return _prepare_df(frame)


def _save_metadata(store, houses):
    with TemporaryDirectory(prefix="nilmtk-gelap-") as directory:
        target = Path(directory)
        with (METADATA_PATH / "dataset.yaml").open(encoding="utf-8") as handle:
            dataset = yaml.safe_load(handle)
        dataset["number_of_buildings"] = len(houses)
        (target / "dataset.yaml").write_text(
            yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8"
        )
        copyfile(METADATA_PATH / "meter_devices.yaml", target / "meter_devices.yaml")
        for house in houses:
            name = f"building{house}.yaml"
            copyfile(METADATA_PATH / name, target / name)
        save_yaml_to_datastore(str(target), store)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("--format", choices=("HDF",), default="HDF")
    parser.add_argument("--extracted-only", action="store_true")
    args = parser.parse_args()
    convert_gelap(
        args.input_path,
        args.output_path,
        uncompress_tars=not args.extracted_only,
        format=args.format,
    )
