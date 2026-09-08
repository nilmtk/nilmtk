"""Convert MORED whole-premises and individual-load ground-truth data."""

import re
from pathlib import Path
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
TIMEZONE = "Africa/Casablanca"


def convert_mored(mored_path, output_filename, format="HDF", *, source_timezone="UTC"):
    """Convert downloaded MORED WPILGT premises to a NILMTK datastore.

    Parameters
    ----------
    mored_path : str or pathlib.Path
        Directory containing one or more complete ``Premises_<number>`` folders.
    output_filename : str or pathlib.Path
        Destination HDF5 file.
    format : {'HDF'}, default 'HDF'
        Output datastore format. Only HDF5 is supported.
    source_timezone : str, default 'UTC'
        Timezone of the naive source timestamps. UTC preserves the original
        converter's interpretation; the published CSVs contain no UTC offset.
        Set this explicitly if the export uses local wall time. Output timestamps
        always use Africa/Casablanca. Ambiguous/nonexistent local times raise.

    Notes
    -----
    Readings remain in volts and watts, without resampling or interpolation.
    Duplicate timestamps keep the first valid reading in source-file order.
    Premises 9 has appliance readings but no measured mains channel.
    Each CSV is loaded separately. A failed conversion closes the datastore but
    may leave a partial output, which must not be used as a complete dataset.
    """
    input_path = Path(mored_path).expanduser()
    premises = _list_premises(input_path)
    locations = _load_data_location(METADATA_PATH)
    unknown = set(premises) - set(locations)
    if unknown:
        raise ValueError(f"No MORED metadata for premises {sorted(unknown)}")

    # Validate all sources before opening (and potentially replacing) output.
    sources = []
    output_path = Path(output_filename).expanduser()
    for house in premises:
        for meter, relative_path in locations[house].items():
            relative_path = Path(relative_path)
            if relative_path.parts != (premises[house], relative_path.name):
                raise ValueError(
                    f"Invalid data location for premises {house}: {relative_path}"
                )
            csv_path = input_path / relative_path
            if not csv_path.is_file():
                raise FileNotFoundError(f"Missing MORED meter file: {csv_path}")
            if csv_path.resolve() == output_path.resolve():
                raise ValueError("Output must not replace a source CSV")
            sources.append((house, meter, csv_path))
    pd.Timestamp("2020-01-01").tz_localize(source_timezone)
    if format != "HDF":
        raise ValueError("Only HDF output is supported for this dataset")

    store = get_datastore(str(output_path), format, mode="w")
    try:
        for house, meter, csv_path in sources:
            print(f"Converting MORED premises {house}, meter {meter}: {csv_path.name}")
            frame = _read_meter_csv(csv_path, source_timezone=source_timezone)
            store.put(str(Key(building=house, meter=meter)), frame)
        _save_metadata(store, premises, source_timezone)
    finally:
        store.close()
    print(f"Done converting MORED to {format}!")


def _list_premises(input_path):
    if not input_path.is_dir():
        raise FileNotFoundError(f"MORED directory does not exist: {input_path}")
    premises = {}
    for path in sorted(input_path.iterdir()):
        match = re.fullmatch(r"Premises_([1-9][0-9]*)", path.name)
        if path.is_dir() and match:
            premises[int(match[1])] = path.name
    if not premises:
        raise ValueError(f"No Premises_<number> directories found in {input_path}")
    return dict(sorted(premises.items()))


def _load_data_location(metadata_path):
    locations = {}
    for path in Path(metadata_path).glob("building*.yaml"):
        with path.open(encoding="utf-8") as handle:
            building = yaml.safe_load(handle)
        locations[building["instance"]] = {
            meter: values["data_location"]
            for meter, values in sorted(building["elec_meters"].items())
        }
    return dict(sorted(locations.items()))


def _read_meter_csv(csv_path, *, source_timezone="UTC"):
    frame = pd.read_csv(csv_path, usecols=["timestamp", "Vrms", "real_power"])
    timestamps = frame.pop("timestamp")
    parsed = pd.to_datetime(timestamps, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    # Premises_10/TV.csv uses English month names and fractional seconds.
    for date_format in ("%d-%b-%Y %H:%M:%S.%f", "%d-%b-%Y %H:%M:%S"):
        missing = parsed.isna() & timestamps.notna()
        parsed.loc[missing] = pd.to_datetime(
            timestamps.loc[missing], format=date_format, errors="coerce"
        )
    invalid = parsed.isna() & timestamps.notna()
    if invalid.any():
        raise ValueError(
            f"Invalid MORED timestamp in {csv_path}: {timestamps[invalid].iloc[0]!r}"
        )

    # Select by name: source column order must not swap watts and volts.
    frame = frame[["Vrms", "real_power"]].astype(np.float32)
    frame.index = pd.DatetimeIndex(parsed, name="timestamp")
    frame = frame.loc[frame.index.notna()]
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    frame = frame.loc[~frame.index.duplicated(keep="first")].sort_index()
    if frame.empty:
        raise ValueError(f"No valid MORED readings in {csv_path}")
    frame.index = frame.index.tz_localize(source_timezone).tz_convert(TIMEZONE)
    frame.columns = pd.MultiIndex.from_tuples(
        [("voltage", ""), ("power", "active")], names=LEVEL_NAMES
    )
    return frame


def _save_metadata(store, premises, source_timezone):
    # The public helper loads every YAML file: stage only converted buildings.
    with TemporaryDirectory(prefix="nilmtk-mored-") as directory:
        target = Path(directory)
        with (METADATA_PATH / "dataset.yaml").open(encoding="utf-8") as handle:
            dataset = yaml.safe_load(handle)
        dataset["number_of_buildings"] = len(premises)
        dataset["description"] += (
            f" The converter interpreted source CSV timestamps in {source_timezone}"
            f" and represents them in {TIMEZONE}."
        )
        (target / "dataset.yaml").write_text(
            yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8"
        )
        copyfile(METADATA_PATH / "meter_devices.yaml", target / "meter_devices.yaml")
        for house in premises:
            name = f"building{house}.yaml"
            copyfile(METADATA_PATH / name, target / name)
        save_yaml_to_datastore(str(target), store)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("--format", choices=("HDF",), default="HDF")
    parser.add_argument("--source-timezone", default="UTC")
    args = parser.parse_args()
    convert_mored(
        args.input_path,
        args.output_path,
        args.format,
        source_timezone=args.source_timezone,
    )
