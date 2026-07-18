"""Command-line dispatcher for NILMTK's path-based dataset converters."""

from __future__ import annotations

import argparse
import importlib
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from nilmtk.version import version


@dataclass(frozen=True)
class ConverterSpec:
    module: str
    function: str
    description: str


CONVERTERS = {
    "ampds": ConverterSpec(
        "nilmtk.dataset_converters.ampds.convert_ampds",
        "convert_ampds",
        "AMPds",
    ),
    "combed": ConverterSpec(
        "nilmtk.dataset_converters.combed.convert_combed",
        "convert_combed",
        "COMBED",
    ),
    "dred": ConverterSpec(
        "nilmtk.dataset_converters.dred.convert_dred",
        "convert_dred",
        "DRED",
    ),
    "hes": ConverterSpec(
        "nilmtk.dataset_converters.hes.convert_hes",
        "convert_hes",
        "HES",
    ),
    "hipe": ConverterSpec(
        "nilmtk.dataset_converters.hipe.convert_hipe",
        "convert_hipe",
        "HIPE",
    ),
    "iawe": ConverterSpec(
        "nilmtk.dataset_converters.iawe.convert_iawe",
        "convert_iawe",
        "iAWE",
    ),
    "ideal": ConverterSpec(
        "nilmtk.dataset_converters.ideal.convert_ideal",
        "convert_ideal",
        "IDEAL",
    ),
    "redd": ConverterSpec(
        "nilmtk.dataset_converters.redd.convert_redd",
        "convert_redd",
        "REDD",
    ),
    "refit": ConverterSpec(
        "nilmtk.dataset_converters.refit.convert_refit",
        "convert_refit",
        "REFIT",
    ),
    "smart": ConverterSpec(
        "nilmtk.dataset_converters.smart.convert_smart",
        "convert_smart",
        "SMART",
    ),
    "ukdale": ConverterSpec(
        "nilmtk.dataset_converters.ukdale.convert_ukdale",
        "convert_ukdale",
        "UK-DALE",
    ),
}

PROGRAMMATIC_ONLY = {
    "caxe": "writes a fixed output and requires dataset-specific metadata layout",
    "dataport": "requires one or more CSV files plus a metadata directory",
    "deddiag": "requires a live database connection",
    "eco": "requires an explicit dataset timezone",
    "greend": "has dataset-specific multiprocessing controls",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nilmtk-convert",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Convert a supported raw energy dataset into a NILMTK datastore. "
            "Use 'nilmtk-convert list' to see CLI and programmatic converters."
        ),
        epilog=(
            "Start here:\n"
            "  nilmtk-convert list\n"
            "  nilmtk-convert redd /path/to/low_freq /path/to/redd.h5"
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {version}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="list available dataset converters")
    for name, spec in CONVERTERS.items():
        command = subparsers.add_parser(
            name,
            help=f"convert {spec.description} data",
            description=f"Convert {spec.description} data to a NILMTK datastore.",
        )
        command.add_argument(
            "input_path", type=Path, help="raw dataset file or directory"
        )
        command.add_argument(
            "output_path", type=Path, help="destination datastore path"
        )
        command.add_argument(
            "--format",
            choices=("HDF", "CSV"),
            default="HDF",
            help="destination datastore format (default: HDF)",
        )
        command.add_argument(
            "--force",
            action="store_true",
            help="allow replacement of an existing destination",
        )
    return parser


def _print_converter_list() -> None:
    print("Command-line converters:")
    for name, spec in CONVERTERS.items():
        print(f"  {name:<10} {spec.description}")
    print("\nProgrammatic converters:")
    for name, reason in PROGRAMMATIC_ONLY.items():
        print(f"  {name:<10} {reason}")


def _validate_paths(
    parser: argparse.ArgumentParser,
    input_path: Path,
    output_path: Path,
    output_format: str,
    force: bool,
) -> None:
    if not input_path.exists():
        parser.error(f"input path does not exist: {input_path}")

    resolved_input = input_path.resolve()
    resolved_output = output_path.resolve()
    if resolved_input == resolved_output:
        parser.error("input and output paths must be different")
    if input_path.is_dir() and resolved_output.is_relative_to(resolved_input):
        parser.error("output path must not be inside the input directory")
    if output_path.is_dir() and resolved_input.is_relative_to(resolved_output):
        parser.error("input path must not be inside the output directory")

    if not output_path.exists():
        return
    protected_paths = {
        Path(resolved_output.anchor),
        Path.cwd().resolve(),
        Path.home().resolve(),
        Path(tempfile.gettempdir()).resolve(),
    }
    if force and resolved_output in protected_paths:
        parser.error(f"refusing to replace protected path: {output_path}")
    if output_format == "HDF" and output_path.is_dir():
        parser.error("HDF output path must be a file, not a directory")
    if output_format == "CSV" and not output_path.is_dir():
        parser.error("CSV output path must be a directory, not a file")
    if not force:
        parser.error(
            f"output already exists (pass --force to replace it): {output_path}"
        )


def _remove_existing_output(output_path: Path) -> None:
    if output_path.is_symlink() or output_path.is_file():
        output_path.unlink()
    elif output_path.is_dir():
        shutil.rmtree(output_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        _print_converter_list()
        return 0

    input_path = args.input_path.expanduser()
    output_path = args.output_path.expanduser()
    _validate_paths(parser, input_path, output_path, args.format, args.force)

    spec = CONVERTERS[args.command]
    try:
        module = importlib.import_module(spec.module)
    except ImportError as error:
        parser.exit(1, f"nilmtk-convert: could not load {args.command}: {error}\n")
    converter = getattr(module, spec.function)

    if args.force:
        try:
            _remove_existing_output(output_path)
        except OSError as error:
            parser.exit(1, f"nilmtk-convert: could not replace output: {error}\n")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    converter(str(input_path), str(output_path), format=args.format)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
