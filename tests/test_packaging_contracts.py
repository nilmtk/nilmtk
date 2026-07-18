import tomllib
from importlib.metadata import version as installed_version
from pathlib import Path

import nilmtk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NILM_METADATA_REQUIREMENT = (
    "nilm_metadata @ git+https://github.com/nilmtk/nilm_metadata.git"
    "@59c9990de4836d77c0dcd807bd4293e39e0cc314"
)


def test_nilm_metadata_is_pinned_to_an_immutable_commit():
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
        dependencies = tomllib.load(handle)["project"]["dependencies"]

    matching = [
        requirement
        for requirement in dependencies
        if requirement.partition(" @ ")[0].replace("_", "-") == "nilm-metadata"
    ]
    assert matching == [NILM_METADATA_REQUIREMENT]


def test_runtime_version_matches_package_metadata():
    assert nilmtk.__version__ == installed_version("nilmtk")
