#!/usr/bin/env python3
"""Check NILMTK's public onboarding and installation contract."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = (
    ROOT / "README.md",
    ROOT / "docs/manual/user_guide/install_user.md",
    ROOT / "docs/manual/user_guide/install_dev.md",
)

REQUIRED_README_TEXT = (
    "Dataset conversion, meter access, preprocessing, and metrics",
    "Appliance taxonomy, synonyms, meter relationships, and dataset schema",
    "Disaggregation model implementation and testing",
    "Fixed T1/T2/T3 evaluation and published result bundles",
    "https://nilmtk.github.io/",
    "https://github.com/nilmtk/nilm_metadata",
    "https://github.com/nilmtk/nilmtk-contrib",
    "https://github.com/nilmtk/nilmbench",
    "https://nilmtk.github.io/nilmtk/master/index.html",
    "10.1145/2602044.2602051",
    "10.1109/COMPSACW.2014.97",
    "10.1145/3360322.3360844",
    "10.1145/3744256.3812587",
    '"nilmtk[deddiag] @ git+https://github.com/nilmtk/nilmtk.git"',
    (
        "NILMTK core is a Python library and does not publish a separate "
        "official core image."
    ),
)

FORBIDDEN_INSTALL_COMMANDS = (
    re.compile(r"^\s*conda\s+install\s+-c\s+nilmtk\b", re.MULTILINE),
    re.compile(r"^\s*python\s+setup\.py\s+develop\b", re.MULTILINE),
    re.compile(r"^\s*nosetests\s*$", re.MULTILINE),
)

MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def check_relative_links(document: Path, source: str, errors: list[str]) -> int:
    count = 0
    for raw_target in MARKDOWN_LINK.findall(source):
        target = raw_target.split(maxsplit=1)[0].strip("<>")
        parsed = urlsplit(target)
        if (
            parsed.scheme
            or parsed.netloc
            or target.startswith(("#", "mailto:", "tel:"))
        ):
            continue
        path = (document.parent / unquote(parsed.path)).resolve()
        count += 1
        if not path.exists():
            errors.append(
                f"{document.relative_to(ROOT)} has a missing local link: {target}"
            )
    return count


def main() -> int:
    errors: list[str] = []
    checked_links = 0

    for document in DOCUMENTS:
        source = document.read_text(encoding="utf-8")
        if source.count("```") % 2:
            errors.append(f"{document.relative_to(ROOT)} has an unclosed code fence")
        if "http://" in source:
            errors.append(f"{document.relative_to(ROOT)} contains an insecure URL")
        if "ghcr.io/enfuego27826" in source:
            errors.append(f"{document.relative_to(ROOT)} advertises a personal image")
        for pattern in FORBIDDEN_INSTALL_COMMANDS:
            if pattern.search(source):
                errors.append(
                    f"{document.relative_to(ROOT)} contains retired install command: "
                    f"{pattern.pattern}"
                )
        checked_links += check_relative_links(document, source, errors)

    readme = DOCUMENTS[0].read_text(encoding="utf-8")
    normalized_readme = " ".join(readme.split())
    for required in REQUIRED_README_TEXT:
        if required not in normalized_readme:
            errors.append(f"README.md is missing required contract text: {required}")

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    if 'requires-python = ">=3.11"' not in pyproject:
        errors.append("pyproject.toml no longer matches the documented Python floor")
    if 'nilmtk-convert = "nilmtk.dataset_converters.cli:main"' not in pyproject:
        errors.append(
            "pyproject.toml no longer defines the documented converter command"
        )
    if not (ROOT / "nilmtk/dataset_converters/cli.py").is_file():
        errors.append("the documented converter entry point module does not exist")

    if errors:
        print("Documentation checks failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(
        f"Documentation checks passed: {len(DOCUMENTS)} documents, "
        f"{checked_links} local links, {len(REQUIRED_README_TEXT)} contract clauses."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
