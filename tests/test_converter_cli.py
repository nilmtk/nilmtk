from types import SimpleNamespace

import pytest

from nilmtk.dataset_converters import cli


def test_list_is_explicit_about_cli_and_programmatic_converters(capsys):
    assert cli.main(["list"]) == 0

    output = capsys.readouterr().out
    assert "Command-line converters:" in output
    assert "redd" in output
    assert "ukdale" in output
    assert "Programmatic converters:" in output
    assert "deddiag" in output
    assert "live database connection" in output


def test_help_and_version_exit_successfully(capsys):
    with pytest.raises(SystemExit) as help_exit:
        cli.main(["--help"])
    assert help_exit.value.code == 0
    assert "nilmtk-convert list" in capsys.readouterr().out

    with pytest.raises(SystemExit) as version_exit:
        cli.main(["--version"])
    assert version_exit.value.code == 0
    assert "nilmtk-convert" in capsys.readouterr().out


def test_missing_input_is_rejected_before_import(tmp_path, monkeypatch, capsys):
    imported = False

    def unexpected_import(_name):
        nonlocal imported
        imported = True

    monkeypatch.setattr(cli.importlib, "import_module", unexpected_import)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["redd", str(tmp_path / "missing"), str(tmp_path / "out.h5")])

    assert exit_info.value.code == 2
    assert not imported
    assert "input path does not exist" in capsys.readouterr().err


def test_existing_output_requires_force(tmp_path, monkeypatch, capsys):
    source = tmp_path / "raw"
    source.mkdir()
    output = tmp_path / "redd.h5"
    output.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(
        cli.importlib,
        "import_module",
        lambda _name: pytest.fail("converter imported before overwrite validation"),
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["redd", str(source), str(output)])

    assert exit_info.value.code == 2
    assert output.read_text(encoding="utf-8") == "keep"
    assert "--force" in capsys.readouterr().err


def test_dispatch_is_lazy_and_forwards_normalized_arguments(tmp_path, monkeypatch):
    source = tmp_path / "raw"
    source.mkdir()
    output = tmp_path / "nested" / "redd.csv"
    calls = []

    def convert_redd(input_path, output_path, format):
        calls.append((input_path, output_path, format))

    imported = []

    def fake_import(name):
        imported.append(name)
        return SimpleNamespace(convert_redd=convert_redd)

    monkeypatch.setattr(cli.importlib, "import_module", fake_import)

    assert cli.main(["redd", str(source), str(output), "--format", "CSV"]) == 0
    assert imported == ["nilmtk.dataset_converters.redd.convert_redd"]
    assert calls == [(str(source), str(output), "CSV")]
    assert output.parent.is_dir()


@pytest.mark.parametrize("name", sorted(cli.CONVERTERS))
def test_registry_points_to_a_specific_converter_function(name):
    spec = cli.CONVERTERS[name]
    assert spec.module.startswith(f"nilmtk.dataset_converters.{name}.")
    assert spec.function == f"convert_{name}"


def test_input_and_output_cannot_be_the_same_path(tmp_path, capsys):
    path = tmp_path / "dataset"
    path.mkdir()

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["redd", str(path), str(path), "--force"])

    assert exit_info.value.code == 2
    assert "must be different" in capsys.readouterr().err
