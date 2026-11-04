import subprocess
import sys
from pathlib import Path

import pytest

import agatha
from agatha.columns import CAMERA_IDS
from agatha.config import load_config

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "stage1.toml"
FILLED = {"root": "/data", "outputs": "/outputs"}
VERSION_SCRIPT = (
    "import sys; from agatha.config import load_config; "
    "print(load_config(sys.argv[1], root='/data', outputs='/outputs').preprocessing_version)"
)


def version_of_a_copy(tmp_path: Path, name: str, edit: bytes = b"") -> str:
    """preprocessing_version from a copy of the package in another directory, with `edit` added to one source file."""
    copy = tmp_path / name / "agatha"
    copy.mkdir(parents=True)
    for source in Path(agatha.__file__).parent.glob("*.py"):
        (copy / source.name).write_bytes(source.read_bytes())
    (copy / "render.py").write_bytes((copy / "render.py").read_bytes() + edit)
    result = subprocess.run([sys.executable, "-c", VERSION_SCRIPT, str(CONFIG_PATH)], capture_output=True, text=True,
                            check=True, env={"PYTHONPATH": str(copy.parent)})
    return result.stdout.strip()


def test_the_version_follows_the_source_even_with_no_commit(tmp_path):
    """sync.sh copies the working tree, so an uncommitted edit on PACE changes results and must change this."""
    same = version_of_a_copy(tmp_path, "one"), version_of_a_copy(tmp_path, "two")
    assert same[0] == same[1]
    assert version_of_a_copy(tmp_path, "edited", b"# an edit\n") != same[0]


def test_the_version_does_not_read_any_repository_near_the_config(tmp_path):
    """The commit of whatever repository holds the config isn't the code's."""
    config = tmp_path / "stage1.toml"
    config.write_bytes(CONFIG_PATH.read_bytes())
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    version = load_config(config, **FILLED).preprocessing_version
    assert version == load_config(CONFIG_PATH, **FILLED).preprocessing_version


def test_the_version_follows_the_config_file(tmp_path):
    config = tmp_path / "stage1.toml"
    config.write_text(CONFIG_PATH.read_text() + "\n# an edit\n")
    edited, original = load_config(config, **FILLED), load_config(CONFIG_PATH, **FILLED)
    assert edited.preprocessing_version != original.preprocessing_version


def test_cameras_override_the_file_and_the_version():
    front = load_config(CONFIG_PATH, **FILLED)
    both = load_config(CONFIG_PATH, **FILLED, cameras=["FRONT", "FRONT_LEFT"])
    assert front.cameras == (CAMERA_IDS["FRONT"],)
    assert both.cameras == (CAMERA_IDS["FRONT"], CAMERA_IDS["FRONT_LEFT"])
    assert both.preprocessing_version != front.preprocessing_version


def test_an_unset_outputs_variable_fails_rather_than_writing_under_the_working_directory(monkeypatch):
    """expandvars leaves it as literal text, and the working directory on PACE is the repo in home."""
    monkeypatch.delenv("WOD_OUTPUTS", raising=False)
    with pytest.raises(ValueError):
        load_config(CONFIG_PATH, root="/data")


def test_outputs_expand_from_the_environment(monkeypatch):
    monkeypatch.setenv("WOD_OUTPUTS", "/scratch/outputs")
    assert load_config(CONFIG_PATH, root="/data").outputs == Path("/scratch/outputs")


def test_an_unset_root_is_left_for_the_schema_check_to_report(monkeypatch):
    """Nothing gets written there, and a run that says its files are missing leaves a report."""
    monkeypatch.delenv("WOD_ROOT", raising=False)
    assert str(load_config(CONFIG_PATH, outputs="/outputs").root) == "$WOD_ROOT"
