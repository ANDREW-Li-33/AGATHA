from pathlib import Path

import pytest

from agatha.config import load_config

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "stage1.toml"


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory):
    """(root, ref) of the default synthetic segment, written once per session."""
    # Imported here so that a broken pipeline module can't stop every other test collecting.
    from agatha.synthetic import write_synthetic_segment

    root = tmp_path_factory.mktemp("wod")
    return root, write_synthetic_segment(root)


@pytest.fixture
def stage1_config(synthetic_root, tmp_path):
    """configs/stage1.toml, reading the synthetic segment and writing to a fresh directory."""
    root, _ = synthetic_root
    return load_config(CONFIG_PATH, root=root, outputs=tmp_path / "outputs")
