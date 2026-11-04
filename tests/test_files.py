from pathlib import Path

import pytest

from agatha.files import write_atomic


def test_write_creates_the_directory_and_leaves_no_partial(tmp_path):
    path = tmp_path / "not" / "yet" / "report.json"
    write_atomic(path, b"new")
    assert path.read_bytes() == b"new"
    assert [item.name for item in path.parent.iterdir()] == ["report.json"]


def test_a_write_killed_before_the_swap_keeps_the_old_file_whole(tmp_path, monkeypatch):
    path = tmp_path / "report.json"
    write_atomic(path, b"old")

    def killed(self, target):
        raise OSError("job killed")

    monkeypatch.setattr(Path, "replace", killed)
    with pytest.raises(OSError):
        write_atomic(path, b"new")
    assert path.read_bytes() == b"old"


def test_files_differing_only_in_suffix_do_not_share_a_partial(tmp_path):
    """Labels hold a colon and windows a double underscore, so the partial's name is the path, not a fixed one."""
    write_atomic(tmp_path / "sha256:0123456789ab.json", b"a")
    write_atomic(tmp_path / "1__clean.jpg", b"b")
    write_atomic(tmp_path / "1__pass1.jpg", b"c")
    assert sorted(path.read_bytes() for path in tmp_path.iterdir()) == [b"a", b"b", b"c"]
