import json
import re
import subprocess
import sys
import textwrap
import traceback
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest

from agatha.columns import ROLLING_SHUTTER
from agatha.config import load_config
from agatha.report import PUBLIC_SEGMENT, Report, context_label, guarded

# Fake, in the real shapes: 22 base64url characters, and <id>_<start>_<end>.
OBJECT_ID = "xK9-qLm2_ZtR7vB4nWc8Yd"
LOWERCASE_ID = "xk9qlm2ztr7vb4nwc8yd0a"
CONTEXT_NAME = "12345678901234567890_1000_000_1020_000"
VERSION = "0.1.0+nogit.0123456789ab"
SECRETS = (OBJECT_ID, CONTEXT_NAME, "12345678901234567890")


def make_report(context_name: str = CONTEXT_NAME) -> Report:
    return Report(context_name, VERSION)


def assert_clean(text: str) -> None:
    for secret in SECRETS:
        assert secret not in text


def forbid_constants(token: str):
    raise AssertionError(f"not valid JSON: {token}")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(), parse_constant=forbid_constants)


@pytest.mark.parametrize("value", [
    OBJECT_ID,
    CONTEXT_NAME,
    "12.5",                     # a string-typed number
    "0",
    "nan",
    "",
    " pass",                    # vocabulary only as an exact match
    "pass\n",
    "PASS",
    "FRONT",                    # real, but nothing emits it, so it's not in the vocabulary
    "vehicle",
    "skipped",
    "sha256:0123456789ab",
    b"pass",
    bytearray(b"pass"),
    (1, 2),                     # a namedtuple of IDs would look like this
    {1, 2},
    Path("/data/segment.parquet"),
    np.array([1.0, 2.0]),
    object(),
    1 + 2j,
])
def test_add_rejects_anything_but_numbers_and_vocabulary(value):
    report = make_report()
    with pytest.raises(ValueError):
        report.add("section", metric=value)
    assert report.sections == {}


@pytest.mark.parametrize("nest", [
    lambda v: [v],
    lambda v: [1, [2.0, [None, v]]],
    lambda v: {"inner": v},
    lambda v: {"inner": {"deeper": [{"deepest": v}]}},
    lambda v: [{"inner": [v]}],
    lambda v: OrderedDict(inner=v),
])
@pytest.mark.parametrize("value", [OBJECT_ID, CONTEXT_NAME, "12.5"])
def test_add_finds_strings_at_any_depth(nest, value):
    report = make_report()
    with pytest.raises(ValueError):
        report.add("section", metric=nest(value))
    assert report.sections == {}


BAD_NAMES = [
    OBJECT_ID,
    CONTEXT_NAME,
    "12345678901234567890",
    "FRONT",
    "Front",
    "camelCase",
    "UPPER_CASE",
    "1st",
    "_private",
    "has space",
    "has-dash",
    "dotted.name",
    "name\n",                   # re.match with $ would let this through
    "",
    "naïve",
    1,
    1.5,
    None,
    True,
    ("a", "b"),
]


@pytest.mark.parametrize("key", BAD_NAMES)
def test_add_rejects_bad_dict_keys(key):
    report = make_report()
    with pytest.raises(ValueError):
        report.add("section", metric={key: 1})
    with pytest.raises(ValueError):
        report.add("section", metric=[{"inner": {key: 1}}])
    assert report.sections == {}


@pytest.mark.parametrize("key", [k for k in BAD_NAMES if isinstance(k, str)])
def test_add_rejects_bad_metric_names(key):
    report = make_report()
    with pytest.raises(ValueError):
        report.add("section", **{key: 1})
    assert report.sections == {}


@pytest.mark.parametrize("section", BAD_NAMES)
def test_add_rejects_bad_section_names(section):
    report = make_report()
    with pytest.raises(ValueError):
        report.add(section, metric=1)
    assert report.sections == {}


def test_rejections_do_not_echo_the_value():
    # A ValueError that repeats its input would carry the value straight into a Slurm log.
    report = make_report()
    calls = [
        lambda: report.add("section", metric=OBJECT_ID),
        lambda: report.add("section", metric=[{"inner": CONTEXT_NAME}]),
        lambda: report.add("section", metric={OBJECT_ID: 1}),
        lambda: report.add(OBJECT_ID, metric=1),
        lambda: report.add("section", metric=Path(CONTEXT_NAME)),
        lambda: report.fail(CONTEXT_NAME, RuntimeError()),
        lambda: Report(CONTEXT_NAME, CONTEXT_NAME),
    ]
    for call in calls:
        with pytest.raises(ValueError) as info:
            call()
        assert_clean("".join(traceback.format_exception(info.value)))


def test_a_rejected_call_keeps_nothing():
    report = make_report()
    report.add("section", kept=1)
    with pytest.raises(ValueError):
        report.add("section", first_is_fine=2, second_is_not=OBJECT_ID)
    with pytest.raises(ValueError):
        report.add("other", metric=OBJECT_ID)
    assert report.sections == {"section": {"kept": 1}}


def test_vocabulary_passes_at_any_depth():
    words = [*ROLLING_SHUTTER.values(), "pass", "fail", "no_data"]
    report = make_report()
    report.add("section", words=words, nested={"status": "pass", "list": [{"direction": "LEFT_TO_RIGHT"}]})
    assert report.sections["section"]["words"] == words
    assert report.sections["section"]["nested"]["list"][0]["direction"] == "LEFT_TO_RIGHT"


@pytest.mark.parametrize("number", [
    -1550000001900.047,        # a pose_timestamp of 0 turned one row's timestamp into an offset in ms
    1_550_000_000_000_000,     # a raw frame timestamp
    np.int64(1_550_000_000_000_000),
    1e12,
])
def test_a_number_the_size_of_a_raw_timestamp_is_refused(number):
    report = make_report()
    with pytest.raises(ValueError) as error:
        report.add("section", metric={"min": number})
    assert report.sections == {}
    assert str(number)[:8] not in str(error.value)


def test_the_largest_aggregates_still_fit():
    report = make_report()
    report.add("section", most_rows=999_999_999, widest_ms=9.9e11)
    assert report.sections["section"] == {"most_rows": 999_999_999, "widest_ms": 9.9e11}


def test_a_list_the_length_of_a_column_is_refused():
    """Numbers are what section 3 bans, and only the shape of a list can tell a column from an aggregate."""
    report = make_report()
    report.add("section", per_window=[1.0] * 32)
    with pytest.raises(ValueError):
        report.add("section", leaked=list(range(33)))
    with pytest.raises(ValueError):
        report.add("section", nested={"leaked": [[0.0] * 16] * 40})
    assert set(report.sections["section"]) == {"per_window"}


@pytest.mark.parametrize("value, expected", [
    (True, True),
    (False, False),
    (0, 0),
    (-3, -3),
    (2.5, 2.5),
    (None, None),
    (float("nan"), None),
    (float("inf"), None),
    (float("-inf"), None),
    (np.float64(2.5), 2.5),
    (np.float32(0.5), 0.5),
    (np.int64(3), 3),
    (np.int8(-3), -3),
    (np.bool_(True), True),
    (np.float64("nan"), None),
    (np.float32("inf"), None),
    (np.str_("pass"), "pass"),
])
def test_numbers_are_stored_as_plain_python(value, expected):
    report = make_report()
    report.add("section", metric=value, nested=[{"inner": value}])
    stored = report.sections["section"]
    assert stored["metric"] == expected and type(stored["metric"]) is type(expected)
    assert stored["nested"][0]["inner"] == expected


def test_numpy_strings_still_need_the_vocabulary():
    with pytest.raises(ValueError):
        make_report().add("section", metric=np.str_(OBJECT_ID))


def test_add_merges_and_copies():
    metrics = {"values": [1, 2]}
    report = make_report()
    report.add("section", first=metrics)
    report.add("section", second=2)
    metrics["values"].append(OBJECT_ID)  # later edits can't reach past the check
    assert report.sections == {"section": {"first": {"values": [1, 2]}, "second": 2}}


@pytest.mark.parametrize("name", [PUBLIC_SEGMENT, "synthetic_cut_in", "synthetic_Cut_In_2"])
def test_context_label_keeps_public_names(name):
    assert context_label(name) == name


@pytest.mark.parametrize("name", [
    CONTEXT_NAME, PUBLIC_SEGMENT + "x", "x" + PUBLIC_SEGMENT, "Synthetic_x", "", "synthetic_",
    f"synthetic_cut_in/../{CONTEXT_NAME}", f"synthetic_cut_in/{CONTEXT_NAME}", "synthetic_cut_in\n",
], ids=["real", "public plus", "plus public", "capital", "empty", "prefix only", "path out", "path in", "newline"])
def test_context_label_hashes_everything_else(name):
    label = context_label(name)
    assert re.fullmatch(r"sha256:[0-9a-f]{12}", label)
    assert label == context_label(name)
    assert name == "" or name not in label


def test_context_labels_differ():
    assert context_label(CONTEXT_NAME) != context_label(CONTEXT_NAME[:-1] + "1")


def test_report_never_holds_the_raw_context_name(tmp_path):
    report = make_report()
    report.add("section", status="pass")
    assert_clean(repr(vars(report)))
    assert_clean(report.write(tmp_path / "report.json").read_text())


def test_a_real_preprocessing_version_passes_the_shape_check(stage1_config):
    # If the format ever gains another character, every real run would die here.
    version = stage1_config.preprocessing_version
    assert Report(CONTEXT_NAME, version).preprocessing_version == version


def test_swapped_constructor_arguments_are_rejected():
    with pytest.raises(ValueError):
        Report(VERSION, CONTEXT_NAME)


def test_write_is_strict_json_and_creates_its_directory(tmp_path):
    report = make_report("synthetic_cut_in")
    report.add("gaps", n_frames=197, max_dt_ms=float("nan"), status="pass", directions=["LEFT_TO_RIGHT"])
    path = report.write(str(tmp_path / "not" / "yet" / "report.json"))
    assert isinstance(path, Path)
    assert read_json(path) == {
        "context": "synthetic_cut_in",
        "preprocessing_version": VERSION,
        "sections": {"gaps": {"n_frames": 197, "max_dt_ms": None, "status": "pass", "directions": ["LEFT_TO_RIGHT"]}},
        "failures": [],
    }


def test_write_refuses_a_value_that_went_around_add(tmp_path):
    report = make_report()
    report.sections["section"] = {"metric": float("nan")}
    with pytest.raises(ValueError):
        report.write(tmp_path / "report.json")


def test_fail_records_type_and_innermost_package_frame(tmp_path):
    # load_config fails inside pathlib: the frame to report is config.py, not pathlib.py,
    # and the message holds the path, which must go nowhere.
    report = make_report()
    with pytest.raises(FileNotFoundError) as info:
        load_config(tmp_path / OBJECT_ID / "stage1.toml")
    assert OBJECT_ID in str(info.value)
    report.fail("load", info.value)

    assert report.failures == [{"step": "load", "exception": "FileNotFoundError",
                                "location": report.failures[0]["location"]}]
    assert re.fullmatch(r"config\.py:\d+", report.failures[0]["location"])
    text = report.write(tmp_path / "report.json").read_text()
    assert_clean(text)
    assert str(tmp_path) not in text and str(Path.home()) not in text


def test_fail_with_no_package_frame_records_no_location():
    report = make_report()
    report.fail("load", ValueError(OBJECT_ID))  # never raised, so it has no traceback
    assert report.failures == [{"step": "load", "exception": "ValueError", "location": None}]


def test_fail_ignores_frames_outside_the_package():
    report = make_report()
    try:
        raise ValueError(OBJECT_ID)  # this frame is in tests/, not agatha
    except ValueError as exc:
        report.fail("load", exc)
    assert report.failures[0]["location"] is None


def test_fail_with_a_bad_step_does_not_print_the_exception_being_handled():
    # fail runs in except blocks, so a plain ValueError here would carry the real exception
    # and its message into the traceback as context.
    report = make_report()
    try:
        try:
            raise RuntimeError(OBJECT_ID)
        except RuntimeError as exc:
            report.fail(CONTEXT_NAME, exc)
    except ValueError as error:
        assert_clean("".join(traceback.format_exception(error)))
    else:
        pytest.fail("a bad step name must raise")


def raise_value_error():
    raise ValueError(f"bad row {OBJECT_ID} in {CONTEXT_NAME}")


def raise_key_error():
    {}[OBJECT_ID]


def raise_explicit_chain():
    try:
        raise_value_error()
    except ValueError as exc:
        raise RuntimeError("step failed") from exc


def raise_implicit_chain():
    try:
        raise_value_error()
    except ValueError:
        raise RuntimeError("step failed")


def raise_chain_with_secret_on_top():
    raise RuntimeError(CONTEXT_NAME) from KeyError(OBJECT_ID)


def raise_with_note():
    error = RuntimeError("step failed")
    error.add_note(f"while reading {CONTEXT_NAME}")
    raise error


def raise_group():
    raise ExceptionGroup(CONTEXT_NAME, [ValueError(OBJECT_ID)])


def raise_custom_str():
    class Echoing(Exception):
        def __str__(self):
            return OBJECT_ID
    raise Echoing()


def raise_exit_with_string():
    sys.exit(f"cannot read {CONTEXT_NAME}")  # Python prints a string exit code to stderr


def raise_keyboard_interrupt():
    raise KeyboardInterrupt(OBJECT_ID)


@pytest.mark.parametrize("raise_it", [
    raise_value_error, raise_key_error, raise_explicit_chain, raise_implicit_chain,
    raise_chain_with_secret_on_top, raise_with_note, raise_group, raise_custom_str,
    raise_exit_with_string, raise_keyboard_interrupt,
])
def test_guarded_keeps_messages_out_of_output_and_report(raise_it, capsys, tmp_path):
    report = make_report()
    with pytest.raises(SystemExit) as info:
        with guarded(report, "load"):
            raise_it()

    assert info.value.code == 1
    assert info.value.__cause__ is None and info.value.__suppress_context__
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
    assert_clean("".join(traceback.format_exception(info.value)))
    assert_clean(report.write(tmp_path / "report.json").read_text())
    assert [failure["step"] for failure in report.failures] == ["load"]


def test_guarded_names_the_failing_type_and_nothing_more():
    report = make_report()
    with pytest.raises(SystemExit):
        with guarded(report, "load"):
            raise_explicit_chain()
    assert set(report.failures[0]) == {"step", "exception", "location"}
    assert report.failures[0]["exception"] == "RuntimeError"


def test_guarded_locates_failures_inside_the_package():
    report = make_report()
    with pytest.raises(SystemExit):
        with guarded(report, "load"):
            load_config("/nonexistent/stage1.toml")
    assert re.fullmatch(r"config\.py:\d+", report.failures[0]["location"])


def test_guarded_with_a_rejected_value_exits_cleanly(capsys):
    # The most likely real failure: a bug hands add an ID inside a guarded step.
    report = make_report()
    with pytest.raises(SystemExit):
        with guarded(report, "checks"):
            report.add("section", metric=OBJECT_ID)
    assert report.sections == {}
    assert report.failures[0]["exception"] == "ValueError"
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_guarded_success_changes_nothing(capsys):
    report = make_report()
    with guarded(report, "load"):
        report.add("section", metric=1)
    assert report.failures == [] and report.sections == {"section": {"metric": 1}}
    assert capsys.readouterr().out == ""


def test_guarded_checks_the_step_before_running_it():
    ran = []
    with pytest.raises(ValueError):
        with guarded(make_report(), CONTEXT_NAME):
            ran.append(True)
    assert not ran


def test_guarded_keeps_earlier_sections_for_the_final_write(tmp_path):
    report = make_report()
    report.add("section", metric=1)
    with pytest.raises(SystemExit):
        with guarded(report, "load"):
            raise_value_error()
    written = read_json(report.write(tmp_path / "report.json"))
    assert written["sections"] == {"section": {"metric": 1}}
    assert written["failures"][0]["step"] == "load"


@pytest.mark.parametrize("raise_line", [
    f"raise ValueError('bad row {OBJECT_ID}')",
    f"raise RuntimeError('step failed') from KeyError('{CONTEXT_NAME}')",
    f"import sys; sys.exit('cannot read {CONTEXT_NAME}')",
])
def test_nothing_reaches_a_real_process_log(raise_line, tmp_path):
    # capsys can't see what the interpreter prints on the way out, which is what Slurm logs.
    path = tmp_path / "report.json"
    script = textwrap.dedent(f"""
        from agatha.report import Report, guarded

        report = Report({CONTEXT_NAME!r}, {VERSION!r})
        try:
            with guarded(report, "load"):
                {raise_line}
        finally:
            report.write({str(path)!r})
    """)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 1
    assert result.stdout == "" and result.stderr == ""
    assert_clean(path.read_text())
    assert read_json(path)["failures"][0]["step"] == "load"


def test_a_lowercase_id_passes_the_key_shape():
    # A known limit of the spec's rule, pinned so it stays visible: the shape can't tell a
    # lowercase ID from a metric name. Real IDs are mixed case, so a dict keyed by ID fails
    # on its first ordinary key, but report code must still not key by ID.
    report = make_report()
    report.add("section", metric={LOWERCASE_ID: 1})
    assert LOWERCASE_ID in report.sections["section"]["metric"]
