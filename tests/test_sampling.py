import numpy as np
import pytest

from agatha.sampling import GapReport, Window, compute_windows, gap_report, nearest_frame, select_frames

T0 = 1_550_083_467_369_629


def _frames(n: int, dropped=()) -> np.ndarray:
    """Timestamps exactly on a 10 Hz grid from T0, without the `dropped` frame indices."""
    return T0 + np.delete(np.arange(n), list(dropped)) * 100_000


def _jittered(n: int, seed: int = 0, dropped=()) -> np.ndarray:
    """10 Hz with sub-ms jitter, as in real spacing (99,856 to 100,707 microseconds)."""
    jitter = np.random.default_rng(seed).integers(-700, 700, n)
    jitter[0] = 0
    return np.delete(T0 + np.arange(n) * 100_000 + jitter, list(dropped))


def _offsets_ms(selection) -> list[float]:
    return ((selection.timestamps - T0) / 1e3).tolist()


def test_gap_report_on_a_segment_with_a_dropped_frame():
    assert gap_report(_frames(11, dropped=[5]), 150.0) == GapReport(10, 1.0, 100.0, 100.0, 200.0, 1)


def test_gap_report_counts_only_spacing_over_the_limit():
    assert gap_report(_frames(11, dropped=[5]), 200.0).n_gaps == 0  # matches has_gap_before, which is strict
    assert gap_report(_frames(11, dropped=[3, 7]), 150.0).n_gaps == 2


def test_gap_report_ignores_row_order_and_per_camera_repeats():
    ts = np.repeat(_frames(11, dropped=[5]), 3)
    np.random.default_rng(0).shuffle(ts)
    assert gap_report(ts, 150.0) == GapReport(10, 1.0, 100.0, 100.0, 200.0, 1)


def test_gap_report_returns_plain_numbers_for_the_report():
    report = gap_report(_frames(5), 150.0)
    assert type(report.n_frames) is int and type(report.n_gaps) is int
    assert all(type(v) is float for v in report[1:5])


@pytest.mark.parametrize("ts", [[], [T0]])
def test_gap_report_with_too_few_frames_to_space(ts):
    report = gap_report(ts, 150.0)
    assert (report.n_frames, report.duration_s, report.n_gaps) == (len(ts), 0.0, 0)
    assert np.isnan([report.min_dt_ms, report.median_dt_ms, report.max_dt_ms]).all()


def test_two_fps_stays_on_the_grid_across_a_200ms_gap():
    ts = _frames(30, dropped=[5])
    selection = select_frames(ts, T0, 2.0, 0.0, 2.5)
    # 0.5 s is missing, so that grid time ties between 0.4 and 0.6 and takes the earlier frame.
    # Every 5th row would have read 0, 0.6, 1.1, 1.6, 2.1: off the grid for the rest of the segment.
    assert _offsets_ms(selection) == [0, 400, 1000, 1500, 2000]
    assert selection.max_gap_ms == 600.0


def test_two_fps_takes_the_nearer_of_two_uneven_neighbours():
    ts = _frames(30, dropped=[5])  # frames at 0.4 s and 0.6 s around the 0.5 s grid time
    ts[5] = T0 + 590_000
    assert 590.0 in _offsets_ms(select_frames(ts, T0, 2.0, 0.0, 2.5))
    ts[5] = T0 + 620_000
    assert 400.0 in _offsets_ms(select_frames(ts, T0, 2.0, 0.0, 2.5))


def test_grid_time_with_no_frame_within_half_a_period_selects_nothing():
    ts = _frames(40, dropped=range(10, 20))  # nothing from 1.0 to 1.9 s
    # 1.5 s is 500 ms from the nearest frame: taking one anyway would invent coverage of the hole.
    assert _offsets_ms(select_frames(ts, T0, 2.0, 0.0, 3.0)) == [0, 500, 900, 2000, 2500]


def test_half_a_period_is_the_limit_even_for_a_frame_no_other_grid_time_can_reach():
    # 0.76 s is 260 ms from the 0.5 s grid time, and the one nearer, 1.0 s, is past the window's end.
    # In the test above a wider limit would pick frames that the next grid time selects anyway.
    ts = T0 + np.array([0, 760_000])
    assert _offsets_ms(select_frames(ts, T0, 2.0, 0.0, 1.0)) == [0]
    assert _offsets_ms(select_frames(T0 + np.array([0, 750_000]), T0, 2.0, 0.0, 1.0)) == [0, 750]


def test_window_inside_a_hole_selects_nothing():
    selection = select_frames(_frames(40, dropped=range(10, 30)), T0, 2.0, 1.5, 2.5)
    assert selection.timestamps.dtype == np.int64 and len(selection.timestamps) == 0
    assert (selection.achieved_fps, selection.max_gap_ms) == (0.0, 0.0)


def test_timestamps_stay_exact_int64():
    selection = select_frames(_jittered(50), T0, 5.0, 0.0, 5.0)
    assert selection.timestamps.dtype == np.int64
    assert set(selection.timestamps) <= set(_jittered(50))


def test_input_order_and_per_camera_repeats_do_not_change_the_selection():
    ts = _frames(50, dropped=[5])  # the tie at 0.5 s is what would depend on row order
    rows = np.repeat(ts, 3)[::-1]
    np.testing.assert_array_equal(select_frames(rows, T0, 2.0, 0.0, 5.0).timestamps,
                                  select_frames(ts, T0, 2.0, 0.0, 5.0).timestamps)


def test_overlapping_windows_share_frames():
    ts = _jittered(100, dropped=[42])
    first = select_frames(ts, T0, 2.0, 0.0, 5.0)
    second = select_frames(ts, T0, 2.0, 4.0, 9.0)
    assert len(np.intersect1d(first.timestamps, second.timestamps)) == 2  # 4.0 and 4.5 s


def test_grid_is_anchored_at_t0_not_at_the_window_start():
    assert _offsets_ms(select_frames(_frames(100), T0, 2.0, 4.2, 6.0)) == [4500, 5000, 5500]


def test_two_and_five_fps_grids_are_not_nested():
    ts = _frames(200)
    two = select_frames(ts, T0, 2.0, 0.0, 20.0)
    five = select_frames(ts, T0, 5.0, 0.0, 20.0)
    assert (len(two.timestamps), len(five.timestamps)) == (40, 100)
    # Half of the 2 FPS frames sit on .5 s, between 5 FPS grid times, so Pass 2 needs force_include.
    assert len(np.setdiff1d(two.timestamps, five.timestamps)) == 20


@pytest.mark.parametrize("seed", range(5))
def test_ten_fps_returns_every_frame(seed):
    ts = _jittered(198, seed, dropped=[50, 120])
    selection = select_frames(ts, T0, 10.0, 0.0, (ts[-1] - T0) / 1e6, closed_end=True)
    np.testing.assert_array_equal(selection.timestamps, ts)


def test_ten_fps_keeps_a_dropped_frame_dropped():
    selection = select_frames(_frames(30, dropped=[12]), T0, 10.0, 0.0, 3.0)
    assert 1200.0 not in _offsets_ms(selection) and selection.max_gap_ms == 200.0


def test_half_open_window_leaves_the_end_to_the_next_window():
    ts = _frames(100)
    assert _offsets_ms(select_frames(ts, T0, 2.0, 0.0, 5.0))[-1] == 4500
    assert _offsets_ms(select_frames(ts, T0, 2.0, 5.0, 9.0))[0] == 5000


def test_closed_end_includes_the_last_frame():
    ts = _frames(196)  # last frame at 19.5 s, on the 2 FPS grid
    last_t_s = (ts[-1] - T0) / 1e6
    assert _offsets_ms(select_frames(ts, T0, 2.0, 16.0, last_t_s))[-1] == 19_000
    assert _offsets_ms(select_frames(ts, T0, 2.0, 16.0, last_t_s, closed_end=True))[-1] == 19_500


@pytest.mark.parametrize("fps, n_frames", [(10.0, 198), (2.0, 201)])
@pytest.mark.parametrize("early_us", [1, 300, 20_000])
def test_closed_end_reaches_a_last_frame_just_short_of_its_grid_time(fps, n_frames, early_us):
    ts = _frames(n_frames)
    ts[-1] -= early_us  # end_s now falls just before the grid time that this frame answers to
    selection = select_frames(ts, T0, fps, 16.0, (ts[-1] - T0) / 1e6, closed_end=True)
    assert selection.timestamps[-1] == ts[-1]


@pytest.mark.parametrize("n_frames", [198, 199])
def test_closed_end_does_not_stretch_to_a_distant_grid_time(n_frames):
    # Segments end near 19.7 to 19.8 s, off the 2 FPS grid, so the last sample stays at 19.5 s.
    ts = _frames(n_frames)
    selection = select_frames(ts, T0, 2.0, 16.0, (ts[-1] - T0) / 1e6, closed_end=True)
    assert _offsets_ms(selection)[-1] == 19_500


def test_force_include_adds_the_last_frame_when_the_grid_misses_it():
    ts = _frames(198)
    selection = select_frames(ts, T0, 2.0, 16.0, 19.7, closed_end=True, force_include=ts[-1:])
    assert _offsets_ms(selection)[-2:] == [19_500, 19_700]


def test_float_noise_in_the_bounds_does_not_move_frames():
    ts = _frames(198)
    assert 3 * 0.1 > 0.3 and 197 * 0.1 > 19.7
    assert _offsets_ms(select_frames(ts, T0, 10.0, 3 * 0.1, 0.6)) == [300, 400, 500]
    assert _offsets_ms(select_frames(ts, T0, 10.0, 19.5, 197 * 0.1)) == [19_500, 19_600]
    assert _offsets_ms(select_frames(ts, T0, 10.0, 19.5, 19.7, closed_end=True))[-1] == 19_700
    assert _offsets_ms(select_frames(ts, T0, 6.0, 1 / 6, 0.3)) == [200]  # a period with no exact microsecond


def test_force_include_adds_a_cited_frame_to_a_later_pass():
    ts = _frames(200)
    cited = select_frames(ts, T0, 2.0, 0.0, 5.0).timestamps
    selection = select_frames(ts, T0, 5.0, 0.0, 5.0, force_include=cited)
    assert set(cited) <= set(selection.timestamps)
    assert selection.timestamps.dtype == np.int64
    assert (np.diff(selection.timestamps) > 0).all()  # sorted, and the shared frames aren't doubled
    assert len(selection.timestamps) == 25 + 5        # the 5 FPS frames, plus the five on .5 s


@pytest.mark.parametrize("value", [
    T0 + 500_000,    # t0 + round(0.5e6), which misses the jittered frame by 109 us
    9.5,             # seconds, which would truncate to timestamp 9
])
def test_force_include_refuses_what_is_not_a_frame(value):
    ts = _frames(100)
    ts[5] += 109
    with pytest.raises(ValueError) as error:
        select_frames(ts, T0, 2.0, 0.0, 1.0, force_include=[value])
    assert str(T0) not in str(error.value)  # the message can reach a Slurm log


def test_nearest_frame_turns_a_time_into_an_exact_timestamp():
    ts = _jittered(100, dropped=[42])
    t_s = (ts[10] - T0) / 1e6 + 0.003
    assert nearest_frame(ts, T0, t_s) == ts[10] and type(nearest_frame(ts, T0, t_s)) is int
    selection = select_frames(ts, T0, 2.0, 0.0, 1.0, force_include=[nearest_frame(ts, T0, 0.7)])
    assert ts[7] in selection.timestamps


def test_a_frame_outside_the_bounds_can_be_forced_and_shows_in_the_spacing():
    """The bounds are for grid times. Only the frames say what a selection really spans."""
    selection = select_frames(_frames(100), T0, 2.0, 0.0, 1.0, force_include=[T0 + 9_000_000])
    assert _offsets_ms(selection)[-1] == 9000 and selection.max_gap_ms == 8500.0


def test_jitter_can_put_a_selected_frame_just_before_the_start():
    ts = _frames(100)
    ts[40] -= 358
    assert select_frames(ts, T0, 2.0, 4.0, 5.0).timestamps[0] == ts[40] < T0 + 4_000_000


def test_force_include_counts_in_the_reported_spacing():
    selection = select_frames(_frames(100), T0, 2.0, 0.0, 1.0, force_include=[T0 + 3_000_000])
    assert _offsets_ms(selection) == [0, 500, 3000]
    assert selection.max_gap_ms == 2500.0


def test_achieved_fps_reads_the_requested_rate_on_a_clean_grid():
    assert select_frames(_frames(100), T0, 2.0, 0.0, 5.0).achieved_fps == pytest.approx(2.0)
    assert select_frames(_frames(200), T0, 10.0, 0.0, 19.9, closed_end=True).achieved_fps == pytest.approx(10.0)


def test_achieved_fps_falls_when_a_sample_is_lost():
    # Nothing within 250 ms of 2.0 s. Frames / span would read exactly 2.0 and hide the loss.
    selection = select_frames(_frames(100, dropped=range(16, 25)), T0, 2.0, 0.0, 5.0)
    assert len(selection.timestamps) == 9
    assert selection.achieved_fps == pytest.approx(8 / 4.5)


def test_selection_of_one_frame_reports_zero_rates():
    selection = select_frames(_frames(100), T0, 2.0, 0.4, 0.9)
    assert _offsets_ms(selection) == [500]
    assert (selection.achieved_fps, selection.max_gap_ms) == (0.0, 0.0)


def test_windows_for_a_real_last_timestamp():
    assert compute_windows(19.7, 5.0, 4.0) == [
        (0, 5, False), (4, 9, False), (8, 13, False), (12, 17, False), (16, 19.7, True)]


def test_windows_for_exactly_twenty_seconds():
    assert compute_windows(20.0, 5.0, 4.0) == [
        (0, 5, False), (4, 9, False), (8, 13, False), (12, 17, False), (16, 20.0, True)]


def test_windows_at_stride_2_5_keep_seven_windows():
    windows = compute_windows(19.7, 5.0, 2.5)
    assert len(windows) == 7 and windows[-1] == (15.0, 19.7, True)


def test_short_tail_merges_into_the_previous_window():
    # Only 0.5 s past (12, 17): a window of its own at 16 s would be almost all repeats.
    assert compute_windows(17.5, 5.0, 4.0) == [
        (0, 5, False), (4, 9, False), (8, 13, False), (12, 17.5, True)]


def test_short_tail_merges_without_overlap_too():
    assert compute_windows(15.5, 5.0, 5.0) == [(0, 5, False), (5, 10, False), (10, 15.5, True)]


def test_tail_of_exactly_min_tail_is_kept():
    assert compute_windows(18.0, 5.0, 4.0)[-1] == (16, 18.0, True)


def test_last_window_that_fits_exactly_ends_the_list():
    assert compute_windows(17.0, 5.0, 4.0)[-2:] == [(8, 13, False), (12, 17.0, True)]
    assert compute_windows(17.0, 5.0, 4.0, min_tail_s=0.0)[-2:] == [(8, 13, False), (12, 17.0, True)]


def test_segment_shorter_than_a_window_gets_one_closed_window():
    assert compute_windows(3.2, 5.0, 4.0) == [(0, 3.2, True)]
    assert compute_windows(5.5, 5.0, 4.0) == [(0, 5.5, True)]


def test_only_the_last_window_is_closed():
    windows = compute_windows(19.7, 5.0, 2.5)
    assert [w.closed_end for w in windows] == [False] * 6 + [True]
    assert all(isinstance(w, Window) for w in windows)


def test_default_windows_cover_a_segment_at_two_fps():
    ts = _frames(198)
    last_t_s = (ts[-1] - T0) / 1e6
    selections = [select_frames(ts, T0, 2.0, w.start_s, w.end_s, w.closed_end)
                  for w in compute_windows(last_t_s, 5.0, 4.0)]
    assert [len(s.timestamps) for s in selections] == [10, 10, 10, 10, 8]
    assert len(np.unique(np.concatenate([s.timestamps for s in selections]))) == 40
