import pytest

from text_localizer.video import evenly_spaced_timestamps, interval_timestamps, parse_frame_rate


def test_parse_fractional_frame_rate() -> None:
    assert parse_frame_rate("30000/1001") == pytest.approx(29.97002997)


def test_evenly_spaced_timestamps_are_bounded() -> None:
    timestamps = evenly_spaced_timestamps(10.0, 4)
    assert len(timestamps) == 4
    assert timestamps == sorted(timestamps)
    assert 0.0 <= timestamps[0] < timestamps[-1] <= 10.0


def test_timestamp_count_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        evenly_spaced_timestamps(10.0, 0)


def test_interval_timestamps_cover_video_without_exceeding_duration() -> None:
    assert interval_timestamps(3.2, 1.0) == [0.0, 1.0, 2.0, 3.0]
    assert interval_timestamps(0.2, 1.0) == [0.0]


def test_interval_must_be_positive() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        interval_timestamps(10.0, 0.0)
