import pytest

from text_localizer.video import evenly_spaced_timestamps, parse_frame_rate


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
