from pathlib import Path

import pytest

from text_localizer.domain import NormalizedBox
from text_localizer.profile import (
    load_profile,
    merge_analysis_options,
    merge_visual_event_options,
)


def test_browser_lecture_profile_loads_relative_glossary() -> None:
    profile = load_profile(Path("config/profiles/browser-lecture.yaml"))

    assert profile.name == "browser-lecture-content"
    assert profile.analysis.roi == NormalizedBox(0.22, 0.16, 0.75, 0.84)
    assert profile.analysis.translation_batch_size == 4
    assert len(profile.analysis.exclude_regions) == 2
    assert profile.scene_detection.hash_size == 16
    assert profile.scene_detection.hamming_threshold == 18
    assert profile.visual_events.sample_interval_seconds == 0.2
    assert profile.visual_events.tile_columns == 12
    assert profile.visual_events.tile_rows == 8
    assert profile.visual_events.exclude_regions == (NormalizedBox(0.84, 0.78, 0.13, 0.22),)
    assert profile.glossary_path == Path("config/glossary.ko-ja.yaml").resolve()


def test_cli_values_override_profile_and_exclusions_append() -> None:
    profile = load_profile(Path("config/profiles/browser-lecture.yaml"))
    extra_exclusion = NormalizedBox(0.8, 0.8, 0.1, 0.1)

    merged = merge_analysis_options(
        profile,
        roi=NormalizedBox(0.1, 0.1, 0.8, 0.8),
        exclude_regions=(extra_exclusion,),
        min_region_height=0.02,
        group_adjacent=False,
        translation_batch_size=2,
    )

    assert merged.roi == NormalizedBox(0.1, 0.1, 0.8, 0.8)
    assert merged.exclude_regions[-1] == extra_exclusion
    assert len(merged.exclude_regions) == 3
    assert merged.min_region_height == 0.02
    assert merged.group_adjacent is False
    assert merged.translation_batch_size == 2


def test_visual_event_cli_values_override_profile_and_exclusions_append() -> None:
    profile = load_profile(Path("config/profiles/browser-lecture.yaml"))
    extra_exclusion = NormalizedBox(0.5, 0.5, 0.1, 0.1)

    merged = merge_visual_event_options(
        profile,
        sample_interval_seconds=0.1,
        tile_columns=16,
        tile_change_threshold=0.03,
        exclude_regions=(extra_exclusion,),
    )

    assert merged.sample_interval_seconds == 0.1
    assert merged.tile_columns == 16
    assert merged.tile_rows == 8
    assert merged.tile_change_threshold == 0.03
    assert merged.exclude_regions[-1] == extra_exclusion
    assert len(merged.exclude_regions) == 2


def test_profile_rejects_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "profile.yaml"
    path.write_text("schema_version: 2\nname: future\n", encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        load_profile(path)
