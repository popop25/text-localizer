from pathlib import Path

from PIL import Image, ImageDraw

from text_localizer.domain import NormalizedBox
from text_localizer.profile import SceneDetectionOptions
from text_localizer.scenes import (
    difference_hash,
    hamming_distance,
    index_frame_directory,
    index_frame_scenes,
)


def test_hamming_distance_counts_different_bits() -> None:
    assert hamming_distance(0b1010, 0b0011) == 2


def test_excluded_pixels_do_not_change_roi_hash(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (100, 100), "white").save(first)
    changed = Image.new("RGB", (100, 100), "white")
    ImageDraw.Draw(changed).rectangle((50, 0, 99, 99), fill="black")
    changed.save(second)
    exclusion = NormalizedBox(0.5, 0.0, 0.5, 1.0)

    first_hash = difference_hash(
        first,
        roi=None,
        exclude_regions=(exclusion,),
        hash_size=8,
    )
    second_hash = difference_hash(
        second,
        roi=None,
        exclude_regions=(exclusion,),
        hash_size=8,
    )

    assert first_hash == second_hash


def test_consecutive_duplicates_share_scene_representative(tmp_path: Path) -> None:
    first = tmp_path / "frame-0001.png"
    second = tmp_path / "frame-0002.png"
    third = tmp_path / "frame-0003.png"
    gradient = Image.linear_gradient("L").rotate(90, expand=True).resize((100, 100))
    gradient.save(first)
    gradient.copy().save(second)
    gradient.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(third)

    payload = index_frame_scenes(
        [first, second, third],
        profile_name="test",
        roi=None,
        exclude_regions=(),
        options=SceneDetectionOptions(hash_size=8, hamming_threshold=0),
    )

    assert payload["total_frames"] == 3
    assert payload["unique_scenes"] == 2
    assert payload["duplicate_frames"] == 1
    assert payload["reuse_ratio"] == 1 / 3
    assert payload["processing_latency_ms"] >= 0
    assert [item["scene_id"] for item in payload["frames"]] == [1, 1, 2]
    assert payload["frames"][1]["anchor_distance"] == 0
    assert payload["frames"][1]["previous_frame_distance"] == 0
    assert payload["frames"][2]["anchor_distance"] > 0


def test_directory_index_writes_auditable_json(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    Image.new("RGB", (50, 50), "white").save(frames / "frame-0001.png")
    Image.new("RGB", (50, 50), "white").save(frames / "frame-0002.png")
    output = tmp_path / "scene-index.json"

    payload = index_frame_directory(
        frames,
        pattern="frame-*.png",
        output_path=output,
        profile_name=None,
        roi=None,
        exclude_regions=(),
        options=SceneDetectionOptions(hash_size=8, hamming_threshold=0),
    )

    assert output.is_file()
    assert payload["unique_scenes"] == 1
