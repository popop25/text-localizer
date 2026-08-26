from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from text_localizer.domain import NormalizedBox
from text_localizer.quality import AnalysisOptions


@dataclass(frozen=True, slots=True)
class SceneDetectionOptions:
    hash_size: int = 16
    hamming_threshold: int = 12

    def __post_init__(self) -> None:
        if not 4 <= self.hash_size <= 64:
            raise ValueError("Scene hash size must be between 4 and 64.")
        bit_count = self.hash_size * self.hash_size
        if not 0 <= self.hamming_threshold <= bit_count:
            raise ValueError(f"Scene Hamming threshold must be between 0 and {bit_count}.")


@dataclass(frozen=True, slots=True)
class VisualEventOptions:
    sample_interval_seconds: float = 0.2
    analysis_width: int = 384
    tile_columns: int = 12
    tile_rows: int = 8
    tile_change_threshold: float = 0.02
    minimum_changed_tiles: int = 1
    global_change_ratio: float = 0.35
    exclude_regions: tuple[NormalizedBox, ...] = ()

    def __post_init__(self) -> None:
        if self.sample_interval_seconds <= 0.0:
            raise ValueError("Visual-event sample interval must be positive.")
        if not 64 <= self.analysis_width <= 2048:
            raise ValueError("Visual-event analysis width must be between 64 and 2048.")
        if not 2 <= self.tile_columns <= 64 or not 2 <= self.tile_rows <= 64:
            raise ValueError("Visual-event tile dimensions must be between 2 and 64.")
        tile_count = self.tile_columns * self.tile_rows
        if not 1 <= self.minimum_changed_tiles <= tile_count:
            raise ValueError(
                f"Visual-event minimum changed tiles must be between 1 and {tile_count}."
            )
        if not 0.0 < self.tile_change_threshold <= 1.0:
            raise ValueError("Visual-event tile change threshold must be between 0 and 1.")
        if not 0.0 <= self.global_change_ratio <= 1.0:
            raise ValueError("Visual-event global change ratio must be between 0 and 1.")


@dataclass(frozen=True, slots=True)
class LocalizationProfile:
    name: str
    analysis: AnalysisOptions
    scene_detection: SceneDetectionOptions
    visual_events: VisualEventOptions
    glossary_path: Path | None
    source_path: Path


def load_profile(path: Path) -> LocalizationProfile:
    if not path.is_file():
        raise FileNotFoundError(f"Localization profile does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Localization profile root must be a mapping.")
    if payload.get("schema_version") != 1:
        raise ValueError("Localization profile schema_version must be 1.")

    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Localization profile requires a non-empty name.")

    analysis_payload = _mapping(payload.get("analysis", {}), "analysis")
    scene_payload = _mapping(payload.get("scene_detection", {}), "scene_detection")
    visual_payload = _mapping(payload.get("visual_events", {}), "visual_events")
    exclude_payload = analysis_payload.get("exclude_regions", [])
    if not isinstance(exclude_payload, list):
        raise ValueError("analysis.exclude_regions must be a list.")
    visual_exclude_payload = visual_payload.get("exclude_regions", [])
    if not isinstance(visual_exclude_payload, list):
        raise ValueError("visual_events.exclude_regions must be a list.")

    glossary_value = payload.get("glossary")
    if glossary_value is not None and not isinstance(glossary_value, str):
        raise ValueError("Profile glossary must be a path string.")
    glossary_path = (path.parent / glossary_value).resolve() if glossary_value is not None else None

    analysis = AnalysisOptions(
        roi=_optional_box(analysis_payload.get("roi"), "analysis.roi"),
        exclude_regions=tuple(
            _box(item, f"analysis.exclude_regions[{index}]")
            for index, item in enumerate(exclude_payload)
        ),
        min_region_height=_number(
            analysis_payload.get("min_region_height", 0.0),
            "analysis.min_region_height",
        ),
        group_adjacent=_boolean(
            analysis_payload.get("group_adjacent", True),
            "analysis.group_adjacent",
        ),
        translation_batch_size=_integer(
            analysis_payload.get("translation_batch_size", 4),
            "analysis.translation_batch_size",
        ),
    )
    scene_detection = SceneDetectionOptions(
        hash_size=_integer(scene_payload.get("hash_size", 16), "scene_detection.hash_size"),
        hamming_threshold=_integer(
            scene_payload.get("hamming_threshold", 12),
            "scene_detection.hamming_threshold",
        ),
    )
    visual_events = VisualEventOptions(
        sample_interval_seconds=_number(
            visual_payload.get("sample_interval_seconds", 0.2),
            "visual_events.sample_interval_seconds",
        ),
        analysis_width=_integer(
            visual_payload.get("analysis_width", 384),
            "visual_events.analysis_width",
        ),
        tile_columns=_integer(
            visual_payload.get("tile_columns", 12),
            "visual_events.tile_columns",
        ),
        tile_rows=_integer(
            visual_payload.get("tile_rows", 8),
            "visual_events.tile_rows",
        ),
        tile_change_threshold=_number(
            visual_payload.get("tile_change_threshold", 0.02),
            "visual_events.tile_change_threshold",
        ),
        minimum_changed_tiles=_integer(
            visual_payload.get("minimum_changed_tiles", 1),
            "visual_events.minimum_changed_tiles",
        ),
        global_change_ratio=_number(
            visual_payload.get("global_change_ratio", 0.35),
            "visual_events.global_change_ratio",
        ),
        exclude_regions=tuple(
            _box(item, f"visual_events.exclude_regions[{index}]")
            for index, item in enumerate(visual_exclude_payload)
        ),
    )
    return LocalizationProfile(
        name=name.strip(),
        analysis=analysis,
        scene_detection=scene_detection,
        visual_events=visual_events,
        glossary_path=glossary_path,
        source_path=path.resolve(),
    )


def merge_analysis_options(
    profile: LocalizationProfile | None,
    *,
    roi: NormalizedBox | None = None,
    exclude_regions: tuple[NormalizedBox, ...] = (),
    min_region_height: float | None = None,
    group_adjacent: bool | None = None,
    translation_batch_size: int | None = None,
) -> AnalysisOptions:
    base = profile.analysis if profile is not None else AnalysisOptions()
    return AnalysisOptions(
        roi=roi if roi is not None else base.roi,
        exclude_regions=base.exclude_regions + exclude_regions,
        min_region_height=(
            min_region_height if min_region_height is not None else base.min_region_height
        ),
        group_adjacent=(group_adjacent if group_adjacent is not None else base.group_adjacent),
        translation_batch_size=(
            translation_batch_size
            if translation_batch_size is not None
            else base.translation_batch_size
        ),
    )


def merge_visual_event_options(
    profile: LocalizationProfile | None,
    *,
    sample_interval_seconds: float | None = None,
    analysis_width: int | None = None,
    tile_columns: int | None = None,
    tile_rows: int | None = None,
    tile_change_threshold: float | None = None,
    minimum_changed_tiles: int | None = None,
    global_change_ratio: float | None = None,
    exclude_regions: tuple[NormalizedBox, ...] = (),
) -> VisualEventOptions:
    base = profile.visual_events if profile is not None else VisualEventOptions()
    return VisualEventOptions(
        sample_interval_seconds=(
            sample_interval_seconds
            if sample_interval_seconds is not None
            else base.sample_interval_seconds
        ),
        analysis_width=analysis_width if analysis_width is not None else base.analysis_width,
        tile_columns=tile_columns if tile_columns is not None else base.tile_columns,
        tile_rows=tile_rows if tile_rows is not None else base.tile_rows,
        tile_change_threshold=(
            tile_change_threshold
            if tile_change_threshold is not None
            else base.tile_change_threshold
        ),
        minimum_changed_tiles=(
            minimum_changed_tiles
            if minimum_changed_tiles is not None
            else base.minimum_changed_tiles
        ),
        global_change_ratio=(
            global_change_ratio if global_change_ratio is not None else base.global_change_ratio
        ),
        exclude_regions=base.exclude_regions + exclude_regions,
    )


def resolve_glossary_path(
    profile: LocalizationProfile | None,
    override: Path | None,
) -> Path:
    if override is not None:
        return override
    if profile is not None and profile.glossary_path is not None:
        return profile.glossary_path
    return Path("config/glossary.ko-ja.yaml")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Profile {name} must be a mapping.")
    return value


def _optional_box(value: Any, name: str) -> NormalizedBox | None:
    if value is None:
        return None
    return _box(value, name)


def _box(value: Any, name: str) -> NormalizedBox:
    if not isinstance(value, dict):
        raise ValueError(f"Profile {name} must be a coordinate mapping.")
    try:
        return NormalizedBox(
            x=_number(value["x"], f"{name}.x"),
            y=_number(value["y"], f"{name}.y"),
            width=_number(value["width"], f"{name}.width"),
            height=_number(value["height"], f"{name}.height"),
        )
    except KeyError as exc:
        raise ValueError(f"Profile {name} requires x, y, width, and height.") from exc


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Profile {name} must be a number.")
    return float(value)


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Profile {name} must be an integer.")
    return int(value)


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"Profile {name} must be true or false.")
    return value
