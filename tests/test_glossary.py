from pathlib import Path

from text_localizer.glossary import Glossary, GlossaryEntry


def test_glossary_normalizes_aliases_protects_and_restores_terms() -> None:
    glossary = Glossary(
        (
            GlossaryEntry(
                "Azure OpenAI",
                "Azure OpenAI",
                ("Azure OpenA", "Azure OpenAl"),
            ),
            GlossaryEntry("학습 목표", "学習目標"),
        )
    )

    prepared = glossary.prepare("Azure OpenA의학습 목표")

    assert prepared.normalized_text == "Azure OpenAI 의학습 목표"
    assert prepared.protected_text == "<TERM0> 의<TERM1>"
    assert prepared.normalization_edits == (
        "Azure OpenA -> Azure OpenAI",
        "restore-script-boundary-spacing",
    )
    restored, flags = glossary.restore("<TERM0> の<TERM1>", prepared)
    assert restored == "Azure OpenAI の学習目標"
    assert flags == ()


def test_glossary_restores_placeholder_changed_to_japanese_quotes() -> None:
    glossary = Glossary((GlossaryEntry("LangChain", "LangChain"),))
    prepared = glossary.prepare("LangChain 기반")

    restored, flags = glossary.restore("「TERM0」ベース", prepared)

    assert restored == "LangChainベース"
    assert flags == ()


def test_alias_result_is_not_reprocessed_by_a_shorter_alias() -> None:
    glossary = Glossary(
        (
            GlossaryEntry(
                "OpenAI API 둘러보기",
                "OpenAI APIを見てみる",
                ("OpenAI API둘러보기",),
            ),
            GlossaryEntry("OpenAI API", "OpenAI API", ("OpenAI AP",)),
        )
    )

    prepared = glossary.prepare("OpenAI API둘러보기")

    assert prepared.normalized_text == "OpenAI API 둘러보기"
    assert prepared.protected_text == "<TERM0>"


def test_glossary_yaml_validation_and_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / "glossary.yaml"
    path.write_text(
        "entries:\n  - source: OpenAI API\n    target: OpenAI API\n    aliases: [OpenAl AP]\n",
        encoding="utf-8",
    )

    first = Glossary.from_yaml(path)
    second = Glossary.from_yaml(path)

    assert first.entries[0].aliases == ("OpenAl AP",)
    assert first.fingerprint == second.fingerprint
    assert first.cache_namespace("fake") != "fake"


def test_alias_prefix_does_not_mutate_already_canonical_terms() -> None:
    glossary = Glossary(
        (
            GlossaryEntry("Azure OpenAI", "Azure OpenAI", ("Azure OpenA",)),
            GlossaryEntry("OpenAI API", "OpenAI API", ("OpenAI AP",)),
        )
    )

    prepared = glossary.prepare("Azure OpenAI \uc640 OpenAI API")

    assert prepared.normalized_text == "Azure OpenAI \uc640 OpenAI API"
    assert "OpenAII" not in prepared.normalized_text
    assert prepared.normalization_edits == ()


def test_bracketed_ocr_alias_can_be_restored_without_model_translation() -> None:
    glossary = Glossary(
        (
            GlossaryEntry("[\uc2e4\uc2b5]", "[\u6f14\u7fd2]", ("[\uc2e4]",)),
            GlossaryEntry("LangGraph", "LangGraph"),
            GlossaryEntry("\uae30\ucd08", "\u57fa\u790e"),
        )
    )

    prepared = glossary.prepare("[\uc2e4]LangGraph\uae30\ucd08")
    restored, flags = glossary.restore(prepared.protected_text, prepared)

    assert prepared.normalized_text == "[\uc2e4\uc2b5] LangGraph \uae30\ucd08"
    assert prepared.requires_model_translation is False
    assert restored == "[\u6f14\u7fd2] LangGraph \u57fa\u790e"
    assert flags == ()


def test_unprotected_hangul_still_requires_full_sentence_translation() -> None:
    prepared = Glossary((GlossaryEntry("Agent", "Agent"),)).prepare(
        "Agent\ub97c \uac1c\ubc1c\ud55c\ub2e4"
    )

    assert prepared.requires_model_translation is True
