"""Tests for the prompt builders (free-form and multiple-choice)."""

from __future__ import annotations

from directme.geometry.poses import SE3
from directme.qa.prompts import (
    DirectMePromptBuilder,
    MultipleChoicePromptBuilder,
)
from directme.retrieval.query_parser import QueryIntent
from directme.retrieval.retriever import RetrievedContext


def _make_context(language: str = "en", question: str = "Where is the cup?") -> RetrievedContext:
    return RetrievedContext(
        question=question,
        intent=QueryIntent(raw_query=question, language=language),
        current_pose=SE3.identity(),
    )


class TestDirectMePromptBuilder:
    def test_build_returns_system_and_parts(self):
        builder = DirectMePromptBuilder()
        system, parts = builder.build(_make_context("en"))
        assert isinstance(system, str)
        assert len(system) > 50
        assert any(p["type"] == "text" for p in parts)

    def test_english_system_prompt(self):
        system, _ = DirectMePromptBuilder().build(_make_context("en"))
        assert "egocentric" in system.lower()

    def test_chinese_system_prompt(self):
        system, _ = DirectMePromptBuilder().build(
            _make_context("zh", question="杯子在哪？")
        )
        assert "第一人称" in system

    def test_legacy_text_prompt(self):
        text = DirectMePromptBuilder().build_text_prompt(_make_context("en"))
        assert "Where is the cup?" in text
        assert "scene graph" in text.lower()


class TestMultipleChoicePromptBuilder:
    def test_build_includes_options(self):
        builder = MultipleChoicePromptBuilder()
        ctx = _make_context("en")
        system, parts = builder.build(
            ctx, options=["Left", "Right", "Front", "Behind", "Above"]
        )
        text = parts[0]["text"]
        assert "A. Left" in text
        assert "E. Above" in text
        assert "ONLY the option letter" in text

    def test_build_chinese(self):
        builder = MultipleChoicePromptBuilder()
        ctx = _make_context("zh", question="杯子在哪？")
        system, parts = builder.build(ctx, options=["左边", "右边", "前面"])
        text = parts[0]["text"]
        assert "A. 左边" in text
        assert "仅回复选项字母" in text

    def test_parse_answer_single_letter(self):
        assert MultipleChoicePromptBuilder.parse_answer("A") == "A"
        assert MultipleChoicePromptBuilder.parse_answer("B") == "B"
        assert MultipleChoicePromptBuilder.parse_answer(" C ") == "C"

    def test_parse_answer_with_period(self):
        assert MultipleChoicePromptBuilder.parse_answer("A.") == "A"
        assert MultipleChoicePromptBuilder.parse_answer("C. In front") == "C"
        assert MultipleChoicePromptBuilder.parse_answer("B)") == "B"

    def test_parse_answer_natural_language(self):
        assert MultipleChoicePromptBuilder.parse_answer("The answer is D") == "D"
        assert MultipleChoicePromptBuilder.parse_answer("答案是 C") == "C"
        assert MultipleChoicePromptBuilder.parse_answer("I choose B") == "B"
        assert MultipleChoicePromptBuilder.parse_answer("选A") == "A"

    def test_parse_answer_bracketed(self):
        assert MultipleChoicePromptBuilder.parse_answer("(A)") == "A"
        assert MultipleChoicePromptBuilder.parse_answer("[B]") == "B"

    def test_parse_answer_invalid(self):
        assert MultipleChoicePromptBuilder.parse_answer("I don't know") is None
        assert MultipleChoicePromptBuilder.parse_answer("") is None
        assert MultipleChoicePromptBuilder.parse_answer("Maybe left or right") is None
        assert MultipleChoicePromptBuilder.parse_answer("Not sure about this") is None
