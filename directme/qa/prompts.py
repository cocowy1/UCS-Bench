"""Prompt builders for DirectMe → MLLM integration.

Two prompt styles are provided:

* :class:`DirectMePromptBuilder` — free-form QA. Returns a text prompt with
  the scene-graph summary, supporting keyframes, and the user question.
* :class:`MultipleChoicePromptBuilder` — UCS-Bench style. Wraps the free-form
  prompt with the multiple-choice options (A–E) and instructs the MLLM to
  output a single option letter.

Both builders are **model-agnostic**: they produce a structured payload
``(system_prompt, user_content_parts)`` that can be converted to OpenAI chat
messages, HuggingFace Transformers ``processor.apply_chat_template``, or any
other chat format.

Usage::

    from directme.qa.prompts import MultipleChoicePromptBuilder
    builder = MultipleChoicePromptBuilder()
    system, parts = builder.build(context, options=["A. ...", "B. ...", ...])
    # parts is a list of dicts: [{"type": "text", ...}, {"type": "image", ...}]
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from directme.retrieval.retriever import GraphRetriever, RetrievedContext


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_EN = """\
You are an egocentric spatial QA assistant. You observe the world through a \
first-person wearable camera. A pose-anchored scene-graph memory has been \
built incrementally from the video stream and is provided below. The graph \
stores objects in a shared world coordinate frame; at question time, each \
object is projected into the current camera frame to compute its egocentric \
relation (front / behind / left / right / front_left / front_right / \
behind_left / behind_right), Euclidean distance, and reachability.

Use ONLY the provided graph summary and supporting keyframes to answer. \
Spatial descriptions such as "to your left" or "behind you" refer to the \
camera wearer's current orientation."""

SYSTEM_PROMPT_ZH = """\
你是一个第一人称空间问答助手。你通过佩戴式摄像头观察世界。一个基于视频流增量构建的\
姿态锚定场景图记忆已提供如下。场景图以统一的世界坐标系存储物体；在提问时刻，每个物\
体被投影到当前相机坐标系以计算其自我中心方位关系（正前方 / 正后方 / 左侧 / 右侧 / \
左前方 / 右前方 / 左后方 / 右后方）、距离和可达性。

请仅依据提供的场景图摘要和关键帧图像来回答。空间描述（如"在您的左侧"）以相机佩戴\
者当前朝向为参考。"""


# ---------------------------------------------------------------------------
# Free-form prompt builder
# ---------------------------------------------------------------------------


@dataclass
class DirectMePromptBuilder:
    """Builds a DirectMe-style multimodal prompt payload.

    The returned structure is model-agnostic. VLM-specific clients can convert
    it to OpenAI, Transformers, vLLM, or custom chat templates.
    """

    max_keyframes: int = 4

    def build_text_prompt(self, context: RetrievedContext) -> str:
        """Legacy text-only prompt (backward compatible)."""
        graph_summary = GraphRetriever.render_summary(context)
        keyframes = "\n".join(f"- {p}" for p in context.keyframes) or "- none"
        return f"""You are an egocentric video QA assistant.

Use only the provided pose-anchored scene graph summary and supporting keyframes.
The graph stores objects in a world coordinate frame and renders them into the
current camera frame at question time.

{graph_summary}

Supporting keyframes:
{keyframes}

Question:
{context.question}

Answer with a concise, spatially grounded response."""

    def build(
        self, context: RetrievedContext
    ) -> tuple[str, list[dict[str, Any]]]:
        """Build a structured prompt payload.

        Returns:
            A tuple of ``(system_prompt, content_parts)`` where
            ``content_parts`` is a list of dicts with ``type`` in
            ``{"text", "image"}``.  Image parts carry a ``path`` key
            pointing to the local file.
        """
        lang = context.intent.language if context.intent else "en"
        system = SYSTEM_PROMPT_ZH if lang.startswith("zh") else SYSTEM_PROMPT_EN

        graph_summary = GraphRetriever.render_summary(context)

        text = f"""{graph_summary}

Question:
{context.question}

Answer with a concise, spatially grounded response."""

        parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for kf_path in context.keyframes[: self.max_keyframes]:
            if kf_path and Path(kf_path).exists():
                parts.append({"type": "image", "path": kf_path})

        return system, parts


# ---------------------------------------------------------------------------
# Multiple-choice prompt builder (UCS-Bench evaluation)
# ---------------------------------------------------------------------------


MC_INSTRUCTION_EN = """\
Select the single best answer from the options below. \
Reply with ONLY the option letter (e.g. A). Do NOT add any explanation."""

MC_INSTRUCTION_ZH = """\
请从以下选项中选出唯一最佳答案。\
仅回复选项字母（例如 A）。不要添加任何解释。"""


@dataclass
class MultipleChoicePromptBuilder:
    """Build a multiple-choice QA prompt for UCS-Bench evaluation.

    Usage::

        builder = MultipleChoicePromptBuilder()
        system, parts = builder.build(
            context,
            options=["On your left", "Behind you", "In front of you", ...],
        )
    """

    max_keyframes: int = 4

    def build(
        self,
        context: RetrievedContext,
        options: list[str],
        option_labels: str = "ABCDE",
    ) -> tuple[str, list[dict[str, Any]]]:
        """Build a multiple-choice prompt payload.

        Args:
            context: the retrieved subgraph + keyframes.
            options: list of answer strings (without letter prefix).
            option_labels: option letters; default ``"ABCDE"`` for 5-way.

        Returns:
            ``(system_prompt, content_parts)`` ready for VLM consumption.
        """
        lang = context.intent.language if context.intent else "en"
        system = SYSTEM_PROMPT_ZH if lang.startswith("zh") else SYSTEM_PROMPT_EN

        graph_summary = GraphRetriever.render_summary(context)
        mc_instruction = MC_INSTRUCTION_ZH if lang.startswith("zh") else MC_INSTRUCTION_EN

        option_lines = "\n".join(
            f"{option_labels[i]}. {opt}" for i, opt in enumerate(options)
        )

        text = f"""{graph_summary}

Question:
{context.question}

{option_lines}

{mc_instruction}"""

        parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for kf_path in context.keyframes[: self.max_keyframes]:
            if kf_path and Path(kf_path).exists():
                parts.append({"type": "image", "path": kf_path})

        return system, parts

    @staticmethod
    def parse_answer(response: str, option_labels: str = "ABCDE") -> str | None:
        """Extract the chosen option letter from a VLM response.

        Handles common formats: ``"A"``, ``"A."``, ``"A. On your left"``,
        ``"The answer is A"``, ``"答案是 B"``, etc.

        Returns:
            The matched letter, or ``None`` if parsing fails.
        """
        text = response.strip()
        upper = text.upper()
        valid = set(option_labels.upper())

        if not text:
            return None

        # Direct single-letter answer (possibly with surrounding whitespace).
        if len(upper) == 1 and upper in valid:
            return upper

        # "A." / "A:" / "A)" / "A," at the very start.
        if len(upper) >= 2 and upper[0] in valid and upper[1] in ".):, ":
            return upper[0]

        # "The answer is A" / "I choose B" / "My answer: C" patterns.
        m = re.search(
            r"(?:ANSWER\s*(?:IS|:)|CHOOSE|SELECT|PICK|CHOSE)\s*([A-E])\b",
            upper,
        )
        if m and m.group(1) in valid:
            return m.group(1)

        # "答案是 A" / "答案为B" / "选A" patterns.
        m = re.search(r"(?:答案\s*(?:是|为|:)?|选择?)\s*([A-E])\b", text)
        if m and m.group(1).upper() in valid:
            return m.group(1).upper()

        # "(A)" / "[A]" standalone pattern.
        m = re.search(r"[\[\(]([A-E])[\]\)]", upper)
        if m and m.group(1) in valid:
            return m.group(1)

        # Strict fallback: option letter at the very beginning of a line.
        m = re.search(r"^([A-E])\b", upper, re.MULTILINE)
        if m and m.group(1) in valid:
            return m.group(1)

        return None
