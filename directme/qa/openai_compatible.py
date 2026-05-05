"""OpenAI-compatible chat completion generators.

Two flavors are provided. Both target endpoints that speak the OpenAI
``/v1/chat/completions`` schema (OpenAI itself, vLLM, Together, DeepInfra,
Ollama with ``--openai-compat``, etc.):

* :class:`OpenAICompatibleGenerator`  — text-only. Sends the rendered scene-
  graph summary as a single user message. Cheapest and most portable.
* :class:`OpenAICompatibleMultimodalGenerator` — sends the scene-graph
  summary *plus* the supporting keyframes as ``image_url`` content parts.
  Targets Qwen-VL, GPT-4o, InternVL, etc., served via an OpenAI-compatible
  multimodal endpoint.

Both classes lazy-import the ``openai`` package so the core DirectMe install
stays light.
"""

from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path

from directme.qa.generator import AnswerGenerator
from directme.qa.prompts import DirectMePromptBuilder
from directme.retrieval.retriever import RetrievedContext


def _image_to_data_url(path: str | Path) -> str | None:
    """Read a local image file as a base64 ``data:`` URL, or return None."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    mime, _ = mimetypes.guess_type(str(p))
    if mime is None or not mime.startswith("image/"):
        # Toy demo keyframes are .txt placeholders; skip non-image files.
        return None
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


@dataclass
class OpenAICompatibleGenerator(AnswerGenerator):
    """Text-only generator for OpenAI-compatible chat completion endpoints.

    Requires::

        pip install directme[vlm]

    Sends only the rendered graph summary. To include images as well, use
    :class:`OpenAICompatibleMultimodalGenerator`.
    """

    model: str
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.0

    def answer(self, context: RetrievedContext) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "Install with `pip install directme[vlm]` to use OpenAICompatibleGenerator."
            ) from exc

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        prompt = DirectMePromptBuilder().build_text_prompt(context)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""


@dataclass
class OpenAICompatibleMultimodalGenerator(AnswerGenerator):
    """Multimodal generator for OpenAI-compatible vision endpoints.

    Sends the rendered graph summary as a text content part PLUS each
    supporting keyframe as an ``image_url`` content part. Local files are
    inlined as base64 ``data:`` URLs; remote URLs are passed through.

    This matches the message format documented for OpenAI's vision API and
    accepted by Qwen-VL / InternVL-served behind an OpenAI-compatible
    server (e.g. vLLM with ``--served-model-name``).

    Args:
        model: served model name (e.g. ``"qwen2.5-vl-72b-instruct"``).
        base_url: endpoint URL. Defaults to OpenAI if ``None``.
        api_key: bearer token. Read from ``OPENAI_API_KEY`` if ``None``.
        max_images: cap how many keyframes to attach (default 4) to avoid
            blowing the context window when the retrieved subgraph is large.
        temperature: sampling temperature; defaults to deterministic.
        extra_image_urls: extra remote image URLs to always attach (e.g.
            a global "you are wearing this camera" reference shot).
    """

    model: str
    base_url: str | None = None
    api_key: str | None = None
    max_images: int = 4
    temperature: float = 0.0
    extra_image_urls: list[str] = field(default_factory=list)

    def answer(self, context: RetrievedContext) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "Install with `pip install directme[vlm]` to use "
                "OpenAICompatibleMultimodalGenerator."
            ) from exc

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        text_prompt = DirectMePromptBuilder().build_text_prompt(context)

        content_parts: list[dict] = [{"type": "text", "text": text_prompt}]

        for path in context.keyframes[: self.max_images]:
            url = _image_to_data_url(path)
            if url is None:
                continue
            content_parts.append({"type": "image_url", "image_url": {"url": url}})
        for url in self.extra_image_urls:
            content_parts.append({"type": "image_url", "image_url": {"url": url}})

        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content_parts}],
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""
