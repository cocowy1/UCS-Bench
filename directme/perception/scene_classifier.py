"""Scene (landmark) classifiers and tagging utilities.

This module defines a pluggable interface for assigning a coarse
"scene" or "landmark" tag to each frame in the video. A scene tag
identifies which type of physical place a frame likely belongs to
(e.g., ``"kitchen"``, ``"living_room"``, etc.) and is meant to aid
analytics such as place induction or timeline summarization. The
default implementation is a lightweight rule-based classifier that
matches object detection labels against a predefined set of
keywords. This baseline requires no additional dependencies and
introduces minimal overhead, but it can be replaced by more
sophisticated models such as CLIP or Qwen3‑VL when available.

Classes
-------
SceneClassifier
    Abstract base class for all scene classifiers.

RuleBasedSceneClassifier
    A simple classifier based on keyword matching of object labels.

QwenSceneClassifier
    Optional classifier using the Qwen3‑VL model via Transformers. This
    class gracefully degrades to a rule-based classifier if the
    required dependencies are unavailable.

Functions
---------
infer_scene_tag
    Public API to infer a scene tag given a list of object labels.

create_scene_classifier
    Factory function that instantiates a classifier from a string name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


class SceneClassifier:
    """Abstract base class for coarse scene/landmark classifiers.

    Concrete subclasses should implement the ``__call__`` method to
    accept a frame image and/or detections and return a string tag.
    """

    def __call__(self, image: np.ndarray | None, labels: Iterable[str]) -> str:
        """Compute a scene tag from the given inputs.

        Parameters
        ----------
        image : np.ndarray | None
            The RGB frame as a NumPy array or ``None`` if unavailable.
        labels : Iterable[str]
            Iterable of object class labels detected in the frame.

        Returns
        -------
        str
            A lowercase scene tag, e.g. ``"kitchen"``, or ``"unknown"``.
        """
        raise NotImplementedError


@dataclass
class RuleBasedSceneClassifier(SceneClassifier):
    """A simple keyword-based scene classifier.

    This classifier matches object detection labels against a static
    mapping of room types to keyword lists. The first matching room
    determines the tag; if none match, ``"unknown"`` is returned. See
    ``room_keywords`` for the default mapping. Users can override the
    keyword dictionary by passing a custom mapping.
    """

    room_keywords: Mapping[str, Sequence[str]] | None = None

    def __post_init__(self) -> None:
        # Default keywords if none provided.
        if self.room_keywords is None:
            self.room_keywords = {
                "kitchen": [
                    "fridge",
                    "refrigerator",
                    "microwave",
                    "oven",
                    "stove",
                    "sink",
                    "pan",
                    "pot",
                    "cutting board",
                    "spoon",
                    "fork",
                    "knife",
                    "bowl",
                ],
                "living_room": [
                    "sofa",
                    "couch",
                    "tv",
                    "television",
                    "remote",
                    "coffee table",
                    "chair",
                    "bookshelf",
                    "lamp",
                ],
                "bedroom": [
                    "bed",
                    "pillow",
                    "blanket",
                    "nightstand",
                    "dresser",
                    "wardrobe",
                    "closet",
                ],
                "bathroom": [
                    "toilet",
                    "sink",
                    "bathtub",
                    "shower",
                    "towel",
                    "toothbrush",
                ],
                "office": [
                    "desk",
                    "computer",
                    "keyboard",
                    "monitor",
                    "mouse",
                    "printer",
                    "laptop",
                    "notebook",
                ],
            }

    def __call__(self, image: np.ndarray | None, labels: Iterable[str]) -> str:
        labels_lower = [str(lbl).lower() for lbl in labels]
        for scene, keywords in self.room_keywords.items():
            for kw in keywords:
                if any(kw in label for label in labels_lower):
                    return scene
        return "unknown"


class QwenSceneClassifier(SceneClassifier):
    """Scene classifier using the Qwen3‑VL model (optional dependency).

    This classifier attempts to load the Qwen3‑VL model via the
    HuggingFace Transformers library. It processes the frame image as a
    chat-style prompt and asks the model to summarize the scene in one
    word. If the ``transformers`` or Qwen model weights are not
    installed, the classifier falls back to a rule-based classifier.

    Note: Running Qwen3‑VL requires significant computational resources
    and is not recommended in constrained environments. See the Qwen
    model card for details【646616952943204†L80-L85】.
    """

    def __init__(self, fallback: SceneClassifier | None = None) -> None:
        self._fallback = fallback or RuleBasedSceneClassifier()
        self._model = None
        self._processor = None

    def _load_qwen(self) -> bool:
        if self._model is not None:
            return True
        try:
            from transformers import (
                Qwen3VLForConditionalGeneration,  # type: ignore
                AutoProcessor,  # type: ignore
            )
        except Exception:
            return False
        try:
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen3-VL-2B-Instruct", dtype="auto", device_map="auto"
            )
            self._processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen3-VL-2B-Instruct"
            )
            return True
        except Exception:
            self._model = None
            self._processor = None
            return False

    def __call__(self, image: np.ndarray | None, labels: Iterable[str]) -> str:
        if image is not None and self._load_qwen():
            from PIL import Image
            pil_img = Image.fromarray(image)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {
                            "type": "text",
                            "text": "Identify the type of room depicted in this image (kitchen, living room, bedroom, bathroom, office, hallway, outdoor, etc.) using a single lowercase word.",
                        },
                    ],
                }
            ]
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self._model.device)
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
            )
            output_ids = generated_ids[0, inputs.input_ids.shape[1] :]
            text = self._processor.decode(
                output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            ).strip().lower()
            tag = text.split()[0] if text else "unknown"
            return tag
        return self._fallback(image, labels)


def infer_scene_tag(object_labels: Iterable[str], image: np.ndarray | None = None) -> str:
    """Public API for scene tag inference.

    Parameters
    ----------
    object_labels : Iterable[str]
        Detected object class labels.
    image : np.ndarray | None, optional
        The frame image as a NumPy array. If provided and a Qwen classifier
        is available, the classifier may use it for richer prediction.

    Returns
    -------
    str
        A scene tag string.
    """
    classifier = RuleBasedSceneClassifier()
    return classifier(image, object_labels)


def create_scene_classifier(name: str | None = None) -> SceneClassifier:
    """Create a scene classifier from a name.

    Parameters
    ----------
    name : str | None
        The classifier name. Supported values:
          - ``"rule"`` or ``None``: use :class:`RuleBasedSceneClassifier`.
          - ``"qwen"``: use :class:`QwenSceneClassifier` with rule-based fallback.

    Returns
    -------
    SceneClassifier
        An instance of the requested classifier.
    """
    if name is None or name.lower() in {"rule", "rules", "default"}:
        return RuleBasedSceneClassifier()
    if name.lower() == "qwen":
        return QwenSceneClassifier()
    raise ValueError(f"Unknown scene classifier name: {name}")