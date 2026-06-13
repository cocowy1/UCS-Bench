from __future__ import annotations

import re as _re
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from directme.qa.prompts import MultipleChoicePromptBuilder
from PIL import Image

from directme.config import DirectMeConfig
from directme.geometry.poses import SE3
from directme.mapping.offline_engine import OfflineMappingEngine
from directme.mapping.scene_graph import SceneGraph
from directme.perception.base import PerceptionBackend, VideoFrame
from directme.perception.artifacts import PerceptionArtifactBackend
from directme.perception.ingest import iter_frames_from_paths, iter_frames_from_video
from directme.qa.generator import AnswerGenerator, RuleBasedAnswerGenerator
from directme.retrieval.retriever import GraphRetriever, RetrievedContext


# ---------------------------------------------------------------------------
# 本地 VLM 生成器基类
# ---------------------------------------------------------------------------
def _infer_model_device(model) -> str:
    try:
        return next(model.parameters()).device
    except Exception:
        return "cuda"


class LocalVLMGenerator(AnswerGenerator):
    """本地多模态大模型生成器的抽象基类。"""

    @abstractmethod
    def answer_multimodal(self, system_prompt: str, text: str, images: list) -> str:
        raise NotImplementedError

    def answer(self, context: RetrievedContext) -> str:
        from directme.qa.prompts import DirectMePromptBuilder
        text = DirectMePromptBuilder().build_text_prompt(context)
        return self.answer_multimodal(system_prompt="", text=text, images=[])


# ---------------------------------------------------------------------------
# Qwen3-VL 生成器
# ---------------------------------------------------------------------------

@dataclass
class QwenVLGenerator(LocalVLMGenerator):
    """Qwen3-VL 本地推理生成器。

    使用 transformers Qwen3VLForConditionalGeneration +
    processor.apply_chat_template() 处理多图 + 文本。
    """

    model_path: str
    device_map: str = "auto"
    max_new_tokens: int = 32
    max_image_size: int = 512
    _model: Any = field(default=None, init=False, repr=False)
    _processor: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._load()

    def _load(self) -> None:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        print(f"[QwenVL] 加载模型：{self.model_path}")
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map,
        ).eval()
        self._processor = AutoProcessor.from_pretrained(self.model_path)
        print("[QwenVL] 模型加载完成。")

    @staticmethod
    def _resize(image, max_size: int):
        w, h = image.size
        if max(w, h) > max_size:
            s = max_size / max(w, h)
            image = image.resize((int(w * s), int(h * s)), resample=3)
        return image

    def answer_multimodal(self, system_prompt: str, text: str, images: list) -> str:
        import torch

        images_resized = [self._resize(img.convert("RGB"), self.max_image_size) for img in images]

        user_content: list[dict] = [
            {"type": "image", "image": img} for img in images_resized
        ]
        user_content.append({"type": "text", "text": text})

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        messages.append({"role": "user", "content": user_content})

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = inputs.to(_infer_model_device(self._model))
                   
        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        return self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]


# ---------------------------------------------------------------------------
# InternVL3 生成器
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


def _internvl_pixel_values(images: list, image_size: int = 448, dtype=None, device=None):
    import torch
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),
        T.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])
    tensors = [transform(img) for img in images]
    pv = torch.stack(tensors)
    if dtype is not None:
        pv = pv.to(dtype=dtype)
    if device is not None:
        pv = pv.to(device=device)
    return pv


@dataclass
class InternVLGenerator(LocalVLMGenerator):
    """InternVL3 本地推理生成器。"""

    model_path: str
    device_map: str = "auto"
    max_new_tokens: int = 32
    image_size: int = 448
    _model: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._load()
    def _load(self) -> None:
        import torch
        import sys
        import os
        from pathlib import Path
        from transformers import AutoModel, AutoTokenizer

        print(f"[InternVL] 开始加载模型（最终强解路径模式）：{self.model_path}")

        # =====================================================================
        # 核心必杀技：通过物理符号链接（Symlink）强行劫持并修复 HF 损坏的缓存路径
        # =====================================================================
        try:
            # 1. 锁死报错中提示的绝对目标缓存目录
            target_cache_dir = Path("/data/ywang/hf_modules_cache/transformers_modules/InternVL3_hyphen_8B/235c253589d1ed24")
            local_model_dir = Path(self.model_path).resolve()

            if not target_cache_dir.exists():
                print(f"[InternVL] 检测到悬空缓存路径不存在，正在手动建立物理软链接...")
                # 创建父级目录
                target_cache_dir.parent.mkdir(parents=True, exist_ok=True)
                # 将 HF 锁死的缓存目录直接软链接到你的本地完整 ckpt 目录
                os.symlink(str(local_model_dir), str(target_cache_dir), target_is_directory=True)
                print(f"[InternVL] 软链接建立成功: {target_cache_dir} -> {local_model_dir}")
            else:
                # 如果它是个残缺的物理目录（导致文件缺失），删掉它并重构为完整的软链接
                if not (target_cache_dir / "configuration_intern_vit.py").exists():
                    print(f"[InternVL] 检测到残缺缓存目录，正在强行修正为全量软链接...")
                    import shutil
                    if target_cache_dir.is_symlink() or target_cache_dir.is_file():
                        os.unlink(target_cache_dir)
                    else:
                        shutil.rmtree(target_cache_dir)
                    os.symlink(str(local_model_dir), str(target_cache_dir), target_is_directory=True)
                    print(f"[InternVL] 修正完成。")
        except Exception as sym_err:
            print(f"[WARN] 建立路径劫持锁时发生异常（可忽略）: {sym_err}")
        # =====================================================================

        # 2. 注入全局离线变量，确保不产生任何外部网络握手
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

        abs_model_path = os.path.abspath(self.model_path)
        if abs_model_path not in sys.path:
            sys.path.insert(0, abs_model_path)

        # 3. 正常加载模型（移除了 local_files_only 防止触发 HF 固有的只读死锁）
        self._model = AutoModel.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval().cuda()

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=False,
        )

        print("[InternVL] 模型加载成功！")


    def answer_multimodal(self, system_prompt: str, text: str, images: list) -> str:
        import torch

        generation_config = dict(max_new_tokens=self.max_new_tokens, do_sample=False)

        if images:
            device = _infer_model_device(self._model)
            image_tokens = "<image>\n" * len(images)
            question = image_tokens + text
            pv = _internvl_pixel_values(
                images,
                image_size=self.image_size,
                dtype=torch.bfloat16,
                device=device,
            )
        else:
            question = text
            pv = None

        response = self._model.chat(
            self._tokenizer,
            pv,
            question,
            generation_config,
            history=None,
            return_history=False,
        )
        return response


# ---------------------------------------------------------------------------
# COUNT 题确定性计数辅助
# ---------------------------------------------------------------------------

_NUMBER_WORDS_EN = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}
_NUMBER_WORDS_ZH = {
    "零": 0, "一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10, "十一": 11, "十二": 12,
    "十三": 13, "十四": 14, "十五": 15,
}


def _extract_count_from_option(text: str) -> int | None:
    """从选项文本中提取整数（阿拉伯数字 > 英文单词 > 中文单词）。"""
    if not text:
        return None
    # 阿拉伯数字
    m = _re.search(r"\d+", text)
    if m:
        return int(m.group(0))
    lower = text.lower()
    # 英文数字单词
    for word, val in _NUMBER_WORDS_EN.items():
        if _re.search(rf"\b{word}\b", lower):
            return val
    # 中文数字
    for word, val in _NUMBER_WORDS_ZH.items():
        if word in text:
            return val
    return None


def _map_count_to_option(count: int, options: list[str], labels: str) -> str | None:
    """将整数 count 映射到数字最接近的选项字母。

    UCS-Bench 的 COUNT 题每个选项含一个数字，例如：
      A. There are six sheets ...   → 6
      B. There are seven sheets ... → 7  ← GT
    选 |opt_n - count| 最小的字母。
    """
    best_label, best_diff = None, None
    for label, opt_text in zip(labels, options):
        opt_n = _extract_count_from_option(opt_text)
        if opt_n is None:
            continue
        diff = abs(opt_n - count)
        if best_diff is None or diff < best_diff:
            best_diff, best_label = diff, label
    return best_label


# ---------------------------------------------------------------------------
# DirectMe 编排器
# ---------------------------------------------------------------------------

@dataclass
class DirectMe:
    """DirectMe 高层编排器。

    创建方式
    --------
    dm = DirectMe.with_empty_graph()                    # 规则生成器
    dm = DirectMe.with_qwen(model_path="...")           # Qwen3-VL
    dm = DirectMe.with_internvl(model_path="...")       # InternVL3

    问答接口
    --------
    dm.answer(question, current_pose)
    raw, label = dm.answer_mc(question, options, current_pose,
                               qtype="COUNT", as_of_timestamp=3.0)
    """

    config: DirectMeConfig
    graph: SceneGraph
    generator: AnswerGenerator

    # ── 工厂方法 ──────────────────────────────────────────────────────────────

    @classmethod
    def with_empty_graph(
        cls,
        config: DirectMeConfig | None = None,
        generator: AnswerGenerator | None = None,
    ) -> "DirectMe":
        config = config or DirectMeConfig()
        graph = SceneGraph(
            reference_frame=config.world.reference_frame,
            merge_threshold_m=config.mapping.merge_threshold_m,
        )
        return cls(config=config, graph=graph, generator=generator or RuleBasedAnswerGenerator())

    @classmethod
    def with_qwen(
        cls,
        model_path: str,
        device_map: str = "auto",
        max_new_tokens: int = 32,
        max_image_size: int = 512,
        config: DirectMeConfig | None = None,
    ) -> "DirectMe":
        config = config or DirectMeConfig()
        graph = SceneGraph(
            reference_frame=config.world.reference_frame,
            merge_threshold_m=config.mapping.merge_threshold_m,
        )
        generator = QwenVLGenerator(
            model_path=model_path,
            device_map=device_map,
            max_new_tokens=max_new_tokens,
            max_image_size=max_image_size,
        )
        return cls(config=config, graph=graph, generator=generator)

    @classmethod
    def with_internvl(
        cls,
        model_path: str,
        device_map: str = "auto",
        max_new_tokens: int = 32,
        image_size: int = 448,
        config: DirectMeConfig | None = None,
    ) -> "DirectMe":
        config = config or DirectMeConfig()
        graph = SceneGraph(
            reference_frame=config.world.reference_frame,
            merge_threshold_m=config.mapping.merge_threshold_m,
        )
        generator = InternVLGenerator(
            model_path=model_path,
            device_map=device_map,
            max_new_tokens=max_new_tokens,
            image_size=image_size,
        )
        return cls(config=config, graph=graph, generator=generator)

    # ── 建图 ─────────────────────────────────────────────────────────────────

    def build_memory(
        self,
        frames,
        backend: PerceptionBackend,
        *,
        chunk_size: int | None = None,
        artifact_dir: str | Path | None = None,
        artifact_video_fps: float | None = None,
    ) -> SceneGraph:
        """Build the scene graph from perception outputs in offline-incremental chunks.

        ``frames`` can be either a list or a generator of ``VideoFrame``. When
        ``chunk_size=60``, the perception backend receives exactly 60 sampled
        frames per call, except the last partial chunk. If ``artifact_dir`` is
        provided, depth maps and detection/tracking overlays are saved per chunk
        and exported as videos by ``PerceptionArtifactBackend``.
        """
        if artifact_dir is not None and not isinstance(backend, PerceptionArtifactBackend):
            backend = PerceptionArtifactBackend(
                backend=backend,
                artifact_dir=artifact_dir,
                video_fps=artifact_video_fps or self.config.stream.fps,
            )
        engine = OfflineMappingEngine(backend=backend, config=self.config, graph=self.graph)
        engine.process_frames(frames, chunk_size=chunk_size)
        self.graph = engine.graph or self.graph
        return self.graph

    def build_memory_from_video(
        self,
        video_path: str | Path,
        backend: PerceptionBackend,
        *,
        target_fps: float = 1.0,
        chunk_size: int = 60,
        frame_dump_dir: str | Path | None = None,
        artifact_dir: str | Path | None = None,
        max_frames: int | None = None,
    ) -> SceneGraph:
        """Decode a video and build DirectMe memory from perception, end-to-end."""
        self.config.stream.fps = target_fps
        self.config.stream.chunk_size_frames = chunk_size
        if frame_dump_dir is None:
            frame_dump_dir = Path(self.config.run_dir) / "frames"
        frames = iter_frames_from_video(
            video_path,
            target_fps=target_fps,
            frame_dump_dir=frame_dump_dir,
            max_frames=max_frames,
        )
        return self.build_memory(
            frames,
            backend,
            chunk_size=chunk_size,
            artifact_dir=artifact_dir,
            artifact_video_fps=target_fps,
        )

    def build_memory_from_frames(
        self,
        image_paths: list[str | Path],
        backend: PerceptionBackend,
        *,
        fps: float = 1.0,
        chunk_size: int = 60,
        artifact_dir: str | Path | None = None,
        max_frames: int | None = None,
    ) -> SceneGraph:
        """Build DirectMe memory from pre-extracted frames and perception."""
        self.config.stream.fps = fps
        self.config.stream.chunk_size_frames = chunk_size
        if max_frames is not None:
            image_paths = image_paths[:max_frames]
        frames = iter_frames_from_paths(image_paths, fps=fps)
        return self.build_memory(
            frames,
            backend,
            chunk_size=chunk_size,
            artifact_dir=artifact_dir,
            artifact_video_fps=fps,
        )

    # ── 检索 ─────────────────────────────────────────────────────────────────

    def _retriever(self) -> GraphRetriever:
        return GraphRetriever(
            self.graph,
            reachable_radius_m=self.config.retrieval.reachable_radius_m,
            lateral_tolerance_ratio=self.config.retrieval.lateral_tolerance_ratio,
        )

    def retrieve(
        self,
        question: str,
        current_pose: SE3,
        language: str | None = None,
        as_of_timestamp: float | None = None,   # 改动 C：透传给 retriever
    ) -> RetrievedContext:
        return self._retriever().retrieve(
            question=question,
            current_pose=current_pose,
            top_k=self.config.retrieval.top_k,
            language=language or self.config.retrieval.language,
            as_of_timestamp=as_of_timestamp,
        )

    # ── 自由文本问答 ──────────────────────────────────────────────────────────

    def answer(
        self,
        question: str,
        current_pose: SE3,
        language: str | None = None,
        as_of_timestamp: float | None = None,
    ) -> str:
        context = self.retrieve(
            question, current_pose, language=language,
            as_of_timestamp=as_of_timestamp,
        )
        return self.generator.answer(context)

    # ── 多选题问答（UCS-Bench 核心路径） ─────────────────────────────────────

    def answer_mc(
        self,
        question: str,
        options: list[str],
        current_pose: SE3,
        language: str | None = None,
        option_labels: str = "ABCDE",
        qtype: str | None = None,               # 改动 C：COUNT 题走确定性计数
        as_of_timestamp: float | None = None,   # 改动 C：在线 QA 时间戳过滤
    ) -> tuple[str, str | None]:
        """多选题问答。

        路径分发
        --------
        1. COUNT 且目标类别稳定 → deterministic scene-graph counting
        2. COUNT 且目标类别易过分裂 → VLM visual counting
        3. RuleBasedAnswerGenerator → Jaccard 降级匹配
        4. 本地 VLM → MC 提示 + keyframe 图像

        Parameters
        ----------
        qtype            : UCS-Bench QA type，如 "COUNT" / "BINARY" / "OTHER"
        as_of_timestamp  : 查询时刻（秒），仅检索此前已观测到的节点
        """


        ctx = self.retrieve(
            question, current_pose, language=language,
            as_of_timestamp=as_of_timestamp,
        )
        labels = option_labels[: len(options)]

        is_count = _is_count_question(qtype, question, ctx)
        is_noisy_count = is_count and _is_noisy_count_target(question, ctx)

        # ── COUNT 题：只对安全类别走 deterministic count ────────────────────
        if is_count and ctx.count > 0:
            if not is_noisy_count:
                count = int(ctx.count)
                predicted = _map_count_to_option(count, options, labels)
                if predicted is not None:
                    raw = f"[deterministic count] scene_graph_count={count}"
                    return raw, predicted
                # 如果无法从选项中抽取数字，不 return，继续走 VLM
        # ─────────────────────────────────────────────────────────────────────

        if isinstance(self.generator, RuleBasedAnswerGenerator):
            raw = self.generator.answer(ctx)
            predicted = _match_option_by_text(raw, dict(zip(labels, options)))
            return raw, predicted

        max_kf = 6 if is_count else 4

        system_prompt, parts = MultipleChoicePromptBuilder(max_keyframes=max_kf).build(
            ctx, options, labels
        )
        text = next((p["text"] for p in parts if p["type"] == "text"), "")

        if is_noisy_count:
            text = _re.sub(
                r"Matched physical count:.*?\n",
                "Matched graph candidate nodes: unreliable for this counting question.\n",
                text,
                flags=_re.IGNORECASE,
            )
            text += (
                "\n\nImportant: For this counting question, the scene-graph node "
                "count may be inflated due to duplicate or split detections. "
                "Do NOT rely on the graph count. Instead, count the actual "
                "distinct physical object instances visible in the keyframe "
                "images and select the closest option letter."
            )

        images: list = []
        for part in parts:
            if part["type"] == "image":
                img_path = Path(part.get("path", ""))
                if img_path.exists():
                    with Image.open(img_path) as im:
                        images.append(im.convert("RGB"))

        assert isinstance(self.generator, LocalVLMGenerator), (
            f"answer_mc() 需要 LocalVLMGenerator，当前为 {type(self.generator)}"
        )
        raw = self.generator.answer_multimodal(system_prompt, text, images)
        predicted = MultipleChoicePromptBuilder.parse_answer(raw, labels)
        return raw, predicted


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------

def _match_option_by_text(answer: str, options: dict[str, str]) -> str | None:
    """Jaccard 降级：将生成答案映射到最近选项字母。"""
    def _tok(s: str) -> set[str]:
        return set(_re.findall(r"[a-z\u4e00-\u9fff]+", s.lower()))

    ans_tok = _tok(answer)
    if not ans_tok:
        return None

    best_label, best_score = None, -1.0
    for label, text in options.items():
        opt_tok = _tok(text)
        if not opt_tok:
            continue
        score = len(ans_tok & opt_tok) / len(ans_tok | opt_tok)
        if score > best_score:
            best_label, best_score = label, score

    return best_label if best_score > 0 else None

def _is_count_question(qtype: str | None, question: str, ctx: Any | None = None) -> bool:
    """判断是否为计数题。不要只依赖 qtype 字段。"""
    if qtype and str(qtype).strip().upper() == "COUNT":
        return True

    q = question.lower()
    if any(x in q for x in ["how many", "number of", "count", "多少", "几个", "几张", "数量"]):
        return True

    if ctx is not None:
        intent = getattr(ctx, "intent", None)
        if intent is not None and getattr(intent, "wants_count", False):
            return True

    return False


def _is_noisy_count_target(question: str, ctx: Any) -> bool:
    """判断 COUNT 题是否不适合直接用 scene graph node count。

    以下类别容易在建图时被过分裂（一个真实物体 → 多个 node），
    导致 ctx.count 系统性偏大，应退回 VLM 视觉计数：

    * paper / sheet / page / poster：感知端常被识别为 Picture/Frame，
      且同一张纸在多帧中反复被检测，fusion 难以完全去重；
    * picture / photo / frame / wall-mounted：与 paper 同属高分裂风险；
    * on the wall：语义约束（仅数墙面物体）不在当前建图中显式建模。

    安全类别（cup、chair、bottle 等实体感强的单体物品）仍走 deterministic count。
    """
    q = question.lower()

    # 关键词命中：问题文本中出现这些词则视为 noisy
    _NOISY_WORDS = {
        "paper", "papers", "sheet", "sheets", "page", "pages",
        "poster", "posters", "picture", "pictures", "photo", "photos",
        "frame", "frames", "painting", "paintings", "artwork",
        "on the wall", "wall",
        "纸", "纸张", "页", "海报", "照片", "图片", "画框", "相框",
        "墙上", "贴", "挂",
    }
    if any(w in q for w in _NOISY_WORDS):
        return True

    # 匹配到的节点标签中含 picture / frame → 高分裂风险
    matched_labels = [str(x).lower() for x in getattr(ctx, "total_matched_labels", [])]
    if any("picture" in x or "frame" in x for x in matched_labels):
        return True

    return False
