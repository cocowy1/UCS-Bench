# UCS-Bench VLM Baseline Evaluation

This folder contains the evaluation scripts for the **VLM baseline rows** of
UCS-Bench (the "uniform" and "stream" frame-sampling protocols with K = 64
frames). They are the scripts that produced the baseline numbers reported in
the paper, ported from our internal experiment setup with light cleanup.

They are **not** for evaluating DirectMe itself — for that, use
[`directme/eval/eval_ucsbench.py`](../eval_ucsbench.py) (pipeline-based
evaluation) or [`examples/evaluate_ucsbench_vlm.py`](../../../examples/evaluate_ucsbench_vlm.py).

## Contents

| File | Models | Notes |
|---|---|---|
| `eval_qwen_vl.py` | Qwen2.5-VL (7B/32B), Qwen3-VL (8B/32B) | transformers + `qwen_vl_utils` |
| `eval_internvl.py` | InternVL3 (8B/38B), InternVL3.5 (8B/38B) | `trust_remote_code` + dynamic tiling |
| `eval_cambrian.py` | Cambrian-S | requires the Cambrian-S codebase |
| `eval_vgllm.py` | VG-LLM (8B, QA-finetuned) | requires the VG-LLM codebase; `streaming_window` mode |
| `eval_api_vlm.py` | GPT-5, DeepSeek-V3, Doubao-seed-1.6V, Gemini-3-Flash, ... | OpenAI-compatible endpoint, up to 50 images |
| `eval_internvideo25.py` | InternVideo2.5-Chat-8B | InternVL-style `chat` interface |
| `eval_llava_video.py` | LLaVA-Video-7B | requires the LLaVA-NeXT codebase (`--llava_repo`) |
| `eval_vilasr.py` | ViLaSR-7B | Qwen2.5-VL based; interleaved frame-index captions |
| `eval_spatial_mllm.py` | Spatial-MLLM (Qwen2.5-VL based) | fine-tuned `Spatial-MLLM-subset-sft` |
| `eval_sensenova.py` | SenseNova-SI-1.1-InternVL3-8B | InternVL-style tiling with `max_num=6` |
| `extract_frames_1fps.py` | — | helper to build the 1 fps frame dump |
| `envs/` | — | conda environment exports used in our runs |

Each model family keeps its **own self-contained script** on purpose: the
three inference stacks are different, and a few protocol details (sampling
rounding, stream window, prompt ending, answer parsing) intentionally differ
per model. Unifying them into one code path was exactly how the first public
release lost reproducibility, so duplication here is deliberate.

## 1. Data preparation

Download the dataset as described in the main README (Hugging Face
`cocowy1/UCS-Bench` → `data/UCS-Bench/`). Its layout is:

```text
data/UCS-Bench/
├── QAs/                        # per-video MC QA files: <video_uid>-mcq.json
├── *.mp4                       # videos at the repository root
└── metadata.jsonl
```

### Frames (1 fps dump)

All scripts read pre-extracted frames at 1 fps. Build them from the
downloaded videos (this takes a while; ~170 h of video):

```bash
python extract_frames_1fps.py \
    --video_root data/UCS-Bench \
    --out_dir data/UCS-Bench/frames_1fps
```

This produces the expected layout, consumed via `--frames_dir
data/UCS-Bench/frames_1fps` (the scripts' default):

```
<frames_dir>/<video_uid>/frame_000000.png
<frames_dir>/<video_uid>/frame_000001.png
...
<frames_dir>/<video_uid>/index.jsonl
```

`index.jsonl` has one JSON object per line; both relative and absolute paths
are accepted, and several key aliases work (`sec`/`timestamp_sec`/... for the
timestamp, `file`/`path`/... for the image):

```json
{"file": "frame_000000.png", "sec": 0}
```

### QA files (UCS-Bench MC format)

`--qa_dir` (default `data/UCS-Bench/QAs`) points to a directory of per-video
JSON files, each a list of:

```json
{
  "video_uid": "002d2729-df71-438d-8396-5895b349e8fd",
  "qid": "002d2729-..._0",
  "question": "Where is the cup?",
  "options": {"A": "...", "B": "...", "C": "...", "D": "...", "E": "..."},
  "answer_label": "C",
  "qtype": "...", "category": "...", "subcategory": "...",
  "question_timestamps": "00:04:12"
}
```

## 2. Environments

Our runs used two conda environments; `--from-history` exports are included:

```bash
conda env create -f envs/video_chat_env.yml -n video_chat   # Qwen + InternVL
conda env create -f envs/qwen_vl_env.yml   -n qwen_vl       # alternative Qwen env
pip install opencv-python                                    # needed by the OpenCV-based scripts
```

For Cambrian, additionally clone the [Cambrian-S repository](https://github.com/Cambrian-VLM/Cambrian-S)
and either `pip install -e` it or pass its path via `--cambrian_repo`.

Select the GPU(s) with `CUDA_VISIBLE_DEVICES`; the 32B/38B checkpoints load
with `device_map="auto"` and typically need more than one GPU.

## 3. Running

Common flags:

- `--frame_select_mode uniform` — sample K frames uniformly over `[0, t_q]`.
- `--frame_select_mode clip_window` — the "stream" protocol: the most recent
  frames within `[t_q - clip_window_sec, t_q]` (see the table below; despite
  the legacy name, **no CLIP model is used** in any reported result).
- `--num_samples 64` — K, the frame budget.
- `--qa_dir / --frames_dir / --output`.

### Qwen2.5-VL / Qwen3-VL (`video_chat` env)

```bash
# Qwen2.5-VL-32B, uniform (note: --max_image_size 448 was used in our run)
CUDA_VISIBLE_DEVICES=0 python eval_qwen_vl.py \
    --model_path /path/to/Qwen2.5-VL-32B-Instruct \
    --qa_dir /path/to/MC_QAs --frames_dir /path/to/data_frames_1fps \
    --output results/qwen2.5-vl-32b/64_uniform.json \
    --num_samples 64 --frame_select_mode uniform --max_image_size 448

# Qwen2.5-VL-32B, stream (max_image_size left at default 512)
CUDA_VISIBLE_DEVICES=0 python eval_qwen_vl.py \
    --model_path /path/to/Qwen2.5-VL-32B-Instruct \
    --qa_dir /path/to/MC_QAs --frames_dir /path/to/data_frames_1fps \
    --output results/qwen2.5-vl-32b/64_stream.json \
    --num_samples 64 --frame_select_mode clip_window

# Qwen3-VL-32B: same commands with /path/to/Qwen3-VL-32B-Instruct
```

### InternVL3 / InternVL3.5 (`video_chat` env)

```bash
# InternVL3-38B, uniform
CUDA_VISIBLE_DEVICES=1 python eval_internvl.py \
    --model_path /path/to/InternVL3-38B-Instruct \
    --qa_dir /path/to/MC_QAs --frames_dir /path/to/data_frames_1fps \
    --output results/internvl3-38b/64_uniform.json \
    --num_samples 64 --frame_select_mode uniform

# InternVL3-38B, stream
CUDA_VISIBLE_DEVICES=1 python eval_internvl.py \
    --model_path /path/to/InternVL3-38B-Instruct \
    --qa_dir /path/to/MC_QAs --frames_dir /path/to/data_frames_1fps \
    --output results/internvl3-38b/64_stream.json \
    --num_samples 64 --frame_select_mode clip_window

# InternVL3.5-38B: same commands with /path/to/InternVL3_5-38B-Instruct
```

`--max_image_size` is accepted but has **no effect** for InternVL: frames are
fed at native resolution into the dynamic tiling (`input_size=448,
max_num=1`). It is kept so the original commands run unmodified.

### Cambrian-S (`cambrian` env)

```bash
# Cambrian-S-7B, uniform
CUDA_VISIBLE_DEVICES=2 python eval_cambrian.py \
    --model_path /path/to/Cambrian-S-7B --cambrian_repo /path/to/cambrian-s \
    --qa_dir /path/to/MC_QAs --frames_dir /path/to/data_frames_1fps \
    --output results/cambrian-s-7b/64_uniform.json \
    --num_samples 64 --frame_select_mode uniform

# Cambrian-S-7B, stream (window default is 64 s for Cambrian; see table)
CUDA_VISIBLE_DEVICES=2 python eval_cambrian.py \
    --model_path /path/to/Cambrian-S-7B --cambrian_repo /path/to/cambrian-s \
    --qa_dir /path/to/MC_QAs --frames_dir /path/to/data_frames_1fps \
    --output results/cambrian-s-7b/64_stream.json \
    --num_samples 64 --frame_select_mode clip_window \
    --resume_file results/cambrian-s-7b/64_stream_partial.json   # optional
```

`--resume_file` skips QIDs already present in a previous (partial) result
file and carries their statistics over — useful for interrupted runs.

### Proprietary API models (`eval_api_vlm.py`, no GPU needed)

GPT-5, DeepSeek-V3, Doubao-seed-1.6V, Gemini-3-Flash etc. are evaluated
through an **OpenAI-compatible chat.completions endpoint** with up to K=50
images per question. Point `--base_url` at your endpoint and pass
`--api_key` (or set `OPENAI_API_KEY`):

```bash
export OPENAI_API_KEY=sk-...

# GPT-5, uniform 50 frames
python eval_api_vlm.py \
    --base_url https://your-endpoint/v1 --model gpt-5 \
    --qa_dir /path/to/MC_QAs --frames_dir /path/to/data_frames_1fps \
    --output results/gpt5/50_uniform.json \
    --num_samples 50 --frame_select_mode uniform

# stream variant: most recent 50 frames ending at the question frame
python eval_api_vlm.py \
    --base_url https://your-endpoint/v1 --model gpt-5 \
    --qa_dir /path/to/MC_QAs --frames_dir /path/to/data_frames_1fps \
    --output results/gpt5/50_stream.json \
    --num_samples 50 --frame_select_mode clip_window --clip_window_sec 50
```

Swap `--model` (and `--base_url`) for the other API models. Failed requests
are retried up to `--max_retries` times and counted as incorrect if they
keep failing. `--disable_thinking` (default on) sends the
`thinking: disabled` extra body that the proxies used in our runs required;
use `--no-disable_thinking` if your endpoint rejects it.

### VG-LLM (`eval_vgllm.py`)

VG-LLM needs its repository (for the
`Qwen2_5_VLForConditionalGenerationWithVGGT` model class) and the
QA-finetuned checkpoint (e.g. `vgllm-qa-vggt-8b`):

```bash
# streaming window (the mode behind the paper's VG-LLM row)
CUDA_VISIBLE_DEVICES=0 python eval_vgllm.py \
    --model_path /path/to/vgllm-qa-vggt-8b \
    --vgllm_repo /path/to/VG-LLM \
    --qa_dir /path/to/MC_QAs --frames_dir /path/to/data_frames_1fps \
    --output results/vgllm/64_streaming_window.json \
    --num_samples 64 --frame_select_mode streaming_window
```

Frames are resized to max(H, W) <= 412 and cropped to the processor's
patch/merge grid. If the official `vg_llm` package is not installed, the
script falls back to built-in vision helpers — **this fallback is the path
our reported run took**. Single-GPU loading (`device_map={"": 0}`),
flash-attention-2 by default (`--attn_impl sdpa` to avoid flash-attn).

### Other open-source baselines

InternVideo2.5, LLaVA-Video, ViLaSR, Spatial-MLLM and SenseNova-SI follow
the same conventions. Their reported rows use the stream protocol
(`--frame_select_mode clip_window`); see the tables below for per-script
defaults and quirks. Examples:

```bash
# LLaVA-Video-7B (needs the LLaVA-NeXT repo)
CUDA_VISIBLE_DEVICES=0 python eval_llava_video.py \
    --model_path /path/to/LLaVA-Video-7B --llava_repo /path/to/LLaVA-NeXT \
    --qa_dir data/UCS-Bench/QAs --frames_dir data/UCS-Bench/frames_1fps \
    --output results/llava-video/64_stream.json \
    --num_samples 64 --frame_select_mode clip_window

# ViLaSR-7B
CUDA_VISIBLE_DEVICES=0 python eval_vilasr.py \
    --model_path /path/to/ViLaSR-7B \
    --output results/vilasr/64_stream.json \
    --num_samples 64 --frame_select_mode clip_window

# Spatial-MLLM
CUDA_VISIBLE_DEVICES=0 python eval_spatial_mllm.py \
    --model_path /path/to/Spatial-MLLM-subset-sft \
    --output results/spatial-mllm/64_stream.json \
    --num_samples 64 --frame_select_mode clip_window

# InternVideo2.5-Chat-8B
CUDA_VISIBLE_DEVICES=0 python eval_internvideo25.py \
    --model_path /path/to/InternVideo2_5_Chat_8B \
    --output results/internvideo25/64_stream.json \
    --num_samples 64 --frame_select_mode clip_window

# SenseNova-SI-1.1-InternVL3-8B (uniform default, as in the original run)
CUDA_VISIBLE_DEVICES=0 python eval_sensenova.py \
    --model_path /path/to/SenseNova-SI-1.1-InternVL3-8B \
    --output results/sensenova-si/results_1fps.json
```

All five accept `--resume_file` (dedup by qid) and abort on a question with
no decodable frames (ViLaSR / Spatial-MLLM / InternVideo2.5) or skip it
(LLaVA-Video) — matching the original runs.

Each script prints running accuracy per video and writes a JSON report with
overall / per-qtype / per-category / per-subcategory accuracy and per-question
details.

## 4. Protocol reference

Shared by all models: 1 fps frames; K = `--num_samples`; greedy decoding
(`do_sample=False, temperature=0, max_new_tokens=32`); seed 42; questions
sorted by timestamp per video; a question whose inference raises an exception
is counted as **incorrect** (stays in the denominator).

| | `eval_qwen_vl.py` | `eval_internvl.py` | `eval_cambrian.py` | `eval_api_vlm.py` | `eval_vgllm.py` |
|---|---|---|---|---|---|
| uniform sampling | `linspace(0, t_q, K)`, `floor()` to index, **dedup** → may send < K frames | same as Qwen | `linspace(0, t_q, K)`, `round()`, **no dedup**, pad by repeating last index → always exactly K frames | same as Qwen | same as Qwen |
| stream mode | most recent K valid frames in `[t_q − W, t_q]` (`clip_window`) | same as Qwen | all valid frames in the window, keep first K, pad with repeats of the last frame | most recent K valid frames in `[t_q − W, t_q]`, **last frame forced to be the question frame** | seconds `[q−K+1, ..., q]` exactly (`streaming_window`), optional left-pad with frame 0 |
| stream window W (default) | 300 s | 300 s | **64 s** | **50 s** | = K (64 s) |
| K (default) | 32 (paper rows: 64) | 32 (paper rows: 64) | 64 | **50** (API image cap) | 64 |
| frame resize | `max(H, W) ≤ --max_image_size` (default 512) | no pre-resize (dynamic tiling 448, 1 tile/frame) | `max(H, W) ≤ --max_image_size` (default 512) | `max(H, W) ≤ 512`, JPEG q90 → base64 data URL | `max(H, W) ≤ 412` + crop to patch/merge grid |
| prompt ending | "Only output the letter of the best option ... only one best option." | same as Qwen, plus an `Image-N: <image>` token block prefix | "Keep answer short. Output format: answer is ()." | same as Qwen + a system message | same as Qwen (sampling description in parentheses) |
| answer parsing | `Answer: X` → `(X)` → `The answer is X` → first A–E char → raw | same as Qwen | same, but falls back to `"A"` instead of raw | same as Qwen | same as Qwen |
| failed inference | not counted (skipped) if no frames decode; exception ⇒ incorrect | same | same | retried up to 6×; still failing ⇒ incorrect | same as Qwen |

Protocol details of the remaining five scripts (all: greedy decoding, seed
42, OpenCV frame loading with INTER_LINEAR resize, multi-pattern answer
parsing, `--resume_file` support, and **no per-question exception catch** —
an inference error aborts the run):

| | `eval_internvideo25.py` | `eval_llava_video.py` | `eval_vilasr.py` | `eval_spatial_mllm.py` | `eval_sensenova.py` |
|---|---|---|---|---|---|
| base stack | InternVL-style `chat`, tiling 448×1 tile | LLaVA-NeXT `llava_qwen`, one packed video tensor, conv `qwen_1_5` | Qwen2.5-VL `generate` | Qwen2.5-VL `generate` | InternVL-style `chat`, tiling 448×**6** tiles |
| uniform sampling | `linspace(0, t_q, K)`, `floor()`, dedup | same | same | same | same |
| stream mode | last K valid frames in `[t_q−W, t_q]`, precomputed | same, precomputed | same, precomputed | same, precomputed | same (per-question) |
| stream window W (default) | 300 s | 64 s | 64 s | 64 s | 300 s |
| frame resize | cv2, `max(H,W) ≤ 512` | cv2, `max(H,W) ≤ 512` | cv2, `max(H,W) ≤ 512` | cv2, `max(H,W) ≤ 512` | none (native into tiling) |
| prompt style | egocentric-QA prompt, "selected by stream ..." + trailing `Output format: answer is :...` | LLaVA time-instruction ("...frames are uniformly sampled from it..." — also in stream mode, kept verbatim) + question, "Answer with the option letter only." | egocentric-QA prompt + trailing `Output format: answer is :...`, frames interleaved with index captions | egocentric-QA prompt + trailing `Output format: answer is :...` | classic egocentric-QA prompt ("selected by CLIP within the last N seconds") |
| max_new_tokens | 32 | 200 | 200 | 200 | 32 |
| default mode / K | `clip_window` / 64 | `clip_window` / 64 | `clip_window` / 64 | `clip_window` / 64 | `uniform` / 32 |
| no-frame question | aborts | skips | aborts | aborts | skips |

## 5. Reproducibility notes

These quirks are preserved on purpose so the scripts reproduce the original
numbers; do not "fix" them without re-running everything:

1. **`clip_window` never uses CLIP.** The mode name and the prompt sentence
   "selected by CLIP within the last N seconds" are historical. In every
   reported run the selection was simply the most recent valid frames in the
   window. The prompt text is kept byte-identical because changing it changes
   model outputs.
2. **The stream window default differs per model** (64 s for Cambrian vs.
   300 s for Qwen/InternVL). Pass `--clip_window_sec` explicitly if in doubt.
3. **InternVL ignores `--max_image_size`** (frames are not pre-resized); the
   flag exists only for command compatibility.
4. **Reported reference numbers** below were obtained with
   `--qa_dir MC_QAs_v15_norm` (7,404 questions). Re-running on a different
   QA-set version will produce different numbers.

## 6. Reference results

From the runs behind the paper's baseline table (uniform / stream, K = 64,
QA set `MC_QAs_v15_norm`, 7,404 questions):

| Model | uniform | stream |
|---|---|---|
| Cambrian-S-7B | 43.26 | 42.44 |
| Qwen2.5-VL-32B-Instruct | 47.29 | 47.41 |
| Qwen3-VL-32B-Instruct | 48.88 | 49.30 |
| InternVL3.5-38B-Instruct | 48.00 | 48.20 |
| InternVL3-38B-Instruct | 49.91 | 50.27 |

The smaller variants (Qwen2.5-VL-7B, Qwen3-VL-8B, InternVL3-8B,
InternVL3.5-8B) run through the same two scripts with a different
`--model_path`.

Other rows of the paper's main table produced with the scripts in this
folder (overall accuracy, from the paper):

| Model | Protocol | Script | Overall |
|---|---|---|---|
| DeepSeek-V3 | API, up to 50 images | `eval_api_vlm.py` | 35.6 |
| Spatial-MLLM (3B) | stream, K=64 | `eval_spatial_mllm.py` | 34.3 |
| Gemini-3-Flash | API, up to 50 images | `eval_api_vlm.py` | 42.2 |
| GPT-5 | API, up to 50 images | `eval_api_vlm.py` | 43.7 |
| Doubao-seed-1.6V | API, up to 50 images | `eval_api_vlm.py` | 47.3 |
| VG-LLM (8B, finetuned) | `streaming_window`, K=64 | `eval_vgllm.py` | 38.3 |
| ViLaSR-7B | stream, K=64 | `eval_vilasr.py` | 40.1 |
| LLaVA-Video-7B | stream, K=64 | `eval_llava_video.py` | 41.7 |
| InternVideo2.5-8B | stream, K=64 | `eval_internvideo25.py` | 43.8 |
| SenseNova-SI-8B | uniform | `eval_sensenova.py` | 37.8 |
