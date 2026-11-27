#!/usr/bin/env python3
"""Join the improved subset E2E results with the old-code full-run baseline.

Usage: python tools/compare_e2e.py [improved_json] [baseline_json]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

improved_path = sys.argv[1] if len(sys.argv) > 1 else \
    "ucsbench_output/results_qwen_improved_subset.json"
baseline_path = sys.argv[2] if len(sys.argv) > 2 else \
    "/data/ywang/my_projects/VideoUnderstanding/ucsbench_output/results_qwen.json"

imp = json.load(open(improved_path))
base_list = json.load(open(baseline_path))["details"]
base = {r["qid"]: r for r in base_list}

rows = []
for r in imp["details"]:
    qid = r.get("qid")
    b = base.get(str(qid))
    if b is None:
        continue
    rows.append({
        "qid": qid,
        "category": r.get("category") or b.get("category"),
        "b_ok": bool(b.get("is_correct")),
        "n_ok": bool(r.get("is_correct")),
        "n_kf": r.get("num_keyframes", 0) or 0,
        "n_graph_kf": r.get("num_graph_keyframes", 0) or 0,
        "n_fallback_kf": r.get("num_fallback_keyframes", 0) or 0,
        "zero_retrieval": bool(r.get("zero_retrieval")),
        "b_zero_kf": (b.get("num_keyframes") or 0) == 0,
        "b_matched": b.get("matched_count", 0) or 0,
    })

n = len(rows)
b_acc = sum(r["b_ok"] for r in rows) / n
n_acc = sum(r["n_ok"] for r in rows) / n
print(f"joined questions: {n}")
print(f"baseline: {b_acc:.1%}   improved: {n_acc:.1%}   delta: {100*(n_acc-b_acc):+.1f}pp")

print("\n-- by category --")
cats = defaultdict(lambda: [0, 0, 0])
for r in rows:
    c = cats[r["category"]]
    c[0] += int(r["b_ok"]); c[1] += int(r["n_ok"]); c[2] += 1
for cat, (bk, nk, cn) in sorted(cats.items(), key=lambda kv: -kv[1][2]):
    print(f"  {cat[:52]:52s} {bk/cn:.1%} -> {nk/cn:.1%}  ({100*(nk-bk)/cn:+.1f}pp, n={cn})")

print("\n-- by question slice --")
slices = {
    "baseline zero-frame q (was blind)": lambda r: r["b_zero_kf"],
    "baseline had frames": lambda r: not r["b_zero_kf"],
    "improved used fallback frames": lambda r: r["n_fallback_kf"] > 0,
    "improved zero-retrieval": lambda r: r["zero_retrieval"],
    "improved graph frames only": lambda r: r["n_fallback_kf"] == 0 and not r["zero_retrieval"],
}
for name, fn in slices.items():
    sel = [r for r in rows if fn(r)]
    if not sel:
        print(f"  {name}: n=0")
        continue
    bk = sum(r["b_ok"] for r in sel) / len(sel)
    nk = sum(r["n_ok"] for r in sel) / len(sel)
    print(f"  {name:42s} n={len(sel):4d}  {bk:.1%} -> {nk:.1%}  ({100*(nk-bk):+.1f}pp)")

up = sum(1 for r in rows if r["n_ok"] and not r["b_ok"])
down = sum(1 for r in rows if not r["n_ok"] and r["b_ok"])
print(f"\nfixed: {up}   broken: {down}")

# Position/orientation deep-dive: where did it regress?
print("\n-- position/orientation breakdown --")
po = [r for r in rows if "Position" in (r["category"] or "")]
for name, fn in slices.items():
    sel = [r for r in po if fn(r)]
    if not sel:
        continue
    bk = sum(r["b_ok"] for r in sel) / len(sel)
    nk = sum(r["n_ok"] for r in sel) / len(sel)
    print(f"  {name:42s} n={len(sel):4d}  {bk:.1%} -> {nk:.1%}  ({100*(nk-bk):+.1f}pp)")
