#!/usr/bin/env python3
r"""Dump JoyAI-Echo's config (from the checkpoint's safetensors HEADER only — KB,
no weights) to a small json the loader nodes read. Run once.

  python dump_joyai_config.py --src D:\JoyAI-Echo-release.safetensors --out D:\joyai_parts\joyai_echo_config.json
"""
import argparse, json, os
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", default=None)
a = ap.parse_args()
out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.src)), "joyai_echo_config.json")
with safe_open(a.src, framework="pt") as f:
    meta = f.metadata() or {}
if "config" not in meta:
    raise SystemExit("No 'config' key in checkpoint metadata.")
cfg = json.loads(meta["config"])
with open(out, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
print(f"wrote {out}  ({len(cfg)} top-level keys: {', '.join(list(cfg)[:8])})")
