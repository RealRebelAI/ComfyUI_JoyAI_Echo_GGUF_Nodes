#!/usr/bin/env python3
r"""Dump JoyAI-Echo's architecture config from the checkpoint's safetensors HEADER
only (a few KB, no weights are read) to a small JSON that the loader nodes read.

Run this ONCE on the machine that has the full checkpoint. By default it writes to
the node pack's own  configs/joyai_echo_config.json  so you can ship that JSON with
the pack and your users never need the 46 GB checkpoint.

  python dump_joyai_config.py --src <path-to>/JoyAI-Echo-release.safetensors

To write somewhere else:
  python dump_joyai_config.py --src <path-to>/JoyAI-Echo-release.safetensors --out <path-to>/config.json
"""
import argparse, json, os
from safetensors import safe_open

# Default output: <this script's folder>/configs/joyai_echo_config.json
_NODE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_OUT = os.path.join(_NODE_DIR, "configs", "joyai_echo_config.json")

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="Path to JoyAI-Echo-release.safetensors")
ap.add_argument("--out", default=_DEFAULT_OUT,
                help="Output JSON path (default: <node>/configs/joyai_echo_config.json)")
a = ap.parse_args()

with safe_open(a.src, framework="pt") as f:
    meta = f.metadata() or {}
if "config" not in meta:
    raise SystemExit("No 'config' key in checkpoint metadata — wrong file?")
cfg = json.loads(meta["config"])

os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
with open(a.out, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
print(f"wrote {a.out}  ({len(cfg)} top-level keys: {', '.join(list(cfg)[:8])})")
print("Ship this JSON inside the node pack's configs/ folder; the nodes default to it.")
