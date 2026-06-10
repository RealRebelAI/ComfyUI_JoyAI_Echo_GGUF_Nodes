"""ComfyUI nodes for JoyAI-Echo: minute-level multi-shot audio-video generation.

Registers:
  - the 7 upstream JoyEcho nodes (nodes.py)
  - the discrete Rebels loader nodes (rebels_loaders.py)
  - the Rebels staged single-node pipeline for 16GB RAM (rebels_staged.py)

The libs/ folder MUST be on sys.path before anything imports ltx_core /
ltx_distillation, so that setup happens first.
"""

import sys
from pathlib import Path

_NODE_ROOT = Path(__file__).resolve().parent
_LIBS = str(_NODE_ROOT / "libs")

if _LIBS not in sys.path:
    sys.path.insert(0, _LIBS)

# ---------------------------------------------------------------- base maps
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# ---------------------------------------------------------------- upstream JoyEcho nodes
from .nodes import (
    JoyEcho_ModelLoader,
    JoyEcho_TextEncode,
    JoyEcho_Generate,
    JoyEcho_SingleShotGenerate,
    JoyEcho_PromptFormat,
    JoyEcho_LLMEnhance,
    JoyEcho_PromptAtIndex,
)

NODE_CLASS_MAPPINGS.update({
    "JoyEcho_ModelLoader": JoyEcho_ModelLoader,
    "JoyEcho_TextEncode": JoyEcho_TextEncode,
    "JoyEcho_Generate": JoyEcho_Generate,
    "JoyEcho_SingleShotGenerate": JoyEcho_SingleShotGenerate,
    "JoyEcho_PromptFormat": JoyEcho_PromptFormat,
    "JoyEcho_LLMEnhance": JoyEcho_LLMEnhance,
    "JoyEcho_PromptAtIndex": JoyEcho_PromptAtIndex,
})
NODE_DISPLAY_NAME_MAPPINGS.update({
    "JoyEcho_ModelLoader": "JoyEcho Model Loader",
    "JoyEcho_TextEncode": "JoyEcho Text Encode",
    "JoyEcho_Generate": "JoyEcho Generate (Multi-Shot)",
    "JoyEcho_SingleShotGenerate": "JoyEcho Single Shot Generate",
    "JoyEcho_PromptFormat": "JoyEcho Prompt Format (Helper)",
    "JoyEcho_LLMEnhance": "JoyEcho LLM Enhance",
    "JoyEcho_PromptAtIndex": "JoyEcho Prompt At Index",
})

# ---------------------------------------------------------------- Rebels discrete loaders
try:
    from .rebels_loaders import (
        NODE_CLASS_MAPPINGS as _RL_CM,
        NODE_DISPLAY_NAME_MAPPINGS as _RL_DM,
    )
    NODE_CLASS_MAPPINGS.update(_RL_CM)
    NODE_DISPLAY_NAME_MAPPINGS.update(_RL_DM)
except Exception as e:
    print(f"[Rebels JE] rebels_loaders failed to load: {e!r}", flush=True)

# ---------------------------------------------------------------- Rebels staged pipeline (16GB)
try:
    from .rebels_staged import (
        NODE_CLASS_MAPPINGS as _ST_CM,
        NODE_DISPLAY_NAME_MAPPINGS as _ST_DM,
    )
    NODE_CLASS_MAPPINGS.update(_ST_CM)
    NODE_DISPLAY_NAME_MAPPINGS.update(_ST_DM)
except Exception as e:
    print(f"[Rebels JE] rebels_staged failed to load: {e!r}", flush=True)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
