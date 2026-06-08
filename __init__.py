"""ComfyUI nodes for JoyAI-Echo + Rebels low-VRAM loaders.

- nodes.py            : original JoyAI-Echo nodes (TextEncode / Generate / SingleShot / etc.)
- rebels_gguf_dit.py  : GGUF DiT loader that builds JoyAI's REAL architecture (-> MODEL)
- rebels_loaders.py   : optional discrete loaders (Config / TextEncoder / VAE / Assemble)
"""

import sys
from pathlib import Path

_NODE_ROOT = Path(__file__).resolve().parent
_LIBS = str(_NODE_ROOT / "libs")
if _LIBS not in sys.path:
    sys.path.insert(0, _LIBS)

# --- original JoyAI-Echo nodes ---
from .nodes import (
    JoyEcho_ModelLoader,
    JoyEcho_TextEncode,
    JoyEcho_Generate,
    JoyEcho_SingleShotGenerate,
    JoyEcho_PromptFormat,
    JoyEcho_LLMEnhance,
    JoyEcho_PromptAtIndex,
)

NODE_CLASS_MAPPINGS = {
    "JoyEcho_ModelLoader": JoyEcho_ModelLoader,
    "JoyEcho_TextEncode": JoyEcho_TextEncode,
    "JoyEcho_Generate": JoyEcho_Generate,
    "JoyEcho_SingleShotGenerate": JoyEcho_SingleShotGenerate,
    "JoyEcho_PromptFormat": JoyEcho_PromptFormat,
    "JoyEcho_LLMEnhance": JoyEcho_LLMEnhance,
    "JoyEcho_PromptAtIndex": JoyEcho_PromptAtIndex,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "JoyEcho_ModelLoader": "JoyEcho Model Loader",
    "JoyEcho_TextEncode": "JoyEcho Text Encode",
    "JoyEcho_Generate": "JoyEcho Generate (Multi-Shot)",
    "JoyEcho_SingleShotGenerate": "JoyEcho Single Shot Generate",
    "JoyEcho_PromptFormat": "JoyEcho Prompt Format (Helper)",
    "JoyEcho_LLMEnhance": "JoyEcho LLM Enhance",
    "JoyEcho_PromptAtIndex": "JoyEcho Prompt At Index",
}

# --- Rebels GGUF DiT loader (the real-arch replacement for UnetLoaderGGUF) ---
from .rebels_gguf_dit import (
    NODE_CLASS_MAPPINGS as _GG_C,
    NODE_DISPLAY_NAME_MAPPINGS as _GG_D,
)
NODE_CLASS_MAPPINGS.update(_GG_C)
NODE_DISPLAY_NAME_MAPPINGS.update(_GG_D)

# --- Rebels discrete loaders (optional; only if the file is present) ---
try:
    from .rebels_loaders import (
        NODE_CLASS_MAPPINGS as _RL_C,
        NODE_DISPLAY_NAME_MAPPINGS as _RL_D,
    )
    NODE_CLASS_MAPPINGS.update(_RL_C)
    NODE_DISPLAY_NAME_MAPPINGS.update(_RL_D)
except Exception as e:
    print(f"[Rebels] rebels_loaders not loaded ({e}); GGUF DiT loader still active.", flush=True)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
