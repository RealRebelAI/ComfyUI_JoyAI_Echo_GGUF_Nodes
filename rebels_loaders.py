"""
rebels_loaders.py — discrete ComfyUI loader nodes for JoyAI-Echo on low VRAM.

Houses YOUR pre-made files as separate nodes (nothing reads the 46GB checkpoint
weights). Mirrors a normal Comfy graph: a config provider, a UNet-style DiT GGUF
loader, a dual text-encoder (Gemma fp8 + connector), and a VAE loader -> assemble
-> feed the repo's existing Generate / SingleShot nodes.

WHY A CONFIG NODE: ltx_core's configurators build EMPTY networks (e.g. decoder_blocks=[])
unless given the model config dict, which normally lives in the checkpoint's
safetensors __metadata__["config"]. Your GGUF / extracted files don't carry it, so
one small shared config source feeds every builder. Point it at a dumped config.json
(no checkpoint at runtime) or, once, at the checkpoint itself (HEADER read only —
KB, never the weights).

Place in the repo root next to nodes.py and register in __init__.py (see bottom).

v1 — architecture verified against the repo; RAM-safe DiT path uses the Builder's
meta_model+module_ops seam so 41GB bf16 never materializes. The two seams to shake
out on the GPU are tagged `# VERIFY:` (DiT Linear->GGUFLinear swap; Gemma fp8 swap).
"""
from __future__ import annotations
import os, json
from dataclasses import replace
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gguf
from gguf import GGUFReader, GGMLQuantizationType as QT

from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
from ltx_core.loader.primitives import StateDict
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.model.transformer import LTXV_MODEL_COMFY_RENAMING_MAP, LTXModelConfigurator, X0Model
from ltx_core.model.video_vae import (VAE_DECODER_COMFY_KEYS_FILTER, VAE_ENCODER_COMFY_KEYS_FILTER,
                                      VideoDecoderConfigurator, VideoEncoderConfigurator)
from ltx_core.model.audio_vae import (AUDIO_VAE_DECODER_COMFY_KEYS_FILTER, AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
                                      VOCODER_COMFY_KEYS_FILTER, AudioDecoderConfigurator,
                                      AudioEncoderConfigurator, VocoderConfigurator)
from ltx_core.text_encoders.gemma import (EMBEDDINGS_PROCESSOR_KEY_OPS, EmbeddingsProcessorConfigurator)
from ltx_distillation.models.ltx_wrapper import LTX2DiffusionWrapper
from ltx_distillation.models.vae_wrapper import VideoVAEWrapper, AudioVAEWrapper
from ltx_distillation.models.text_encoder_wrapper import GemmaTextEncoderWrapper

CAT = "Rebels/JoyAI-Echo"

# ---------------------------------------------------------------- config
def _full_config(src: str) -> dict:
    src = src.strip().strip('"')
    if src.lower().endswith(".json"):
        with open(src, "r", encoding="utf-8") as f: return json.load(f)
    from safetensors import safe_open                       # checkpoint: HEADER read only
    with safe_open(src, framework="pt") as f:
        meta = f.metadata() or {}
    if "config" not in meta:
        raise ValueError(f"No 'config' in metadata of {src}. Point at the checkpoint or a config.json.")
    return json.loads(meta["config"])

class _CfgLoader(SafetensorsModelStateDictLoader):
    """Stock safetensors weight load, but metadata() returns the shared config.
       Includes dynamic key remapping for Comfy-Org Gemma3 single files."""
    def __init__(self, config: dict, map_gemma=False, *a, **k):
        super().__init__(*a, **k)
        self._cfg = config
        self._map_gemma = map_gemma
        
    def metadata(self, path): return self._cfg
    
    def load(self, paths, sd_ops=None, device=None):
        sd_obj = super().load(paths, sd_ops, device)
        
        # Intercept and remap keys for single-file Gemma3 intake
        if self._map_gemma:
            new_sd = {}
            for k, v in sd_obj.sd.items():
                # Strip external ComfyUI wrappers if they exist
                if k.startswith("cond_stage_model."): k = k[17:]
                elif k.startswith("text_model."): k = k[11:]
                elif k.startswith("text_encoder."): k = k[13:]
                
                # Map standard Gemma paths to Gemma3 Multimodal nested paths
                if k.startswith("model.embed_tokens"): k = k.replace("model.embed_tokens", "model.model.language_model.embed_tokens")
                elif k.startswith("model.layers"): k = k.replace("model.layers", "model.model.language_model.layers")
                elif k.startswith("model.norm"): k = k.replace("model.norm", "model.model.language_model.norm")
                
                new_sd[k] = v
            sd_obj.sd = new_sd
            
        return sd_obj

# ---------------------------------------------------------------- gguf dit
def _gguf_entries(path):
    r = GGUFReader(path); out = {}
    for t in r.tensors:
        out[t.name] = {"data": np.asarray(t.data), "qtype": t.tensor_type,
                       "shape": tuple(int(d) for d in reversed(t.shape))}
    return out

def _dequant(entry, dtype):
    deq = gguf.quants.dequantize(entry["data"], entry["qtype"]).astype(np.float32)
    return torch.from_numpy(deq.reshape(entry["shape"])).to(dtype)

class GGUFLinear(nn.Module):
    def __init__(self, entry, bias=None, compute_dtype=torch.bfloat16):
        super().__init__()
        self.qtype_value = int(entry["qtype"]); self.weight_shape = tuple(entry["shape"])
        self.register_buffer("qweight", torch.from_numpy(entry["data"].copy()))
        self.bias = nn.Parameter(bias.to(compute_dtype), requires_grad=False) if bias is not None else None
    def forward(self, x):
        raw = self.qweight.detach().cpu().numpy()
        w = torch.from_numpy(gguf.quants.dequantize(raw, QT(self.qtype_value)).astype(np.float32))
        w = w.reshape(self.weight_shape).to(device=x.device, dtype=x.dtype)
        return F.linear(x, w, self.bias.to(x.dtype) if self.bias is not None else None)

def _set_sub(root, dotted, new):
    *parents, leaf = dotted.split("."); p = root
    for a in parents: p = getattr(p, a)
    setattr(p, leaf, new)

class _GGUFDiTLoader(SafetensorsModelStateDictLoader):
    """metadata()->config; load()->only the NON-Linear tensors (dequant, small).
    Linear weights stay quantized inside GGUFLinear (swapped via module_ops)."""
    def __init__(self, config, entries, consumed, dtype):
        super().__init__(); self._cfg = config; self._e = entries; self._consumed = consumed; self._dt = dtype
    def metadata(self, path): return self._cfg
    def load(self, paths, sd_ops=None, device=None):
        sd, size = {}, 0
        for k, e in self._e.items():
            if k in self._consumed: continue            # big Linear weights -> GGUFLinear, skip
            t = _dequant(e, self._dt); sd[k] = t; size += t.numel() * t.element_size()
        return StateDict(sd=sd, device=device or torch.device("cpu"), size=size, dtype=self._dt)

def _dit_module_ops(entries, consumed, compute_dtype):
    def mutator(model):
        for name, mod in list(model.named_modules()):
            if isinstance(mod, nn.Linear) and (name + ".weight") in entries:   # VERIFY: DiT keys==JD names
                bk = name + ".bias"
                bias = _dequant(entries[bk], compute_dtype) if bk in entries else None
                _set_sub(model, name, GGUFLinear(entries[name + ".weight"], bias, compute_dtype))
                consumed.add(name + ".weight")
                if bk in entries: consumed.add(bk)
        return model
    return (ModuleOps(name="gguf_linear_swap", matcher=lambda m: True, mutator=mutator),)

# ---------------------------------------------------------------- gemma fp8
class Fp8Linear(nn.Module):
    def __init__(self, qweight_u8, shape, scale, bias=None, compute_dtype=torch.bfloat16):
        super().__init__(); self.weight_shape = tuple(shape)
        self.register_buffer("qweight", qweight_u8)
        self.register_buffer("scale_weight", torch.tensor(float(scale), dtype=torch.float32))
        self.bias = nn.Parameter(bias.to(compute_dtype), requires_grad=False) if bias is not None else None
    def forward(self, x):
        w = self.qweight.view(torch.float8_e4m3fn).reshape(self.weight_shape).to(torch.float32) * self.scale_weight
        w = w.to(device=x.device, dtype=x.dtype)
        return F.linear(x, w, self.bias.to(x.dtype) if self.bias is not None else None)

def _swap_gemma_fp8(model, fp8_dir, compute_dtype):   # VERIFY: our fp8 convention (<k> fp8 + <k>.scale_weight)
    from safetensors import safe_open
    from pathlib import Path
    shards = sorted(Path(fp8_dir).glob("model*.safetensors"))
    scales = {}
    for sh in shards:
        with safe_open(str(sh), framework="pt") as f:
            for k in f.keys():
                if k.endswith(".scale_weight"): scales[k[:-len(".scale_weight")]] = float(f.get_tensor(k))
    n = 0
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Linear) and name in scales:
            for sh in shards:
                with safe_open(str(sh), framework="pt") as f:
                    if name in f.keys():
                        qw = f.get_tensor(name); bias = mod.bias.detach() if mod.bias is not None else None
                        _set_sub(model, name, Fp8Linear(qw.view(torch.uint8), qw.shape, scales[name], bias, compute_dtype))
                        n += 1; break
    print(f"[Rebels] Gemma: swapped {n} Linears to Fp8Linear", flush=True)
    return model

# ================================================================ NODES
def _dev(): return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class RebelsJE_Config:
    CATEGORY = CAT; RETURN_TYPES = ("JOYECHO_CONFIG",); RETURN_NAMES = ("config",); FUNCTION = "run"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"config_source": ("STRING", {"default": r"D:\joyai_parts\joyai_echo_config.json",
            "tooltip": "config.json (preferred) OR the checkpoint (header read only, no weights)"})}}
    def run(self, config_source):
        cfg = _full_config(config_source)
        print(f"[Rebels] config loaded ({len(cfg)} top-level keys)", flush=True)
        return (cfg,)

class RebelsJE_DiTLoader:
    CATEGORY = CAT; RETURN_TYPES = ("JOYECHO_GENERATOR",); RETURN_NAMES = ("generator",); FUNCTION = "run"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "config": ("JOYECHO_CONFIG",),
            "dit_gguf": ("STRING", {"default": r"D:\joyai_echo_quant\JoyAI-Echo-DiT-Q4_K_M.gguf"}),
            "video_height": ("INT", {"default": 736}), "video_width": ("INT", {"default": 1280})}}
    def run(self, config, dit_gguf, video_height, video_width):
        if not os.path.exists(dit_gguf): raise FileNotFoundError(dit_gguf)
        dtype = torch.bfloat16
        entries = _gguf_entries(dit_gguf); consumed = set()
        builder = Builder(
            model_class_configurator=LTXModelConfigurator,
            model_path=dit_gguf,
            model_sd_ops=None,                       # DiT GGUF keys already match JD names
            module_ops=_dit_module_ops(entries, consumed, dtype),
            model_loader=_GGUFDiTLoader(config, entries, consumed, dtype),
        )
        transformer = builder.build(device=torch.device("cpu"), dtype=dtype)   # meta+swap, RAM-safe
        gen = LTX2DiffusionWrapper(model=X0Model(transformer), video_height=video_height, video_width=video_width)
        gen.eval()
        print("[Rebels] DiT generator ready (GGUF, no bf16 materialization)", flush=True)
        return (gen,)

class RebelsJE_TextEncoder:
    CATEGORY = CAT; RETURN_TYPES = ("JOYECHO_TEXTENC",); RETURN_NAMES = ("text_encoder",); FUNCTION = "run"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "config": ("JOYECHO_CONFIG",),
            "gemma_path": ("STRING", {"default": r"D:\gemma-3-12b-it-fp8"}),
            "gemma_format": (["our_fp8", "bf16"], {"default": "our_fp8"}),
            "connector_path": ("STRING", {"default": r"D:\joyai_parts\joyai_echo_embeddings_processor.safetensors"}),
            "low_vram": ("BOOLEAN", {"default": True})}}
    def run(self, config, gemma_path, gemma_format, connector_path, low_vram):
        from ltx_core.text_encoders.gemma import GemmaTextEncoderConfigurator, GEMMA_MODEL_OPS, module_ops_from_gemma_root
        from ltx_core.utils import find_matching_file
        import os
        from pathlib import Path
        
        dtype = torch.bfloat16
        dev = torch.device("cpu") if low_vram else _dev()
        
        # --- SINGLE FILE INTAKE PATCH START ---
        gemma_path_str = str(gemma_path)
        if os.path.isfile(gemma_path_str):
            print(f"\n[JoyAI-Echo] Single file intake engaged. Faking HF folder structure...", flush=True)
            parent_dir = os.path.dirname(gemma_path_str)
            temp_folder = os.path.join(parent_dir, ".gemma_virtual_folder")
            os.makedirs(temp_folder, exist_ok=True)
            
            # 1. Link the safetensors file so JD's loader finds "model.safetensors"
            temp_model = os.path.join(temp_folder, "model.safetensors")
            if os.path.exists(temp_model):
                try: os.remove(temp_model)
                except OSError: pass
            
            # Use os.link to create a zero-space hardlink
            try: os.link(gemma_path_str, temp_model)
            except OSError:
                import shutil
                shutil.copyfile(gemma_path_str, temp_model)
                
            # 2. Link tokenizer AND preprocessor files into the virtual folder
            hf_files = [
                "tokenizer.model", 
                "tokenizer_config.json", 
                "config.json", 
                "special_tokens_map.json",
                "preprocessor_config.json"
            ]
            
            for t_file in hf_files:
                src = os.path.join(parent_dir, t_file)
                dst = os.path.join(temp_folder, t_file)
                if os.path.exists(src) and not os.path.exists(dst):
                    try: os.link(src, dst)
                    except OSError:
                        import shutil
                        shutil.copyfile(src, dst)
                        
            model_folder = Path(temp_folder)
            gemma_op_path = str(temp_folder)
        else:
            # Original fallback if you actually select a folder
            model_folder = find_matching_file(gemma_path, "model*.safetensors").parent
            gemma_op_path = gemma_path
        # --- SINGLE FILE INTAKE PATCH END ---

        # base Gemma (built via the repo's gemma module-ops from the folder/virtual folder)
        weight_paths = tuple(str(p) for p in model_folder.rglob("*.safetensors"))
        te_builder = Builder(
            model_class_configurator=GemmaTextEncoderConfigurator,
            model_path=weight_paths,
            module_ops=(GEMMA_MODEL_OPS, *module_ops_from_gemma_root(gemma_op_path)),
            model_loader=_CfgLoader(config, map_gemma=True), # <--- Enabled key remapper
        )
        text_encoder = te_builder.build(device=dev, dtype=dtype)
        if gemma_format == "our_fp8":
            _swap_gemma_fp8(text_encoder, str(model_folder), dtype)        # VERIFY
            
        # connector / embeddings processor from the standalone file
        ep_builder = Builder(
            model_class_configurator=EmbeddingsProcessorConfigurator,
            model_path=connector_path, model_sd_ops=EMBEDDINGS_PROCESSOR_KEY_OPS,
            model_loader=_CfgLoader(config, map_gemma=False),
        )
        embeddings_processor = ep_builder.build(device=dev, dtype=dtype)
        wrapper = GemmaTextEncoderWrapper(text_encoder=text_encoder, embeddings_processor=embeddings_processor,
                                          device=_dev(), dtype=dtype)
        print("[Rebels] text encoder ready (gemma + connector)", flush=True)
        return (wrapper,)

class RebelsJE_VAELoader:
    CATEGORY = CAT; RETURN_TYPES = ("JOYECHO_VVAE", "JOYECHO_AVAE", "INT"); RETURN_NAMES = ("video_vae", "audio_vae", "audio_sample_rate"); FUNCTION = "run"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "config": ("JOYECHO_CONFIG",),
            "video_vae_path": ("STRING", {"default": r"D:\joyai_parts\joyai_echo_video_vae.safetensors"}),
            "audio_vae_path": ("STRING", {"default": r"D:\joyai_parts\joyai_echo_audio_vae.safetensors"}),
            "vocoder_path": ("STRING", {"default": r"D:\joyai_parts\joyai_echo_vocoder.safetensors"}),
            "with_encoders": ("BOOLEAN", {"default": True})}}
    def _build(self, cfg, configurator, sd_ops, path):
        return Builder(model_class_configurator=configurator, model_path=path,
                       model_sd_ops=sd_ops, model_loader=_CfgLoader(cfg)).build(
                       device=torch.device("cpu"), dtype=torch.bfloat16)
    def run(self, config, video_vae_path, audio_vae_path, vocoder_path, with_encoders):
        dtype = torch.bfloat16
        v_dec = self._build(config, VideoDecoderConfigurator, VAE_DECODER_COMFY_KEYS_FILTER, video_vae_path)
        a_dec = self._build(config, AudioDecoderConfigurator, AUDIO_VAE_DECODER_COMFY_KEYS_FILTER, audio_vae_path)
        voc   = self._build(config, VocoderConfigurator, VOCODER_COMFY_KEYS_FILTER, vocoder_path)
        v_enc = self._build(config, VideoEncoderConfigurator, VAE_ENCODER_COMFY_KEYS_FILTER, video_vae_path) if with_encoders else None
        a_enc = self._build(config, AudioEncoderConfigurator, AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER, audio_vae_path) if with_encoders else None
        video_vae = VideoVAEWrapper(encoder=v_enc, decoder=v_dec, device=_dev(), dtype=dtype)
        audio_vae = AudioVAEWrapper(encoder=a_enc, decoder=a_dec, vocoder=voc, device=_dev(), dtype=dtype)
        video_vae.eval(); audio_vae.eval()
        sr = audio_vae.get_output_sample_rate() or 24000
        print("[Rebels] VAEs ready (video + audio + vocoder)", flush=True)
        return (video_vae, audio_vae, sr)

class RebelsJE_Assemble:
    CATEGORY = CAT; RETURN_TYPES = ("JOYECHO_MODEL",); RETURN_NAMES = ("model",); FUNCTION = "run"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"generator": ("JOYECHO_GENERATOR",), "text_encoder": ("JOYECHO_TEXTENC",),
                             "video_vae": ("JOYECHO_VVAE",), "audio_vae": ("JOYECHO_AVAE",),
                             "audio_sample_rate": ("INT", {"default": 24000})}}
    def run(self, generator, text_encoder, video_vae, audio_vae, audio_sample_rate):
        model = {"text_encoder": text_encoder, "generator": generator, "video_vae": video_vae,
                 "audio_vae": audio_vae, "audio_sample_rate": audio_sample_rate,
                 "device": _dev(), "dtype": torch.bfloat16}
        print("[Rebels] JOYECHO_MODEL assembled from discrete loaders.", flush=True)
        return (model,)

NODE_CLASS_MAPPINGS = {
    "RebelsJE_Config": RebelsJE_Config,
    "RebelsJE_DiTLoader": RebelsJE_DiTLoader,
    "RebelsJE_TextEncoder": RebelsJE_TextEncoder,
    "RebelsJE_VAELoader": RebelsJE_VAELoader,
    "RebelsJE_Assemble": RebelsJE_Assemble,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RebelsJE_Config": "Rebels JE • Config",
    "RebelsJE_DiTLoader": "Rebels JE • DiT GGUF Loader (UNet)",
    "RebelsJE_TextEncoder": "Rebels JE • Text Encoder (Gemma fp8 + Connector)",
    "RebelsJE_VAELoader": "Rebels JE • VAE Loader (video+audio)",
    "RebelsJE_Assemble": "Rebels JE • Assemble Model",
}