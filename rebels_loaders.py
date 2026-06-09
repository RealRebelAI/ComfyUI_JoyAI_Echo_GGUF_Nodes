"""
rebels_loaders.py — discrete ComfyUI loader nodes for JoyAI-Echo on low VRAM.
Patched for Single-File Gemma intake and Key Remapping.
"""
from __future__ import annotations
import os, json, gc
_LOADER_DIR = os.path.dirname(os.path.abspath(__file__))
_LOADER_CFG = os.path.join(_LOADER_DIR, "configs", "joyai_echo_config.json")
import dataclasses
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
    from safetensors import safe_open
    with safe_open(src, framework="pt") as f:
        meta = f.metadata() or {}
    if "config" not in meta:
        raise ValueError(f"No 'config' in metadata of {src}. Point at the checkpoint or a config.json.")
    return json.loads(meta["config"])

class _CfgLoader(SafetensorsModelStateDictLoader):
    """Stock safetensors weight load, but metadata() returns the shared config."""
    def __init__(self, config: dict, map_gemma=False, *a, **k):
        super().__init__(*a, **k)
        self._cfg = config
        self._map_gemma = map_gemma
        
    def metadata(self, path): return self._cfg
    
    def load(self, paths, sd_ops=None, device=None):
        sd_obj = super().load(paths, sd_ops, device)
        
        if self._map_gemma:
            new_sd = {}
            for k, v in sd_obj.sd.items():
                # Aggressive remapping to capture varying Gemma3 key structures
                nk = k.replace("cond_stage_model.", "").replace("text_model.", "").replace("text_encoder.", "")
                
                # Standardize to language_model path
                if "embed_tokens" in nk: nk = "model.model.language_model.embed_tokens.weight"
                elif "layers" in nk: nk = nk.replace("model.layers", "model.model.language_model.layers")
                elif "norm" in nk and "language_model" not in nk: nk = nk.replace("model.norm", "model.model.language_model.norm")
                
                new_sd[nk] = v
            return dataclasses.replace(sd_obj, sd=new_sd)
        return sd_obj

# ---------------------------------------------------------------- gguf dit
# Keep GGUFReaders alive for the process lifetime so their memory-mapped data
# (which the GGUFLinear weights below reference WITHOUT copying) stays valid.
_OPEN_GGUF_READERS = []


def _gguf_entries(path):
    r = GGUFReader(path)
    _OPEN_GGUF_READERS.append(r)
    out = {}
    for t in r.tensors:
        out[t.name] = {"data": np.asarray(t.data), "qtype": t.tensor_type,
                       "shape": tuple(int(d) for d in reversed(t.shape))}
    return out

def _dequant(entry, dtype):
    deq = gguf.quants.dequantize(entry["data"], entry["qtype"]).astype(np.float32)
    t = torch.from_numpy(deq).to(dtype)
    del deq; del entry["data"]; gc.collect()
    return t

class GGUFLinear(nn.Module):
    def __init__(self, entry, bias=None, compute_dtype=torch.bfloat16):
        super().__init__()
        self.qtype_value = int(entry["qtype"]); self.weight_shape = tuple(entry["shape"])
        # Do NOT .copy() here. Copying pulled every packed linear out of the
        # memory-mapped GGUF into real RAM, duplicating the whole quantized DiT in
        # memory (~the GGUF size) and blowing the RAM budget. torch.from_numpy on
        # the memmap view shares the file-backed buffer, so the weights stay paged
        # by the OS instead of resident -- the same reason city96's GGUF loader is
        # light on RAM. The reader is held alive in _OPEN_GGUF_READERS.
        self.register_buffer("qweight", torch.from_numpy(entry["data"]))
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
    def __init__(self, config, entries, consumed, dtype):
        super().__init__(); self._cfg = config; self._e = entries; self._consumed = consumed; self._dt = dtype
    def metadata(self, path): return self._cfg
    def load(self, paths, sd_ops=None, device=None):
        sd, size = {}, 0
        for k, e in self._e.items():
            if k in self._consumed: continue
            t = _dequant(e, self._dt); sd[k] = t; size += t.numel() * t.element_size()
            gc.collect()
        return StateDict(sd=sd, device=device or torch.device("cpu"), size=size, dtype=self._dt)

# The bare LTXModel modules are named e.g. "transformer_blocks.0.attn1.to_q",
# but the GGUF keys keep JD's checkpoint prefix "model.diffusion_model.". The
# configurator strips that prefix at load time, so we must match across it here
# or the swap fires on nothing and the whole DiT dequantizes into RAM -> OOM.
_DIT_PREFIXES = ("", "model.diffusion_model.", "diffusion_model.")


def _find_entry(entries, base):
    for p in _DIT_PREFIXES:
        k = p + base
        if k in entries:
            return k
    return None


def _dit_module_ops(entries, consumed, compute_dtype):
    def mutator(model):
        n_lin = n_hit = 0
        miss = []
        for name, mod in list(model.named_modules()):
            if not isinstance(mod, nn.Linear):
                continue
            n_lin += 1
            wk = _find_entry(entries, name + ".weight")
            if wk is None:
                if len(miss) < 5:
                    miss.append(name)
                continue
            bk = _find_entry(entries, name + ".bias")
            bias = _dequant(entries[bk], compute_dtype) if bk else None
            _set_sub(model, name, GGUFLinear(entries[wk], bias, compute_dtype))
            consumed.add(wk)
            if bk:
                consumed.add(bk)
            n_hit += 1
        print(f"[Rebels JE] DiT GGUF swap: matched {n_hit}/{n_lin} Linear layers "
              f"({len(consumed)} tensors kept packed).", flush=True)
        if n_hit == 0 and n_lin:
            print(f"[Rebels JE]   NO matches -> whole DiT would dequantize. "
                  f"sample model Linears={miss}", flush=True)
            print(f"[Rebels JE]   sample GGUF keys={list(entries.keys())[:5]}", flush=True)
        return model
    return (ModuleOps("gguf_linear_swap", matcher=lambda m: True, mutator=mutator),)

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

def _swap_gemma_fp8(model, fp8_dir, compute_dtype):
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
    return model

# ================================================================ NODES
def _dev(): return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class RebelsJE_Config:
    CATEGORY = CAT; RETURN_TYPES = ("JOYECHO_CONFIG",); RETURN_NAMES = ("config",); FUNCTION = "run"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"config_source": ("STRING", {"default": _LOADER_CFG})}}
    def run(self, config_source): return (_full_config(config_source),)

class RebelsJE_DiTLoader:
    CATEGORY = CAT; RETURN_TYPES = ("JOYECHO_GENERATOR",); RETURN_NAMES = ("generator",); FUNCTION = "run"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "config": ("JOYECHO_CONFIG",),
            "dit_gguf": ("STRING", {"default": ""}),
            "video_height": ("INT", {"default": 736}), "video_width": ("INT", {"default": 1280})}}
    def run(self, config, dit_gguf, video_height, video_width):
        dtype = torch.bfloat16
        entries = _gguf_entries(dit_gguf); consumed = set()
        builder = Builder(
            model_class_configurator=LTXModelConfigurator,
            model_path=dit_gguf,
            model_sd_ops=None,
            module_ops=_dit_module_ops(entries, consumed, dtype),
            model_loader=_GGUFDiTLoader(config, entries, consumed, dtype),
        )
        transformer = builder.build(device=torch.device("cpu"), dtype=dtype)
        gen = LTX2DiffusionWrapper(model=X0Model(transformer), video_height=video_height, video_width=video_width)
        gen.eval()
        return (gen,)

class RebelsJE_TextEncoder:
    CATEGORY = CAT; RETURN_TYPES = ("JOYECHO_TEXTENC",); RETURN_NAMES = ("text_encoder",); FUNCTION = "run"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "config": ("JOYECHO_CONFIG",),
            "gemma_path": ("STRING", {"default": ""}),
            "gemma_format": (["our_fp8", "bf16"], {"default": "our_fp8"}),
            "connector_path": ("STRING", {"default": ""}),
            "low_vram": ("BOOLEAN", {"default": True})}}
    def run(self, config, gemma_path, gemma_format, connector_path, low_vram):
        from ltx_core.text_encoders.gemma import GemmaTextEncoderConfigurator, GEMMA_MODEL_OPS, module_ops_from_gemma_root
        from ltx_core.utils import find_matching_file
        from pathlib import Path
        
        dtype = torch.bfloat16; dev = torch.device("cpu") if low_vram else _dev()
        
        # --- SINGLE FILE INTAKE PATCH START ---
        gemma_path_str = str(gemma_path)
        if os.path.isfile(gemma_path_str):
            parent_dir = os.path.dirname(gemma_path_str)
            temp_folder = os.path.join(parent_dir, ".gemma_virtual_folder")
            os.makedirs(temp_folder, exist_ok=True)
            
            temp_model = os.path.join(temp_folder, "model.safetensors")
            if os.path.exists(temp_model):
                try: os.remove(temp_model)
                except OSError: pass
            
            try: os.link(gemma_path_str, temp_model)
            except OSError:
                import shutil
                shutil.copyfile(gemma_path_str, temp_model)
                
            # Gemma needs its HF sidecar files (tokenizer + config jsons) next to
            # the weights. A single-file fp8 download has none of them, so we search
            # several places, in order: the weights' own folder, a 'gemma_assets' or
            # 'gemma' subfolder beside them, and a 'gemma_assets' folder bundled in
            # this node pack (so they can ship with the pack).
            _node_dir = os.path.dirname(os.path.abspath(__file__))
            sidecar_sources = [
                parent_dir,
                os.path.join(parent_dir, "gemma_assets"),
                os.path.join(parent_dir, "gemma"),
                os.path.join(_node_dir, "gemma_assets"),
            ]
            sidecar_files = ["tokenizer.model", "tokenizer_config.json", "config.json",
                             "special_tokens_map.json", "preprocessor_config.json"]
            for t_file in sidecar_files:
                dst = os.path.join(temp_folder, t_file)
                if os.path.exists(dst):
                    continue
                for srcdir in sidecar_sources:
                    src = os.path.join(srcdir, t_file)
                    if os.path.exists(src):
                        try: os.link(src, dst)
                        except OSError:
                            import shutil
                            shutil.copyfile(src, dst)
                        break

            # Fail with a clear, actionable message instead of a cryptic one later.
            missing = [f for f in sidecar_files
                       if not os.path.exists(os.path.join(temp_folder, f))]
            if missing:
                raise FileNotFoundError(
                    "Gemma sidecar files missing: " + ", ".join(missing) + ".\n"
                    "Put them in one of:\n"
                    f"  - the same folder as your Gemma file ({parent_dir})\n"
                    f"  - {os.path.join(_node_dir, 'gemma_assets')}  (ships with the node pack)\n"
                    "Get them from the google/gemma-3-12b-it repo (the small json/tokenizer "
                    "files, not the weights)."
                )
                        
            model_folder = Path(temp_folder)
            gemma_op_path = str(temp_folder)
        else:
            model_folder = find_matching_file(gemma_path, "model*.safetensors").parent
            gemma_op_path = gemma_path
        # --- SINGLE FILE INTAKE PATCH END ---

        weight_paths = tuple(str(p) for p in model_folder.rglob("*.safetensors"))
        te_builder = Builder(
            model_class_configurator=GemmaTextEncoderConfigurator,
            model_path=weight_paths,
            module_ops=(GEMMA_MODEL_OPS, *module_ops_from_gemma_root(gemma_op_path)),
            model_loader=_CfgLoader(config, map_gemma=True),
        )
        text_encoder = te_builder.build(device=dev, dtype=dtype)
        if gemma_format == "our_fp8": _swap_gemma_fp8(text_encoder, str(model_folder), dtype)
            
        ep_builder = Builder(
            model_class_configurator=EmbeddingsProcessorConfigurator,
            model_path=connector_path, model_sd_ops=EMBEDDINGS_PROCESSOR_KEY_OPS,
            model_loader=_CfgLoader(config, map_gemma=False),
        )
        embeddings_processor = ep_builder.build(device=dev, dtype=dtype)
        wrapper = GemmaTextEncoderWrapper(text_encoder=text_encoder, embeddings_processor=embeddings_processor,
                                          device=_dev(), dtype=dtype)
        return (wrapper,)

class RebelsJE_VAELoader:
    CATEGORY = CAT; RETURN_TYPES = ("JOYECHO_VVAE", "JOYECHO_AVAE", "INT"); RETURN_NAMES = ("video_vae", "audio_vae", "audio_sample_rate"); FUNCTION = "run"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "config": ("JOYECHO_CONFIG",),
            "video_vae_path": ("STRING", {"default": ""}),
            "audio_vae_path": ("STRING", {"default": ""}),
            "vocoder_path": ("STRING", {"default": ""}),
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