"""Rebels JE - Staged single-node pipeline.

Why this exists
---------------
The discrete loader graph (Config -> DiTLoader + TextEncoder + VAELoader ->
Assemble -> TextEncode -> SingleShotGenerate) loads EVERY component and holds
it for the whole ComfyUI run. On a 16GB-RAM box that means the fp8 Gemma
(~12GB) and the DiT (~9GB quantized) sit in RAM at the same time -> >16GB ->
pagefile thrash -> numpy ArrayMemoryError during DiT dequant.

This node fixes that by STAGING inside a single function so we control the
load/free order:

    Stage 1  build Gemma + connector, encode the prompt -> small cond, FREE Gemma
    Stage 2  load DiT (GGUF) + VAEs   (Gemma is already gone)
    Stage 3  denoise + decode using the precomputed cond (no text encoder needed)

Peak system RAM becomes the largest SINGLE component (~12GB for Gemma during
encode, ~9-12GB for the DiT during denoise) instead of the sum (~25GB).

It reuses the existing, working loader classes from rebels_loaders.py and the
real BidirectionalAVInferencePipeline denoise/decode path from nodes.py
(generate_shot, Phase A + Phase B) -- only the in-line encode block is dropped,
because we already have the conditioning.

NOTE: single-shot, no incoming memory bank (fresh, empty -> base pipeline path,
so VAE encoders are not needed -> with_encoders=False saves more RAM).
"""

from __future__ import annotations

import gc
import torch

# Reuse the working loaders + helpers already in the package.
from .rebels_loaders import (
    RebelsJE_Config,
    RebelsJE_TextEncoder,
    RebelsJE_DiTLoader,
    RebelsJE_VAELoader,
    _dev,
)
from .nodes import SequentialOffloader, DENOISING_SIGMAS, _empty_cache, _move

try:
    from .rebels_loaders import CAT as _CAT
except Exception:
    _CAT = "JoyAI-Echo/Rebels"


class RebelsJE_StagedPipeline:
    """One node: encode -> free Gemma -> load DiT/VAE -> denoise -> decode."""

    CATEGORY = _CAT
    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config_source": ("STRING", {"default": r"D:\joyai_parts\joyai_echo_config.json"}),
                "dit_gguf": ("STRING", {"default": r"D:\AI_Tools\ComfyUI_windows_portable\ComfyUI\models\unet\JoyAI-Echo-DiT-Q2_K.gguf"}),
                "gemma_path": ("STRING", {"default": r"D:\AI_Tools\ComfyUI_windows_portable\ComfyUI\models\text_encoders\gemma\gemma_3_12B_it_fp8_scaled.safetensors"}),
                "gemma_format": (["our_fp8", "bf16"], {"default": "our_fp8"}),
                "connector_path": ("STRING", {"default": r"D:\joyai_parts\joyai_echo_embeddings_processor.safetensors"}),
                "video_vae_path": ("STRING", {"default": r"D:\joyai_parts\joyai_echo_video_vae.safetensors"}),
                "audio_vae_path": ("STRING", {"default": r"D:\joyai_parts\joyai_echo_audio_vae.safetensors"}),
                "vocoder_path": ("STRING", {"default": r"D:\joyai_parts\joyai_echo_vocoder.safetensors"}),
                "prompt": ("STRING", {"default": "a woman walks down a busy street at sunset", "multiline": True}),
                "seed": ("INT", {"default": 12345}),
                "num_frames": ("INT", {"default": 25, "min": 9, "max": 257}),
                "video_height": ("INT", {"default": 512, "min": 64, "max": 1280, "step": 32}),
                "video_width": ("INT", {"default": 768, "min": 64, "max": 1280, "step": 32}),
                "video_fps": ("INT", {"default": 24}),
                "audio_sample_rate": ("INT", {"default": 24000}),
                # Gemma fp8 (~12GB) cannot fit 8GB VRAM, so encode on CPU. Slow but fits RAM.
                "encode_on_cpu": ("BOOLEAN", {"default": True}),
                # Stream one DiT block to GPU at a time so denoise fits 8GB VRAM.
                "sequential_offload": ("BOOLEAN", {"default": True}),
            }
        }

    # ------------------------------------------------------------------ run
    def run(self, config_source, dit_gguf, gemma_path, gemma_format, connector_path,
            video_vae_path, audio_vae_path, vocoder_path, prompt, seed, num_frames,
            video_height, video_width, video_fps, audio_sample_rate,
            encode_on_cpu, sequential_offload):

        device = _dev()
        dtype = torch.bfloat16
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("Prompt is empty.")

        cfg = RebelsJE_Config().run(config_source)[0]

        # ============================================================ STAGE 1
        # Build Gemma + connector, encode prompt, then FREE Gemma from RAM.
        print("[StagedJE] Stage 1/3: building text encoder (Gemma)...", flush=True)
        te = RebelsJE_TextEncoder().run(cfg, gemma_path, gemma_format, connector_path, True)[0]

        if encode_on_cpu:
            # Force CPU encode: 12GB fp8 Gemma will not fit 8GB VRAM.
            try:
                te.device = torch.device("cpu")
            except Exception:
                pass
            for attr in ("text_encoder", "embeddings_processor"):
                m = getattr(te, attr, None)
                if m is not None and hasattr(m, "to"):
                    try:
                        m.to("cpu")
                    except Exception:
                        pass

        print("[StagedJE] Stage 1/3: encoding prompt (this is the slow part on CPU)...", flush=True)
        cond = te([prompt])
        cond_cpu = {
            k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
            for k, v in cond.items()
        }
        del cond, te
        gc.collect()
        _empty_cache()
        print("[StagedJE] Stage 1/3: Gemma released from RAM.", flush=True)

        # ============================================================ STAGE 2
        # Now that Gemma is gone, load the DiT + VAEs.
        print("[StagedJE] Stage 2/3: loading DiT (GGUF) + VAEs...", flush=True)
        generator = RebelsJE_DiTLoader().run(cfg, dit_gguf, video_height, video_width)[0]
        video_vae, audio_vae, sr = RebelsJE_VAELoader().run(
            cfg, video_vae_path, audio_vae_path, vocoder_path, False  # with_encoders=False
        )
        audio_sample_rate = audio_sample_rate or sr

        # ============================================================ STAGE 3
        # Denoise + decode using the precomputed conditioning. Lifted from
        # nodes.py JoyEcho_SingleShotGenerate.generate_shot Phase A + Phase B,
        # with the in-line text-encode block removed.
        print("[StagedJE] Stage 3/3: denoise + decode...", flush=True)
        from ltx_distillation.inference.bidirectional_pipeline import BidirectionalAVInferencePipeline
        from ltx_distillation.inference.memory_multishot import normalize_audio_waveform_for_media
        from ltx_distillation.utils import add_noise, compute_latent_shapes, decode_benchmark_sample

        # validate num_frames -> 1 + multiple of 8
        if (num_frames - 1) % 8 != 0:
            num_frames = 1 + ((num_frames - 1) // 8) * 8

        # generator resolution wiring (matches generate_shot)
        generator.video_height = video_height
        generator.video_width = video_width
        generator.latent_height = video_height // 32
        generator.latent_width = video_width // 32
        generator.video_frame_seqlen = generator.latent_height * generator.latent_width

        video_shape, audio_shape = compute_latent_shapes(
            num_frames=num_frames,
            video_height=video_height,
            video_width=video_width,
            batch_size=1,
            video_fps=float(video_fps),
        )

        conditional_dict = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in cond_cpu.items()
        }
        del cond_cpu

        denoising_sigmas = torch.tensor(DENOISING_SIGMAS, device=device, dtype=torch.float32)
        base_pipeline = BidirectionalAVInferencePipeline(
            generator=generator,
            add_noise_fn=add_noise,
            denoising_sigmas=denoising_sigmas,
        )

        offloader = SequentialOffloader(generator, device) if sequential_offload else None

        # ---- Phase A: denoise (generator on GPU, VAEs off) ----
        _move(video_vae.decoder, "cpu")
        _move(audio_vae.decoder, "cpu")
        _move(audio_vae.vocoder, "cpu")
        if getattr(video_vae, "encoder", None) is not None:
            _move(video_vae.encoder, "cpu")
        if getattr(audio_vae, "encoder", None) is not None:
            _move(audio_vae.encoder, "cpu")
        if sequential_offload:
            offloader.install()
        else:
            _move(generator, device)
        _empty_cache()

        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(seed)
            video_latent, audio_latent = base_pipeline.generate(
                video_shape=tuple(video_shape),
                audio_shape=tuple(audio_shape),
                conditional_dict=conditional_dict,
                seed=seed,
            )

        if device.type == "cuda":
            torch.cuda.synchronize()
        del conditional_dict
        _empty_cache()

        # ---- Phase B: decode (generator off, VAE decoders on GPU) ----
        if sequential_offload:
            offloader.remove()
        _move(generator, "cpu")
        _empty_cache()
        _move(video_vae.decoder, device)
        _move(audio_vae.decoder, device)
        _move(audio_vae.vocoder, device)

        video_uint8, audio_waveform = decode_benchmark_sample(
            video_vae, audio_vae, video_latent, audio_latent
        )

        if device.type == "cuda":
            torch.cuda.synchronize()
        _move(video_vae.decoder, "cpu")
        _move(audio_vae.decoder, "cpu")
        _move(audio_vae.vocoder, "cpu")
        _empty_cache()

        images = video_uint8.float() / 255.0  # [F, H, W, 3]

        audio_out = None
        if audio_waveform is not None:
            audio_norm = normalize_audio_waveform_for_media(audio_waveform)
            audio_out = {"waveform": audio_norm.unsqueeze(0), "sample_rate": audio_sample_rate}

        del video_latent, audio_latent, video_uint8, audio_waveform
        gc.collect()
        _empty_cache()
        print(f"[StagedJE] done. {images.shape[0]} frames.", flush=True)
        return (images, audio_out)


NODE_CLASS_MAPPINGS = {"RebelsJE_StagedPipeline": RebelsJE_StagedPipeline}
NODE_DISPLAY_NAME_MAPPINGS = {"RebelsJE_StagedPipeline": "Rebels JE - Staged Pipeline (16GB)"}
