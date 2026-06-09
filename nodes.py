"""JoyAI-Echo ComfyUI node implementations.

Six nodes faithful to the official inference.py:
1. JoyEcho_ModelLoader          — load text encoder + DiT + VAEs (bf16)
2. JoyEcho_TextEncode           — encode prompts, auto-release text encoder
3. JoyEcho_Generate             — multi-shot denoise + decode with memory bank
4. JoyEcho_SingleShotGenerate   — single-shot with per-shot text box and memory chaining
5. JoyEcho_PromptFormat         — get system prompt for LLM-based prompt enhancement
6. JoyEcho_LLMEnhance           — call LLM API to generate shot prompts from a story idea
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch


DENOISING_SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]


def _empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _move(module, device):
    if module is not None:
        module.to(device)


class SequentialOffloader:
    """Layer-by-layer GPU offloading for the DiT transformer blocks.

    Hooks into each transformer block so that only the currently-executing block
    resides on GPU. All other blocks stay on CPU/pinned memory.
    Peak VRAM for the generator drops from ~30GB to ~2-3GB (1 block + activations).
    """

    def __init__(self, generator, device: torch.device, pin_memory: bool = True):
        self._generator = generator
        self._device = device
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._pin_memory = pin_memory
        self._installed = False

    def install(self):
        """Install forward hooks on transformer blocks and move them to CPU."""
        if self._installed:
            return
        self._installed = True

        velocity_model = self._generator.model.velocity_model
        blocks = velocity_model.transformer_blocks

        # Keep pre/post processing layers on GPU (small footprint)
        for name, param in velocity_model.named_parameters():
            if "transformer_blocks" not in name:
                param.data = param.data.to(self._device)
        for name, buf in velocity_model.named_buffers():
            if "transformer_blocks" not in name:
                buf.data = buf.data.to(self._device)

        # Move all blocks to CPU (optionally pinned)
        for block in blocks:
            block.to("cpu")
            if self._pin_memory and torch.cuda.is_available():
                for param in block.parameters():
                    param.data = param.data.pin_memory()
                for buf in block.buffers():
                    buf.data = buf.data.pin_memory()

        # Also keep the wrapper's patchifiers and X0Model's non-block params on GPU
        for name, param in self._generator.named_parameters():
            if "velocity_model.transformer_blocks" not in name and "velocity_model" not in name:
                param.data = param.data.to(self._device)
        for name, buf in self._generator.named_buffers():
            if "velocity_model.transformer_blocks" not in name and "velocity_model" not in name:
                buf.data = buf.data.to(self._device)

        def make_pre_hook(block_module):
            def hook(module, args):
                block_module.to(self._device, non_blocking=True)
                if torch.cuda.is_available():
                    torch.cuda.current_stream().synchronize()
            return hook

        def make_post_hook(block_module):
            def hook(module, args, output):
                block_module.to("cpu", non_blocking=True)
            return hook

        for block in blocks:
            h1 = block.register_forward_pre_hook(make_pre_hook(block))
            h2 = block.register_forward_hook(make_post_hook(block))
            self._hooks.extend([h1, h2])

        print(f"[JoyEcho] Sequential offloading installed: {len(blocks)} blocks", flush=True)

    def remove(self):
        """Remove all hooks and move entire generator back to CPU."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._installed = False
        self._generator.to("cpu")


class JoyEcho_ModelLoader:
    """Load JoyAI-Echo model components: text encoder, DiT generator, and VAEs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to echo-longvideo-release.safetensors",
                }),
                "gemma_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to gemma-3-12b-it directory (bf16 safetensors)",
                }),
            },
            "optional": {
                "lora_path": ("STRING", {"default": ""}),
                "lora_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                }),
                "low_vram": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Load text encoder on CPU for 24GB GPUs. "
                               "Encoding will be slower but uses no GPU memory.",
                }),
            },
        }

    RETURN_TYPES = ("JOYECHO_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "JoyAI-Echo"

    def load_model(self, checkpoint_path: str, gemma_path: str,
                   lora_path: str = "", lora_strength: float = 1.0,
                   low_vram: bool = False):
        from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
        from ltx_distillation.models.ltx_wrapper import create_ltx2_wrapper
        from ltx_distillation.models.text_encoder_wrapper import create_text_encoder_wrapper
        from ltx_distillation.models.vae_wrapper import create_vae_wrappers

        checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        gemma_path = str(Path(gemma_path).expanduser().resolve())

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        dtype = torch.bfloat16

        # Load text encoder
        text_encoder_device = torch.device("cpu") if low_vram else device
        print(f"[JoyEcho] Loading text encoder (bf16) on {text_encoder_device}...", flush=True)
        text_encoder = create_text_encoder_wrapper(
            checkpoint_path=checkpoint_path,
            gemma_path=gemma_path,
            device=text_encoder_device,
            dtype=dtype,
        )
        text_encoder.eval()

        # Load generator
        print("[JoyEcho] Loading DiT generator...", flush=True)
        loras = ()
        if lora_path and lora_path.strip():
            loras = (
                LoraPathStrengthAndSDOps(
                    str(Path(lora_path).expanduser()),
                    float(lora_strength),
                    LTXV_LORA_COMFY_RENAMING_MAP,
                ),
            )

        generator = create_ltx2_wrapper(
            checkpoint_path=checkpoint_path,
            gemma_path=gemma_path,
            device=torch.device("cpu"),
            dtype=dtype,
            video_height=736,
            video_width=1280,
            loras=loras,
        )
        generator.eval()

        # Load VAEs to CPU
        print("[JoyEcho] Loading VAEs...", flush=True)
        video_vae, audio_vae = create_vae_wrappers(
            checkpoint_path=checkpoint_path,
            device=torch.device("cpu"),
            dtype=dtype,
            with_video_encoder=True,
            with_audio_encoder=True,
            decoder_device=torch.device("cpu"),
        )
        video_vae.eval()
        audio_vae.eval()

        audio_sample_rate = audio_vae.get_output_sample_rate() or 24000

        model = {
            "text_encoder": text_encoder,
            "generator": generator,
            "video_vae": video_vae,
            "audio_vae": audio_vae,
            "audio_sample_rate": audio_sample_rate,
            "device": device,
            "dtype": dtype,
            "checkpoint_path": checkpoint_path,
            "gemma_path": gemma_path,
        }

        print(f"[JoyEcho] Model loaded. Audio sample rate: {audio_sample_rate}", flush=True)
        return (model,)


class JoyEcho_TextEncode:
    """Encode text prompts using Gemma-3-12b.

    Supports:
    - One prompt per line (multi-line text, each line = one shot)
    - JSON format: {"prompts": ["shot1", "shot2", ...]} (official format)
    - JSON file path (*.json)

    After encoding, the text encoder is released from GPU to free ~24GB VRAM.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("JOYECHO_MODEL",),
                "prompts": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "One prompt per line, JSON object, or path to .json file",
                }),
            },
            "optional": {
                "release_text_encoder": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("JOYECHO_MODEL", "JOYECHO_COND",)
    RETURN_NAMES = ("model", "conditioning",)
    FUNCTION = "encode"
    CATEGORY = "JoyAI-Echo"

    @staticmethod
    def _parse_prompts(prompts: str) -> list[str]:
        """Parse prompts from text, JSON string, or JSON file path."""
        text = prompts.strip()

        # Check if it's a file path to a .json
        if text.endswith(".json") and not text.startswith("{"):
            p = Path(text).expanduser()
            if not p.is_absolute():
                p = Path(__file__).resolve().parent / p
            p = p.resolve()
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return JoyEcho_TextEncode._extract_from_json(data)

        # Check if it's a JSON object
        if text.startswith("{"):
            try:
                data = json.loads(text)
                return JoyEcho_TextEncode._extract_from_json(data)
            except json.JSONDecodeError:
                pass

        # Fall back to one-prompt-per-line
        return [line.strip() for line in text.split("\n") if line.strip()]

    @staticmethod
    def _extract_from_json(data: dict) -> list[str]:
        """Extract prompt list from JSON (supports 'prompts' or 'shots' key)."""
        if isinstance(data.get("prompts"), list):
            return [str(p).strip() for p in data["prompts"] if str(p).strip()]
        if isinstance(data.get("shots"), list):
            return [str(p).strip() for p in data["shots"] if str(p).strip()]
        raise ValueError("JSON must contain a 'prompts' or 'shots' array.")

    def encode(self, model: dict, prompts: str, release_text_encoder: bool = True):
        text_encoder = model.get("text_encoder")
        if text_encoder is None:
            raise RuntimeError(
                "Text encoder not available. It may have been released already. "
                "Reload the model to encode new prompts."
            )

        prompt_list = self._parse_prompts(prompts)
        if not prompt_list:
            raise ValueError("No prompts provided. Enter text, JSON, or a .json file path.")

        device = model["device"]
        print(f"[JoyEcho] Encoding {len(prompt_list)} prompt(s)...", flush=True)

        cached_conds = []
        for i, prompt in enumerate(prompt_list):
            cond = text_encoder([prompt])
            cached_conds.append(
                {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
                 for k, v in cond.items()}
            )
            del cond
            print(f"[JoyEcho]   Encoded shot {i+1}/{len(prompt_list)}", flush=True)

        if release_text_encoder:
            print("[JoyEcho] Releasing text encoder to free VRAM...", flush=True)
            del text_encoder
            model["text_encoder"] = None
            gc.collect()
            _empty_cache()

        return (model, cached_conds,)


class JoyEcho_Generate:
    """Generate multi-shot video + audio using DMD few-step denoising with memory bank.

    Implements the same hot-swap memory management as official inference.py:
    - Denoise phase: generator on GPU, VAE on CPU
    - Decode phase: generator on CPU, VAE on GPU
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("JOYECHO_MODEL",),
                "conditioning": ("JOYECHO_COND",),
                "seed": ("INT", {"default": 12345, "min": 0, "max": 2**31 - 1}),
                "num_frames": ("INT", {"default": 241, "min": 9, "max": 481, "step": 8,
                                       "tooltip": "Must be 1 + 8*k (e.g. 121, 241, 361)"}),
                "video_height": ("INT", {"default": 736, "min": 256, "max": 1088, "step": 32}),
                "video_width": ("INT", {"default": 1280, "min": 256, "max": 1920, "step": 32}),
            },
            "optional": {
                "video_fps": ("INT", {"default": 25, "min": 1, "max": 60}),
                "v2a_grad_scale": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "memory_max_size": ("INT", {"default": 7, "min": 0, "max": 20}),
                "num_fix_frames": ("INT", {"default": 3, "min": 0, "max": 10}),
                "enable_audio_memory": ("BOOLEAN", {"default": True}),
                "audio_memory_window_size": ("INT", {"default": 96, "min": 16, "max": 256}),
                "sequential_offload": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable layer-by-layer GPU offloading for DiT. "
                               "Reduces VRAM from ~30GB to ~3GB at the cost of slower inference.",
                }),
                "output_prefix": ("STRING", {
                    "default": "joyecho/shot",
                    "tooltip": "Prefix for per-shot video files saved immediately after each shot completes.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO",)
    RETURN_NAMES = ("images", "audio",)
    FUNCTION = "generate"
    CATEGORY = "JoyAI-Echo"
    OUTPUT_NODE = True

    def generate(
        self,
        model: dict,
        conditioning: list,
        seed: int = 12345,
        num_frames: int = 241,
        video_height: int = 736,
        video_width: int = 1280,
        video_fps: int = 25,
        v2a_grad_scale: float = 2.0,
        memory_max_size: int = 7,
        num_fix_frames: int = 3,
        enable_audio_memory: bool = True,
        audio_memory_window_size: int = 96,
        sequential_offload: bool = False,
        output_prefix: str = "joyecho/shot",
    ):
        from ltx_distillation.inference.bidirectional_pipeline import BidirectionalAVInferencePipeline
        from ltx_distillation.inference.memory_bidirectional_pipeline import BidirectionalMemoryAVInferencePipeline
        from ltx_distillation.inference.memory_multishot import (
            PairedAudioVideoMemoryBank,
            build_paired_audio_memory_kwargs,
            video_uint8_to_pil_frames,
        )
        from ltx_distillation.utils import (
            add_noise,
            compute_latent_shapes,
            decode_benchmark_sample,
            encode_memory_frames_batch,
        )

        generator = model["generator"]
        video_vae = model["video_vae"]
        audio_vae = model["audio_vae"]
        audio_sample_rate = model["audio_sample_rate"]
        device = model["device"]
        dtype = model["dtype"]

        # Validate num_frames
        if (num_frames - 1) % 8 != 0:
            num_frames = 1 + ((num_frames - 1) // 8) * 8
            print(f"[JoyEcho] Adjusted num_frames to {num_frames} (must be 1 + 8*k)", flush=True)

        # Update generator resolution if changed
        generator.video_height = video_height
        generator.video_width = video_width
        generator.latent_height = video_height // 32
        generator.latent_width = video_width // 32
        generator.video_frame_seqlen = generator.latent_height * generator.latent_width

        # Compute latent shapes
        video_shape, audio_shape = compute_latent_shapes(
            num_frames=num_frames,
            video_height=video_height,
            video_width=video_width,
            batch_size=1,
            video_fps=float(video_fps),
        )

        # Build pipelines
        denoising_sigmas = torch.tensor(DENOISING_SIGMAS, device=device, dtype=torch.float32)
        base_pipeline = BidirectionalAVInferencePipeline(
            generator=generator,
            add_noise_fn=add_noise,
            denoising_sigmas=denoising_sigmas,
        )
        memory_pipeline = BidirectionalMemoryAVInferencePipeline(
            generator=generator,
            add_noise_fn=add_noise,
            denoising_sigmas=denoising_sigmas,
            memory_downscale_factor=1,
        )

        # Memory bank
        memory_bank = PairedAudioVideoMemoryBank(
            max_size=memory_max_size,
            save_mode="random_every_shot_frame",
            num_fix_frames=num_fix_frames,
        )

        all_video_frames = []
        all_audio_waveforms = []

        num_shots = len(conditioning)
        offloader = None
        if sequential_offload:
            offloader = SequentialOffloader(generator, device)

        print(f"[JoyEcho] Generating {num_shots} shot(s) at {video_width}x{video_height}, "
              f"{num_frames} frames{' [sequential offload]' if sequential_offload else ''}...",
              flush=True)

        for shot_idx in range(num_shots):
            prompt_seed = seed + shot_idx
            conditional_dict = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in conditioning[shot_idx].items()
            }

            print(f"[JoyEcho] Shot {shot_idx+1}/{num_shots}, seed={prompt_seed}, "
                  f"memory_size={len(memory_bank)}", flush=True)

            # --- Phase A: Denoise (generator on GPU, VAE on CPU) ---
            _move(video_vae.encoder, "cpu")
            _move(video_vae.decoder, "cpu")
            _move(audio_vae.encoder, "cpu")
            _move(audio_vae.decoder, "cpu")
            _move(audio_vae.vocoder, "cpu")
            if sequential_offload:
                offloader.install()
            else:
                _move(generator, device)
            _empty_cache()

            with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
                torch.manual_seed(prompt_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed(prompt_seed)

                if len(memory_bank) > 0:
                    # Encode memory frames (briefly bring video encoder to GPU)
                    _move(video_vae.encoder, device)
                    memory_video = encode_memory_frames_batch(
                        video_vae=video_vae,
                        batch_memory_frames=[memory_bank.get_memory_frames()],
                        target_h=video_height,
                        target_w=video_width,
                        device=device,
                        dtype=dtype,
                    )
                    _move(video_vae.encoder, "cpu")
                    _empty_cache()

                    memory_audio_kwargs = build_paired_audio_memory_kwargs(
                        memory_bank,
                        enable_audio_memory=enable_audio_memory,
                        v2a_grad_scale=v2a_grad_scale,
                        memory_position_mode="reference",
                    )

                    video_latent, audio_latent = memory_pipeline.generate(
                        video_shape=tuple(video_shape),
                        audio_shape=tuple(audio_shape),
                        conditional_dict=conditional_dict,
                        memory_video=memory_video,
                        seed=prompt_seed,
                        **memory_audio_kwargs,
                    )
                    del memory_video
                else:
                    video_latent, audio_latent = base_pipeline.generate(
                        video_shape=tuple(video_shape),
                        audio_shape=tuple(audio_shape),
                        conditional_dict=conditional_dict,
                        seed=prompt_seed,
                    )

            if device.type == "cuda":
                torch.cuda.synchronize()

            del conditional_dict
            _empty_cache()

            # Save audio latent for memory before decode moves things around
            audio_memory_latent = (
                audio_latent.detach().cpu().contiguous()
                if enable_audio_memory and audio_latent is not None
                else None
            )

            # --- Phase B: Decode (generator off GPU, VAE on GPU) ---
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

            # Move VAE back to CPU
            _move(video_vae.decoder, "cpu")
            _move(audio_vae.decoder, "cpu")
            _move(audio_vae.vocoder, "cpu")
            _empty_cache()

            # Update memory bank
            memory_frames_pil = video_uint8_to_pil_frames(video_uint8)
            if audio_memory_latent is not None:
                memory_bank.save_memory_slot(
                    memory_frames_pil,
                    audio_memory_latent,
                    audio_window_size=audio_memory_window_size,
                    video_clip_num_frames=9,
                    audio_waveform=audio_waveform,
                    audio_sample_rate=16000,
                    video_fps=float(video_fps),
                    audio_window_selection_mode="max_response",
                    video_frame_selection_mode="center",
                    audio_memory_mel_bins=128,
                    audio_memory_mel_hop_length=160,
                    audio_memory_n_fft=1024,
                    audio_memory_downsample_factor=4,
                    audio_memory_is_causal=True,
                )

            # Collect outputs
            # video_uint8: [F, H, W, 3] uint8 -> [F, H, W, 3] float32 [0, 1]
            video_float = video_uint8.float() / 255.0
            all_video_frames.append(video_float)

            if audio_waveform is not None:
                from ltx_distillation.inference.memory_multishot import normalize_audio_waveform_for_media
                audio_norm = normalize_audio_waveform_for_media(audio_waveform)
                all_audio_waveforms.append(audio_norm)

            # Save per-shot video immediately for real-time preview
            self._save_shot_video(
                video_uint8, audio_waveform, shot_idx,
                video_fps, audio_sample_rate, output_prefix
            )

            del video_latent, audio_latent, audio_memory_latent, video_uint8, audio_waveform
            _empty_cache()

            print(f"[JoyEcho] Shot {shot_idx+1}/{num_shots} done.", flush=True)

        # Concatenate all shots
        images = torch.cat(all_video_frames, dim=0)  # [total_frames, H, W, 3]

        audio_out = None
        if all_audio_waveforms:
            combined_waveform = torch.cat(all_audio_waveforms, dim=-1)  # [2, total_samples]
            audio_out = {
                "waveform": combined_waveform.unsqueeze(0),  # [1, 2, samples]
                "sample_rate": audio_sample_rate,
            }

        print(f"[JoyEcho] Generation complete. {images.shape[0]} frames, "
              f"{num_shots} shot(s).", flush=True)

        return (images, audio_out,)

    @staticmethod
    def _save_shot_video(video_uint8, audio_waveform, shot_idx, fps, audio_sr, prefix):
        """Save a single shot as mp4 immediately after generation."""
        import av
        import numpy as np

        try:
            import folder_paths
            output_dir = folder_paths.get_output_directory()
        except Exception:
            output_dir = Path("/root/ComfyUI/output")

        # Build output path
        parts = prefix.rsplit("/", 1)
        if len(parts) == 2:
            sub_dir = Path(output_dir) / parts[0]
            name_prefix = parts[1]
        else:
            sub_dir = Path(output_dir)
            name_prefix = prefix

        sub_dir.mkdir(parents=True, exist_ok=True)
        out_path = sub_dir / f"{name_prefix}_{shot_idx:03d}.mp4"

        frames_np = video_uint8.cpu().numpy() if isinstance(video_uint8, torch.Tensor) else video_uint8

        container = av.open(str(out_path), mode="w")
        stream = container.add_stream("h264", rate=fps)
        stream.height = frames_np.shape[1]
        stream.width = frames_np.shape[2]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "fast"}

        for frame_data in frames_np:
            frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()

        # Save audio sidecar
        if audio_waveform is not None:
            import torchaudio
            wav_path = sub_dir / f"{name_prefix}_{shot_idx:03d}.wav"
            waveform = audio_waveform.cpu()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            torchaudio.save(str(wav_path), waveform, sample_rate=audio_sr)

        print(f"[JoyEcho] Shot {shot_idx} saved → {out_path}", flush=True)


class JoyEcho_SingleShotGenerate:
    """Generate a single shot with memory bank input/output for chaining.

    Each instance has its own editable prompt text box and outputs video frames
    that can be previewed immediately via CreateVideo → SaveVideo.
    Chain multiple instances via the memory output → next shot's memory input.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("JOYECHO_MODEL",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Single shot prompt text",
                }),
                "seed": ("INT", {"default": 12345, "min": 0, "max": 2**31 - 1}),
                "num_frames": ("INT", {"default": 241, "min": 9, "max": 481, "step": 8,
                                       "tooltip": "Must be 1 + 8*k (e.g. 121, 241, 361)"}),
                "video_height": ("INT", {"default": 736, "min": 256, "max": 1088, "step": 32}),
                "video_width": ("INT", {"default": 1280, "min": 256, "max": 1920, "step": 32}),
            },
            "optional": {
                "memory": ("JOYECHO_MEMORY",),
                "video_fps": ("INT", {"default": 25, "min": 1, "max": 60}),
                "v2a_grad_scale": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "memory_max_size": ("INT", {"default": 7, "min": 0, "max": 20}),
                "num_fix_frames": ("INT", {"default": 3, "min": 0, "max": 10}),
                "enable_audio_memory": ("BOOLEAN", {"default": True}),
                "audio_memory_window_size": ("INT", {"default": 96, "min": 16, "max": 256}),
                "sequential_offload": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable layer-by-layer GPU offloading for DiT.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "JOYECHO_MEMORY", "JOYECHO_MODEL",)
    RETURN_NAMES = ("images", "audio", "memory", "model",)
    FUNCTION = "generate_shot"
    CATEGORY = "JoyAI-Echo"

    def generate_shot(
        self,
        model: dict,
        prompt: str,
        seed: int = 12345,
        num_frames: int = 241,
        video_height: int = 736,
        video_width: int = 1280,
        memory: dict | None = None,
        video_fps: int = 25,
        v2a_grad_scale: float = 2.0,
        memory_max_size: int = 7,
        num_fix_frames: int = 3,
        enable_audio_memory: bool = True,
        audio_memory_window_size: int = 96,
        sequential_offload: bool = False,
    ):
        from ltx_distillation.inference.bidirectional_pipeline import BidirectionalAVInferencePipeline
        from ltx_distillation.inference.memory_bidirectional_pipeline import BidirectionalMemoryAVInferencePipeline
        from ltx_distillation.inference.memory_multishot import (
            PairedAudioVideoMemoryBank,
            build_paired_audio_memory_kwargs,
            video_uint8_to_pil_frames,
        )
        from ltx_distillation.utils import (
            add_noise,
            compute_latent_shapes,
            decode_benchmark_sample,
            encode_memory_frames_batch,
        )

        if not prompt.strip():
            raise ValueError("Prompt is empty. Enter a shot description.")

        text_encoder = model.get("text_encoder")
        if text_encoder is None:
            raise RuntimeError(
                "Text encoder not available. It may have been released by a previous shot. "
                "Set release_text_encoder=False on earlier shots."
            )

        generator = model["generator"]
        video_vae = model["video_vae"]
        audio_vae = model["audio_vae"]
        audio_sample_rate = model["audio_sample_rate"]
        device = model["device"]
        dtype = model["dtype"]

        # Validate num_frames
        if (num_frames - 1) % 8 != 0:
            num_frames = 1 + ((num_frames - 1) // 8) * 8

        # Update generator resolution
        generator.video_height = video_height
        generator.video_width = video_width
        generator.latent_height = video_height // 32
        generator.latent_width = video_width // 32
        generator.video_frame_seqlen = generator.latent_height * generator.latent_width

        # Compute latent shapes
        video_shape, audio_shape = compute_latent_shapes(
            num_frames=num_frames,
            video_height=video_height,
            video_width=video_width,
            batch_size=1,
            video_fps=float(video_fps),
        )

        # Get or create memory bank
        if memory is not None:
            memory_bank = memory["bank"]
        else:
            memory_bank = PairedAudioVideoMemoryBank(
                max_size=memory_max_size,
                save_mode="random_every_shot_frame",
                num_fix_frames=num_fix_frames,
            )

        print(f"[JoyEcho] SingleShot: encoding prompt, seed={seed}, "
              f"memory_size={len(memory_bank)}", flush=True)

        # --- Phase 0: Encode (text encoder on GPU, everything else off) ---
        _move(generator, "cpu")
        _move(video_vae.encoder, "cpu")
        _move(video_vae.decoder, "cpu")
        _move(audio_vae.encoder, "cpu")
        _move(audio_vae.decoder, "cpu")
        _move(audio_vae.vocoder, "cpu")
        _move(text_encoder, device)
        _empty_cache()

        cond = text_encoder([prompt.strip()])
        conditional_dict = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in cond.items()
        }
        del cond

        # Offload text encoder immediately after encoding
        _move(text_encoder, "cpu")
        _empty_cache()

        # Build pipelines
        denoising_sigmas = torch.tensor(DENOISING_SIGMAS, device=device, dtype=torch.float32)
        base_pipeline = BidirectionalAVInferencePipeline(
            generator=generator,
            add_noise_fn=add_noise,
            denoising_sigmas=denoising_sigmas,
        )
        memory_pipeline = BidirectionalMemoryAVInferencePipeline(
            generator=generator,
            add_noise_fn=add_noise,
            denoising_sigmas=denoising_sigmas,
            memory_downscale_factor=1,
        )

        offloader = None
        if sequential_offload:
            offloader = SequentialOffloader(generator, device)

        # --- Phase A: Denoise (generator on GPU, everything else off) ---
        if sequential_offload:
            offloader.install()
        else:
            _move(generator, device)
        _empty_cache()

        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(seed)

            if len(memory_bank) > 0:
                _move(video_vae.encoder, device)
                memory_video = encode_memory_frames_batch(
                    video_vae=video_vae,
                    batch_memory_frames=[memory_bank.get_memory_frames()],
                    target_h=video_height,
                    target_w=video_width,
                    device=device,
                    dtype=dtype,
                )
                _move(video_vae.encoder, "cpu")
                _empty_cache()

                memory_audio_kwargs = build_paired_audio_memory_kwargs(
                    memory_bank,
                    enable_audio_memory=enable_audio_memory,
                    v2a_grad_scale=v2a_grad_scale,
                    memory_position_mode="reference",
                )

                video_latent, audio_latent = memory_pipeline.generate(
                    video_shape=tuple(video_shape),
                    audio_shape=tuple(audio_shape),
                    conditional_dict=conditional_dict,
                    memory_video=memory_video,
                    seed=seed,
                    **memory_audio_kwargs,
                )
                del memory_video
            else:
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

        audio_memory_latent = (
            audio_latent.detach().cpu().contiguous()
            if enable_audio_memory and audio_latent is not None
            else None
        )

        # --- Phase B: Decode ---
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

        # Update memory bank
        memory_frames_pil = video_uint8_to_pil_frames(video_uint8)
        if audio_memory_latent is not None:
            memory_bank.save_memory_slot(
                memory_frames_pil,
                audio_memory_latent,
                audio_window_size=audio_memory_window_size,
                video_clip_num_frames=9,
                audio_waveform=audio_waveform,
                audio_sample_rate=16000,
                video_fps=float(video_fps),
                audio_window_selection_mode="max_response",
                video_frame_selection_mode="center",
                audio_memory_mel_bins=128,
                audio_memory_mel_hop_length=160,
                audio_memory_n_fft=1024,
                audio_memory_downsample_factor=4,
                audio_memory_is_causal=True,
            )

        # Build outputs
        images = video_uint8.float() / 255.0  # [F, H, W, 3]

        audio_out = None
        if audio_waveform is not None:
            from ltx_distillation.inference.memory_multishot import normalize_audio_waveform_for_media
            audio_norm = normalize_audio_waveform_for_media(audio_waveform)
            audio_out = {
                "waveform": audio_norm.unsqueeze(0),  # [1, C, samples]
                "sample_rate": audio_sample_rate,
            }

        memory_out = {"bank": memory_bank}

        del video_latent, audio_latent, audio_memory_latent, video_uint8, audio_waveform
        _empty_cache()

        print(f"[JoyEcho] SingleShot done. {images.shape[0]} frames.", flush=True)

        return (images, audio_out, memory_out, model,)


_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

_DEFAULT_LONG_STORY_SYSTEM_PROMPT = ""
_long_sp_path = _PROMPTS_DIR / "long_story_writer_system_prompt.md"
if _long_sp_path.exists():
    _DEFAULT_LONG_STORY_SYSTEM_PROMPT = _long_sp_path.read_text(encoding="utf-8").strip()


def _load_system_prompt(mode: str) -> str:
    """Load the full system prompt from the bundled markdown file."""
    if "long" in mode:
        fp = _PROMPTS_DIR / "long_story_writer_system_prompt.md"
    else:
        fp = _PROMPTS_DIR / "short_story_writer_system_prompt.md"
    if fp.exists():
        return fp.read_text(encoding="utf-8").strip()
    raise FileNotFoundError(f"System prompt not found: {fp}")


class JoyEcho_PromptFormat:
    """Helper node providing the official prompt writing system prompts.

    Use this with any LLM node in ComfyUI to generate properly formatted
    shot prompts from a short story description.

    The output can be fed directly into JoyEcho_TextEncode.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["long_story (multi-shot)", "short_story (single-shot)"],),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("system_prompt",)
    FUNCTION = "get_prompt"
    CATEGORY = "JoyAI-Echo"

    def get_prompt(self, mode: str):
        return (_load_system_prompt(mode),)


class JoyEcho_LLMEnhance:
    """Call a cloud LLM API to expand a short story idea into JoyAI-Echo shot prompts.

    Supports OpenAI-compatible APIs (OpenAI, DeepSeek, etc.).
    The output JSON can be fed directly into JoyEcho_TextEncode or split via JoyEcho_PromptAtIndex.
    Uses only cloud API calls — zero local GPU memory.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "story_idea": ("STRING", {
                    "multiline": True,
                    "default": "A young woman records a quiet evening vlog in her cozy room, reflecting on life and finding warmth in small things.",
                    "tooltip": "Describe your story or scene idea in a few sentences.",
                }),
                "mode": (["long_story (multi-shot)", "short_story (single-shot)"],),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Your API key (OpenAI, DeepSeek, etc.)",
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": _DEFAULT_LONG_STORY_SYSTEM_PROMPT,
                    "tooltip": "System prompt for the LLM. Edit to customize prompt generation style.",
                }),
            },
            "optional": {
                "base_url": ("STRING", {
                    "default": "https://api.openai.com/v1",
                    "tooltip": "API base URL. Use https://api.deepseek.com/v1 for DeepSeek, etc.",
                }),
                "model_name": ("STRING", {
                    "default": "gpt-4o",
                    "tooltip": "Model name (gpt-4o, deepseek-chat, claude-3-5-sonnet, etc.)",
                }),
                "num_shots": ("INT", {
                    "default": 0, "min": 0, "max": 30,
                    "tooltip": "Number of shots to generate (0 = let LLM decide, default 15 for long story).",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05,
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompts_json",)
    FUNCTION = "enhance"
    CATEGORY = "JoyAI-Echo"

    def enhance(
        self,
        story_idea: str,
        mode: str,
        api_key: str,
        system_prompt: str,
        base_url: str = "https://api.openai.com/v1",
        model_name: str = "gpt-4o",
        num_shots: int = 0,
        temperature: float = 0.7,
    ):
        import urllib.request
        import urllib.error

        if not api_key.strip():
            raise ValueError("API key is required. Enter your OpenAI/DeepSeek/etc. API key.")

        if system_prompt.strip():
            sys_prompt = system_prompt.strip()
        else:
            sys_prompt = _load_system_prompt(mode)

        user_msg = story_idea.strip()
        if num_shots > 0:
            user_msg += f"\n\nGenerate exactly {num_shots} shots."

        url = base_url.rstrip("/") + "/chat/completions"
        payload = json.dumps({
            "model": model_name,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg},
            ],
            "temperature": temperature,
            "max_tokens": 16384,
        }).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key.strip()}",
        }

        print(f"[JoyEcho] Calling LLM ({model_name}) to enhance prompt...", flush=True)
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM API error {e.code}: {body}")

        content = result["choices"][0]["message"]["content"].strip()

        # Strip markdown code fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines).strip()

        # Validate JSON
        try:
            data = json.loads(content)
            if "prompts" not in data or not isinstance(data["prompts"], list):
                raise ValueError("LLM output missing 'prompts' array")
            num = len(data["prompts"])
        except (json.JSONDecodeError, ValueError) as e:
            raise RuntimeError(
                f"LLM returned invalid JSON: {e}\n\nRaw output:\n{content[:500]}"
            )

        print(f"[JoyEcho] LLM generated {num} shot prompt(s).", flush=True)
        return (content,)


class JoyEcho_PromptAtIndex:
    """Extract a single prompt from a JSON prompts array by index.

    Connect the output to a SingleShotGenerate node's prompt input to override
    the text box with LLM-generated content. This is optional — if not connected,
    the SingleShot node uses its own text box.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompts_json": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "JSON string with 'prompts' array (from LLM Enhance or file)",
                }),
                "index": ("INT", {
                    "default": 0, "min": 0, "max": 29,
                    "tooltip": "0-based shot index to extract",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "extract"
    CATEGORY = "JoyAI-Echo"

    def extract(self, prompts_json: str, index: int):
        text = prompts_json.strip()
        if not text:
            raise ValueError("No prompts JSON provided.")

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

        prompt_list = data.get("prompts") or data.get("shots") or []
        if not prompt_list:
            raise ValueError("JSON must contain a 'prompts' or 'shots' array.")

        if index >= len(prompt_list):
            raise ValueError(
                f"Index {index} out of range (only {len(prompt_list)} prompts available)."
            )

        return (str(prompt_list[index]).strip(),)
