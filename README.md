
# ComfyUI_JoyAI_Echo


<img width="1501" height="757" alt="Screenshot (156)" src="https://github.com/user-attachments/assets/e17a0e5e-e2f5-4761-b4b4-2f0da6633197" />



Run JD's JoyAI-Echo (text → video + audio) on consumer hardware.
A faithful GGUF implementation of JoyAI-Echo for ComfyUI, built so the model can run on an 8 GB GPU with 16 GB system RAM instead of the ~48 GB the full checkpoint expects.
> ⚠️ **Work in progress / experimental.** This is a heavy model on small hardware. Loads and the first encode are slow (see [Performance](#performance)). It works, but it asks for patience.
<!-- Add a preview image at assets/preview.png and it will show here -->

---
What this is:
JoyAI-Echo is a modified LTX-2.3 diffusion transformer that generates video and matching audio from a text prompt. The full release is a single ~46 GB bf16 checkpoint designed for big GPUs.
This pack makes it runnable on a typical gaming PC by:
Loading the DiT as a GGUF (Q2_K / Q4_K_M) instead of full bf16.
Building the transformer with JoyAI's own `LTXModelConfigurator` so the architecture is correct (stock ComfyUI GGUF loaders build the wrong shapes for this model).
Running the Gemma-3 text encoder in fp8, on CPU, and freeing it from RAM before the DiT loads — so the encoder and the DiT never sit in memory at the same time.
Streaming DiT blocks to the GPU one at a time during denoising.
All in a single node so you don't have to wire the memory dance yourself.
---
Requirements
ComfyUI (recent build).
ComfyUI-GGUF (city96) — provides the GGUF tooling this pack relies on.
A CUDA GPU. Designed and tested target: RTX 3070, 8 GB VRAM + 16 GB system RAM. More is better; less will struggle.
The model files (see Models).
---
Installation
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/RealRebelAI/ComfyUI_JoyAI_Echo_GGUF_Nodes.git
```
Install Python dependencies into your ComfyUI environment (portable users: use the embedded Python):
```bash
python_embeded\python.exe -m pip install -r ComfyUI_JoyAI_Echo_GGUF_Nodes\requirements.txt
```
Restart ComfyUI.
---
Models
Download the quantized files from the model repo:
➡️ https://huggingface.co/realrebelai/JoyAI-Echo_GGUF
Place each file in the matching ComfyUI folder. The node reads these folders directly, so the files appear as dropdowns:
File	Put it in
`JoyAI-Echo-DiT-Q2_K.gguf` (or Q4_K_M)	`ComfyUI/models/unet`
`joyai_echo_video_vae.safetensors`	`ComfyUI/models/vae`
`joyai_echo_audio_vae.safetensors`	`ComfyUI/models/vae`
`joyai_echo_vocoder.safetensors`	`ComfyUI/models/audio_encoders`
`joyai_echo_embeddings_processor.safetensors` (connector)	`ComfyUI/models/text_encoders`
Text encoder (Gemma-3, downloaded separately)
The text encoder is the standard Gemma-3-12B fp8 file — not rehosted here. Get `gemma_3_12B_it_fp8_scaled.safetensors` (Comfy-Org) and place it in:
```
ComfyUI/models/text_encoders/gemma/
```
Gemma also needs its small sidecar files. Put these five in a `gemma_assets/` folder inside the node pack or next to the Gemma `.safetensors`:
```
tokenizer.model
tokenizer_config.json
config.json
special_tokens_map.json
preprocessor_config.json
```
Get them from `google/gemma-3-12b-it` (the small config/tokenizer files only — not the weight shards). If any are missing, the node tells you exactly which ones and where to put them.
Architecture config
A `configs/joyai_echo_config.json` ships with the pack, and the node defaults to it automatically — no setup needed. If you ever need to regenerate it from the full checkpoint:
```bash
python dump_joyai_config.py --src D:\path\to\JoyAI-Echo-release.safetensors
```
---
Usage
Build a minimal graph:
```
RebelsJE_StagedPipeline  →  CreateVideo  →  SaveVideo
```
Select your files in the dropdowns, type a prompt, queue. The node runs three internal stages: encode → free Gemma → load DiT + VAEs → denoise → decode, and prints `[StagedJE]` progress lines to the console.
Key settings
Setting	What it does
`dit_gguf`	The DiT GGUF. Q2_K = smallest/fastest-to-load, Q4_K_M = higher quality, more RAM.
`gemma_path`	The Gemma-3 fp8 encoder.
`gemma_format`	`our_fp8` for the Comfy-Org fp8 file; `bf16` if you point at full-precision weights.
`encode_on_cpu`	Keep on. The 12 B encoder won't fit 8 GB VRAM, so it runs on CPU (slow but fits).
`sequential_offload`	Keep on. Streams DiT blocks to the GPU one at a time.
`num_frames` / `video_height` / `video_width`	Start small (e.g. 25 frames, 512×768). Raise only if you have headroom.
> First successful run? Start with the smallest settings, confirm you get output, *then* push resolution and frames.
---
Performance
Be realistic about an 8 GB / 16 GB box:
Text encode runs a 12 B model on CPU. Expect several minutes, with one CPU core pegged and ~12 GB RAM in use. It looks like a hang — it isn't.
DiT load reads several GB from disk. Put your models on an SSD; a spinning HDD makes this painful.
Denoising streams blocks to the GPU and is slow per step at this quantization. Keep frames/resolution low while testing.
Watch the console for `[Rebels JE] DiT GGUF swap: matched N/N Linear layers` — that confirms the DiT stayed quantized in RAM (the whole point). If it ever prints `0/N`, something's pointed at the wrong file.
---
Troubleshooting
Symptom	Fix
Dropdowns show the wrong files after updating	Combo widgets reset when fields change type. Re-select the correct file in every dropdown.
`No files matching 'preprocessor_config.json'`	Gemma sidecar files missing — add the five files to `gemma_assets/` (see Models).
`Cannot copy out of meta tensor`	You're on an old node version; update — the text-only Gemma path is handled internally now.
`ArrayMemoryError` during DiT load	RAM exhausted from full dequant. Check the `matched N/N` line; make sure `dit_gguf` points at the real JoyAI-Echo DiT GGUF.
DiT GGUF not showing in dropdown	It scans `models/unet` and `models/diffusion_models`; confirm the `.gguf` is actually there, then restart ComfyUI.
Encode seems frozen	CPU encode of a 12 B model is just slow. Give it a few minutes.
---
Acknowledgements
JoyAI-Echo — Echo Team @ Joy Future Academy, JD
LTX-2.3 — Lightricks
Gemma-3 — Google
ComfyUI-GGUF — city96
Forked from zhuang2002/ComfyUI_JoyAI_Echo.
---
License
For academic research and non-commercial use only, following the upstream JoyAI-Echo license. Gemma-3 components are subject to Google's Gemma license; review those terms before redistributing any Gemma files.
