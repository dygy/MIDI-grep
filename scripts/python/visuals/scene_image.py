"""Paint the scene with a self-hosted, one-step distilled diffusion model (diffusers; MPS on Apple Silicon).

Default ``stabilityai/sd-turbo`` (~2.6 GB fp16, 512-px native). ``stabilityai/sdxl-turbo`` (~7 GB) paints more detail
and is selected with ``--image-model``. Turbo models are trained without classifier-free guidance, so guidance is 0
and the negative prompt is unused by them; it is still recorded for models that take it.
"""

from __future__ import annotations

from pathlib import Path


def _device():
    import torch
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def paint(prompt: str, out: Path, model: str, seed: int, width: int, height: int, steps: int,
          negative_prompt: str = "") -> dict:
    import torch
    from diffusers import AutoPipelineForText2Image

    device, dtype = _device()
    kw = {"variant": "fp16"} if dtype == torch.float16 else {}
    try:
        # the local cache first: an online load re-checked the Hub and rewrote a verified weight file (2026-10-30)
        pipe = AutoPipelineForText2Image.from_pretrained(model, torch_dtype=dtype, local_files_only=True, **kw)
    except OSError:
        pipe = AutoPipelineForText2Image.from_pretrained(model, torch_dtype=dtype, **kw)  # first use: download
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    gen = torch.Generator("cpu").manual_seed(seed)  # CPU generator: the same seed paints the same image on any device
    turbo = "turbo" in model
    image = pipe(prompt=prompt, negative_prompt=None if turbo else (negative_prompt or None),
                 num_inference_steps=steps, guidance_scale=0.0 if turbo else 5.0,
                 width=width, height=height, generator=gen).images[0]
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out, quality=88)
    return {"model": model, "device": device, "seed": seed, "steps": steps, "size": [width, height], "file": out.name}
