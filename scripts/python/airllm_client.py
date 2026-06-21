#!/usr/bin/env python3
"""AirLLM-backed LLM client that mirrors the Ollama HTTP surface.

AirLLM (https://github.com/lyogavin/airllm) runs large models layer-by-layer
to fit in low VRAM. Tradeoff: per-token inference is much slower than Ollama
because each layer is paged in from disk. Use it when you want a much larger
model (e.g. Qwen2.5-Coder-32B, Llama-3.1-70B) than your GPU can hold.

This module exposes one public function:

    generate_completion(prompt: str, model: str, options: dict) -> dict

Returned dict has the same {"response": "..."} shape as Ollama's
/api/generate, so callers can swap backends without changing parsing code.

The actual airllm import is lazy — installing this project does NOT require
the airllm package. If LLM_BACKEND is left at the default ("ollama"), this
module is never loaded.

Usage:
    LLM_BACKEND=airllm \\
    AIRLLM_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct \\
    ./bin/midi-grep extract --url ...

Recommended models for music+code understanding:
    - Qwen/Qwen2.5-Coder-32B-Instruct       (good balance)
    - meta-llama/Meta-Llama-3.1-70B-Instruct (broader knowledge, slower)
    - Qwen/Qwen2.5-72B-Instruct             (largest, requires ~150GB disk)
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Any, Dict, Optional

DEFAULT_MODEL = os.environ.get("AIRLLM_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")
DEFAULT_COMPRESSION = os.environ.get("AIRLLM_COMPRESSION", "")  # "4bit", "8bit", or ""
DEFAULT_PROFILING = os.environ.get("AIRLLM_PROFILING_MODE", "0") == "1"

# Singleton model — loading is expensive (downloads model files, builds layer index).
_model_lock = threading.Lock()
_loaded_model: Optional[Any] = None
_loaded_model_id: Optional[str] = None


class AirLLMUnavailable(RuntimeError):
    """Raised when the airllm package isn't installed or model fails to load.

    Callers should catch this and fall back to Ollama or another backend
    rather than crashing the pipeline.
    """


def _load_model(model_id: str) -> Any:
    """Load the airllm AutoModel, caching by model_id."""
    global _loaded_model, _loaded_model_id

    with _model_lock:
        if _loaded_model is not None and _loaded_model_id == model_id:
            return _loaded_model

        try:
            # Lazy import — airllm is optional
            from airllm import AutoModel  # type: ignore
        except ImportError as e:
            raise AirLLMUnavailable(
                "airllm package not installed. Run: "
                "pip install airllm  (or set LLM_BACKEND=ollama)"
            ) from e

        kwargs: Dict[str, Any] = {}
        if DEFAULT_COMPRESSION in ("4bit", "8bit"):
            kwargs["compression"] = DEFAULT_COMPRESSION
        if DEFAULT_PROFILING:
            kwargs["profiling_mode"] = True

        print(f"[airllm] Loading {model_id}... (this can take minutes on first run)",
              file=sys.stderr)
        try:
            model = AutoModel.from_pretrained(model_id, **kwargs)
        except Exception as e:
            raise AirLLMUnavailable(f"Failed to load airllm model {model_id}: {e}") from e

        _loaded_model = model
        _loaded_model_id = model_id
        print(f"[airllm] Loaded {model_id}", file=sys.stderr)
        return model


def generate_completion(
    prompt: str,
    model: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate a completion via AirLLM.

    Args mirror the parts of the Ollama /api/generate body that the rest of the
    codebase actually uses:
        prompt: the user prompt
        model: model id (default: AIRLLM_MODEL env var)
        options:
            num_predict: max new tokens (default 4096)
            temperature: sampling temperature (default 0.7)

    Returns:
        {"response": "<generated text>"}  — same shape as Ollama's response.

    Raises:
        AirLLMUnavailable: if airllm is missing or model can't be loaded.
    """
    options = options or {}
    model_id = model or DEFAULT_MODEL
    max_new_tokens = int(options.get("num_predict", 4096))
    temperature = float(options.get("temperature", 0.7))

    m = _load_model(model_id)

    # AirLLM works directly with the underlying HF tokenizer.
    enc = m.tokenizer(
        [prompt],
        return_tensors="pt",
        return_attention_mask=False,
        truncation=True,
        max_length=int(options.get("num_ctx", 32768)),
        padding=False,
    )

    out_ids = m.generate(
        enc["input_ids"],
        max_new_tokens=max_new_tokens,
        use_cache=True,
        return_dict_in_generate=False,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-5),
    )

    # AirLLM returns the prompt + completion concatenated; strip the prompt back off.
    full_text = m.tokenizer.decode(out_ids[0], skip_special_tokens=True)
    if full_text.startswith(prompt):
        completion = full_text[len(prompt):]
    else:
        # Some tokenizers add/eat whitespace — fall back to the full decode
        completion = full_text

    return {"response": completion}


def is_available() -> bool:
    """Cheap probe: does the airllm package import cleanly?"""
    try:
        import airllm  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Dispatcher: lets callers stay backend-agnostic
# ---------------------------------------------------------------------------

def dispatch_generate(
    prompt: str,
    *,
    model: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    ollama_url: str = "http://localhost:11434",
    timeout: int = 300,
) -> Dict[str, Any]:
    """Generate a completion via the backend selected by LLM_BACKEND.

    LLM_BACKEND values:
        "ollama" (default) — POST to Ollama's /api/generate
        "airllm"           — call AirLLM in-process

    Returns the same dict shape regardless of backend: {"response": "<text>"}.

    On AirLLM failure, raises AirLLMUnavailable so the caller can fall back.
    """
    backend = (os.environ.get("LLM_BACKEND") or "ollama").lower()

    if backend == "airllm":
        return generate_completion(prompt, model=model, options=options)

    # Default: Ollama HTTP
    import requests
    response = requests.post(
        f"{ollama_url}/api/generate",
        json={
            "model": model or os.environ.get("OLLAMA_MODEL", "midi-grep-strudel-mistral"),
            "prompt": prompt,
            "stream": False,
            "options": options or {},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


__all__ = [
    "AirLLMUnavailable",
    "dispatch_generate",
    "generate_completion",
    "is_available",
]
