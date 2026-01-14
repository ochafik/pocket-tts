# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Utility functions for loading weights and downloading files."""

import hashlib
import logging
from pathlib import Path

import mlx.core as mx
import requests
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

_voices_names = ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]
PREDEFINED_VOICES = {
    x: f"hf://kyutai/pocket-tts-without-voice-cloning/embeddings/{x}.safetensors"
    for x in _voices_names
}


def make_cache_directory() -> Path:
    """Create and return the cache directory."""
    cache_dir = Path.home() / ".cache" / "pocket_tts_mlx"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_if_necessary(file_path: str) -> Path:
    """Download file if it's a URL or HuggingFace path.

    Args:
        file_path: Local path, HTTP URL, or HuggingFace path (hf://...)

    Returns:
        Path to the local file
    """
    if file_path.startswith("http://") or file_path.startswith("https://"):
        cache_dir = make_cache_directory()
        cached_file = cache_dir / (
            hashlib.sha256(file_path.encode()).hexdigest() + "." + file_path.split(".")[-1]
        )
        if not cached_file.exists():
            logger.info("Downloading %s", file_path)
            response = requests.get(file_path)
            response.raise_for_status()
            with open(cached_file, "wb") as f:
                f.write(response.content)
        return cached_file
    elif file_path.startswith("hf://"):
        file_path = file_path.removeprefix("hf://")
        splitted = file_path.split("/")
        repo_id = "/".join(splitted[:2])
        filename = "/".join(splitted[2:])
        if "@" in filename:
            filename, revision = filename.split("@")
        else:
            revision = None
        cached_file = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
        return Path(cached_file)
    else:
        return Path(file_path)


_voice_cache: dict[str, "mx.array"] = {}


def load_predefined_voice(voice_name: str) -> mx.array:
    """Load a predefined voice embedding (cached).

    Args:
        voice_name: Name of the predefined voice

    Returns:
        Audio prompt tensor
    """
    # Check cache first
    if voice_name in _voice_cache:
        return _voice_cache[voice_name]

    if voice_name not in PREDEFINED_VOICES:
        raise ValueError(
            f"Predefined voice '{voice_name}' not found, "
            f"available voices are {list(PREDEFINED_VOICES)}."
        )
    voice_file = download_if_necessary(PREDEFINED_VOICES[voice_name])
    weights = mx.load(str(voice_file))
    audio_prompt = weights["audio_prompt"]

    # Cache for future requests
    _voice_cache[voice_name] = audio_prompt
    return audio_prompt


def load_safetensors_weights(file_path: str) -> dict[str, mx.array]:
    """Load weights from a safetensors file.

    Args:
        file_path: Path to the safetensors file

    Returns:
        Dictionary of weight tensors
    """
    file_path = download_if_necessary(file_path)
    return mx.load(str(file_path))


def convert_conv_weights(key: str, weight: mx.array) -> mx.array:
    """Convert Conv1d/ConvTranspose1d weights from PyTorch to MLX format.

    PyTorch Conv1d: (outC, inC, kSize)
    MLX Conv1d: (outC, kSize, inC)

    PyTorch ConvTranspose1d: (inC, outC, kSize)
    MLX ConvTranspose1d: (outC, kSize, inC)
    """
    if "conv.weight" in key or "output_proj.weight" in key:
        # Conv1d: swap last two axes
        return weight.swapaxes(-1, -2)
    elif "convtr.weight" in key:
        # ConvTranspose1d: transpose (in, out, k) -> (out, k, in)
        return weight.transpose(1, 2, 0)
    return weight


def remap_weight_key(key: str) -> str | None:
    """Remap PyTorch weight key to MLX format.

    Args:
        key: Original PyTorch weight key

    Returns:
        Remapped key for MLX model, or None to skip
    """
    import re

    # Skip certain keys
    skip_prefixes = [
        "flow.w_s_t.",  # Not used
        "condition_provider.conditioners.transcript_in_segment.learnt_padding",
        "condition_provider.conditioners.speaker_wavs.learnt_padding",
        "model.quantizer.vq.",  # VQ weights not used
        "model.quantizer.logvar_proj.",
    ]
    for prefix in skip_prefixes:
        if key.startswith(prefix) or key == prefix:
            return None  # Signal to skip this key

    # Remove leading underscores from module names
    parts = key.split(".")
    parts = [p.removeprefix("_") for p in parts]
    key = ".".join(parts)

    # FlowLM conditioner remapping
    if key == "condition_provider.conditioners.transcript_in_segment.embed.weight":
        return "flow_lm.conditioner.embed.weight"
    if key == "condition_provider.conditioners.speaker_wavs.output_proj.weight":
        return "flow_lm.speaker_proj_weight"

    # Mimi model prefix
    if key.startswith("model."):
        key = "mimi." + key.removeprefix("model.")

    # FlowLM flow_net MLP remappings (PyTorch Sequential -> MLX named attributes)
    # TimestepEmbedder: mlp.0 -> linear1, mlp.2 -> linear2, mlp.3 -> norm
    key = re.sub(r"flow_lm\.flow_net\.time_embed\.(\d+)\.mlp\.0\.",
                 r"flow_lm.flow_net.time_embed.\1.linear1.", key)
    key = re.sub(r"flow_lm\.flow_net\.time_embed\.(\d+)\.mlp\.2\.",
                 r"flow_lm.flow_net.time_embed.\1.linear2.", key)
    key = re.sub(r"flow_lm\.flow_net\.time_embed\.(\d+)\.mlp\.3\.",
                 r"flow_lm.flow_net.time_embed.\1.norm.", key)

    # ResBlock: mlp.0 -> mlp_linear1, mlp.2 -> mlp_linear2
    key = re.sub(r"flow_lm\.flow_net\.res_blocks\.(\d+)\.mlp\.0\.",
                 r"flow_lm.flow_net.res_blocks.\1.mlp_linear1.", key)
    key = re.sub(r"flow_lm\.flow_net\.res_blocks\.(\d+)\.mlp\.2\.",
                 r"flow_lm.flow_net.res_blocks.\1.mlp_linear2.", key)

    # ResBlock: adaLN_modulation.1 -> adaLN_linear
    key = re.sub(r"flow_lm\.flow_net\.res_blocks\.(\d+)\.adaLN_modulation\.1\.",
                 r"flow_lm.flow_net.res_blocks.\1.adaLN_linear.", key)

    # FinalLayer: adaLN_modulation.1 -> adaLN_linear
    key = re.sub(r"flow_lm\.flow_net\.final_layer\.adaLN_modulation\.1\.",
                 r"flow_lm.flow_net.final_layer.adaLN_linear.", key)

    # SEANet model indices match directly since MLX model now includes ELU layers
    # No index remapping needed

    # SEANet resblock internal remappings
    # PyTorch: block.1 (conv after ELU at 0), block.3 (conv after ELU at 2)
    # MLX: block.0 (first conv), block.1 (second conv)
    key = re.sub(r"(mimi\.(encoder|decoder)\.model\.\d+)\.block\.1\.",
                 r"\1.block.0.", key)
    key = re.sub(r"(mimi\.(encoder|decoder)\.model\.\d+)\.block\.3\.",
                 r"\1.block.1.", key)

    return key
