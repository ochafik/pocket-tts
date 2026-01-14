#!/usr/bin/env python3
# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""CLI for Pocket-TTS MLX."""

import logging

import mlx.core as mx
import typer
from typing_extensions import Annotated

from pocket_tts_mlx.models.tts_model import (
    TTSModel,
    DEFAULT_VARIANT,
    DEFAULT_TEMPERATURE,
    DEFAULT_LSD_DECODE_STEPS,
    DEFAULT_NOISE_CLAMP,
    DEFAULT_EOS_THRESHOLD,
)
from pocket_tts_mlx.utils.audio import stream_audio_chunks
from pocket_tts_mlx.utils.loaders import PREDEFINED_VOICES

logger = logging.getLogger(__name__)

cli_app = typer.Typer(
    help="Kyutai Pocket TTS MLX - Text-to-Speech generation tool for Apple Silicon",
    pretty_exceptions_show_locals=False,
)


@cli_app.command()
def generate(
    text: Annotated[
        str, typer.Option(help="Text to generate")
    ] = "Hello world. I am Kyutai's Pocket TTS running on MLX. I hope you'll like me.",
    voice: Annotated[
        str, typer.Option(help="Voice name or path to audio file for voice cloning")
    ] = "cosette",
    quiet: Annotated[bool, typer.Option("-q", "--quiet", help="Disable logging")] = False,
    variant: Annotated[str, typer.Option(help="Model variant")] = DEFAULT_VARIANT,
    lsd_decode_steps: Annotated[
        int, typer.Option(help="Number of LSD decoding steps")
    ] = DEFAULT_LSD_DECODE_STEPS,
    temperature: Annotated[
        float, typer.Option(help="Sampling temperature")
    ] = DEFAULT_TEMPERATURE,
    noise_clamp: Annotated[float, typer.Option(help="Noise clamp value")] = DEFAULT_NOISE_CLAMP,
    eos_threshold: Annotated[float, typer.Option(help="EOS detection threshold")] = DEFAULT_EOS_THRESHOLD,
    frames_after_eos: Annotated[
        int, typer.Option(help="Frames to generate after EOS")
    ] = 3,
    output_path: Annotated[
        str, typer.Option("-o", "--output", help="Output path (use '-' for stdout)")
    ] = "./tts_output.wav",
    seed: Annotated[
        int | None, typer.Option(help="Random seed for reproducibility")
    ] = None,
):
    """Generate speech using Pocket TTS on MLX."""
    # Setup logging
    log_level = logging.ERROR if quiet else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Set random seed if provided
    if seed is not None:
        mx.random.seed(seed)
        logger.info(f"Random seed set to {seed}")

    logger.info("Loading TTS model...")
    tts_model = TTSModel.load_model(
        variant=variant,
        temp=temperature,
        lsd_decode_steps=lsd_decode_steps,
        noise_clamp=noise_clamp,
        eos_threshold=eos_threshold,
    )
    logger.info("Model loaded successfully")

    # Generate audio stream
    logger.info(f"Generating speech for: '{text[:50]}...' with voice '{voice}'")

    # Handle voice
    voice_arg = voice if voice in PREDEFINED_VOICES else voice

    audio_chunks = tts_model.generate_audio_stream(
        text=text,
        voice=voice_arg,
        frames_after_eos=frames_after_eos,
    )

    # Stream to output
    stream_audio_chunks(output_path, audio_chunks, tts_model.sample_rate)

    if output_path != "-":
        logger.info(f"Audio saved to {output_path}")


@cli_app.command()
def list_voices():
    """List available predefined voices."""
    print("Available predefined voices:")
    for voice in sorted(PREDEFINED_VOICES.keys()):
        print(f"  - {voice}")


def main():
    cli_app()


if __name__ == "__main__":
    main()
