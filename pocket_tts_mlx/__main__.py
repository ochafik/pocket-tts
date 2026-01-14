#!/usr/bin/env python3
# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""CLI for Pocket-TTS MLX."""

import io
import logging
import os
import tempfile
import threading
from pathlib import Path
from queue import Queue

import mlx.core as mx
import typer
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
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

# ------------------------------------------------------
# The pocket-tts-mlx server implementation
# ------------------------------------------------------

# Global model instance
tts_model: TTSModel | None = None
default_voice: str = "cosette"

web_app = FastAPI(
    title="Kyutai Pocket TTS MLX API",
    description="Text-to-Speech generation API (MLX)",
    version="1.0.0",
)
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8000",
        "https://kyutai.org",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@web_app.get("/")
async def root():
    """Serve the frontend (reuses pocket_tts static files)."""
    # Try to find static files from pocket_tts package
    static_path = Path(__file__).parent.parent / "pocket_tts" / "static" / "index.html"
    if not static_path.exists():
        # Fallback: try installed package location
        import pocket_tts
        static_path = Path(pocket_tts.__file__).parent / "static" / "index.html"
    if not static_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(static_path)


@web_app.get("/health")
async def health():
    return {"status": "healthy", "backend": "mlx"}


def _generate_audio_to_queue(queue: Queue, text: str, voice: str | None):
    """Generate audio in a thread and write chunks to queue."""
    try:
        voice_to_use = voice if voice else default_voice

        class QueueWriter(io.IOBase):
            def write(self, data):
                queue.put(data)
            def flush(self):
                pass
            def close(self):
                queue.put(None)

        audio_chunks = tts_model.generate_audio_stream(
            text=text,
            voice=voice_to_use,
        )
        stream_audio_chunks(QueueWriter(), audio_chunks, tts_model.sample_rate)
    except Exception as e:
        logger.exception(f"Error generating audio: {e}")
        queue.put(None)


def _stream_audio(text: str, voice: str | None):
    """Stream audio chunks as they're generated."""
    queue = Queue()
    thread = threading.Thread(
        target=_generate_audio_to_queue,
        args=(queue, text, voice),
        daemon=True,
    )
    thread.start()

    while True:
        data = queue.get()
        if data is None:
            break
        yield data

    thread.join(timeout=5.0)


@web_app.post("/tts")
def text_to_speech(
    text: str = Form(...),
    voice_url: str | None = Form(None),
    voice_wav: UploadFile | None = File(None),
):
    """Generate speech from text.

    Args:
        text: Text to convert to speech
        voice_url: Optional predefined voice name or URL (http://, https://, hf://)
        voice_wav: Optional uploaded voice file (mutually exclusive with voice_url)
    """
    if tts_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if voice_url is not None and voice_wav is not None:
        raise HTTPException(
            status_code=400,
            detail="Cannot provide both voice_url and voice_wav",
        )

    # Determine voice to use
    voice = None
    if voice_url is not None:
        if voice_url in PREDEFINED_VOICES:
            voice = voice_url
        elif voice_url.startswith(("http://", "https://", "hf://")):
            voice = voice_url
        else:
            raise HTTPException(
                status_code=400,
                detail=f"voice_url must be a predefined voice {list(PREDEFINED_VOICES.keys())} "
                       "or start with http://, https://, or hf://",
            )
    elif voice_wav is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = voice_wav.file.read()
            temp_file.write(content)
            temp_file.flush()
            voice = temp_file.name
        # Note: temp file will be cleaned up after generation

    return StreamingResponse(
        _stream_audio(text, voice),
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=generated_speech.wav",
            "Transfer-Encoding": "chunked",
        },
    )


# ------------------------------------------------------
# CLI commands
# ------------------------------------------------------

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


@cli_app.command()
def serve(
    voice: Annotated[
        str, typer.Option(help="Default voice for TTS generation")
    ] = "cosette",
    host: Annotated[str, typer.Option(help="Host to bind to")] = "localhost",
    port: Annotated[int, typer.Option(help="Port to bind to")] = 8000,
    variant: Annotated[str, typer.Option(help="Model variant")] = DEFAULT_VARIANT,
    lsd_decode_steps: Annotated[
        int, typer.Option(help="Number of LSD decoding steps")
    ] = DEFAULT_LSD_DECODE_STEPS,
    temperature: Annotated[
        float, typer.Option(help="Sampling temperature")
    ] = DEFAULT_TEMPERATURE,
    noise_clamp: Annotated[float, typer.Option(help="Noise clamp value")] = DEFAULT_NOISE_CLAMP,
    eos_threshold: Annotated[float, typer.Option(help="EOS detection threshold")] = DEFAULT_EOS_THRESHOLD,
):
    """Start the FastAPI server for TTS generation."""
    global tts_model, default_voice

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Loading TTS model...")
    tts_model = TTSModel.load_model(
        variant=variant,
        temp=temperature,
        lsd_decode_steps=lsd_decode_steps,
        noise_clamp=noise_clamp,
        eos_threshold=eos_threshold,
    )
    logger.info("Model loaded successfully")

    default_voice = voice
    logger.info(f"Default voice set to: {default_voice}")

    logger.info(f"Starting server at http://{host}:{port}")
    uvicorn.run(web_app, host=host, port=port)


def main():
    cli_app()


if __name__ == "__main__":
    main()
