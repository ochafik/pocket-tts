# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Audio I/O utilities for MLX."""

import logging
import os
import sys
import wave
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterator

import mlx.core as mx
import numpy as np
from scipy.signal import resample_poly
from math import gcd

logger = logging.getLogger(__name__)

FIRST_CHUNK_LENGTH_SECONDS = float(os.environ.get("FIRST_CHUNK_LENGTH_SECONDS", "0"))


def audio_read(filepath: str | Path) -> tuple[mx.array, int]:
    """Read audio from WAV file.

    Args:
        filepath: Path to WAV file

    Returns:
        Tuple of (audio tensor [1, T], sample_rate)
    """
    with wave.open(str(filepath), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        n_channels = wav_file.getnchannels()

        # Read all audio data as 16-bit signed integers
        raw_data = wav_file.readframes(-1)
        samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0

        # Convert to mono if stereo
        if n_channels > 1:
            samples = samples.reshape(-1, n_channels).mean(axis=1)

        # Return as [1, T] tensor
        return mx.array(samples.reshape(1, -1)), sample_rate


def convert_audio(
    wav: mx.array,
    from_rate: int,
    to_rate: int,
    to_channels: int,
) -> mx.array:
    """Convert audio sample rate and channels.

    Args:
        wav: Audio tensor [C, T] or [T]
        from_rate: Source sample rate
        to_rate: Target sample rate
        to_channels: Target number of channels

    Returns:
        Converted audio tensor
    """
    # Ensure 2D
    if wav.ndim == 1:
        wav = wav.reshape(1, -1)

    # Resample if needed
    if from_rate != to_rate:
        # Convert to numpy for scipy resampling
        wav_np = np.array(wav)
        factor = gcd(from_rate, to_rate)
        up = to_rate // factor
        down = from_rate // factor
        resampled = resample_poly(wav_np, up, down, axis=-1)
        wav = mx.array(resampled)

    # Handle channels
    current_channels = wav.shape[0]
    if to_channels == 1 and current_channels > 1:
        wav = wav.mean(axis=0, keepdims=True)
    elif to_channels > current_channels:
        wav = mx.repeat(wav, to_channels, axis=0)

    return wav


class StreamingWAVWriter:
    """WAV writer for streaming output."""

    def __init__(self, output_stream, sample_rate: int):
        self.output_stream = output_stream
        self.sample_rate = sample_rate
        self.wave_writer = None
        self.first_chunk_buffer = []
        self.total_frames = 0

    def write_header(self, sample_rate: int):
        """Initialize WAV writer with header."""
        self.wave_writer = wave.open(self.output_stream, "wb")
        self.wave_writer.setnchannels(1)  # Mono
        self.wave_writer.setsampwidth(2)  # 16-bit
        self.wave_writer.setframerate(sample_rate)
        self.wave_writer.setnframes(1_000_000_000)  # Large placeholder for streaming

    def write_pcm_data(self, audio_chunk: mx.array):
        """Write PCM data chunk."""
        # Convert MLX array to int16 PCM bytes
        chunk_np = np.array(audio_chunk)
        chunk_np = np.clip(chunk_np, -1, 1)
        chunk_int16 = (chunk_np * 32767).astype(np.int16)
        chunk_bytes = chunk_int16.tobytes()
        self.total_frames += len(chunk_int16)

        if self.first_chunk_buffer is not None:
            self.first_chunk_buffer.append(chunk_bytes)
            total_length = sum(len(c) for c in self.first_chunk_buffer)
            target_length = int(self.sample_rate * FIRST_CHUNK_LENGTH_SECONDS) * 2
            if total_length < target_length:
                return
            self._flush()
            return

        self.wave_writer.writeframesraw(chunk_bytes)

    def _flush(self):
        if self.first_chunk_buffer is not None:
            self.wave_writer.writeframesraw(b"".join(self.first_chunk_buffer))
            self.first_chunk_buffer = None

    def finalize(self):
        """Close the wave writer."""
        self._flush()

        # Add 200ms of silence for proper playback
        silence_duration_sec = 0.2
        num_silence_samples = int(self.sample_rate * silence_duration_sec)
        self.wave_writer.writeframesraw(bytes(num_silence_samples * 2))

        if self.wave_writer:
            # Don't try to patch the header for streaming
            self.wave_writer._patchheader = lambda: None
            self.wave_writer.close()


def is_file_like(obj):
    """Check if object has basic file-like methods."""
    return all(hasattr(obj, attr) for attr in ["write", "close"])


def stream_audio_chunks(
    path: str | Path | None | Any,
    audio_chunks: Iterator[mx.array],
    sample_rate: int,
):
    """Stream audio chunks to a WAV file or stdout.

    Args:
        path: Output path, "-" for stdout, or None to discard
        audio_chunks: Iterator of audio chunk tensors
        sample_rate: Sample rate for output
    """
    if path == "-":
        f = sys.stdout.buffer
    elif path is None:
        f = nullcontext()
    elif is_file_like(path):
        f = path
    else:
        f = open(path, "wb")

    with f:
        if path is not None:
            writer = StreamingWAVWriter(f, sample_rate)
            writer.write_header(sample_rate)

        for chunk in audio_chunks:
            if path is not None:
                writer.write_pcm_data(chunk)

        if path is not None:
            writer.finalize()
