import asyncio
import base64
import io
import logging
import os
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from typing import Literal

import torch
import typer
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing_extensions import Annotated

from pocket_tts.data.audio import stream_audio_chunks
from pocket_tts.default_parameters import (
    DEFAULT_AUDIO_PROMPT,
    DEFAULT_EOS_THRESHOLD,
    DEFAULT_FRAMES_AFTER_EOS,
    DEFAULT_LSD_DECODE_STEPS,
    DEFAULT_NOISE_CLAMP,
    DEFAULT_TEMPERATURE,
    DEFAULT_VARIANT,
)
from pocket_tts.models.tts_model import TTSModel, prepare_text_prompt
from pocket_tts.streaming import StreamingTextChunker
from pocket_tts.utils.logging_utils import enable_logging
from pocket_tts.utils.utils import PREDEFINED_VOICES, size_of_dict

logger = logging.getLogger(__name__)

cli_app = typer.Typer(
    help="Kyutai Pocket TTS - Text-to-Speech generation tool", pretty_exceptions_show_locals=False
)


# ------------------------------------------------------
# The pocket-tts server implementation
# ------------------------------------------------------

# Global model instance
tts_model = None
global_model_state = None

web_app = FastAPI(
    title="Kyutai Pocket TTS API", description="Text-to-Speech generation API", version="1.0.0"
)
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://pod1-10007.internal.kyutai.org",
        "https://kyutai.org",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@web_app.get("/")
async def root():
    """Serve the frontend."""
    static_path = Path(__file__).parent / "static" / "index.html"
    return FileResponse(static_path)


@web_app.get("/health")
async def health():
    return {"status": "healthy"}


def write_to_queue(
    queue,
    text_to_generate,
    model_state,
    temperature: float | None = None,
    lsd_decode_steps: int | None = None,
    noise_clamp: float | None = None,
    eos_threshold: float | None = None,
):
    """Allows writing to the StreamingResponse as if it were a file."""

    class FileLikeToQueue(io.IOBase):
        def __init__(self, queue):
            self.queue = queue

        def write(self, data):
            self.queue.put(data)

        def flush(self):
            pass

        def close(self):
            self.queue.put(None)

    audio_chunks = tts_model.generate_audio_stream(
        model_state=model_state,
        text_to_generate=text_to_generate,
        temperature=temperature,
        lsd_decode_steps=lsd_decode_steps,
        noise_clamp=noise_clamp,
        eos_threshold=eos_threshold,
    )
    stream_audio_chunks(FileLikeToQueue(queue), audio_chunks, tts_model.config.mimi.sample_rate)


def generate_data_with_state(
    text_to_generate: str,
    model_state: dict,
    temperature: float | None = None,
    lsd_decode_steps: int | None = None,
    noise_clamp: float | None = None,
    eos_threshold: float | None = None,
):
    queue = Queue()

    # Run your function in a thread
    thread = threading.Thread(
        target=write_to_queue,
        args=(queue, text_to_generate, model_state, temperature, lsd_decode_steps, noise_clamp, eos_threshold),
    )
    thread.start()

    # Yield data as it becomes available
    i = 0
    while True:
        data = queue.get()
        if data is None:
            break
        i += 1
        yield data

    thread.join()


@web_app.post("/tts")
def text_to_speech(
    text: str = Form(...),
    voice_url: str | None = Form(None),
    voice_wav: UploadFile | None = File(None),
    temperature: float | None = Form(None),
    lsd_decode_steps: int | None = Form(None),
    noise_clamp: float | None = Form(None),
    eos_threshold: float | None = Form(None),
):
    """
    Generate speech from text using the pre-loaded voice prompt or a custom voice.

    Args:
        text: Text to convert to speech
        voice_url: Optional voice URL (http://, https://, or hf://)
        voice_wav: Optional uploaded voice file (mutually exclusive with voice_url)
        temperature: Sampling temperature (overrides server default)
        lsd_decode_steps: LSD decoding steps (overrides server default)
        noise_clamp: Noise clamp value (overrides server default)
        eos_threshold: EOS detection threshold (overrides server default)
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if voice_url is not None and voice_wav is not None:
        raise HTTPException(status_code=400, detail="Cannot provide both voice_url and voice_wav")

    # Use the appropriate model state
    if voice_url is not None:
        if not (
            voice_url.startswith("http://")
            or voice_url.startswith("https://")
            or voice_url.startswith("hf://")
            or voice_url in PREDEFINED_VOICES
        ):
            raise HTTPException(
                status_code=400, detail="voice_url must start with http://, https://, or hf://"
            )
        model_state = tts_model._cached_get_state_for_audio_prompt(voice_url, truncate=True)
        logging.warning("Using voice from URL: %s", voice_url)
    elif voice_wav is not None:
        # Use uploaded voice file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = voice_wav.file.read()
            temp_file.write(content)
            temp_file.flush()

            try:
                model_state = tts_model.get_state_for_audio_prompt(
                    Path(temp_file.name), truncate=True
                )
            finally:
                os.unlink(temp_file.name)
    else:
        # Use default global model state
        model_state = global_model_state

    return StreamingResponse(
        generate_data_with_state(
            text,
            model_state,
            temperature=temperature,
            lsd_decode_steps=lsd_decode_steps,
            noise_clamp=noise_clamp,
            eos_threshold=eos_threshold,
        ),
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=generated_speech.wav",
            "Transfer-Encoding": "chunked",
        },
    )


# ------------------------------------------------------
# WebSocket streaming endpoint
# ------------------------------------------------------


@web_app.websocket("/tts/stream")
async def websocket_tts_stream(websocket: WebSocket):
    """WebSocket endpoint for streaming text-to-speech.

    Protocol:
    1. Client sends {"type": "start", "voice": "...", ...} to initialize
    2. Client sends {"type": "text", "content": "..."} as text arrives
    3. Client sends {"type": "end"} when text stream completes
    4. Server sends binary audio frames (PCM int16, 24kHz mono) as generated
    5. Server sends {"type": "done", ...} when complete
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())[:8]
    logger.info(f"WebSocket session {session_id} connected")

    try:
        # Wait for start message
        start_msg = await websocket.receive_json()
        if start_msg.get("type") != "start":
            await websocket.send_json({
                "type": "error",
                "message": "First message must be type 'start'"
            })
            await websocket.close()
            return

        # Extract settings
        voice = start_msg.get("voice", DEFAULT_AUDIO_PROMPT)
        settings = {
            k: start_msg.get(k)
            for k in ["temperature", "lsd_decode_steps", "noise_clamp", "eos_threshold"]
            if start_msg.get(k) is not None
        }

        # Get model state for voice
        model_state = tts_model._cached_get_state_for_audio_prompt(voice, truncate=True)

        # Send started confirmation
        await websocket.send_json({
            "type": "started",
            "session_id": session_id,
            "sample_rate": tts_model.config.mimi.sample_rate,
            "channels": 1,
            "bit_depth": 16,
        })

        # Create chunker
        chunker = StreamingTextChunker(tts_model.flow_lm.conditioner.tokenizer)

        chunk_index = 0
        char_offset = 0  # Track cumulative character offset for karaoke sync
        total_audio_samples = 0

        # Process incoming messages
        while True:
            msg = await websocket.receive_json()
            msg_type = msg.get("type")

            if msg_type == "text":
                content = msg.get("content", "")
                chunks = chunker.add_text(content)

                for chunk_text in chunks:
                    chunk_index += 1
                    char_length = len(chunk_text)
                    samples = await _process_chunk_websocket(
                        websocket, chunk_text, chunk_index, char_offset, model_state, settings
                    )
                    total_audio_samples += samples
                    char_offset += char_length

            elif msg_type == "end":
                # Flush remaining text
                chunks = chunker.flush()
                for chunk_text in chunks:
                    chunk_index += 1
                    char_length = len(chunk_text)
                    samples = await _process_chunk_websocket(
                        websocket, chunk_text, chunk_index, char_offset, model_state, settings
                    )
                    total_audio_samples += samples
                    char_offset += char_length

                # Send completion
                sample_rate = tts_model.config.mimi.sample_rate
                total_duration_ms = int(total_audio_samples * 1000 / sample_rate)
                await websocket.send_json({
                    "type": "done",
                    "total_chunks": chunk_index,
                    "total_audio_duration_ms": total_duration_ms,
                })
                break

            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}"
                })

    except WebSocketDisconnect:
        logger.info(f"WebSocket session {session_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket session {session_id} error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


async def _process_chunk_websocket(
    websocket: WebSocket,
    text: str,
    chunk_index: int,
    char_offset: int,
    model_state: dict,
    settings: dict,
) -> int:
    """Process a text chunk and send audio over WebSocket. Returns total samples sent."""

    char_length = len(text)

    # Send processing notification with character position info for karaoke sync
    await websocket.send_json({
        "type": "processing",
        "chunk_index": chunk_index,
        "char_offset": char_offset,
        "char_length": char_length,
        "text": text,
    })

    loop = asyncio.get_event_loop()
    audio_queue: asyncio.Queue = asyncio.Queue()
    total_samples = 0

    def generate_sync():
        """Run synchronous generation in thread."""
        nonlocal total_samples
        try:
            _, frames_after_eos = prepare_text_prompt(text)
            frames_after_eos += 2

            for audio_chunk in tts_model._generate_audio_stream_short_text(
                model_state=model_state,
                text_to_generate=text,
                frames_after_eos=frames_after_eos,
                copy_state=True,
                **settings,
            ):
                # Convert to int16 bytes
                audio_int16 = (audio_chunk * 32767).to(torch.int16)
                audio_bytes = audio_int16.cpu().numpy().tobytes()
                total_samples += len(audio_chunk)
                loop.call_soon_threadsafe(audio_queue.put_nowait, audio_bytes)
        except Exception as e:
            loop.call_soon_threadsafe(audio_queue.put_nowait, ("error", e))
        finally:
            loop.call_soon_threadsafe(audio_queue.put_nowait, None)

    # Start generation in thread pool
    gen_future = loop.run_in_executor(None, generate_sync)

    # Stream audio as it arrives
    while True:
        item = await audio_queue.get()
        if item is None:
            break
        if isinstance(item, tuple) and item[0] == "error":
            await websocket.send_json({
                "type": "error",
                "message": str(item[1]),
                "chunk_index": chunk_index,
            })
            break
        await websocket.send_bytes(item)

    await gen_future

    # Send chunk completion with audio duration for karaoke sync
    sample_rate = tts_model.config.mimi.sample_rate
    audio_duration_ms = int(total_samples * 1000 / sample_rate) if total_samples > 0 else 0
    await websocket.send_json({
        "type": "chunk_done",
        "chunk_index": chunk_index,
        "char_offset": char_offset,
        "char_length": char_length,
        "audio_duration_ms": audio_duration_ms,
        "audio_samples": total_samples,
    })

    return total_samples


# ------------------------------------------------------
# Polling-based streaming endpoints (no WebSocket)
# ------------------------------------------------------


@dataclass
class AudioChunkData:
    """Audio chunk with timing metadata for karaoke sync."""
    index: int
    audio_bytes: bytes
    char_start: int
    char_end: int
    duration_ms: float


@dataclass
class PollingGenerationState:
    """State for a polling-based TTS generation session."""
    id: str
    voice: str
    sample_rate: int
    settings: dict
    status: Literal["active", "complete", "error"] = "active"
    error_message: str | None = None

    # Text queue: strings to process, None = end signal
    text_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    end_signaled: bool = False

    # Audio output: accumulated chunks
    audio_chunks: list[AudioChunkData] = field(default_factory=list)
    chunks_delivered: int = 0

    # Tracking
    created_at: float = field(default_factory=time.time)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # Background task handle
    task: asyncio.Task | None = None


# Active polling generations (in production, use Redis or similar)
polling_generations: dict[str, PollingGenerationState] = {}


class CreateGenerationRequest(BaseModel):
    voice: str = "cosette"
    temperature: float | None = None
    lsd_decode_steps: int | None = None
    noise_clamp: float | None = None
    eos_threshold: float | None = None


class TextChunkRequest(BaseModel):
    text: str


class PollResponse(BaseModel):
    chunks: list[dict]
    done: bool
    status: str
    # BACKPRESSURE: Add these fields to help client throttle:
    # queued_text_chunks: int  # How many text chunks waiting to process
    # queued_chars: int        # Approximate chars in queue
    # Client can slow down feeding if queued_chars > threshold


@web_app.post("/tts/polling/create")
async def create_polling_generation(req: CreateGenerationRequest):
    """Create a new polling-based TTS generation session.

    Returns a generation_id to use for subsequent requests.
    """
    gen_id = uuid.uuid4().hex[:12]

    settings = {
        k: getattr(req, k)
        for k in ["temperature", "lsd_decode_steps", "noise_clamp", "eos_threshold"]
        if getattr(req, k) is not None
    }

    state = PollingGenerationState(
        id=gen_id,
        voice=req.voice,
        sample_rate=tts_model.config.mimi.sample_rate,
        settings=settings,
    )
    polling_generations[gen_id] = state

    # Start background TTS processing task
    state.task = asyncio.create_task(_run_polling_tts_loop(state))

    logger.info(f"Created polling generation {gen_id}")

    return {
        "generation_id": gen_id,
        "sample_rate": state.sample_rate,
    }


@web_app.post("/tts/polling/{gen_id}/text")
async def feed_polling_text(gen_id: str, req: TextChunkRequest):
    """Feed text to an active generation.

    Text is queued and processed by the background TTS task.
    """
    state = polling_generations.get(gen_id)
    if not state:
        raise HTTPException(404, "Generation not found")
    if state.end_signaled:
        raise HTTPException(400, "Generation already ended")
    if state.status == "error":
        raise HTTPException(400, f"Generation failed: {state.error_message}")

    await state.text_queue.put(req.text)

    # BACKPRESSURE: Return queue depth so client can throttle:
    # return {"queued": state.text_queue.qsize()}
    return {"queued": True}


@web_app.post("/tts/polling/{gen_id}/end")
async def end_polling_generation(gen_id: str):
    """Signal that no more text will be sent.

    The generation will complete once all queued text is processed.
    """
    state = polling_generations.get(gen_id)
    if not state:
        raise HTTPException(404, "Generation not found")
    if state.end_signaled:
        return {"already_ended": True}

    state.end_signaled = True
    await state.text_queue.put(None)  # EOF marker

    return {"ended": True}


@web_app.post("/tts/polling/{gen_id}/poll")
async def poll_polling_audio(gen_id: str):
    """Poll for available audio chunks.

    Returns any new audio chunks since the last poll.
    Call repeatedly until done=true.
    """
    state = polling_generations.get(gen_id)
    if not state:
        raise HTTPException(404, "Generation not found")

    # Get new chunks (thread-safe)
    async with state.lock:
        new_chunks = state.audio_chunks[state.chunks_delivered:]
        state.chunks_delivered = len(state.audio_chunks)

    # Check if fully done
    done = state.status == "complete" and state.chunks_delivered >= len(state.audio_chunks)

    return PollResponse(
        chunks=[
            {
                "index": c.index,
                "audio_base64": base64.b64encode(c.audio_bytes).decode(),
                "char_start": c.char_start,
                "char_end": c.char_end,
                "duration_ms": c.duration_ms,
            }
            for c in new_chunks
        ],
        done=done,
        status=state.status,
    )


async def _run_polling_tts_loop(state: PollingGenerationState):
    """Background task: consume text queue, produce audio chunks."""

    # Get model state for voice
    model_state = tts_model._cached_get_state_for_audio_prompt(state.voice, truncate=True)
    sample_rate = state.sample_rate

    chunker = StreamingTextChunker(tts_model.flow_lm.conditioner.tokenizer)
    chunk_index = 0
    char_offset = 0

    try:
        while True:
            # Get text from queue (blocking)
            text_item = await state.text_queue.get()

            if text_item is None:
                # EOF - flush remaining text
                remaining_chunks = chunker.flush()
                for chunk_text in remaining_chunks:
                    await _process_polling_chunk(
                        state, chunk_text, chunk_index, char_offset, model_state
                    )
                    char_offset += len(chunk_text)
                    chunk_index += 1

                state.status = "complete"
                logger.info(f"Polling generation {state.id} complete: {chunk_index} chunks")
                break

            # Feed text to chunker
            ready_chunks = chunker.add_text(text_item)

            # Process any complete chunks
            for chunk_text in ready_chunks:
                await _process_polling_chunk(
                    state, chunk_text, chunk_index, char_offset, model_state
                )
                char_offset += len(chunk_text)
                chunk_index += 1

    except Exception as e:
        logger.error(f"Polling generation {state.id} error: {e}")
        state.status = "error"
        state.error_message = str(e)


async def _process_polling_chunk(
    state: PollingGenerationState,
    text: str,
    chunk_index: int,
    char_offset: int,
    model_state: dict,
):
    """Process a text chunk and add audio to state."""

    loop = asyncio.get_event_loop()
    audio_bytes_list: list[bytes] = []
    total_samples = 0

    def generate_sync():
        nonlocal total_samples
        _, frames_after_eos = prepare_text_prompt(text)
        frames_after_eos += 2

        for audio_chunk in tts_model._generate_audio_stream_short_text(
            model_state=model_state,
            text_to_generate=text,
            frames_after_eos=frames_after_eos,
            copy_state=True,
            **state.settings,
        ):
            audio_int16 = (audio_chunk * 32767).to(torch.int16)
            audio_bytes_list.append(audio_int16.cpu().numpy().tobytes())
            total_samples += len(audio_chunk)

    # Run TTS in thread pool
    await loop.run_in_executor(None, generate_sync)

    # Combine all audio bytes
    combined_audio = b"".join(audio_bytes_list)
    duration_ms = (total_samples / state.sample_rate) * 1000

    # Add chunk to state (thread-safe)
    chunk_data = AudioChunkData(
        index=chunk_index,
        audio_bytes=combined_audio,
        char_start=char_offset,
        char_end=char_offset + len(text),
        duration_ms=duration_ms,
    )

    async with state.lock:
        state.audio_chunks.append(chunk_data)

    logger.debug(f"Generation {state.id}: chunk {chunk_index} ready ({duration_ms:.0f}ms)")


# BACKPRESSURE: Add a cleanup task to remove old generations
# async def _cleanup_polling_generations():
#     """Background task to clean up expired generations."""
#     while True:
#         await asyncio.sleep(60)
#         now = time.time()
#         expired = [
#             gid for gid, state in polling_generations.items()
#             if now - state.created_at > 300  # 5 min TTL
#             or (state.status == "complete" and now - state.created_at > 60)
#         ]
#         for gid in expired:
#             del polling_generations[gid]
#             logger.info(f"Cleaned up polling generation {gid}")


@cli_app.command()
def serve(
    voice: Annotated[
        str, typer.Option(help="Path to voice prompt audio file (voice to clone)")
    ] = DEFAULT_AUDIO_PROMPT,
    host: Annotated[str, typer.Option(help="Host to bind to")] = "localhost",
    port: Annotated[int, typer.Option(help="Port to bind to")] = 8000,
    reload: Annotated[bool, typer.Option(help="Enable auto-reload")] = False,
    variant: Annotated[str, typer.Option(help="Model variant")] = DEFAULT_VARIANT,
    lsd_decode_steps: Annotated[
        int, typer.Option(help="Number of LSD decoding steps")
    ] = DEFAULT_LSD_DECODE_STEPS,
    temperature: Annotated[
        float, typer.Option(help="Sampling temperature")
    ] = DEFAULT_TEMPERATURE,
    noise_clamp: Annotated[float, typer.Option(help="Noise clamp value")] = DEFAULT_NOISE_CLAMP,
    eos_threshold: Annotated[float, typer.Option(help="EOS detection threshold")] = DEFAULT_EOS_THRESHOLD,
    compile: Annotated[
        bool, typer.Option("--compile", help="Compile model with torch.compile for faster CPU inference")
    ] = False,
):
    """Start the FastAPI server."""

    global tts_model, global_model_state
    tts_model = TTSModel.load_model(
        variant,
        temperature,
        lsd_decode_steps,
        noise_clamp,
        eos_threshold,
    )

    if compile:
        tts_model.compile_model()

    # Pre-load the voice prompt
    global_model_state = tts_model.get_state_for_audio_prompt(voice)
    logger.info(f"The size of the model state is {size_of_dict(global_model_state) // 1e6} MB")

    # When reload is disabled, pass the app object directly to preserve global state.
    # When reload is enabled, pass as string (but globals won't work - require per-request voice).
    if reload:
        uvicorn.run("pocket_tts.main:web_app", host=host, port=port, reload=True)
    else:
        uvicorn.run(web_app, host=host, port=port)


# ------------------------------------------------------
# The pocket-tts single generation CLI implementation
# ------------------------------------------------------


@cli_app.command()
def generate(
    text: Annotated[
        str, typer.Option(help="Text to generate")
    ] = "Hello world. I am Kyutai's Pocket TTS. I'm fast enough to run on small CPUs. I hope you'll like me.",
    voice: Annotated[
        str, typer.Option(help="Path to audio conditioning file (voice to clone)")
    ] = DEFAULT_AUDIO_PROMPT,
    quiet: Annotated[bool, typer.Option("-q", "--quiet", help="Disable logging output")] = False,
    variant: Annotated[str, typer.Option(help="Model signature")] = DEFAULT_VARIANT,
    lsd_decode_steps: Annotated[
        int, typer.Option(help="Number of generation steps")
    ] = DEFAULT_LSD_DECODE_STEPS,
    temperature: Annotated[
        float, typer.Option(help="Temperature for generation")
    ] = DEFAULT_TEMPERATURE,
    noise_clamp: Annotated[float, typer.Option(help="Noise clamp value")] = DEFAULT_NOISE_CLAMP,
    eos_threshold: Annotated[float, typer.Option(help="EOS threshold")] = DEFAULT_EOS_THRESHOLD,
    frames_after_eos: Annotated[
        int, typer.Option(help="Number of frames to generate after EOS")
    ] = DEFAULT_FRAMES_AFTER_EOS,
    output_path: Annotated[
        str, typer.Option("-o", "--output-path", help="Output path for generated audio")
    ] = "./tts_output.wav",
    device: Annotated[str, typer.Option(help="Device to use")] = "cpu",
    seed: Annotated[int | None, typer.Option(help="Random seed for reproducibility")] = None,
    compile: Annotated[
        bool, typer.Option("--compile", help="Compile model with torch.compile for faster CPU inference")
    ] = False,
):
    """Generate speech using Kyutai Pocket TTS."""
    if "cuda" in device:
        # Cuda graphs capturing does not play nice with multithreading.
        os.environ["NO_CUDA_GRAPH"] = "1"

    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)

    log_level = logging.ERROR if quiet else logging.INFO
    with enable_logging("pocket_tts", log_level):
        tts_model = TTSModel.load_model(
            variant, temperature, lsd_decode_steps, noise_clamp, eos_threshold
        )
        tts_model.to(device)

        if compile:
            tts_model.compile_model()

        model_state_for_voice = tts_model.get_state_for_audio_prompt(voice)
        # Stream audio generation directly to file or stdout
        audio_chunks = tts_model.generate_audio_stream(
            model_state=model_state_for_voice,
            text_to_generate=text,
            frames_after_eos=frames_after_eos,
        )

        stream_audio_chunks(output_path, audio_chunks, tts_model.config.mimi.sample_rate)

        # Only print the result message if not writing to stdout
        if output_path != "-":
            logger.info("Results written in %s", output_path)
        logger.info(
            "If you want to try multiple voices and prompts quickly, try the `serve` command."
        )


if __name__ == "__main__":
    cli_app()
