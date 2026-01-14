# Pocket TTS MLX

MLX port of Kyutai's Pocket TTS for Apple Silicon. This implementation provides native acceleration on M1/M2/M3 Macs using Apple's MLX framework.

## Installation

```bash
# Install with MLX support
pip install pocket-tts[mlx]

# Or with uv
uv pip install pocket-tts[mlx]
```

## CLI Usage

### Generate Speech

```bash
# Basic generation
uv run python -m pocket_tts_mlx generate --text "Hello world"

# With custom voice
uv run python -m pocket_tts_mlx generate --text "Hello" --voice sarah

# Output to stdout (for piping)
uv run python -m pocket_tts_mlx generate --text "Hello" -o -

# With reproducible seed
uv run python -m pocket_tts_mlx generate --text "Hello" --seed 42

# Full options
uv run python -m pocket_tts_mlx generate \
  --text "Hello world" \
  --voice cosette \
  --temperature 0.9 \
  --lsd-decode-steps 1 \
  -o output.wav
```

### List Available Voices

```bash
uv run python -m pocket_tts_mlx list-voices
```

### Start Server

```bash
# Start with default settings (localhost:8000)
uv run python -m pocket_tts_mlx serve

# Custom host and port
uv run python -m pocket_tts_mlx serve --host 0.0.0.0 --port 8080

# With custom default voice
uv run python -m pocket_tts_mlx serve --voice sarah

# Full options
uv run python -m pocket_tts_mlx serve \
  --host localhost \
  --port 8000 \
  --voice cosette \
  --temperature 0.9 \
  --lsd-decode-steps 1
```

## Server API

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves the web frontend |
| `/health` | GET | Health check, returns `{"status":"healthy","backend":"mlx"}` |
| `/tts` | POST | Generate speech from text |

### TTS Endpoint

**POST /tts**

Generate speech from text with optional voice customization.

**Form Parameters:**
- `text` (required): Text to convert to speech
- `voice_url` (optional): Predefined voice name or URL (`http://`, `https://`, `hf://`)
- `voice_wav` (optional): Uploaded WAV file for voice cloning

**Example requests:**

```bash
# Basic request with default voice
curl -X POST "http://localhost:8000/tts" \
  -F "text=Hello world" \
  -o output.wav

# With predefined voice
curl -X POST "http://localhost:8000/tts" \
  -F "text=Hello world" \
  -F "voice_url=sarah" \
  -o output.wav

# With custom voice file
curl -X POST "http://localhost:8000/tts" \
  -F "text=Hello world" \
  -F "voice_wav=@my_voice.wav" \
  -o output.wav

# With HuggingFace voice URL
curl -X POST "http://localhost:8000/tts" \
  -F "text=Hello world" \
  -F "voice_url=hf://kyutai/pocket-tts/voices/custom.safetensors" \
  -o output.wav
```

## Configuration Options

### Generate Command

| Option | Default | Description |
|--------|---------|-------------|
| `--text` | "Hello world..." | Text to generate |
| `--voice` | cosette | Voice name or path to audio file |
| `--variant` | b6369a24 | Model variant |
| `--temperature` | 0.9 | Sampling temperature |
| `--lsd-decode-steps` | 1 | Number of LSD decoding steps |
| `--noise-clamp` | 3.0 | Noise clamp value |
| `--eos-threshold` | -4.0 | EOS detection threshold |
| `--frames-after-eos` | 3 | Frames to generate after EOS |
| `-o, --output` | ./tts_output.wav | Output path (use `-` for stdout) |
| `--seed` | None | Random seed for reproducibility |
| `-q, --quiet` | False | Disable logging |

### Serve Command

| Option | Default | Description |
|--------|---------|-------------|
| `--voice` | cosette | Default voice for TTS generation |
| `--host` | localhost | Host to bind to |
| `--port` | 8000 | Port to bind to |
| `--variant` | b6369a24 | Model variant |
| `--temperature` | 0.9 | Sampling temperature |
| `--lsd-decode-steps` | 1 | Number of LSD decoding steps |
| `--noise-clamp` | 3.0 | Noise clamp value |
| `--eos-threshold` | -4.0 | EOS detection threshold |

## Differences from PyTorch Version

- Uses MLX framework for Apple Silicon acceleration
- Random number generation differs from PyTorch (outputs are similar but not identical)
- No CUDA support (MLX is Apple Silicon only)
- Server returns `{"backend":"mlx"}` in health endpoint

## Benchmarks

### Server Performance (M3 Max)

Comparison between PyTorch and MLX servers:

```bash
# Run the benchmark
uv run python scripts/benchmark_servers.py --auto-start --runs 5
```

| Text Length | PyTorch | MLX | Speedup |
|-------------|---------|-----|---------|
| Short (~2 words) | 224ms | 342ms | 0.66x |
| Medium (~12 words) | 716ms | 566ms | **1.26x** |
| Long (~35 words) | 2065ms | 1469ms | **1.41x** |
| **Overall** | 1002ms | 793ms | **1.26x** |

**Key findings:**
- MLX is **1.26x faster overall** for total generation time
- MLX scales better with text length (1.41x faster for long text)
- Both achieve **6-8x faster than real-time** playback
- PyTorch has faster time-to-first-byte (61ms vs 217ms) due to MLX's lazy evaluation

### CLI Performance

```bash
hyperfine --runs 5 --warmup 2 \
  -L variant pocket_tts,pocket_tts_mlx \
  'uv run {variant} generate --text "Hello world" -o /dev/null'
```

Note: CLI benchmarks include model loading time. Server mode amortizes this cost across requests.

## Requirements

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.10+
- MLX 0.22.0+
