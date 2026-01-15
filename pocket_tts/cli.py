"""Unified CLI entry point with automatic backend detection."""

import sys
from typing import Optional

import typer
from typing_extensions import Annotated

from pocket_tts.backend import Backend, detect_best_backend, get_backend_info, is_mlx_available

cli_app = typer.Typer(
    help="Kyutai Pocket TTS - Text-to-Speech generation tool (auto-selects best backend)",
    pretty_exceptions_show_locals=False,
)


def _get_backend_display_name(backend: Backend) -> str:
    """Get display name for backend."""
    if backend == Backend.MLX:
        return "MLX (Apple Silicon)"
    return "PyTorch"


@cli_app.command()
def serve(
    backend: Annotated[
        Backend, typer.Option(help="Backend to use (auto, pytorch, mlx)")
    ] = Backend.AUTO,
    voice: Annotated[
        str, typer.Option(help="Default voice for TTS generation")
    ] = "cosette",
    host: Annotated[str, typer.Option(help="Host to bind to")] = "localhost",
    port: Annotated[int, typer.Option(help="Port to bind to")] = 8000,
    variant: Annotated[str, typer.Option(help="Model variant")] = "b6369a24",
    lsd_decode_steps: Annotated[
        int, typer.Option(help="Number of LSD decoding steps")
    ] = 1,
    temperature: Annotated[
        float, typer.Option(help="Sampling temperature")
    ] = 0.9,
    noise_clamp: Annotated[float, typer.Option(help="Noise clamp value")] = 3.0,
    eos_threshold: Annotated[float, typer.Option(help="EOS detection threshold")] = -4.0,
):
    """Start the TTS server (auto-selects MLX on Apple Silicon)."""
    # Resolve backend
    resolved_backend = backend if backend != Backend.AUTO else detect_best_backend()
    print(f"Using backend: {_get_backend_display_name(resolved_backend)}", file=sys.stderr)

    if resolved_backend == Backend.MLX:
        if not is_mlx_available():
            print("MLX not available, falling back to PyTorch", file=sys.stderr)
            resolved_backend = Backend.PYTORCH

    if resolved_backend == Backend.MLX:
        try:
            from pocket_tts_mlx.__main__ import serve as mlx_serve
            mlx_serve(
                voice=voice,
                host=host,
                port=port,
                variant=variant,
                lsd_decode_steps=lsd_decode_steps,
                temperature=temperature,
                noise_clamp=noise_clamp,
                eos_threshold=eos_threshold,
            )
            return
        except ImportError as e:
            print(f"MLX import failed ({e}), falling back to PyTorch", file=sys.stderr)
            resolved_backend = Backend.PYTORCH

    # PyTorch fallback
    from pocket_tts.main import serve as pytorch_serve
    pytorch_serve(
            voice=voice,
            host=host,
            port=port,
            variant=variant,
            lsd_decode_steps=lsd_decode_steps,
            temperature=temperature,
            noise_clamp=noise_clamp,
            eos_threshold=eos_threshold,
        )


@cli_app.command()
def generate(
    backend: Annotated[
        Backend, typer.Option(help="Backend to use (auto, pytorch, mlx)")
    ] = Backend.AUTO,
    text: Annotated[
        str, typer.Option(help="Text to generate")
    ] = "Hello world. I am Kyutai's Pocket TTS.",
    voice: Annotated[
        str, typer.Option(help="Voice name or path to audio file")
    ] = "cosette",
    output_path: Annotated[
        str, typer.Option("-o", "--output-path", help="Output path (use - for stdout)")
    ] = "./tts_output.wav",
    quiet: Annotated[bool, typer.Option("-q", "--quiet", help="Disable logging")] = False,
    variant: Annotated[str, typer.Option(help="Model variant")] = "b6369a24",
    lsd_decode_steps: Annotated[
        int, typer.Option(help="Number of LSD decoding steps")
    ] = 1,
    temperature: Annotated[
        float, typer.Option(help="Sampling temperature")
    ] = 0.9,
    noise_clamp: Annotated[float, typer.Option(help="Noise clamp value")] = 3.0,
    eos_threshold: Annotated[float, typer.Option(help="EOS detection threshold")] = -4.0,
    seed: Annotated[Optional[int], typer.Option(help="Random seed")] = None,
):
    """Generate speech (auto-selects MLX on Apple Silicon)."""
    # Resolve backend
    resolved_backend = backend if backend != Backend.AUTO else detect_best_backend()
    if not quiet:
        print(f"Using backend: {_get_backend_display_name(resolved_backend)}", file=sys.stderr)

    if resolved_backend == Backend.MLX:
        if not is_mlx_available():
            if not quiet:
                print("MLX not available, falling back to PyTorch", file=sys.stderr)
            resolved_backend = Backend.PYTORCH

    if resolved_backend == Backend.MLX:
        try:
            from pocket_tts_mlx.__main__ import generate as mlx_generate
            mlx_generate(
                text=text,
                voice=voice,
                output=output_path,
                quiet=quiet,
                variant=variant,
                lsd_decode_steps=lsd_decode_steps,
                temperature=temperature,
                noise_clamp=noise_clamp,
                eos_threshold=eos_threshold,
                seed=seed,
            )
            return
        except ImportError as e:
            if not quiet:
                print(f"MLX import failed ({e}), falling back to PyTorch", file=sys.stderr)
            resolved_backend = Backend.PYTORCH

    # PyTorch fallback
    from pocket_tts.main import generate as pytorch_generate
    pytorch_generate(
            text=text,
            voice=voice,
            output_path=output_path,
            quiet=quiet,
            variant=variant,
            lsd_decode_steps=lsd_decode_steps,
            temperature=temperature,
            noise_clamp=noise_clamp,
            eos_threshold=eos_threshold,
            seed=seed,
        )


@cli_app.command()
def info():
    """Show backend information."""
    info = get_backend_info()
    print(f"Apple Silicon: {info['apple_silicon']}")
    print(f"MLX available: {info['mlx_available']}")
    print(f"Recommended backend: {info['recommended']}")


@cli_app.command()
def list_voices():
    """List available predefined voices."""
    from pocket_tts.utils.utils import PREDEFINED_VOICES
    print("Available predefined voices:")
    for voice in sorted(PREDEFINED_VOICES.keys()):
        print(f"  - {voice}")


def main():
    cli_app()


if __name__ == "__main__":
    main()
