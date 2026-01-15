"""Backend detection and routing for pocket_tts."""

import platform
import sys
from enum import Enum
from typing import Callable


class Backend(str, Enum):
    """Available TTS backends."""
    PYTORCH = "pytorch"
    MLX = "mlx"
    AUTO = "auto"


def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def is_mlx_available() -> bool:
    """Check if MLX is available and usable."""
    if not is_apple_silicon():
        return False
    try:
        import mlx.core  # noqa: F401
        return True
    except ImportError:
        return False


def detect_best_backend() -> Backend:
    """Detect the best available backend.

    Returns MLX on Apple Silicon if available, otherwise PyTorch.
    """
    if is_mlx_available():
        return Backend.MLX
    return Backend.PYTORCH


def get_backend_info() -> dict:
    """Get information about available backends."""
    return {
        "apple_silicon": is_apple_silicon(),
        "mlx_available": is_mlx_available(),
        "recommended": detect_best_backend().value,
    }


def run_with_backend(
    backend: Backend,
    pytorch_fn: Callable,
    mlx_fn: Callable,
    *args,
    **kwargs
):
    """Run a function with the specified backend.

    Args:
        backend: Which backend to use (auto, pytorch, mlx)
        pytorch_fn: Function to call for PyTorch backend
        mlx_fn: Function to call for MLX backend
        *args, **kwargs: Arguments to pass to the function
    """
    if backend == Backend.AUTO:
        backend = detect_best_backend()

    if backend == Backend.MLX:
        if not is_mlx_available():
            print("Warning: MLX requested but not available, falling back to PyTorch",
                  file=sys.stderr)
            backend = Backend.PYTORCH
        else:
            return mlx_fn(*args, **kwargs)

    return pytorch_fn(*args, **kwargs)
