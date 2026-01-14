#!/usr/bin/env python3
"""Benchmark script comparing PyTorch and MLX TTS servers.

Usage:
    # Start servers manually first:
    uv run pocket_tts serve --port 8000 &
    uv run pocket_tts_mlx serve --port 8001 &

    # Then run benchmark:
    uv run python scripts/benchmark_servers.py

    # Or with auto-start (starts and stops servers automatically):
    uv run python scripts/benchmark_servers.py --auto-start
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Iterator

import requests

# Test sentences of varying lengths
TEST_SENTENCES = [
    ("short", "Hello world."),
    ("medium", "The quick brown fox jumps over the lazy dog near the riverbank."),
    ("long", "Pocket TTS is a fast text-to-speech model that can run efficiently on both CPU and GPU. It uses a transformer-based architecture with flow matching for high-quality audio generation."),
]

WARMUP_TEXT = "Warming up the model."


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    total_time: float  # Total request time in seconds
    ttfb: float  # Time to first byte in seconds
    audio_bytes: int  # Size of audio response

    @property
    def audio_duration_estimate(self) -> float:
        """Estimate audio duration assuming 24kHz 16-bit mono WAV."""
        # WAV header is 44 bytes, then 2 bytes per sample at 24kHz
        audio_samples = (self.audio_bytes - 44) / 2
        return audio_samples / 24000

    @property
    def realtime_factor(self) -> float:
        """Real-time factor (< 1 means faster than real-time)."""
        if self.audio_duration_estimate <= 0:
            return float('inf')
        return self.total_time / self.audio_duration_estimate


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results."""
    name: str
    sentence_type: str
    metrics: list[RequestMetrics]

    @property
    def avg_total_time(self) -> float:
        return statistics.mean(m.total_time for m in self.metrics)

    @property
    def std_total_time(self) -> float:
        if len(self.metrics) < 2:
            return 0.0
        return statistics.stdev(m.total_time for m in self.metrics)

    @property
    def avg_ttfb(self) -> float:
        return statistics.mean(m.ttfb for m in self.metrics)

    @property
    def avg_realtime_factor(self) -> float:
        return statistics.mean(m.realtime_factor for m in self.metrics)


def make_tts_request(url: str, text: str) -> RequestMetrics:
    """Make a TTS request and measure timing."""
    start = time.perf_counter()
    ttfb = None
    audio_bytes = 0

    with requests.post(
        url,
        data={"text": text},
        stream=True,
    ) as response:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=4096):
            if ttfb is None:
                ttfb = time.perf_counter() - start
            audio_bytes += len(chunk)

    total_time = time.perf_counter() - start
    return RequestMetrics(
        total_time=total_time,
        ttfb=ttfb or total_time,
        audio_bytes=audio_bytes,
    )


def warmup_server(url: str, name: str, num_requests: int = 2) -> None:
    """Warm up a server with a few requests."""
    print(f"  Warming up {name}...", end=" ", flush=True)
    for _ in range(num_requests):
        try:
            make_tts_request(url, WARMUP_TEXT)
        except Exception as e:
            print(f"FAILED: {e}")
            return
    print("done")


def check_server(url: str, name: str) -> bool:
    """Check if a server is running."""
    try:
        response = requests.get(url.replace("/tts", "/health"), timeout=2)
        return response.status_code == 200
    except:
        return False


def run_benchmark(
    url: str,
    name: str,
    sentences: list[tuple[str, str]],
    num_runs: int = 5,
) -> Iterator[BenchmarkResult]:
    """Run benchmark for a server."""
    for sentence_type, text in sentences:
        metrics = []
        for i in range(num_runs):
            try:
                m = make_tts_request(url, text)
                metrics.append(m)
                print(f"    {name} [{sentence_type}] run {i+1}/{num_runs}: "
                      f"{m.total_time*1000:.0f}ms total, {m.ttfb*1000:.0f}ms TTFB, "
                      f"{m.realtime_factor:.2f}x RT")
            except Exception as e:
                print(f"    {name} [{sentence_type}] run {i+1}/{num_runs}: FAILED - {e}")

        if metrics:
            yield BenchmarkResult(name=name, sentence_type=sentence_type, metrics=metrics)


def print_comparison(pytorch_results: list[BenchmarkResult], mlx_results: list[BenchmarkResult]) -> None:
    """Print a comparison table."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    # Group by sentence type
    pytorch_by_type = {r.sentence_type: r for r in pytorch_results}
    mlx_by_type = {r.sentence_type: r for r in mlx_results}

    print(f"\n{'Sentence':<10} {'Backend':<10} {'Total (ms)':<15} {'TTFB (ms)':<12} {'RT Factor':<10}")
    print("-" * 60)

    for sentence_type, _ in TEST_SENTENCES:
        pt = pytorch_by_type.get(sentence_type)
        mlx = mlx_by_type.get(sentence_type)

        if pt:
            print(f"{sentence_type:<10} {'PyTorch':<10} "
                  f"{pt.avg_total_time*1000:>6.0f} ± {pt.std_total_time*1000:>4.0f}  "
                  f"{pt.avg_ttfb*1000:>8.0f}    "
                  f"{pt.avg_realtime_factor:>6.2f}x")
        if mlx:
            print(f"{'':<10} {'MLX':<10} "
                  f"{mlx.avg_total_time*1000:>6.0f} ± {mlx.std_total_time*1000:>4.0f}  "
                  f"{mlx.avg_ttfb*1000:>8.0f}    "
                  f"{mlx.avg_realtime_factor:>6.2f}x")

        if pt and mlx:
            speedup = pt.avg_total_time / mlx.avg_total_time
            ttfb_speedup = pt.avg_ttfb / mlx.avg_ttfb
            print(f"{'':<10} {'Speedup':<10} "
                  f"{speedup:>6.2f}x          "
                  f"{ttfb_speedup:>6.2f}x")
        print()

    # Overall summary
    if pytorch_results and mlx_results:
        pt_avg = statistics.mean(r.avg_total_time for r in pytorch_results)
        mlx_avg = statistics.mean(r.avg_total_time for r in mlx_results)
        overall_speedup = pt_avg / mlx_avg

        pt_ttfb = statistics.mean(r.avg_ttfb for r in pytorch_results)
        mlx_ttfb = statistics.mean(r.avg_ttfb for r in mlx_results)
        ttfb_speedup = pt_ttfb / mlx_ttfb

        print("-" * 60)
        print(f"{'OVERALL':<10} {'PyTorch':<10} {pt_avg*1000:>6.0f} ms avg    {pt_ttfb*1000:>8.0f} ms")
        print(f"{'':<10} {'MLX':<10} {mlx_avg*1000:>6.0f} ms avg    {mlx_ttfb*1000:>8.0f} ms")
        print(f"{'':<10} {'Speedup':<10} {overall_speedup:>6.2f}x          {ttfb_speedup:>6.2f}x")


def start_server(module: str, port: int) -> subprocess.Popen:
    """Start a TTS server."""
    return subprocess.Popen(
        ["uv", "run", "python", "-m", module, "serve", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark TTS servers")
    parser.add_argument("--pytorch-port", type=int, default=8000, help="PyTorch server port")
    parser.add_argument("--mlx-port", type=int, default=8001, help="MLX server port")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per test")
    parser.add_argument("--auto-start", action="store_true", help="Automatically start servers")
    parser.add_argument("--pytorch-only", action="store_true", help="Only benchmark PyTorch")
    parser.add_argument("--mlx-only", action="store_true", help="Only benchmark MLX")
    args = parser.parse_args()

    pytorch_url = f"http://localhost:{args.pytorch_port}/tts"
    mlx_url = f"http://localhost:{args.mlx_port}/tts"

    servers = []

    try:
        if args.auto_start:
            print("Starting servers...")
            if not args.mlx_only:
                print(f"  Starting PyTorch server on port {args.pytorch_port}...")
                servers.append(start_server("pocket_tts", args.pytorch_port))
            if not args.pytorch_only:
                print(f"  Starting MLX server on port {args.mlx_port}...")
                servers.append(start_server("pocket_tts_mlx", args.mlx_port))

            # Wait for servers to start
            print("  Waiting for servers to be ready...")
            time.sleep(15)

        # Check servers
        print("\nChecking servers...")
        pytorch_ok = not args.mlx_only and check_server(pytorch_url, "PyTorch")
        mlx_ok = not args.pytorch_only and check_server(mlx_url, "MLX")

        if not args.mlx_only:
            print(f"  PyTorch (port {args.pytorch_port}): {'OK' if pytorch_ok else 'NOT RUNNING'}")
        if not args.pytorch_only:
            print(f"  MLX (port {args.mlx_port}): {'OK' if mlx_ok else 'NOT RUNNING'}")

        if not pytorch_ok and not mlx_ok:
            print("\nNo servers running. Start them with:")
            print(f"  uv run pocket_tts serve --port {args.pytorch_port} &")
            print(f"  uv run pocket_tts_mlx serve --port {args.mlx_port} &")
            print("\nOr use --auto-start to start them automatically.")
            sys.exit(1)

        # Warmup
        print("\nWarming up servers...")
        if pytorch_ok:
            warmup_server(pytorch_url, "PyTorch")
        if mlx_ok:
            warmup_server(mlx_url, "MLX")

        # Run benchmarks
        print(f"\nRunning benchmarks ({args.runs} runs each)...")

        pytorch_results = []
        mlx_results = []

        if pytorch_ok:
            print("\n  PyTorch server:")
            pytorch_results = list(run_benchmark(pytorch_url, "PyTorch", TEST_SENTENCES, args.runs))

        if mlx_ok:
            print("\n  MLX server:")
            mlx_results = list(run_benchmark(mlx_url, "MLX", TEST_SENTENCES, args.runs))

        # Print comparison
        print_comparison(pytorch_results, mlx_results)

    finally:
        # Cleanup servers if we started them
        for server in servers:
            server.terminate()
            server.wait()


if __name__ == "__main__":
    main()
