"""Time `rectified_tanh` vs `rectified_sigmoid` forward + backward.

The two activations are mathematically identical (paper's Eq. 1 vs the
opentxfm reference repo's `RectifiedSigmoid`), so the choice between them
is purely runtime/numerical. Run this on each backend we care about (CPU,
MPS, and later CUDA) to see whether one form is meaningfully faster.

Usage:

    uv run python scripts/bench_activations.py
    uv run python scripts/bench_activations.py --device cpu
    uv run python scripts/bench_activations.py --device mps
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from txfm_repro.lit_model import rectified_sigmoid, rectified_tanh

ACTIVATIONS = {"tanh": rectified_tanh, "sigmoid": rectified_sigmoid}


def _resolve_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def time_op(fn, *args, n_warmup: int = 3, n_runs: int = 25, device: torch.device) -> float:
    """Return median seconds per call across `n_runs` runs after `n_warmup`."""
    for _ in range(n_warmup):
        out = fn(*args)
    _sync(device)
    samples: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        out = fn(*args)
        _sync(device)
        samples.append(time.perf_counter() - t0)
    del out
    return statistics.median(samples)


def bench_forward(B: int, G: int, device: torch.device, library_size_L: float) -> None:
    print(f"\n[forward] shape=(B={B}, G={G})  device={device}")
    z = torch.randn(B, G, device=device)
    times: dict[str, float] = {}
    for name, fn in ACTIVATIONS.items():
        sec = time_op(fn, z, library_size_L, device=device)
        times[name] = sec
        print(f"  {name:>9s}  {sec * 1e3:8.3f} ms")
    ratio = times["sigmoid"] / times["tanh"]
    print(f"  ratio sigmoid/tanh: {ratio:.2f}x  ({'sigmoid faster' if ratio < 1 else 'tanh faster'})")


def bench_forward_backward(B: int, G: int, device: torch.device, library_size_L: float) -> None:
    print(f"\n[forward+backward] shape=(B={B}, G={G})  device={device}")
    times: dict[str, float] = {}
    for name, fn in ACTIVATIONS.items():
        z_template = torch.randn(B, G, device=device)

        def run() -> None:
            z = z_template.detach().clone().requires_grad_(True)
            out = fn(z, library_size_L)
            out.sum().backward()

        sec = time_op(run, device=device)
        times[name] = sec
        print(f"  {name:>9s}  {sec * 1e3:8.3f} ms")
    ratio = times["sigmoid"] / times["tanh"]
    print(f"  ratio sigmoid/tanh: {ratio:.2f}x  ({'sigmoid faster' if ratio < 1 else 'tanh faster'})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="torch device override (e.g. cpu, mps, cuda)")
    ap.add_argument("--library-size-l", type=float, default=1e5)
    args = ap.parse_args()

    device = _resolve_device(args.device)
    print(f"device: {device}")
    print(f"torch:  {torch.__version__}")

    shapes = [
        (16, 2_000),       # phase-0 mock-data scale
        (32, 20_000),      # roughly protein-coding TCGA scale
        (64, 60_000),      # full-genome TCGA-ish
    ]
    for B, G in shapes:
        bench_forward(B, G, device=device, library_size_L=args.library_size_l)
        bench_forward_backward(B, G, device=device, library_size_L=args.library_size_l)


if __name__ == "__main__":
    main()
