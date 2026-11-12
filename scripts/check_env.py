"""Print a report of the local environment (Python, packages, CPU/GPU, RAM, disk).

Run:  .venv\\Scripts\\python scripts\\check_env.py
The output is copied into docs/environment.md as evidence for the report.
"""

from __future__ import annotations

import importlib
import platform
import shutil
import sys
from pathlib import Path

PACKAGES = [
    "streamlit", "PIL", "numpy", "cv2", "pandas", "yaml", "requests",
    "mediapipe", "torch", "transformers", "pytest", "fastapi", "uvicorn", "httpx", "nbformat",
]


def package_versions() -> list[tuple[str, str]]:
    rows = []
    for name in PACKAGES:
        try:
            module = importlib.import_module(name)
            rows.append((name, getattr(module, "__version__", "installed")))
        except Exception as exc:  # report, don't crash: this is a diagnostic script
            rows.append((name, f"MISSING ({type(exc).__name__}: {exc})"))
    return rows


def torch_report() -> list[str]:
    try:
        import torch
    except ImportError:
        return ["torch not installed"]
    lines = [f"torch {torch.__version__}", f"CUDA available: {torch.cuda.is_available()}"]
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        lines.append(f"GPU: {props.name}, {props.total_memory / 1024**3:.1f} GB")
    return lines


def memory_report() -> str:
    try:
        import psutil  # optional
        vm = psutil.virtual_memory()
        return f"RAM total {vm.total / 1024**3:.1f} GB, available {vm.available / 1024**3:.1f} GB"
    except ImportError:
        return "RAM: psutil not installed (use Task Manager)"


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    print(f"Python      : {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform    : {platform.platform()}")
    print(f"Processor   : {platform.processor()}")
    print(memory_report())
    usage = shutil.disk_usage(root)
    print(f"Disk (project drive): free {usage.free / 1024**3:.1f} GB")
    print("\nPackages:")
    for name, version in package_versions():
        print(f"  {name:<13} {version}")
    print("\nPyTorch:")
    for line in torch_report():
        print(f"  {line}")


if __name__ == "__main__":
    main()
