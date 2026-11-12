"""Run one try-on from the command line through the same service the app uses.

    .venv\\Scripts\\python scripts\\tryon_cli.py --url https://xxx.trycloudflare.com --key KEY ^
        --person path\\to\\person.jpg --garment path\\to\\shirt.jpg --type upper --steps 30

Prints the quality-gate report, progress, timings, and where the result was saved
(outputs/runs/<timestamp>/). Used for Phase 3 testing and for batch runs in evaluation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.validation import ImageValidationError, validate_image_bytes  # noqa: E402
from config import settings  # noqa: E402
from virtual_tryon import quality_check  # noqa: E402
from virtual_tryon.backends.base import TryOnError  # noqa: E402
from virtual_tryon.backends.remote import RemoteBackend  # noqa: E402
from virtual_tryon.postprocessing import comparison_sheet  # noqa: E402
from virtual_tryon.service import TryOnService  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=settings.TRYON_SERVER_URL, help="server URL printed by the Colab notebook")
    parser.add_argument("--key", default=None, help="API key printed by the Colab notebook")
    parser.add_argument("--person", required=True)
    parser.add_argument("--garment", required=True)
    parser.add_argument("--type", default="upper", choices=settings.GARMENT_TYPES)
    parser.add_argument("--steps", type=int, default=settings.DEFAULT_STEPS)
    parser.add_argument("--guidance", type=float, default=settings.DEFAULT_GUIDANCE)
    parser.add_argument("--seed", type=int, default=settings.DEFAULT_SEED)
    parser.add_argument("--no-repaint", action="store_true")
    parser.add_argument("--no-pose-check", action="store_true")
    args = parser.parse_args()

    try:
        person = validate_image_bytes(Path(args.person).read_bytes(), args.person, "person").image
        garment = validate_image_bytes(Path(args.garment).read_bytes(), args.garment, "garment").image
    except (OSError, ImageValidationError) as exc:
        print("INPUT ERROR:", getattr(exc, "user_message", exc))
        return 2

    for label, report in (("person", quality_check.check_person_image(person, args.type, use_pose=not args.no_pose_check)),
                          ("garment", quality_check.check_garment_image(garment))):
        print(f"Quality ({label}): {report.metrics}")
        for issue in report.issues:
            print(f"  [{issue.level}] {issue.message}")

    backend = RemoteBackend(args.url, args.key)
    status = backend.health()
    print("Server:", status.message)
    if not status.reachable:
        return 3

    def progress(stage: str, step: int, total: int) -> None:
        print(f"\r  {stage:<10} {step:>3}/{total}", end="", flush=True)

    try:
        outcome = TryOnService(backend).run(
            person, garment, args.type, args.steps, args.guidance, args.seed, repaint=not args.no_repaint,
            garment_label=Path(args.garment).name, progress=progress,
        )
    except TryOnError as exc:
        print("\nTRY-ON ERROR:", exc.user_message, f"({exc.detail})" if exc.detail else "")
        return 4
    print()
    result = outcome.result
    print(f"Engine: {result.engine} | server timings: {result.timings} | round trip: {result.round_trip_s} s"
          f" | peak GPU memory: {result.peak_gpu_memory_gb} GB")
    if outcome.saved:
        sheet_path = outcome.saved.folder / "comparison.jpg"
        comparison_sheet(outcome.prepared.person_preview, outcome.prepared.garment, result).save(sheet_path, quality=92)
        print("Saved to:", outcome.saved.folder)
    return 0


if __name__ == "__main__":
    sys.exit(main())
