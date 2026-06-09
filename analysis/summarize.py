"""Regenerate the benchmark charts from the committed result files.

This is the entry point the README points at. It rebuilds every figure in
`assets/` straight from the ground-truth CSVs/JSON, so the charts are always
reproducible and never drift from the data.

Usage:
    python -m analysis.summarize                 # regenerate all charts -> assets/
    python -m analysis.summarize --only kv_cache # regenerate one chart
    python -m analysis.summarize --outdir /tmp/preview   # render elsewhere to diff first

Charts:
    engine_comparison, thread_sweep, kv_cache, offload_cliff, batch_size

Note: assets/prefill_decode.png is a hand-drawn architecture diagram, not a data
chart, so it is intentionally NOT regenerated here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis.plots import ASSETS_DIR, CHARTS


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate benchmark charts from result files.")
    parser.add_argument(
        "--only",
        choices=sorted(CHARTS),
        help="Regenerate a single chart instead of all of them.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ASSETS_DIR,
        help="Directory to write charts into (default: assets/). "
        "Use a scratch dir to preview before overwriting published assets.",
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    selected = {args.only: CHARTS[args.only]} if args.only else CHARTS

    print(f"Writing {len(selected)} chart(s) to {args.outdir}\n")
    for key, (fn, filename) in selected.items():
        out_path = args.outdir / filename
        try:
            fn(out_path)
            print(f"  [ok]   {key:18s} -> {out_path}")
        except FileNotFoundError as exc:
            print(f"  [skip] {key:18s} -> missing input: {exc}")
        except Exception as exc:  # noqa: BLE001 - surface any plotting failure clearly
            print(f"  [FAIL] {key:18s} -> {type(exc).__name__}: {exc}")

    print(
        "\nDone. If you wrote into assets/, visually diff against the previously "
        "published charts before committing — confirm they match what the article shows."
    )


if __name__ == "__main__":
    main()