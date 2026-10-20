"""Derives quantized corpora from an existing float32 embedding file.

Nothing is re-embedded and no model is loaded: this reads the float32 matrix off disk and
writes float16 / int8 / binary variants of it. Pair a binary primary index with the float32
file as the rescore store to get a 32x smaller scan that still returns exact scores.

    python -m zcgvs_tools.quantize --input data/scifact.bin --dtype binary int8 float16
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from zcgvs_tools import formats


def variant_path(source: Path, dtype: str) -> Path:
    return source.with_name(f"{source.stem}.{dtype}{source.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="float32 embedding file (.bin)")
    parser.add_argument(
        "--dtype",
        nargs="+",
        default=["float16", "int8", "binary"],
        choices=["float16", "int8", "binary"],
    )
    parser.add_argument("--output-dir", default=None, help="defaults to the input directory")
    args = parser.parse_args()

    source = Path(args.input)
    header = formats.read_embedding_header(source)
    if header["dtype"] != "float32":
        raise SystemExit(
            f"{source} stores {header['dtype']}; quantization needs the float32 corpus"
        )

    matrix, normalized = formats.read_embeddings(source)
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)

    output_dir = Path(args.output_dir) if args.output_dir else source.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"source     : {source} ({header['doc_count']} x {header['dim']}, "
          f"{header['bytes'] / 1e6:.1f} MB)")

    for dtype in args.dtype:
        target = output_dir / variant_path(source, dtype).name
        formats.write_embeddings(target, matrix, normalized=normalized, dtype=dtype)
        written = formats.read_embedding_header(target)
        ratio = header["bytes"] / max(written["bytes"], 1)
        print(
            f"{dtype:<10} : {target} ({written['bytes'] / 1e6:.1f} MB, {ratio:.1f}x smaller, "
            f"scale={written['scale']:.6g})"
        )


if __name__ == "__main__":
    main()
