"""Embeds a BEIR corpus subset with the bi-encoder and writes the engine artifacts.

Run once, offline. Produces ``<dataset>.bin`` (the VRAM-resident embedding matrix) and
``<dataset>.meta`` (doc ids and passage text for the reranker). Use
``zcgvs_tools.quantize`` afterwards to derive float16 / int8 / binary variants without
re-embedding anything.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from zcgvs_tools import beir_loader, formats

DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--model", default=DEFAULT_ENCODER)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "int8", "binary"],
        default="float32",
        help="on-disk storage precision of the embedding matrix",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    doc_ids, texts = beir_loader.load_corpus(args.dataset, args.limit)

    model = SentenceTransformer(args.model, device=args.device)
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)

    embeddings_path = output_dir / f"{args.dataset}.bin"
    metadata_path = output_dir / f"{args.dataset}.meta"
    formats.write_embeddings(embeddings_path, embeddings, normalized=True, dtype=args.dtype)
    formats.write_metadata(metadata_path, doc_ids, texts)

    header = formats.read_embedding_header(embeddings_path)
    print(f"embeddings : {embeddings_path} ({header['doc_count']} x {header['dim']}, "
          f"{header['dtype']}, {header['bytes'] / 1e6:.1f} MB)")
    print(f"metadata   : {metadata_path}")


if __name__ == "__main__":
    main()
