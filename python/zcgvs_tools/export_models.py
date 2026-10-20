"""Exports the bi-encoder and cross-encoder to ONNX and writes the WordPiece vocab.

The engine consumes ``model.onnx`` for each model and a single ``vocab.txt`` (both
checkpoints share the same BERT uncased vocabulary).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from optimum.onnxruntime import (
    ORTModelForFeatureExtraction,
    ORTModelForSequenceClassification,
)
from transformers import AutoTokenizer

DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L6-v2"


def export_encoder(model_id: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
    model.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(model_id).save_pretrained(out_dir)
    return out_dir / "model.onnx"


def export_reranker(model_id: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = ORTModelForSequenceClassification.from_pretrained(model_id, export=True)
    model.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(model_id).save_pretrained(out_dir)
    return out_dir / "model.onnx"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--encoder-model", default=DEFAULT_ENCODER)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER)
    args = parser.parse_args()

    root = Path(args.output_dir)
    encoder_onnx = export_encoder(args.encoder_model, root / "encoder")
    reranker_onnx = export_reranker(args.reranker_model, root / "reranker")

    vocab_source = root / "encoder" / "vocab.txt"
    vocab_target = root / "vocab.txt"
    shutil.copyfile(vocab_source, vocab_target)

    print(f"encoder  : {encoder_onnx}")
    print(f"reranker : {reranker_onnx}")
    print(f"vocab    : {vocab_target}")


if __name__ == "__main__":
    main()
