"""Builds the CSR inverted index that backs the hybrid BM25 stage.

Terms are WordPiece ids from the same ``vocab.txt`` the engine tokenizes queries with, so no
separate analyzer has to be kept in sync across the Python/C++ boundary.

Every posting stores its *complete* BM25 contribution — idf, term frequency saturation and
length normalization are all folded in offline — which reduces query-time device work to a
scatter-add over the postings of the query's terms.

    python -m zcgvs_tools.build_lexical --metadata data/scifact.meta \\
        --vocab models/vocab.txt --output data/scifact.lex
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from zcgvs_tools import formats


def load_vocab(path: str | Path) -> dict[str, int]:
    vocab: dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            vocab[line.rstrip("\n").rstrip("\r")] = index
    if not vocab:
        raise ValueError(f"vocab file '{path}' is empty")
    return vocab


def build_tokenizer(vocab_path: str | Path, tokenizer_dir: str | None):
    """Prefers the exported HuggingFace tokenizer; falls back to a local WordPiece pass.

    The fallback mirrors ``src/text/tokenizer.cpp`` (lowercase, punctuation split, greedy
    longest-match WordPiece) so ids agree with the engine even without ``transformers``.
    """
    if tokenizer_dir:
        from transformers import AutoTokenizer

        hf = AutoTokenizer.from_pretrained(tokenizer_dir)

        def encode(text: str) -> list[int]:
            return hf(text, add_special_tokens=False, truncation=False)["input_ids"]

        return encode

    vocab = load_vocab(vocab_path)
    unk_id = vocab.get("[UNK]", 0)

    def basic_tokens(text: str) -> list[str]:
        tokens: list[str] = []
        current: list[str] = []
        for character in text:
            if character.isspace():
                if current:
                    tokens.append("".join(current))
                    current = []
            elif not character.isalnum() and ord(character) < 128:
                if current:
                    tokens.append("".join(current))
                    current = []
                tokens.append(character)
            else:
                current.append(character.lower())
        if current:
            tokens.append("".join(current))
        return tokens

    def wordpiece(token: str) -> list[int]:
        if len(token) > 200:
            return [unk_id]
        pieces: list[int] = []
        start = 0
        while start < len(token):
            end = len(token)
            matched = None
            while start < end:
                piece = token[start:end]
                if start > 0:
                    piece = "##" + piece
                if piece in vocab:
                    matched = vocab[piece]
                    break
                end -= 1
            if matched is None:
                return [unk_id]
            pieces.append(matched)
            start = end
        return pieces

    def encode(text: str) -> list[int]:
        ids: list[int] = []
        for token in basic_tokens(text):
            ids.extend(wordpiece(token))
        return ids

    return encode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, help="corpus metadata (.meta)")
    parser.add_argument("--vocab", required=True, help="WordPiece vocab.txt")
    parser.add_argument("--output", required=True, help="destination .lex file")
    parser.add_argument(
        "--tokenizer-dir",
        default=None,
        help="exported HuggingFace tokenizer directory (e.g. models/encoder); "
        "omit to use the built-in WordPiece pass",
    )
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--max-doc-tokens", type=int, default=512)
    args = parser.parse_args()

    vocab = load_vocab(args.vocab)
    vocab_size = len(vocab)
    encode = build_tokenizer(args.vocab, args.tokenizer_dir)

    _, texts = formats.read_metadata(args.metadata)
    doc_count = len(texts)

    term_frequencies: list[dict[int, int]] = []
    lengths = np.zeros(doc_count, dtype=np.float64)
    document_frequency: dict[int, int] = defaultdict(int)

    for index, text in enumerate(texts):
        ids = encode(text)[: args.max_doc_tokens]
        counts: dict[int, int] = defaultdict(int)
        for term in ids:
            if 0 <= term < vocab_size:
                counts[term] += 1
        term_frequencies.append(counts)
        lengths[index] = float(len(ids))
        for term in counts:
            document_frequency[term] += 1

    average_length = float(lengths.mean()) if doc_count else 0.0
    if average_length <= 0.0:
        average_length = 1.0

    postings: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for index, counts in enumerate(term_frequencies):
        normalizer = args.k1 * (1.0 - args.b + args.b * lengths[index] / average_length)
        for term, frequency in counts.items():
            df = document_frequency[term]
            idf = math.log(1.0 + (doc_count - df + 0.5) / (df + 0.5))
            weight = idf * (frequency * (args.k1 + 1.0)) / (frequency + normalizer)
            postings[term].append((index, weight))

    offsets = np.zeros(vocab_size + 1, dtype=np.int64)
    posting_docs: list[int] = []
    posting_weights: list[float] = []
    for term in range(vocab_size):
        for doc, weight in postings.get(term, ()):
            posting_docs.append(doc)
            posting_weights.append(weight)
        offsets[term + 1] = len(posting_docs)

    formats.write_lexical_index(
        args.output,
        doc_count=doc_count,
        vocab_size=vocab_size,
        offsets=offsets,
        posting_docs=np.asarray(posting_docs, dtype=np.int32),
        posting_weights=np.asarray(posting_weights, dtype=np.float32),
        k1=args.k1,
        b=args.b,
        average_doc_length=average_length,
    )

    size_mb = Path(args.output).stat().st_size / 1e6
    print(f"documents : {doc_count}")
    print(f"vocab     : {vocab_size}")
    print(f"postings  : {len(posting_docs)}")
    print(f"avg length: {average_length:.1f} tokens")
    print(f"index     : {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
