"""SciFact claim-verification labels (Wadden et al., EMNLP 2020).

BEIR's SciFact "test" split is the original SciFact *dev* split: the same 300 claims, with the
abstracts each claim cites as relevance judgements. The original release additionally labels
each claim as SUPPORT, CONTRADICT or NOT ENOUGH INFO, which is what turns the retrieval
benchmark into an end-to-end fact-checking benchmark for the Llama stage.
"""

from __future__ import annotations

import json
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

RELEASE_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "scifact_claims"


@dataclass
class Claim:
    claim_id: str
    text: str
    label: str
    evidence_doc_ids: list[str]
    cited_doc_ids: list[str]


def ensure_release(root: Path = DEFAULT_DIR) -> Path:
    claims = root / "data" / "claims_dev.jsonl"
    if claims.exists():
        return claims
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "data.tar.gz"
    if not archive.exists():
        urllib.request.urlretrieve(RELEASE_URL, archive)
    with tarfile.open(archive) as bundle:
        bundle.extractall(root, filter="data")
    return claims


def claim_label(evidence: dict) -> str:
    labels = {sentence["label"] for sets in evidence.values() for sentence in sets}
    if not labels:
        return "NOT ENOUGH INFO"
    if len(labels) > 1:
        return "MIXED"
    return labels.pop()


def load_claims(split: str = "dev", root: Path = DEFAULT_DIR) -> list[Claim]:
    """``dev`` is the evaluation split (= BEIR test); ``train`` is for fitting calibration."""
    path = ensure_release(root).parent / f"claims_{split}.jsonl"
    claims: list[Claim] = []
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            claims.append(Claim(
                claim_id=str(row["id"]),
                text=row["claim"],
                label=claim_label(row["evidence"]),
                evidence_doc_ids=[str(doc_id) for doc_id in row["evidence"]],
                cited_doc_ids=[str(doc_id) for doc_id in row["cited_doc_ids"]],
            ))
    return claims


def load_dev_claims(root: Path = DEFAULT_DIR) -> list[Claim]:
    return load_claims("dev", root)
