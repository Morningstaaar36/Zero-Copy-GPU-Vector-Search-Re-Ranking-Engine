"""Builds a configured ``Engine`` from a JSON file or a plain dict.

Keeps the C++ ``EngineConfig`` struct free of a JSON dependency: parsing lives here on the
Python side. Relative paths in a config file are resolved against the file's directory, so a
config can be run from anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import zcgvs_engine

METRICS = {
    "cosine": zcgvs_engine.Metric.Cosine,
    "dot": zcgvs_engine.Metric.DotProduct,
    "dotproduct": zcgvs_engine.Metric.DotProduct,
    "ip": zcgvs_engine.Metric.DotProduct,
    "l2": zcgvs_engine.Metric.L2,
    "euclidean": zcgvs_engine.Metric.L2,
}

BACKENDS = {
    "custom": zcgvs_engine.SimilarityBackend.Custom,
    "cublas": zcgvs_engine.SimilarityBackend.Cublas,
}

PATH_FIELDS = (
    "embeddings_path",
    "metadata_path",
    "vocab_path",
    "rescore_embeddings_path",
    "lexical_index_path",
    "encoder_model_path",
    "reranker_model_path",
)


def metric_from_name(name: str) -> "zcgvs_engine.Metric":
    try:
        return METRICS[name.lower()]
    except KeyError as error:
        raise ValueError(f"unknown metric '{name}'") from error


def backend_from_name(name: str) -> "zcgvs_engine.SimilarityBackend":
    try:
        return BACKENDS[name.lower()]
    except KeyError as error:
        raise ValueError(f"unknown similarity backend '{name}'") from error


def _resolve_path(value: str, search_roots: tuple[Path, ...]) -> str:
    """Accepts artifact paths relative to the working directory, the config file, or the
    repository root, so the same config works from anywhere."""
    candidate = Path(value)
    if candidate.is_absolute() or candidate.exists():
        return str(candidate)
    for root in search_roots:
        resolved = root / candidate
        if resolved.exists():
            return str(resolved)
    return str(candidate)


def config_from_dict(values: dict, search_roots: tuple[Path, ...] = ()) -> "zcgvs_engine.EngineConfig":
    config = zcgvs_engine.EngineConfig()
    for key, value in values.items():
        if key == "metric":
            config.metric = metric_from_name(str(value))
        elif key == "similarity_backend":
            config.similarity_backend = backend_from_name(str(value))
        elif key in PATH_FIELDS and search_roots and value:
            setattr(config, key, _resolve_path(str(value), search_roots))
        elif hasattr(config, key):
            setattr(config, key, value)
        else:
            raise KeyError(f"unknown config field: {key}")
    return config


def config_from_json(path: str | Path) -> "zcgvs_engine.EngineConfig":
    config_path = Path(path).resolve()
    values = json.loads(config_path.read_text())
    return config_from_dict(values, search_roots=(config_path.parent, config_path.parent.parent))


def build_engine(source: str | Path | dict) -> "zcgvs_engine.Engine":
    if isinstance(source, dict):
        config = config_from_dict(source)
    elif isinstance(source, (str, Path)):
        config = config_from_json(source)
    else:
        config = source
    return zcgvs_engine.Engine(config)


def describe(engine: "zcgvs_engine.Engine") -> dict:
    """Flattens ``Engine.stats()`` into a JSON-serializable dict."""
    stats = engine.stats()
    return {
        "corpus_size": stats.corpus_size,
        "dim": stats.dim,
        "storage_dtype": stats.storage_dtype,
        "corpus_mb": round(stats.corpus_bytes / 1e6, 2),
        "rescore": {
            "enabled": stats.has_rescore_store,
            "dtype": stats.rescore_dtype,
            "mb": round(stats.rescore_bytes / 1e6, 2),
        },
        "lexical": {
            "enabled": stats.has_lexical_index,
            "postings": stats.lexical_postings,
            "mb": round(stats.lexical_bytes / 1e6, 2),
        },
        "reranker": stats.has_reranker,
        "graph": {"captures": stats.graph_captures, "launches": stats.graph_launches},
        "device": {
            "name": stats.device_name,
            "compute": f"{stats.compute_major}.{stats.compute_minor}",
            "total_mb": round(stats.device_total_memory / 1e6, 2),
        },
    }
