"""FastAPI microservice exposing the zcgvs engine over HTTP.

Endpoints:
  GET  /health          liveness plus corpus size and storage dtype
  GET  /stats           full engine, index and device statistics
  POST /search          one query; returns ranked hits and per-stage timing
  POST /search/batch    many queries in a single device pass
  POST /retrieve        search stage only, no cross-encoder
  POST /documents       add documents to the live VRAM index
  POST /verify          fact-check a claim: retrieval + Llama verdict + cited explanation
  POST /ask             grounded question answering with Llama, cited to retrieved sources

Run with an engine config JSON (``--llama`` enables /verify and /ask):
  PYTHONPATH=build/bindings:python python -m zcgvs_tools.service --config configs/scifact.json \
      --llama meta-llama/Llama-3.2-3B-Instruct
"""

from __future__ import annotations

import argparse
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import zcgvs_engine
from zcgvs_tools.config import backend_from_name, build_engine, describe, metric_from_name
from zcgvs_tools.factcheck import ClaimVerifier, GroundedAnswerer


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    candidates: int = 100
    metric: Optional[str] = None
    allow: Optional[list[int]] = None
    hybrid: bool = False
    dense_weight: Optional[float] = None
    rescore_candidates: Optional[int] = None
    rerank: bool = True
    backend: Optional[str] = None
    use_graph: Optional[bool] = None


class BatchSearchRequest(SearchRequest):
    queries: list[str] = []
    query: str = ""


class RetrieveRequest(BaseModel):
    query: str
    k: int = 100
    metric: Optional[str] = None
    allow: Optional[list[int]] = None


class AddRequest(BaseModel):
    ids: list[str]
    texts: list[str]


class VerifyRequest(BaseModel):
    claim: str
    evidence_k: int = 3
    explain: bool = True


class AskRequest(BaseModel):
    question: str
    k: int = 5


def _options(engine, request: SearchRequest) -> "zcgvs_engine.SearchOptions":
    options = engine.default_options()
    options.top_k = request.top_k
    options.candidates = request.candidates
    options.allow = request.allow or []
    options.hybrid = request.hybrid
    options.rerank = request.rerank

    if request.metric is not None:
        options.use_config_metric = False
        options.metric = metric_from_name(request.metric)
    if request.dense_weight is not None:
        options.dense_weight = request.dense_weight
    if request.rescore_candidates is not None:
        options.rescore_candidates = request.rescore_candidates
    if request.backend is not None:
        options.use_config_backend = False
        options.backend = backend_from_name(request.backend)
    if request.use_graph is not None:
        options.use_config_graphs = False
        options.use_graph = request.use_graph
    return options


def _hits(hits) -> list[dict]:
    return [
        {
            "doc_id": hit.doc_id,
            "corpus_index": hit.corpus_index,
            "similarity": hit.similarity,
            "rerank_score": hit.rerank_score,
            "text": hit.text,
        }
        for hit in hits
    ]


def _timing(timing) -> dict:
    return {
        "tokenize_ms": timing.tokenize_ms,
        "embed_ms": timing.embed_ms,
        "search_gpu_ms": timing.search_gpu_ms,
        "copy_ms": timing.copy_ms,
        "rerank_ms": timing.rerank_ms,
        "total_ms": timing.total_ms,
        "batch": timing.batch,
        "graph_replay": timing.graph_replay,
    }


def _evidence(evidence) -> list[dict]:
    return [
        {
            "rank": item.rank,
            "doc_id": item.doc_id,
            "similarity": item.similarity,
            "rerank_score": item.rerank_score,
            "text": item.text,
        }
        for item in evidence
    ]


def create_app(engine, llama=None) -> FastAPI:
    app = FastAPI(title="Zero-Copy GPU Vector Search & Re-Ranking Engine")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "corpus_size": engine.corpus_size,
            "dim": engine.dim,
            "storage_dtype": engine.storage_dtype,
            "hybrid_available": engine.has_lexical_index,
            "rescore_available": engine.has_rescore_store,
            "reranker": engine.has_reranker,
            "llama": llama.name if llama is not None else None,
        }

    @app.get("/stats")
    def stats() -> dict:
        return describe(engine)

    @app.post("/search")
    def search(request: SearchRequest) -> dict:
        try:
            response = engine.search_with(request.query, _options(engine, request))
        except (ValueError, RuntimeError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"hits": _hits(response.hits), "timing": _timing(response.timing)}

    @app.post("/search/batch")
    def search_batch(request: BatchSearchRequest) -> dict:
        queries = request.queries or ([request.query] if request.query else [])
        if not queries:
            raise HTTPException(status_code=400, detail="queries must not be empty")
        try:
            response = engine.search_batch(queries, _options(engine, request))
        except (ValueError, RuntimeError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {
            "results": [_hits(hits) for hits in response.results],
            "timing": _timing(response.timing),
        }

    @app.post("/retrieve")
    def retrieve(request: RetrieveRequest) -> dict:
        metric = metric_from_name(request.metric) if request.metric else zcgvs_engine.Metric.Cosine
        try:
            candidates = engine.retrieve(request.query, request.k, metric, request.allow or [])
        except (ValueError, RuntimeError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {
            "candidates": [
                {
                    "doc_id": candidate.doc_id,
                    "corpus_index": candidate.corpus_index,
                    "score": candidate.score,
                }
                for candidate in candidates
            ]
        }

    @app.post("/documents")
    def add_documents(request: AddRequest) -> dict:
        try:
            engine.add_documents(request.ids, request.texts)
        except (ValueError, RuntimeError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"corpus_size": engine.corpus_size}

    def _require_llama():
        if llama is None:
            raise HTTPException(status_code=503, detail="start the service with --llama")
        return llama

    @app.post("/verify")
    def verify(request: VerifyRequest) -> dict:
        verifier = ClaimVerifier(engine, _require_llama(), evidence_k=request.evidence_k)
        verdict = verifier.verify(request.claim, explain=request.explain)
        return {
            "claim": verdict.claim,
            "verdict": verdict.label,
            "probabilities": verdict.probabilities,
            "explanation": verdict.explanation,
            "evidence": _evidence(verdict.evidence),
            "timing_ms": verdict.timing_ms,
        }

    @app.post("/ask")
    def ask(request: AskRequest) -> dict:
        answerer = GroundedAnswerer(engine, _require_llama(), k=request.k)
        generation, evidence, timing = answerer.answer(request.question)
        return {"answer": generation.text, "sources": _evidence(evidence), "timing_ms": timing}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="engine config JSON")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--llama", default=None,
                        help="Llama model id or local dir; enables /verify and /ask")
    args = parser.parse_args()

    engine = build_engine(args.config)
    llama = None
    if args.llama:
        from zcgvs_tools.llama import Llama

        llama = Llama(args.llama)
    uvicorn.run(create_app(engine, llama), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
