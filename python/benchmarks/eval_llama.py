"""Evaluates the Llama reasoning stages on SciFact, end to end through the GPU engine.

Three experiments, each runnable on its own via ``--tasks``:

verify   Claim verification on the 300 SciFact dev claims (= BEIR SciFact test queries).
         *open*: evidence is whatever the engine retrieves (top-k after the cross-encoder).
         *oracle*: evidence is the abstracts the claim's authors cited — an upper bound that
         separates Llama's reasoning errors from retrieval errors.
         Reports accuracy, macro-F1, per-class F1 and the confusion matrix, raw and after
         contextual calibration fitted on the disjoint SciFact *train* claims.

rerank   nDCG@10 / Recall@10 of: the dense scan alone, the ONNX MiniLM cross-encoder, Llama as a
         zero-shot reranker (query likelihood and yes/no), a cross-encoder → Llama cascade, and
         reciprocal-rank fusion of the two rerankers.

latency  Per-stage wall-clock of ``ClaimVerifier.verify`` with explanation: CUDA retrieval,
         Llama verdict (one forward pass), time to first explanation token, decode rate.

  PYTHONPATH=build/bindings:python python python/benchmarks/eval_llama.py \\
      --config configs/scifact.json --llama meta-llama/Llama-3.2-3B-Instruct
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from eval_quality import ndcg_at_k, recall_at_k
from zcgvs_tools import beir_loader, config, scifact_claims
from zcgvs_tools.factcheck import (
    LABELS,
    ClaimVerifier,
    Evidence,
    LlamaReranker,
    calibration_path,
    fit_label_bias,
    macro_f1,
    reciprocal_rank_fusion as rrf,
    retrieve_evidence,
)
from zcgvs_tools.llama import Llama


def classification_report(gold: list[str], predicted: list[str]) -> dict:
    confusion = {g: {p: 0 for p in LABELS} for g in LABELS}
    for g, p in zip(gold, predicted):
        confusion[g][p] += 1
    per_class = {}
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[g][label] for g in LABELS if g != label)
        fn = sum(confusion[label][p] for p in LABELS if p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
    accuracy = sum(g == p for g, p in zip(gold, predicted)) / max(len(gold), 1)
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean([per_class[label]["f1"] for label in LABELS])),
        "per_class": per_class,
        "confusion": confusion,
        "n": len(gold),
    }


def run_verify(llama: Llama, engine, claims, corpus: dict[str, str], open_evidence,
               args) -> dict:
    verifier = ClaimVerifier(engine, llama, evidence_k=args.evidence_k, bias=False)
    gold = [claim.label for claim in claims]
    gold_index = np.array([LABELS.index(label) for label in gold])

    # Calibration is fitted on SciFact *train* claims (disjoint from the 300 evaluated here),
    # with the same open-retrieval evidence the dev claims get, then frozen.
    train = [c for c in scifact_claims.load_claims("train") if c.label in LABELS]
    train = train[: args.calibration_claims]
    train_evidence = [retrieve_evidence(engine, c.text, args.evidence_k)[0] for c in train]
    train_logprobs = verifier.label_logprobs([c.text for c in train], train_evidence,
                                             args.verify_batch)
    train_gold = np.array([LABELS.index(c.label) for c in train])
    bias = fit_label_bias(train_logprobs, train_gold)
    calibrated_train = macro_f1(train_gold, np.argmax(train_logprobs + bias, axis=1))

    path = calibration_path(llama.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "model": llama.name,
        "method": "contextual calibration: additive label log-prob offsets, grid-searched "
                  "for macro-F1",
        "fitted_on": f"SciFact train, {len(train)} claims, open retrieval top-"
                     f"{args.evidence_k} evidence",
        "train_macro_f1": {
            "raw": macro_f1(train_gold, np.argmax(train_logprobs, axis=1)),
            "calibrated": calibrated_train,
        },
        "bias": {label: float(value) for label, value in zip(LABELS, bias)},
    }, indent=2))
    verifier.bias = bias

    oracle_evidence = [
        [Evidence(rank, doc_id, corpus[doc_id], 0.0, 0.0)
         for rank, doc_id in enumerate(claim.cited_doc_ids[: args.evidence_k], start=1)]
        for claim in claims
    ]

    results = {"calibration": {"bias": dict(zip(LABELS, map(float, bias))),
                               "train_claims": len(train)}}
    for setting, evidence_sets in (("open", open_evidence), ("oracle", oracle_evidence)):
        start = time.perf_counter()
        logprobs = verifier.label_logprobs([claim.text for claim in claims], evidence_sets,
                                           args.verify_batch)
        elapsed = time.perf_counter() - start
        for variant, scores in (("raw", logprobs), ("calibrated", logprobs + bias)):
            predicted = [LABELS[int(i)] for i in scores.argmax(axis=1)]
            report = classification_report(gold, predicted)
            report["claims_per_s"] = len(claims) / elapsed
            results[f"{setting}_{variant}"] = report

    majority = max(LABELS, key=gold.count)
    results["majority_baseline"] = classification_report(gold, [majority] * len(gold))
    return results


def run_rerank(llama: Llama, engine, queries, qrels, args) -> dict:
    rerankers = {
        mode: LlamaReranker(llama, mode=mode, batch_size=args.rerank_batch)
        for mode in args.rerank_modes
    }
    ndcg: dict[str, list[float]] = {}
    recall: dict[str, list[float]] = {}
    seconds = {mode: 0.0 for mode in rerankers}

    pairs = 0
    for query_id, query_text in queries:
        relevant = qrels[query_id]
        dense, _ = retrieve_evidence(engine, query_text, args.rerank_depth, rerank=False)
        ce, _ = retrieve_evidence(engine, query_text, args.rerank_depth, rerank=True)

        texts = {item.doc_id: item.text for item in dense + ce}
        doc_ids = list(texts)
        pairs += len(doc_ids)

        dense_ids = [item.doc_id for item in dense]
        ce_ids = [item.doc_id for item in ce]
        rankings = {"dense": dense_ids, "cross_encoder": ce_ids}

        for mode, reranker in rerankers.items():
            start = time.perf_counter()
            scored = reranker.score(query_text, [texts[d] for d in doc_ids])
            seconds[mode] += time.perf_counter() - start
            llama_scores = dict(zip(doc_ids, scored))

            def by_llama(ids: list[str]) -> list[str]:
                return sorted(ids, key=lambda d: -llama_scores[d])

            rankings[f"llama[{mode}]"] = by_llama(dense_ids)
            rankings[f"ce→llama[{mode}]"] = (by_llama(ce_ids[: args.cascade_depth])
                                             + ce_ids[args.cascade_depth :])
            rankings[f"rrf(ce,llama[{mode}])"] = rrf([ce_ids, by_llama(ce_ids)])

        for name, ranked in rankings.items():
            ndcg.setdefault(name, []).append(ndcg_at_k(ranked, relevant, 10))
            value = recall_at_k(ranked, relevant, 10)
            if value is not None:
                recall.setdefault(name, []).append(value)

    return {
        "queries": len(queries),
        "depth": args.rerank_depth,
        "cascade_depth": args.cascade_depth,
        "pairs_scored": pairs,
        "pairs_per_s": {mode: pairs / max(s, 1e-9) for mode, s in seconds.items()},
        "ndcg@10": {name: float(np.mean(values)) for name, values in ndcg.items()},
        "recall@10": {name: float(np.mean(values)) for name, values in recall.items()},
        "per_query_ndcg@10": {name: [float(v) for v in values] for name, values in ndcg.items()},
    }


def run_latency(llama: Llama, engine, claims, args) -> dict:
    verifier = ClaimVerifier(engine, llama, evidence_k=args.evidence_k)
    for claim in claims[:3]:
        verifier.verify(claim.text, explain=True)

    samples: dict[str, list[float]] = {}
    for claim in claims[3 : 3 + args.latency_claims]:
        verdict = verifier.verify(claim.text, explain=True)
        for stage, value in verdict.timing_ms.items():
            samples.setdefault(stage, []).append(value)
    return {stage: statistics.median(values) for stage, values in samples.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/scifact.json")
    parser.add_argument("--llama", nargs="+", default=["meta-llama/Llama-3.2-3B-Instruct"],
                        help="one or more Llama model ids or local directories")
    parser.add_argument("--tasks", nargs="+", default=["verify", "rerank", "latency"],
                        choices=["verify", "rerank", "latency"])
    parser.add_argument("--max-claims", type=int, default=300)
    parser.add_argument("--evidence-k", type=int, default=3)
    parser.add_argument("--verify-batch", type=int, default=4)
    parser.add_argument("--calibration-claims", type=int, default=809,
                        help="SciFact train claims used to fit the label calibration")
    parser.add_argument("--rerank-depth", type=int, default=20)
    parser.add_argument("--rerank-modes", nargs="+", default=list(LlamaReranker.MODES),
                        choices=list(LlamaReranker.MODES))
    parser.add_argument("--cascade-depth", type=int, default=10)
    parser.add_argument("--rerank-batch", type=int, default=16)
    parser.add_argument("--latency-claims", type=int, default=20)
    parser.add_argument("--out", default="docs/results")
    args = parser.parse_args()

    engine = config.build_engine(args.config)
    stats = engine.stats()

    claims = [c for c in scifact_claims.load_dev_claims() if c.label in LABELS][: args.max_claims]
    corpus_ids, corpus_texts = beir_loader.load_corpus("scifact")
    corpus = dict(zip(corpus_ids, corpus_texts))
    qrels = beir_loader.load_qrels("scifact")
    queries = [(c.claim_id, c.text) for c in claims if c.claim_id in qrels]

    # Retrieval is independent of the Llama model, so it is done once and shared.
    open_evidence = [retrieve_evidence(engine, c.text, args.evidence_k)[0] for c in claims]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_id in args.llama:
        torch.cuda.reset_peak_memory_stats()
        llama = Llama(model_id)
        result = {
            "model": llama.name,
            "weights": llama.model_id,
            "parameters": llama.parameter_count,
            "weight_mb": llama.weight_bytes / 1e6,
            "load_s": llama.load_s,
            "engine": {
                "config": Path(args.config).name,
                "corpus_size": stats.corpus_size,
                "storage": stats.storage_dtype,
                "corpus_mb": stats.corpus_bytes / 1e6,
                "device": stats.device_name,
            },
        }
        print(f"\n=== {llama.name} ({llama.parameter_count / 1e9:.2f}B params, "
              f"{llama.weight_bytes / 1e9:.1f} GB bf16, loaded in {llama.load_s:.1f}s) ===")

        if "verify" in args.tasks:
            result["verify"] = run_verify(llama, engine, claims, corpus, open_evidence, args)
            for setting in ("majority_baseline", "open_raw", "open_calibrated", "oracle_raw",
                            "oracle_calibrated"):
                row = result["verify"][setting]
                print(f"verify {setting:<18} acc={row['accuracy']:.3f} "
                      f"macroF1={row['macro_f1']:.3f} "
                      + " ".join(f"{label[:3]}F1={row['per_class'][label]['f1']:.3f}"
                                 for label in LABELS))

        if "rerank" in args.tasks:
            result["rerank"] = run_rerank(llama, engine, queries, qrels, args)
            for name, value in result["rerank"]["ndcg@10"].items():
                print(f"rerank {name:<34} nDCG@10={value:.4f} "
                      f"Recall@10={result['rerank']['recall@10'][name]:.4f}")
            for mode, rate in result["rerank"]["pairs_per_s"].items():
                print(f"rerank throughput [{mode}]: {rate:.0f} pairs/s")

        if "latency" in args.tasks:
            result["latency_ms"] = run_latency(llama, engine, claims, args)
            print("latency (median ms): " + ", ".join(
                f"{stage}={value:.2f}" for stage, value in result["latency_ms"].items()))

        result["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 1e6
        path = out_dir / f"llama_eval_{llama.name}.json"
        # Merge, so re-running one task keeps the results of the others.
        merged = json.loads(path.read_text()) if path.exists() else {}
        merged.update(result)
        path.write_text(json.dumps(merged, indent=2))
        print(f"wrote {path}")

        del llama
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
