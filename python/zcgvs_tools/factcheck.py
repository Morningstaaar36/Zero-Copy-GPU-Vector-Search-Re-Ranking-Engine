"""Retrieval-augmented reasoning with Llama on top of the zero-copy GPU engine.

Three stages, each usable on its own:

``LlamaReranker``
    Pointwise relevance with an instruction-tuned Llama: one forward pass per (query, passage)
    pair, scored as ``log P("Yes") - log P("No")`` at the first answer position. Drop-in
    alternative or complement to the engine's ONNX cross-encoder.

``ClaimVerifier``
    Scientific fact-checking. The engine retrieves evidence abstracts (CUDA scan + on-device
    top-K + cross-encoder), Llama classifies the claim as SUPPORT / CONTRADICT /
    NOT ENOUGH INFO from the label logits — a calibrated probability, not parsed text — and
    then writes a short explanation that cites the abstracts it used.

``GroundedAnswerer``
    Question answering restricted to retrieved sources, with ``[n]`` citations.

The engine owns retrieval; this module only sees the ``k`` hits that cross to the host.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from zcgvs_tools.llama import Generation, Llama

LABELS = ("SUPPORT", "CONTRADICT", "NOT ENOUGH INFO")

CALIBRATION_DIR = Path(__file__).resolve().parents[2] / "configs" / "calibration"

VERIFY_SYSTEM = (
    "You are a meticulous biomedical fact-checker. You judge scientific claims strictly "
    "against the research abstracts you are given, never against outside knowledge."
)

ANSWER_SYSTEM = (
    "You are a precise scientific research assistant. Answer only from the numbered sources. "
    "Cite every factual sentence with the source number in square brackets, like [2]. If the "
    "sources do not contain the answer, say so plainly instead of guessing."
)


@dataclass
class Evidence:
    rank: int
    doc_id: str
    text: str
    similarity: float
    rerank_score: float


@dataclass
class Verdict:
    claim: str
    label: str
    probabilities: dict[str, float]
    evidence: list[Evidence]
    explanation: str = ""
    timing_ms: dict[str, float] = field(default_factory=dict)

    @property
    def confidence(self) -> float:
        return self.probabilities[self.label]


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " …"


def format_sources(evidence: list[Evidence], max_words: int) -> str:
    return "\n\n".join(
        f"[{item.rank}] {truncate_words(item.text, max_words)}" for item in evidence
    )


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    """RRF (Cormack et al., 2009): fuses rankings by summed ``1 / (k + rank)``. Rank-based,
    so it needs no score normalization between rerankers on different scales."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda doc_id: -scores[doc_id])


def retrieve_evidence(engine, query: str, k: int, candidates: int = 100,
                      rerank: bool = True) -> tuple[list[Evidence], dict[str, float]]:
    options = engine.default_options()
    options.top_k = k
    options.candidates = candidates
    options.rerank = rerank
    response = engine.search_with(query, options)
    evidence = [
        Evidence(rank, hit.doc_id, hit.text, hit.similarity, hit.rerank_score)
        for rank, hit in enumerate(response.hits, start=1)
    ]
    timing = response.timing
    return evidence, {
        "embed": timing.embed_ms,
        "gpu_search": timing.search_gpu_ms,
        "copy": timing.copy_ms,
        "cross_encoder": timing.rerank_ms,
        "retrieval_total": timing.total_ms,
    }


class LlamaReranker:
    """Zero-shot passage reranking with Llama, in one of two modes.

    ``query_likelihood`` (default) — UPR (Sachan et al., 2022): score a passage by the mean
    log-probability Llama assigns to the *query* when asked to write a claim the passage is
    evidence about. Measured on SciFact it is the better mode for Llama-3.2-1B.

    ``yes_no`` — pointwise relevance judgement, ``log P("Yes") - log P("No")``. Useless at 1B
    (scrambles the ranking it is given), useful from 3B. See docs/LLAMA.md for the numbers.
    """

    CHOICES = ("Yes", "No")
    MODES = ("query_likelihood", "yes_no")

    def __init__(self, llama: Llama, mode: str = "query_likelihood", max_words: int = 220,
                 batch_size: int = 16) -> None:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self.llama = llama
        self.mode = mode
        self.max_words = max_words
        self.batch_size = batch_size

    def _judge_prompt(self, query: str, passage: str) -> str:
        return self.llama.prompt([
            {"role": "system", "content": VERIFY_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n\nAbstract: {truncate_words(passage, self.max_words)}\n\n"
                    "Is this abstract direct evidence about the query — does it support it, "
                    "refute it, or answer it? Reply with Yes or No."
                ),
            },
        ])

    def _generation_prompt(self, passage: str) -> str:
        return self.llama.prompt([
            {
                "role": "user",
                "content": (
                    f"Abstract: {truncate_words(passage, self.max_words)}\n\n"
                    "Write one short scientific claim that this abstract provides evidence "
                    "about."
                ),
            },
        ])

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        if self.mode == "query_likelihood":
            prompts = [self._generation_prompt(passage) for passage in passages]
            return self.llama.continuation_logprob(prompts, query, self.batch_size)
        prompts = [self._judge_prompt(query, passage) for passage in passages]
        logprobs = self.llama.choice_logprobs(prompts, list(self.CHOICES), self.batch_size)
        return logprobs[:, 0] - logprobs[:, 1]

    def rerank(self, query: str, evidence: list[Evidence]) -> list[Evidence]:
        scores = self.score(query, [item.text for item in evidence])
        order = np.argsort(-scores, kind="stable")
        return [
            Evidence(new_rank, evidence[i].doc_id, evidence[i].text, evidence[i].similarity,
                     float(scores[i]))
            for new_rank, i in enumerate(order, start=1)
        ]


def macro_f1(gold: np.ndarray, predicted: np.ndarray, classes: int = len(LABELS)) -> float:
    scores = []
    for c in range(classes):
        tp = np.sum((predicted == c) & (gold == c))
        denominator = np.sum(predicted == c) + np.sum(gold == c)
        scores.append(2 * tp / denominator if denominator else 0.0)
    return float(np.mean(scores))


def fit_label_bias(logprobs: np.ndarray, gold: np.ndarray, span: float = 8.0,
                   step: float = 0.1) -> np.ndarray:
    """Contextual calibration (Zhao et al., 2021): additive per-label offsets on the label
    log-probabilities, chosen by grid search to maximize macro-F1. SUPPORT is the reference
    (offset 0). Fit on *training* claims only, then frozen."""
    grid = np.arange(-span, span + step / 2, step)
    best_bias, best_score = np.zeros(len(LABELS)), -1.0
    for contradict in grid:
        shifted = logprobs + np.array([0.0, contradict, 0.0])
        for nei in grid:
            predicted = np.argmax(shifted + np.array([0.0, 0.0, nei]), axis=1)
            score = macro_f1(gold, predicted)
            if score > best_score:
                best_score, best_bias = score, np.array([0.0, contradict, nei])
    return best_bias


def calibration_path(model_name: str) -> Path:
    return CALIBRATION_DIR / f"{model_name}.json"


def load_label_bias(model_name: str) -> np.ndarray | None:
    path = calibration_path(model_name)
    if not path.exists():
        return None
    values = json.loads(path.read_text())["bias"]
    return np.array([values[label] for label in LABELS], dtype=np.float64)


class ClaimVerifier:
    """``bias`` defaults to the frozen calibration for this Llama model, if one was fitted
    (``configs/calibration/<model>.json``); pass ``bias=False`` for the raw model."""

    def __init__(self, engine, llama: Llama, evidence_k: int = 3, candidates: int = 100,
                 max_words: int = 230, bias=None) -> None:
        self.engine = engine
        self.llama = llama
        self.evidence_k = evidence_k
        self.candidates = candidates
        self.max_words = max_words
        if bias is None:
            bias = load_label_bias(llama.name)
        self.bias = None if bias is False or bias is None else np.asarray(bias, dtype=np.float64)

    def _messages(self, claim: str, evidence: list[Evidence]) -> list[dict]:
        return [
            {"role": "system", "content": VERIFY_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Claim: {claim}\n\nAbstracts:\n\n"
                    f"{format_sources(evidence, self.max_words)}\n\n"
                    "Based only on these abstracts, does the evidence SUPPORT the claim, "
                    "CONTRADICT it, or is there NOT ENOUGH INFO to decide? Answer with exactly "
                    "one of: SUPPORT, CONTRADICT, NOT ENOUGH INFO."
                ),
            },
        ]

    def label_logprobs(self, claims: list[str], evidence_sets: list[list[Evidence]],
                       batch_size: int = 4) -> np.ndarray:
        """Raw (uncalibrated) label log-probabilities, shape ``[len(claims), 3]``."""
        prompts = [
            self.llama.prompt(self._messages(claim, evidence))
            for claim, evidence in zip(claims, evidence_sets)
        ]
        return self.llama.choice_logprobs(prompts, list(LABELS), batch_size)

    def calibrate(self, logprobs: np.ndarray) -> np.ndarray:
        """Applies the label bias and renormalizes into probabilities."""
        shifted = logprobs + (self.bias if self.bias is not None else 0.0)
        shifted = shifted - shifted.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def classify(self, claims: list[str], evidence_sets: list[list[Evidence]],
                 batch_size: int = 4) -> np.ndarray:
        """Calibrated label probabilities for many claims at once, shape ``[len(claims), 3]``."""
        return self.calibrate(self.label_logprobs(claims, evidence_sets, batch_size))

    def verify(self, claim: str, explain: bool = True, on_text=None,
               max_new_tokens: int = 160) -> Verdict:
        start = time.perf_counter()
        evidence, timing = retrieve_evidence(self.engine, claim, self.evidence_k, self.candidates)

        classify_start = time.perf_counter()
        probabilities = self.classify([claim], [evidence])[0]
        timing["llama_verdict"] = (time.perf_counter() - classify_start) * 1000.0

        label = LABELS[int(np.argmax(probabilities))]
        verdict = Verdict(
            claim=claim,
            label=label,
            probabilities={name: float(p) for name, p in zip(LABELS, probabilities)},
            evidence=evidence,
            timing_ms=timing,
        )

        if explain:
            self.explain(verdict, on_text=on_text, max_new_tokens=max_new_tokens)

        timing["end_to_end"] = (time.perf_counter() - start) * 1000.0
        return verdict

    def explain(self, verdict: Verdict, on_text=None, max_new_tokens: int = 160) -> Generation:
        """Streams a short, cited rationale for an already-classified verdict."""
        messages = self._messages(verdict.claim, verdict.evidence)
        messages.append({"role": "assistant", "content": verdict.label})
        messages.append({
            "role": "user",
            "content": (
                "In two or three sentences, explain that verdict. Cite the abstracts you "
                "rely on as [1], [2] or [3], and quote the key finding."
            ),
        })
        generation = self.llama.stream(messages, max_new_tokens=max_new_tokens, on_text=on_text)
        verdict.explanation = generation.text
        verdict.timing_ms["llama_ttft"] = generation.ttft_ms
        verdict.timing_ms["llama_explain"] = generation.total_ms
        verdict.timing_ms["llama_tokens_per_s"] = generation.tokens_per_s
        return generation


class GroundedAnswerer:
    def __init__(self, engine, llama: Llama, k: int = 5, candidates: int = 100,
                 max_words: int = 200) -> None:
        self.engine = engine
        self.llama = llama
        self.k = k
        self.candidates = candidates
        self.max_words = max_words

    def answer(self, question: str, on_text=None,
               max_new_tokens: int = 320) -> tuple[Generation, list[Evidence], dict]:
        start = time.perf_counter()
        evidence, timing = retrieve_evidence(self.engine, question, self.k, self.candidates)
        messages = [
            {"role": "system", "content": ANSWER_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Sources:\n\n{format_sources(evidence, self.max_words)}\n\n"
                    f"Question: {question}"
                ),
            },
        ]
        generation = self.llama.stream(messages, max_new_tokens=max_new_tokens, on_text=on_text)
        timing["llama_ttft"] = generation.ttft_ms
        timing["llama_generate"] = generation.total_ms
        timing["llama_tokens_per_s"] = generation.tokens_per_s
        timing["end_to_end"] = (time.perf_counter() - start) * 1000.0
        return generation, evidence, timing
