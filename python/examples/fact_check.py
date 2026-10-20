"""Interactive scientific fact-checker: zero-copy CUDA retrieval + Llama reasoning.

  PYTHONPATH=build/bindings:python python python/examples/fact_check.py \\
      --claim "Vitamin D supplementation reduces the risk of fractures in older adults."

With no ``--claim`` it reads claims from stdin, one per line. ``--ask`` switches to grounded
question answering with citations instead of verdicts.
"""

from __future__ import annotations

import argparse
import sys
import time

from zcgvs_tools.config import build_engine
from zcgvs_tools.factcheck import LABELS, ClaimVerifier, GroundedAnswerer
from zcgvs_tools.llama import DEFAULT_MODEL, Llama

COLORS = {"SUPPORT": "\033[92m", "CONTRADICT": "\033[91m", "NOT ENOUGH INFO": "\033[93m"}
DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"


def bar(probability: float, width: int = 24) -> str:
    filled = round(probability * width)
    return "█" * filled + "░" * (width - filled)


def print_evidence(evidence) -> None:
    print(f"{BOLD}evidence{RESET} {DIM}(CUDA scan → on-device top-K → cross-encoder){RESET}")
    for item in evidence:
        print(f"  [{item.rank}] {DIM}doc {item.doc_id}  ce={item.rerank_score:+.2f}{RESET}  "
              f"{item.text[:110]}…")


def print_quote_checks(verdict) -> None:
    print(f"\n{BOLD}quote check{RESET} {DIM}(is each quoted passage in the cited abstract?){RESET}")
    if not verdict.quote_checks:
        print("  no verbatim quotations to check")
        return
    for check in verdict.quote_checks:
        mark = f"{COLORS['SUPPORT']}✓ verified{RESET}" if check.verified else \
            f"{COLORS['CONTRADICT']}✗ NOT VERIFIED{RESET}"
        where = f"[{check.cited}]" if check.cited else f"in {check.found_in or 'no abstract'}"
        print(f"  {mark}  {where}  “{check.quote[:90]}{'…' if len(check.quote) > 90 else ''}”")


def print_timing(timing: dict) -> None:
    order = [
        ("gpu_search", "CUDA scan + top-K", "ms"),
        ("copy", "device→host (k results)", "ms"),
        ("embed", "query embedding (ONNX)", "ms"),
        ("cross_encoder", "cross-encoder rerank", "ms"),
        ("llama_verdict", "Llama verdict (1 forward)", "ms"),
        ("llama_ttft", "Llama first token", "ms"),
        ("llama_explain", "Llama explanation", "ms"),
        ("llama_generate", "Llama answer", "ms"),
        ("llama_tokens_per_s", "Llama decode rate", "tok/s"),
        ("end_to_end", "end to end", "ms"),
    ]
    print(f"\n{BOLD}timing{RESET}")
    for key, label, unit in order:
        if key in timing:
            print(f"  {label:<28} {timing[key]:>9.2f} {unit}")


def stream(piece: str) -> None:
    sys.stdout.write(piece)
    sys.stdout.flush()


def verify(verifier: ClaimVerifier, claim: str) -> None:
    print(f"\n{BOLD}claim{RESET}  {claim}\n")

    # The verdict is one forward pass, so it prints before the explanation starts streaming.
    start = time.perf_counter()
    verdict = verifier.verify(claim, explain=False)
    print_evidence(verdict.evidence)
    color = COLORS[verdict.label]
    print(f"\n{BOLD}verdict{RESET} {color}{BOLD}{verdict.label}{RESET} "
          f"({verdict.confidence:.1%} confidence)")
    for label in LABELS:
        print(f"  {label:<16} {COLORS[label]}{bar(verdict.probabilities[label])}{RESET} "
              f"{verdict.probabilities[label]:6.1%}")

    print(f"\n{BOLD}why{RESET}  ", end="")
    verifier.explain(verdict, on_text=stream)
    verdict.timing_ms["end_to_end"] = (time.perf_counter() - start) * 1000.0
    print()
    print_quote_checks(verdict)
    print_timing(verdict.timing_ms)


def ask(answerer: GroundedAnswerer, question: str) -> None:
    print(f"\n{BOLD}question{RESET}  {question}\n")
    print(f"{BOLD}answer{RESET}  ", end="")
    _, evidence, timing = answerer.answer(question, on_text=stream)
    print("\n")
    print_evidence(evidence)
    print_timing(timing)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/scifact.json")
    parser.add_argument("--llama", default=DEFAULT_MODEL, help="Llama model id or local dir")
    parser.add_argument("--claim", action="append", help="claim to verify (repeatable)")
    parser.add_argument("--ask", action="append", help="question to answer (repeatable)")
    parser.add_argument("--evidence-k", type=int, default=3)
    args = parser.parse_args()

    engine = build_engine(args.config)
    llama = Llama(args.llama)
    stats = engine.stats()
    print(f"{DIM}index: {stats.corpus_size} abstracts, {stats.storage_dtype}, "
          f"{stats.corpus_bytes / 1e6:.1f} MB VRAM | model: {llama.name} "
          f"({llama.weight_bytes / 1e9:.1f} GB bf16) | {stats.device_name}{RESET}")

    # One throwaway pass pays the ONNX Runtime, CUDA-module and Llama kernel warmup up front,
    # so the timings printed below are steady-state rather than first-call.
    ClaimVerifier(engine, llama).verify("Warmup claim about insulin.", explain=True,
                                        max_new_tokens=8)

    if args.ask:
        answerer = GroundedAnswerer(engine, llama)
        for question in args.ask:
            ask(answerer, question)
        return

    verifier = ClaimVerifier(engine, llama, evidence_k=args.evidence_k)
    claims = args.claim or [line.strip() for line in sys.stdin if line.strip()]
    for claim in claims:
        verify(verifier, claim)


if __name__ == "__main__":
    main()
