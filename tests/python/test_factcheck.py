"""Checks the model-free logic of the Llama layer: calibration, macro-F1, label handling,
reciprocal-rank fusion and SciFact label derivation. No GPU, no model weights, no build.

Runs standalone (``python tests/python/test_factcheck.py``) or under pytest.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from zcgvs_tools.factcheck import (  # noqa: E402
    LABELS,
    ClaimVerifier,
    fit_label_bias,
    macro_f1,
    reciprocal_rank_fusion,
    truncate_words,
)
from zcgvs_tools.scifact_claims import claim_label  # noqa: E402


def test_macro_f1_perfect_and_degenerate():
    gold = np.array([0, 1, 2, 0, 1, 2])
    assert macro_f1(gold, gold) == 1.0
    # Always predicting class 0: F1 for class 0 is 2*2/(6+2) = 0.5, the others 0.
    assert abs(macro_f1(gold, np.zeros_like(gold)) - 0.5 / 3) < 1e-12


def test_fit_label_bias_recovers_a_skewed_prior():
    rng = np.random.default_rng(0)
    gold = rng.integers(0, 3, size=600)
    logits = rng.normal(scale=0.5, size=(600, 3))
    logits[np.arange(600), gold] += 1.5          # informative signal
    logits[:, 0] += 3.0                          # a strong SUPPORT prior that hides it
    logprobs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))

    raw = macro_f1(gold, logprobs.argmax(axis=1))
    bias = fit_label_bias(logprobs, gold)
    calibrated = macro_f1(gold, (logprobs + bias).argmax(axis=1))

    assert bias[0] == 0.0, "SUPPORT is the reference label"
    assert bias[1] > 2.0 and bias[2] > 2.0, f"offsets should undo the +3 prior, got {bias}"
    assert calibrated > raw + 0.3, (raw, calibrated)


def test_calibrate_renormalizes():
    verifier = ClaimVerifier.__new__(ClaimVerifier)
    verifier.bias = np.array([0.0, -1.0, 2.0])
    probabilities = verifier.calibrate(np.log(np.array([[0.5, 0.3, 0.2]])))
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    expected = np.array([0.5, 0.3 * np.exp(-1.0), 0.2 * np.exp(2.0)])
    assert np.allclose(probabilities[0], expected / expected.sum())


def test_labels_have_distinct_order_and_names():
    assert LABELS == ("SUPPORT", "CONTRADICT", "NOT ENOUGH INFO")


def test_truncate_words():
    assert truncate_words("a b c", 5) == "a b c"
    assert truncate_words("a b c d e f", 3) == "a b c …"


def test_scifact_claim_label():
    assert claim_label({}) == "NOT ENOUGH INFO"
    assert claim_label({"1": [{"sentences": [0], "label": "SUPPORT"}]}) == "SUPPORT"
    assert claim_label({"1": [{"sentences": [0], "label": "CONTRADICT"}],
                        "2": [{"sentences": [3], "label": "CONTRADICT"}]}) == "CONTRADICT"
    assert claim_label({"1": [{"sentences": [0], "label": "SUPPORT"}],
                        "2": [{"sentences": [3], "label": "CONTRADICT"}]}) == "MIXED"


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "c"]])
    assert set(fused[:2]) == {"a", "b"} and fused[2] == "c"
    assert reciprocal_rank_fusion([["x", "y"], ["y", "z"]])[0] == "y"


def main() -> int:
    failures = 0
    for name, test in sorted(globals().items()):
        if not name.startswith("test_") or not callable(test):
            continue
        try:
            test()
            print(f"  PASS  {name}")
        except AssertionError as error:
            failures += 1
            print(f"  FAIL  {name}: {error}")
    print(f"\nfactcheck: {'PASS' if failures == 0 else f'FAIL ({failures})'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
