#!/usr/bin/env python3
"""Regenerates the README figures in docs/assets/ from measured results.

  python scripts/make_figures.py

Standard library only. Llama numbers (macro-F1, latency, quote audit) are read from
docs/results/llama_eval_*.json, so re-running `eval_llama.py` and then this script keeps the
figures honest. The kernel numbers come from `scripts/bench_all.sh`; they are constants below
(KERNEL_*) because that benchmark prints a table rather than a file. Update them after a
re-benchmark, and the same numbers in docs/BENCHMARKS.md.

Colour use follows one rule across the figures: blue = code written from scratch in CUDA (and
"after" in before/after charts), orange = Meta Llama 3.2 (and "before"), gray = off-the-shelf
parts. The blue/orange pair was checked for colour-blind separation and 3:1 contrast on both the
light and the dark surface.
"""
from __future__ import annotations

import json
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets"
RESULTS = ROOT / "docs" / "results"

# -- kernel benchmark: `build/apps/zcgvs_bench_kernels --docs 1000000 --dim 384 --k 100`
# (name, v1 % of peak, v1 ms, v2 % of peak, v2 ms); v1 = one block per document.
KERNEL_ROWS = [("float32", 47, 7.279, 94, 3.650), ("float16", 22, 7.769, 92, 1.872),
               ("int8", 11, 7.938, 89, 0.959), ("binary", 2, 5.425, 87, 0.124)]
KERNEL_PEAK_RANGE = "87–94%"
CUBLAS_MS, CUSTOM_FP32_MS = 3.625, 3.650
TESTS_SUITES = 9
SEARCH_DOCS = 5183


def load(model: str) -> dict:
    return json.loads((RESULTS / f"llama_eval_Llama-3.2-{model}-Instruct.json").read_text())


r1, r3 = load("1B"), load("3B")
lat = r3["latency_ms"]
audit = r3["grounding"]
mf1 = lambda r, k: r["verify"][k]["macro_f1"]
BASELINE_F1 = r3["verify"]["majority_baseline"]["macro_f1"]

FONT = "-apple-system,'Segoe UI',Helvetica,Arial,sans-serif"
STYLE = """<style>
.bg{fill:#fcfcfb}.panel{fill:#f4f3f0;stroke:#e4e3df}.line{stroke:#c9c8c2;fill:none}
.t1{fill:#0b0b0b}.t2{fill:#52514e}.t3{fill:#6b6a65}
.blue{fill:#2a78d6}.orange{fill:#eb6834}.gray{fill:#8a8982}.good{fill:#0ca30c}
.arrow{stroke:#52514e;fill:none}
@media (prefers-color-scheme:dark){
.bg{fill:#1a1a19}.panel{fill:#232321;stroke:#34342f}.line{stroke:#4a4a45}
.t1{fill:#f0efec}.t2{fill:#c3c2b7}.t3{fill:#a09f95}
.blue{fill:#3987e5}.orange{fill:#d95926}.gray{fill:#6b6a65}.good{fill:#0ca30c}
.arrow{stroke:#c3c2b7}}
</style>"""


def svg(width, height, title, desc, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}" role="img" aria-labelledby="t d" '
            f'font-family="{FONT}">\n<title id="t">{escape(title)}</title>\n'
            f'<desc id="d">{escape(desc)}</desc>\n{STYLE}\n'
            f'<rect class="bg" width="{width}" height="{height}" rx="10"/>\n{body}\n</svg>\n')


def text(x, y, s, cls="t1", size=13, weight=400, anchor="start", extra=""):
    return (f'<text class="{cls}" x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
            f'text-anchor="{anchor}" {extra}>{escape(s)}</text>')


def hero():
    W, H = 900, 412
    ce, verdict, toks = round(lat["cross_encoder"]), round(lat["llama_verdict"]), round(lat["llama_tokens_per_s"])
    b = [text(30, 46, "Zero-Copy GPU Vector Search & Re-Ranking Engine", size=25, weight=700),
         text(30, 72, "Hand-written CUDA search at memory speed, plus Meta Llama 3.2 that reads the "
                      "evidence and shows its sources.", "t2", 14)]
    cards = [
        ("1 · Claim", "gray", ["“Aspirin reduces", "colorectal cancer risk.”"], "text in", None),
        ("2 · GPU search", "blue", ["Custom CUDA scan and", "top-K, all in VRAM"], f"{lat['gpu_search']:.2f} ms",
         f"{SEARCH_DOCS:,} abstracts"),
        ("3 · Re-rank", "gray", ["A cross-encoder reads", "100 results, keeps 3"], f"{ce} ms", "ONNX model"),
        ("4 · Verdict", "orange", ["Llama 3.2 reads the 3", "abstracts in ONE pass"], f"{verdict} ms", "calibrated"),
        ("5 · Explain", "orange", ["Cited, quote-checked,", "streamed live"], f"{toks} tok/s", "3B, bf16"),
    ]
    cw, gap, x0, y0, ch = 148, 28, 24, 100, 156
    for i, (head, color, lines, metric, sub) in enumerate(cards):
        x = x0 + i * (cw + gap)
        b.append(f'<rect class="panel" x="{x}" y="{y0}" width="{cw}" height="{ch}" rx="9"/>')
        b.append(f'<path class="{color}" d="M{x},{y0 + 9} a9,9 0 0 1 9,-9 h{cw - 18} a9,9 0 0 1 9,9 v1 h-{cw} z"/>')
        b.append(f'<rect class="{color}" x="{x}" y="{y0 + 8}" width="{cw}" height="3"/>')
        b.append(text(x + 14, y0 + 34, head, size=13.5, weight=700))
        for j, line in enumerate(lines):
            b.append(text(x + 14, y0 + 58 + j * 16, line, "t2", 12))
        b.append(text(x + 14, y0 + 120, metric, size=23, weight=700))
        if sub:
            b.append(text(x + 14, y0 + 140, sub, "t3", 11.5))
        if i < 4:
            ax = x + cw + 5
            b.append(f'<path class="arrow" d="M{ax},{y0 + ch / 2 - 7} l9,7 l-9,7" stroke-width="2" '
                     f'stroke-linecap="round" stroke-linejoin="round"/>')
    ly = y0 + ch + 28
    b.append(f'<rect class="blue" x="24" y="{ly - 10}" width="12" height="12" rx="2"/>')
    b.append(text(42, ly, "written from scratch in CUDA", "t2", 12.5))
    b.append(f'<rect class="orange" x="246" y="{ly - 10}" width="12" height="12" rx="2"/>')
    b.append(text(264, ly, "Meta Llama 3.2", "t2", 12.5))
    b.append(f'<rect class="gray" x="388" y="{ly - 10}" width="12" height="12" rx="2"/>')
    b.append(text(406, ly, "off-the-shelf component", "t2", 12.5))
    b.append(text(876, ly, "median timings, RTX 5060 Ti", "t3", 11.5, anchor="end"))

    chips = [(KERNEL_PEAK_RANGE, "of GPU memory bandwidth,", "every storage precision"),
             ("= cuBLAS", f"fp32 scan: {CUSTOM_FP32_MS:.3f} ms vs {CUBLAS_MS:.3f} ms", "on 1M × 384 vectors"),
             (f"{TESTS_SUITES} suites", "of CUDA tests, each checked", "against a CPU reference"),
             (f"{mf1(r3, 'open_calibrated'):.3f}", "macro-F1 on 300 held-out",
              f"claims (baseline: {BASELINE_F1:.3f})")]
    cy, chw, cgap = 322, 207, 12
    for i, (big, l1, l2) in enumerate(chips):
        x = 24 + i * (chw + cgap)
        b.append(f'<rect class="panel" x="{x}" y="{cy}" width="{chw}" height="72" rx="9"/>')
        b.append(text(x + 14, cy + 30, big, size=22, weight=700))
        b.append(text(x + 14, cy + 48, l1, "t2", 11.5))
        b.append(text(x + 14, cy + 62, l2, "t3", 11))
    return svg(W, H, "Zero-Copy GPU Vector Search and Re-Ranking Engine",
               f"Pipeline from a text claim through a CUDA GPU search ({lat['gpu_search']:.2f} ms), a "
               f"cross-encoder re-rank ({ce} ms), a Llama 3.2 verdict ({verdict} ms) and a streamed, "
               "quote-checked explanation, with four headline numbers.", "\n".join(b))


def zero_copy():
    W, H = 900, 372
    b = [text(30, 40, "What “zero-copy” means here: almost nothing crosses to the CPU", size=20, weight=700),
         text(30, 64, "Per query, searching 1,000,000 documents and returning the 10 best", "t2", 13)]

    def panel(y, title, color, gpu_lines, arrow_label, big, sub):
        b.append(f'<rect class="panel" x="24" y="{y}" width="852" height="118" rx="10"/>')
        b.append(text(44, y + 28, title, size=14, weight=700))
        b.append(f'<rect class="{color}" x="44" y="{y + 44}" width="250" height="58" rx="8"/>')
        for i, line in enumerate(gpu_lines):
            b.append(text(58, y + 68 + i * 17, line, "t1", 12.5, 600, extra='style="fill:#fff"'))
        b.append(f'<line class="line" x1="294" y1="{y + 73}" x2="618" y2="{y + 73}" stroke-width="3"/>')
        b.append(f'<path class="arrow" d="M610,{y + 66} l9,7 l-9,7" stroke-width="2.5" '
                 f'stroke-linecap="round" stroke-linejoin="round"/>')
        b.append(text(456, y + 62, arrow_label, "t2", 12, anchor="middle"))
        b.append(text(456, y + 94, big, "t1", 22, 700, anchor="middle"))
        b.append(f'<rect class="panel" x="632" y="{y + 44}" width="224" height="58" rx="8" style="fill:none"/>')
        b.append(text(744, y + 68, "CPU", "t1", 13, 700, anchor="middle"))
        b.append(text(744, y + 86, sub, "t2", 11.5, anchor="middle"))

    panel(84, "A naive stack", "orange", ["GPU computes a score for", "all 1,000,000 documents"],
          "copies every score to the CPU", "4,000,000 bytes", "sorts a million numbers")
    panel(214, "This engine", "blue", ["GPU scores, sorts, re-scores", "and picks the best 10"],
          "copies only the final 10 ids + scores", "80 bytes", "gets the answer, nothing else")
    b.append(text(450, 352, f"That is 50,000× less data per query. Measured here: the single device-to-host "
                            f"copy takes {lat['copy']:.2f} ms.", "t2", 12.5, anchor="middle"))
    return svg(W, H, "What zero-copy means in this engine",
               "A naive GPU search copies a score for every one of a million documents, 4,000,000 bytes, "
               "to the CPU to sort. This engine selects and re-scores on the GPU and copies only the "
               "final 10 ids and scores, 80 bytes.", "\n".join(b))


def llama_results():
    W, H = 760, 330
    left, right, top, xmax = 150, 700, 112, 0.7
    sx = lambda v: left + (right - left) * v / xmax
    rows = [("Llama 3.2 1B", mf1(r1, "open_raw"), mf1(r1, "open_calibrated")),
            ("Llama 3.2 3B", mf1(r3, "open_raw"), mf1(r3, "open_calibrated"))]
    oracle = mf1(r3, "oracle_calibrated")
    gains = [cal - raw for _, raw, cal in rows]
    b = [text(24, 36, f"Calibrating Llama's label priors lifts macro-F1 by {min(gains):.2f}–{max(gains):.2f}",
              size=18, weight=700),
         text(24, 58, "Claim verification (support / contradict / not enough info) on 300 held-out "
                      "SciFact claims", "t2", 12.5),
         '<rect class="orange" x="24" y="72" width="12" height="12" rx="2"/>', text(42, 83, "raw model", "t2", 12),
         '<rect class="blue" x="132" y="72" width="12" height="12" rx="2"/>',
         text(150, 83, "after calibration (offsets fitted on separate training claims)", "t2", 12)]
    for v in (0, 0.2, 0.4, 0.6):
        b.append(f'<line class="line" x1="{sx(v)}" y1="{top - 2}" x2="{sx(v)}" y2="288" stroke-width="1"/>')
        b.append(text(sx(v), 306, f"{v:.1f}", "t3", 11, anchor="middle"))
    y = top
    for name, raw, cal in rows:
        b.append(text(left - 14, y + 34, name, "t1", 13, 600, anchor="end"))
        for k, (val, cls) in enumerate(((raw, "orange"), (cal, "blue"))):
            by, w = y + k * 30, sx(val) - left
            b.append(f'<path class="{cls}" d="M{left},{by} h{w - 5:.1f} a5,5 0 0 1 5,5 v16 a5,5 0 0 1 -5,5 h-{w - 5:.1f} z"/>')
            b.append(text(sx(val) + 8, by + 19, f"{val:.3f}", "t1", 12.5, 700))
        y += 82
    for v in (BASELINE_F1, oracle):
        b.append(f'<line class="arrow" x1="{sx(v)}" y1="{top - 2}" x2="{sx(v)}" y2="288" stroke-width="1.5" '
                 f'stroke-dasharray="5 4"/>')
    b.append(text(sx(BASELINE_F1) + 6, top - 12, f"always SUPPORT: {BASELINE_F1:.3f}", "t2", 11.5))
    b.append(text(sx(oracle) - 6, top - 12, f"3B + gold abstracts: {oracle:.3f}", "t2", 11.5, anchor="end"))
    return svg(W, H, "Llama 3.2 claim verification, raw versus calibrated",
               "Macro-F1 on 300 held-out SciFact claims. " + " ".join(
                   f"{n}: {raw:.3f} raw, {cal:.3f} calibrated." for n, raw, cal in rows) +
               f" Reference lines: always answering SUPPORT scores {BASELINE_F1:.3f}; the 3B model given the "
               f"gold evidence abstracts scores {oracle:.3f}.", "\n".join(b))


def rigor():
    W, H = 900, 372
    q = [f"Every quotation must appear verbatim", f"in the abstract it cites: {audit['quotes_verified']} of "
         f"{audit['quotes']}", f"passed in a {audit['claims']}-claim audit"]
    cards = [
        ("Kernels tested against the CPU", ["9 CUDA test suites; one sweeps 44", "dtype × dim × metric × mask cases"]),
        ("No peeking at the test set", ["Calibration fitted on 809 train claims,", "scored on 300 held-out claims"]),
        ("Error bars, not vibes", ["Paired bootstrap, 10,000 resamples,", "on every reranking comparison"]),
        ("Negative results published", ["Hybrid BM25 and Llama-only reranking", "lost to the baseline — both reported"]),
        ("Quotes are machine-checked", q),
        ("One command reproduces it", ["scripts/verify.sh, bench_all.sh,", "eval_llama.py · median of 50 warm runs"]),
    ]
    b = [text(30, 40, "Why you can trust the numbers", size=22, weight=700),
         text(30, 64, "Every claim in this README is backed by a test, a held-out split, or a script you can run.",
              "t2", 13)]
    cw, ch, gx, gy = 276, 124, 12, 12
    for i, (head, lines) in enumerate(cards):
        col, row = i % 3, i // 3
        x, y = 24 + col * (cw + gx), 86 + row * (ch + gy)
        b.append(f'<rect class="panel" x="{x}" y="{y}" width="{cw}" height="{ch}" rx="10"/>')
        b.append(f'<circle class="good" cx="{x + 28}" cy="{y + 30}" r="12"/>')
        b.append(f'<path d="M{x + 22},{y + 30} l5,5 l8,-10" stroke="#fff" stroke-width="2.6" fill="none" '
                 f'stroke-linecap="round" stroke-linejoin="round"/>')
        b.append(text(x + 50, y + 35, head, "t1", 14, 700))
        for j, line in enumerate(lines):
            b.append(text(x + 18, y + 70 + j * 18, line, "t2", 12.3))
    return svg(W, H, "Why you can trust the numbers",
               "Six rigor practices: CPU-reference tests for every kernel, a held-out test set, bootstrap "
               "confidence intervals, published negative results, machine-checked quotes and one-command "
               "reproduction.", "\n".join(b))


def roofline():
    W, H = 760, 330
    left, right = 96, 520
    x = lambda pct: left + (right - left) * pct / 100
    b = [text(24, 34, "Rewriting the scan kernel: every precision now runs at memory speed", size=17, weight=700),
         text(24, 56, "Share of DRAM peak achieved scanning 1M × 384 embeddings · RTX 5060 Ti (448 GB/s) · "
                      "median of 50 runs", "t2", 12.5),
         '<rect class="orange" x="24" y="72" width="12" height="12" rx="2"/>',
         text(42, 82, "v1: one block per document", "t2", 12.5),
         '<rect class="blue" x="230" y="72" width="12" height="12" rx="2"/>',
         text(248, 82, "v2: persistent, warp per document, 128-bit loads", "t2", 12.5)]
    top, bottom = 104, 296
    for pct in (0, 25, 50, 75):
        b.append(f'<line class="line" x1="{x(pct)}" y1="{top}" x2="{x(pct)}" y2="{bottom}" stroke-width="1"/>')
        b.append(text(x(pct), bottom + 16, f"{pct}%", "t3", 11, anchor="middle"))
    b.append(f'<line class="arrow" x1="{x(100)}" y1="{top - 4}" x2="{x(100)}" y2="{bottom}" stroke-width="1" '
             f'stroke-dasharray="3 3"/>')
    b.append(text(x(100), bottom + 16, "100% = peak", "t2", 11, anchor="middle"))

    def bar(y, pct, cls):
        w = max(x(pct) - left, 4)
        r = min(4, w / 2)
        return (f'<path class="{cls}" d="M{left},{y} h{w - r:.1f} a{r},{r} 0 0 1 {r},{r} v{16 - 2 * r} '
                f'a{r},{r} 0 0 1 -{r},{r} h-{w - r:.1f} z"/>')

    y = top + 6
    for name, p1, ms1, p2, ms2 in KERNEL_ROWS:
        b.append(text(left - 12, y + 25, name, "t1", 13.5, 600, anchor="end"))
        b.append(bar(y, p1, "orange"))
        b.append(text(x(p1) + 8, y + 12.5, f"{p1}% · {ms1:.2f} ms", "t2", 11.5))
        b.append(bar(y + 18, p2, "blue"))
        speed = ms1 / ms2
        fmt = f"{speed:.0f}×" if speed >= 9.5 else f"{speed:.1f}×"
        b.append(text(right + 12, y + 30.5, f"{p2}% · {ms2:.2f} ms · {fmt} faster", "t1", 11.5, 600))
        y += 46
    return svg(W, H, "Scan kernel rewrite: share of DRAM peak bandwidth",
               "Before: the block-per-document kernel reached " +
               ", ".join(str(r[1]) for r in KERNEL_ROWS[:3]) + f" and {KERNEL_ROWS[3][1]} percent of peak for "
               "float32, float16, int8 and binary. After: the persistent warp-per-document kernel reaches " +
               ", ".join(str(r[3]) for r in KERNEL_ROWS[:3]) + f" and {KERNEL_ROWS[3][3]} percent.",
               "\n".join(b))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("hero", hero), ("zero_copy", zero_copy), ("llama_results", llama_results),
                     ("rigor", rigor), ("kernel_roofline", roofline)):
        (OUT / f"{name}.svg").write_text(fn())
        print("wrote", OUT / f"{name}.svg")
