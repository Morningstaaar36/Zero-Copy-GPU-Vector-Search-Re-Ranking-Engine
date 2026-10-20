"""Llama runtime shared by the reasoning stages that sit on top of the retrieval engine.

Loads a Meta Llama 3.x Instruct checkpoint in bfloat16 onto the same GPU that holds the
vector index, and exposes the two ways the rest of the package uses it:

- ``generate`` / ``stream`` — chat-templated free-text generation (grounded answers,
  rationales), with time-to-first-token and decode throughput measured per call.
- ``choice_logprobs`` — closed-set classification in a **single forward pass**: the logits at
  the first generated position are restricted to the first token of each allowed answer and
  renormalized. No sampling, no output parsing, fully deterministic, and batched — which is
  what makes Llama usable as a reranker and as a verdict classifier at evaluation scale.

The official ``meta-llama/*`` repositories are gated on the Hugging Face Hub. If the current
token has not been granted access, loading falls back to a byte-identical community mirror of
the same release (redistribution is permitted by the Llama 3.2 Community License).
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

MIRRORS = {
    "meta-llama/Llama-3.2-1B-Instruct": "unsloth/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct": "unsloth/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct": "unsloth/Llama-3.1-8B-Instruct",
}


@dataclass
class Generation:
    text: str
    prompt_tokens: int
    new_tokens: int
    ttft_ms: float
    total_ms: float
    tokens_per_s: float = field(init=False)

    def __post_init__(self) -> None:
        decode_ms = max(self.total_ms - self.ttft_ms, 1e-6)
        self.tokens_per_s = max(self.new_tokens - 1, 0) / (decode_ms / 1000.0)


def _is_gated_error(error: Exception) -> bool:
    text = f"{type(error).__name__} {error}".lower()
    return "gated" in text or "401" in text or "403" in text or "restricted" in text


def resolve_model_id(model_id: str) -> str:
    """Returns ``model_id`` if its config is reachable, else the mirror of the same weights."""
    from huggingface_hub import hf_hub_download

    if Path(model_id).is_dir():
        return model_id
    try:
        hf_hub_download(model_id, "config.json")
        return model_id
    except Exception as error:  # noqa: BLE001 - hub raises several unrelated types
        mirror = MIRRORS.get(model_id)
        if mirror is None or not _is_gated_error(error):
            raise
        print(
            f"[llama] no access to gated repo {model_id}; using the identical weights "
            f"from {mirror} (request access on the Hub to use the official repo)",
            file=sys.stderr,
        )
        return mirror


class Llama:
    def __init__(self, model_id: str = DEFAULT_MODEL, device: str = "cuda:0",
                 dtype: torch.dtype = torch.bfloat16, max_context: int = 4096) -> None:
        self.requested_id = model_id
        self.model_id = resolve_model_id(model_id)
        self.device = torch.device(device)
        self.max_context = max_context

        start = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left padding keeps the last real token of every row at position -1, which is where
        # both batched generation and choice scoring read their logits.
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, dtype=dtype, attn_implementation="sdpa"
        ).to(self.device)
        self.model.eval()
        self.load_s = time.perf_counter() - start

        self.eot_ids = [
            token_id
            for token_id in (
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            )
            if isinstance(token_id, int) and token_id >= 0
        ]

    @property
    def name(self) -> str:
        return self.requested_id.split("/")[-1]

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @property
    def weight_bytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.model.parameters())

    def prompt(self, messages: list[dict], assistant_prefix: str = "") -> str:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return text + assistant_prefix

    def _encode(self, prompts: list[str]) -> dict:
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_context,
            add_special_tokens=False,  # the chat template already emits <|begin_of_text|>
        )
        return {key: value.to(self.device) for key, value in encoded.items()}

    def first_token_ids(self, choices: list[str]) -> list[int]:
        ids = [self.tokenizer.encode(choice, add_special_tokens=False)[0] for choice in choices]
        if len(set(ids)) != len(ids):
            raise ValueError(f"choices {choices} share a first token; rephrase them")
        return ids

    @torch.inference_mode()
    def choice_logprobs(self, prompts: list[str], choices: list[str],
                        batch_size: int = 8) -> np.ndarray:
        """Log-probabilities over ``choices`` for each prompt, shape ``[len(prompts), len(choices)]``.

        Only the first token of each choice is scored, so choices must start differently.
        """
        choice_ids = torch.tensor(self.first_token_ids(choices), device=self.device)
        # Only the answer rows of the LM head are needed, and they are applied in float32:
        # bf16 logits near 20 are quantized to steps of 0.125, which ties most candidates.
        head = self.model.lm_head.weight.index_select(0, choice_ids).float()
        rows: list[np.ndarray] = []
        for start in range(0, len(prompts), batch_size):
            encoded = self._encode(prompts[start : start + batch_size])
            hidden = self.model.model(**encoded).last_hidden_state[:, -1, :].float()
            rows.append(torch.log_softmax(hidden @ head.T, dim=-1).cpu().numpy())
        return np.concatenate(rows, axis=0) if rows else np.zeros((0, len(choices)))

    @torch.inference_mode()
    def continuation_logprob(self, prompts: list[str], continuation: str,
                             batch_size: int = 8) -> np.ndarray:
        """Mean per-token log P(continuation | prompt) for each prompt, in float32.

        Teacher-forced in one forward pass per batch: the shared continuation is appended to
        every prompt and only its positions are pushed through the LM head.
        """
        target = self.tokenizer.encode(continuation, add_special_tokens=False)
        target_ids = torch.tensor(target, device=self.device)
        n = len(target)
        scores: list[np.ndarray] = []
        for start in range(0, len(prompts), batch_size):
            batch = [
                self.tokenizer.encode(prompt, add_special_tokens=False)[: self.max_context - n]
                + target
                for prompt in prompts[start : start + batch_size]
            ]
            encoded = self.tokenizer.pad({"input_ids": batch}, return_tensors="pt")
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            hidden = self.model.model(**encoded).last_hidden_state[:, -n - 1 : -1, :]
            logits = self.model.lm_head(hidden).float()
            logprobs = torch.log_softmax(logits, dim=-1)
            picked = logprobs.gather(-1, target_ids.view(1, n, 1).expand(len(batch), n, 1))
            scores.append(picked.squeeze(-1).mean(dim=-1).cpu().numpy())
        return np.concatenate(scores) if scores else np.zeros(0)

    def _generation_kwargs(self, max_new_tokens: int, temperature: float) -> dict:
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "eos_token_id": self.eot_ids,
            "pad_token_id": self.tokenizer.pad_token_id,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            kwargs.update(temperature=temperature, top_p=0.9)
        else:
            kwargs.update(temperature=None, top_p=None)
        return kwargs

    @torch.inference_mode()
    def stream(self, messages: list[dict], max_new_tokens: int = 384, temperature: float = 0.0,
               on_text=None) -> Generation:
        """Generates a reply, calling ``on_text(piece)`` as tokens arrive."""
        prompt = self.prompt(messages)
        encoded = self._encode([prompt])
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        kwargs = dict(encoded, streamer=streamer,
                      **self._generation_kwargs(max_new_tokens, temperature))

        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        worker = threading.Thread(target=self.model.generate, kwargs=kwargs, daemon=True)
        worker.start()

        pieces: list[str] = []
        ttft_ms = None
        for piece in streamer:
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - start) * 1000.0
            pieces.append(piece)
            if on_text is not None:
                on_text(piece)
        worker.join()
        total_ms = (time.perf_counter() - start) * 1000.0

        text = "".join(pieces).strip()
        new_tokens = len(self.tokenizer.encode(text, add_special_tokens=False)) + 1
        return Generation(
            text=text,
            prompt_tokens=int(encoded["input_ids"].shape[1]),
            new_tokens=new_tokens,
            ttft_ms=ttft_ms if ttft_ms is not None else total_ms,
            total_ms=total_ms,
        )

    def generate(self, messages: list[dict], max_new_tokens: int = 384,
                 temperature: float = 0.0) -> Generation:
        return self.stream(messages, max_new_tokens=max_new_tokens, temperature=temperature)
