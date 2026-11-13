"""Llama on ExecuTorch with access to the scores of every next token (constrained decoding).

The official Python TextLLMRunner (executorch 1.6) returns decoded text only: no logits, no sampler hook.
The exported .pte itself, called through executorch.runtime, takes (tokens [1, n], input_pos [1]) and
returns the next-token logits [1, vocab] with its KV cache kept inside the program. That is enough to
do grammar-constrained decoding ourselves:

    feed(text)            append text to the context (prefill, many tokens per call)
    choose(options)       the model picks ONE of the given strings: at every token, only tokens that
                          continue some option are allowed (a prefix tree), and the best allowed token
                          wins. Returns the option, and how decisively it won.

So a field of a customer rule can only ever be a listed defect, a listed unit, or a number that is
actually written in the clause: the model still does the reading, but it cannot answer off the list.
Everything runs in the same ExecuTorch runtime, on the same Llama .pte, on the Arm CPU.
"""
from __future__ import annotations

import base64
import time
from pathlib import Path

import numpy as np

SPECIAL = {"<|begin_of_text|>": 128000, "<|end_of_text|>": 128001, "<|start_header_id|>": 128006,
           "<|end_header_id|>": 128007, "<|eot_id|>": 128009}
PAT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|"
       r"\s*[\r\n]+|\s+(?!\S)|\s+")


def load_tokenizer(path: str | Path):
    """Llama 3 BPE (tiktoken format, the tokenizer.model Meta ships) with the chat special tokens."""
    import tiktoken
    ranks = {base64.b64decode(a): int(b) for a, b in
             (ln.split() for ln in Path(path).read_bytes().splitlines() if ln)}
    return tiktoken.Encoding("llama3", pat_str=PAT, mergeable_ranks=ranks, special_tokens=SPECIAL)


class Trie:
    def __init__(self, seqs: list[list[int]]):
        self.root: dict = {}
        for i, s in enumerate(seqs):
            node = self.root
            for t in s:
                node = node.setdefault(t, {})
            node[None] = i                      # an option ends here


def constrained_pick(step, seqs: list[list[int]]) -> tuple[int, float, list[int]]:
    """Greedy grammar-constrained decoding over a closed set of token sequences.
    step(token | None) -> logits after appending that token (None = logits at the current position).
    Returns (index of the chosen option, margin at the first decision in nats, tokens consumed)."""
    if not seqs or any(len(s) == 0 for s in seqs):
        raise ValueError("options must be non-empty token sequences")
    node = Trie(seqs).root
    logits = step(None)
    used: list[int] = []
    margin = None
    while True:
        allowed = [t for t in node if t is not None]
        if not allowed:
            return node[None], float(margin if margin is not None else np.inf), used
        lp = logits[allowed] - np.logaddexp.reduce(logits[allowed])
        if None in node:
            # one option is a prefix of another ("spur" / "spurious copper"): it ends here unless the
            # model clearly wants to go on. Ending is scored as "next token is not a continuation".
            cont = np.logaddexp.reduce(logits[allowed]) - np.logaddexp.reduce(logits)
            if cont < np.log(0.5):
                return node[None], float(margin if margin is not None else -cont), used
        order = np.argsort(-lp)
        if margin is None:
            margin = float(lp[order[0]] - lp[order[1]]) if len(allowed) > 1 else np.inf
        tok = allowed[int(order[0])]
        used.append(tok)
        node = node[tok]
        if list(node.keys()) == [None]:
            return node[None], float(margin), used
        logits = step(tok)


class LogitLlama:
    """Llama .pte through executorch.runtime, with a KV cache position we control."""
    name = "llama-executorch-logits"

    def __init__(self, pte_path: str | Path, tokenizer_path: str | Path):
        import executorch.extension.llm.custom_ops.custom_ops  # noqa: F401  (KV cache, SDPA kernels)
        import executorch.kernels.quantized  # noqa: F401
        import torch
        from executorch.runtime import Runtime
        self._torch = torch
        self.model = Path(pte_path).stem
        t0 = time.perf_counter()
        prog = Runtime.get().load_program(str(pte_path))
        self.method = prog.load_method("forward")
        self.load_ms = (time.perf_counter() - t0) * 1e3
        self.tok = load_tokenizer(tokenizer_path)
        self.max_ctx = 2048
        self.pos = 0
        self._last: np.ndarray | None = None
        self.n_forward = 0

    def reset(self) -> None:
        self.pos, self._last = 0, None          # stale cache entries beyond pos are never attended to

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, allowed_special="all")

    def _forward(self, tokens: list[int]) -> np.ndarray:
        if self.pos + len(tokens) >= self.max_ctx:
            raise ValueError("context full")
        t = self._torch
        out = self.method.execute([t.tensor([tokens], dtype=t.long), t.tensor([self.pos], dtype=t.long)])
        self.pos += len(tokens)
        self.n_forward += 1
        self._last = np.asarray(out[0].numpy(), dtype=np.float32).reshape(-1).copy()
        return self._last

    def feed(self, text: str) -> None:
        toks = self.encode(text)
        if toks:
            try:
                self._forward(toks)                              # many tokens per call when the export allows it
            except Exception:                                    # noqa: BLE001  (static-shape export: one by one)
                for tk in toks:
                    self._forward([tk])

    def choose(self, options: list[str]) -> tuple[str, float]:
        """The model continues the context with exactly one of `options`. Returns (option, margin)."""
        if self._last is None:
            raise RuntimeError("feed() a prompt first")
        seqs = [self.encode(o) for o in options]

        def step(tok):
            return self._last if tok is None else self._forward([tok])
        i, margin, _ = constrained_pick(step, seqs)
        return options[i], margin
