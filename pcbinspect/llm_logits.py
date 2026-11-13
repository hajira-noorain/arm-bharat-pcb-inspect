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


def _log_softmax(x: np.ndarray) -> np.ndarray:
    return x - np.logaddexp.reduce(x)


def score_sequences(step, seqs: list[list[int]], end: list[int] | None = None) -> list[tuple[float, int]]:
    """Exact log-probability of every option's full token sequence (not just its first token).

    step(pos_offset, token | None) -> logits after `token` was written at that offset from the start
    (None = the logits at the start). Options share work through a prefix tree; a branch is scored by
    rewinding to the branching position, which a KV-cache model allows because a position only ever
    attends to earlier ones. `end`: token ids of a terminator (e.g. a newline); if given, the probability
    that the answer stops right after the option is included, so "spur" and "spurious copper" compete
    fairly. Returns [(log_prob, n_tokens)] in the order of `seqs`."""
    if not seqs or any(len(q) == 0 for q in seqs):
        raise ValueError("options must be non-empty token sequences")
    root = Trie(seqs).root
    out: list = [None] * len(seqs)
    end_tok = end[0] if end else None

    def rec(node, logits, off, lp, n):
        logp = _log_softmax(logits)
        for tok, child in node.items():
            if tok is None:
                out[child] = (lp + (float(logp[end_tok]) if end_tok is not None else 0.0), n)
                continue
            l2 = lp + float(logp[tok])
            if end_tok is None and set(child) == {None}:
                out[child[None]] = (l2, n + 1)          # nothing left to decide below this token
                continue
            rec(child, step(off, tok), off + 1, l2, n + 1)

    rec(root, step(0, None), 0, 0.0, 0)
    return out


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

    def score(self, options: list[str], end: str | None = "\n") -> list[tuple[float, int]]:
        """Log-probability of each option as the continuation of the context (see score_sequences).
        The context is left exactly as it was: feed() the chosen option afterwards."""
        if self._last is None:
            raise RuntimeError("feed() a prompt first")
        base_pos, base = self.pos, self._last.copy()

        def step(off, tok):
            if tok is None:
                return base
            self.pos = base_pos + off                   # rewind to the branching position
            return self._forward([tok]).copy()
        try:
            return score_sequences(step, [self.encode(o) for o in options], self.encode(end) if end else None)
        finally:
            self.pos, self._last = base_pos, base

    def write(self, max_new_tokens: int, stop: tuple[str, ...] = ("\n",), allow=None, top_k: int = 40) -> str:
        """Greedy free text from the current context, until a stop string, an end token or the limit.
        allow(text_so_far, piece) -> bool vetoes a token (e.g. a digit that would form a number that is
        not in the findings); the best allowed token among the top_k is taken instead."""
        if self._last is None:
            raise RuntimeError("feed() a prompt first")
        text = ""
        for _ in range(max_new_tokens):
            pick = None
            for tok in np.argsort(-self._last)[:top_k]:
                tok = int(tok)
                if tok in (SPECIAL["<|eot_id|>"], SPECIAL["<|end_of_text|>"]):
                    return text
                if tok >= 128000:
                    continue
                piece = self.tok.decode([tok])
                if allow is None or allow(text, piece):
                    pick = (tok, piece)
                    break
            if pick is None:
                break
            text += pick[1]
            cut = [text.find(st) for st in stop if st in text]
            if cut:
                return text[:min(cut)]
            self._forward([pick[0]])
        return text

    def generate(self, prompt: str, max_new_tokens: int):
        """Drop-in for the text runner (shift questions, alert wording): reset, prefill, greedy text."""
        from .advisor import GenResult
        t0 = time.perf_counter()
        self.reset()
        self.feed(prompt)
        n_prompt, t1 = self.pos, time.perf_counter()
        text = self.write(max_new_tokens, stop=())
        t2 = time.perf_counter()
        n_gen = self.pos - n_prompt
        return GenResult(text, (t2 - t0) * 1e3, n_prompt / max(t1 - t0, 1e-6),
                         n_gen / max(t2 - t1, 1e-6) if n_gen else None, n_gen, n_prompt)

    def choose(self, options: list[str]) -> tuple[str, float]:
        """The model continues the context with exactly one of `options`. Returns (option, margin)."""
        if self._last is None:
            raise RuntimeError("feed() a prompt first")
        seqs = [self.encode(o) for o in options]

        def step(tok):
            return self._last if tok is None else self._forward([tok])
        i, margin, _ = constrained_pick(step, seqs)
        return options[i], margin
