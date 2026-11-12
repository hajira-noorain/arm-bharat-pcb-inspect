"""Llama 3.2 on ExecuTorch (XNNPACK, KleidiAI micro-kernels on Arm) through the official Python runner.

The model stays loaded for the lifetime of the station (load once, ~2 s on a Pi 5 for 1B), and
per-call speed comes from the runtime's own Stats (not wall-clock guesses).
"""
from __future__ import annotations

import time
from pathlib import Path

from .advisor import GenResult


class ExecuTorchLlama:
    name = "llama-executorch"

    def __init__(self, pte_path: str | Path, tokenizer_path: str | Path, threads: int = 4):
        from executorch.extension.llm.runner import GenerationConfig, TextLLMRunner
        self._GenerationConfig = GenerationConfig
        self.model = Path(pte_path).stem
        self.threads = threads
        t0 = time.perf_counter()
        self.runner = TextLLMRunner(str(pte_path), str(tokenizer_path))
        self.load_ms = (time.perf_counter() - t0) * 1e3
        self.last_stats: dict | None = None

    def generate(self, prompt: str, max_new_tokens: int) -> GenResult:
        pieces: list[str] = []
        stats_box: dict = {}

        def on_token(t: str) -> None:
            pieces.append(t)

        def on_stats(s) -> None:
            stats_box.update(
                prompt_tokens=s.num_prompt_tokens, generated_tokens=s.num_generated_tokens,
                inference_start=s.inference_start_ms, prompt_eval_end=s.prompt_eval_end_ms,
                first_token=s.first_token_ms, inference_end=s.inference_end_ms,
                model_load_ms=s.model_load_end_ms - s.model_load_start_ms)

        cfg = self._GenerationConfig(echo=False, max_new_tokens=max_new_tokens, temperature=0.0)
        t0 = time.perf_counter()
        self.runner.reset()
        self.runner.generate(prompt, cfg, on_token, on_stats)
        wall = (time.perf_counter() - t0) * 1e3
        text = "".join(pieces).split("<|eot_id|>")[0]
        prefill_tps = decode_tps = None
        if stats_box:
            pre_ms = stats_box["prompt_eval_end"] - stats_box["inference_start"]
            dec_ms = stats_box["inference_end"] - stats_box["first_token"]
            if pre_ms > 0:
                prefill_tps = stats_box["prompt_tokens"] / (pre_ms / 1000)
            if dec_ms > 0 and stats_box["generated_tokens"] > 1:
                decode_tps = (stats_box["generated_tokens"] - 1) / (dec_ms / 1000)
        self.last_stats = {**stats_box, "wall_ms": wall}
        return GenResult(text, wall, prefill_tps, decode_tps, stats_box.get("generated_tokens"))
