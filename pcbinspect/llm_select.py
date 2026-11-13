"""Which Llama the station loads, in one place (the app, the soak test and the benchmark all use this).

config/app.yaml:
  llm_pte            the small model (1B): the fallback, and the one the tokens-per-second benchmark times
  advice_llm_pte     the model that decides advice and shift questions when present (3B: it follows the
                     evidence where the 1B model does not, docs/RESULTS.md)
  llm_backend        logits = decisions read from next-token scores (pcbinspect/llm_logits.py);
                     runner = the text runner writing JSON
  advice_cause_mode  pointwise = each candidate cause judged alone with a yes/no question
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("pcbinspect.llm")


def advice_pte(root: Path, cfg: dict) -> Path | None:
    for key in ("advice_llm_pte", "llm_pte"):
        if cfg.get(key) and (root / cfg[key]).is_file():
            return root / cfg[key]
    return None


def load_llama(root: Path, cfg: dict):
    """The backend for advice and questions, or None (the station then shows knowledge-base text)."""
    p = advice_pte(root, cfg)
    if p is None:
        log.warning("no Llama .pte found (%s / %s): advice will use templates",
                    cfg.get("advice_llm_pte"), cfg.get("llm_pte"))
        return None
    tok, threads = root / cfg["llm_tokenizer"], cfg.get("llm_threads", 4)
    if cfg.get("llm_backend", "runner") == "logits":
        try:
            from .llm_logits import LogitLlama
            return LogitLlama(p, tok, threads=threads)
        except Exception as exc:    # noqa: BLE001
            log.exception("Llama (scores) failed to load (%s): trying the text runner", exc)
    from .llm import ExecuTorchLlama
    try:
        return ExecuTorchLlama(p, tok, threads=threads)
    except Exception as exc:        # noqa: BLE001
        log.exception("Llama failed to load (%s): advice will use templates", exc)
        return None
