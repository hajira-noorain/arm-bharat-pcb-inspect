"""Probe: does TextLLMRunner keep the KV cache across generate() calls, and does stop() from the
token callback end generation cleanly? Both are required for step-wise structured generation."""
import sys
import time

import executorch.extension.llm.custom_ops.custom_ops  # noqa: F401
import executorch.kernels.quantized  # noqa: F401
from executorch.extension.llm.runner import GenerationConfig, TextLLMRunner

pte, tok = sys.argv[1], sys.argv[2]
r = TextLLMRunner(pte, tok)


def gen(text, n, stop=None):
    out = []

    def cb(t):
        out.append(t)
        if stop and stop in "".join(out):
            r.stop()
    t0 = time.perf_counter()
    r.generate(text, GenerationConfig(echo=False, max_new_tokens=n, temperature=0.0), cb)
    return "".join(out), (time.perf_counter() - t0) * 1e3


head = ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nThe secret code word is "
        "PERIDOT. Remember it.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nOK.<|eot_id|>")
r.reset()
a, ta = gen(head + "<|start_header_id|>user<|end_header_id|>\n\nCount from 1 to 30, comma separated."
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", 60, stop="5")
print(f"[stop test] output={a!r} ({ta:.0f} ms) -> stopped early: {len(a) < 40}")
b, tb = gen("<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat was the secret code word? "
            "Answer with the word only.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", 8)
print(f"[kv test] output={b!r} ({tb:.0f} ms) -> remembers context: {'PERIDOT' in b.upper()}")
r.reset()
c, tc = gen("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhat was the secret code word? "
            "Answer with the word only.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", 8)
print(f"[after reset] output={c!r} -> forgot: {'PERIDOT' not in c.upper()}")
