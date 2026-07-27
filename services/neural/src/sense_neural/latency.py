"""Per-stage generation latency instrumentation: time-to-first-token and mean
inter-token latency (CLAUDE.md's latency instrumentation requirements). Measured
via TextIteratorStreamer running generate() in a background thread while the main
thread timestamps each decoded piece as it arrives — this is the only place token
arrival time is directly observable, since generate() otherwise blocks until done.

Warm-up runs are not discarded here — that's the caller's responsibility (run once
and throw away the result before measuring), per CLAUDE.md: first-call kernel
compilation dominates and would bias a single-call measurement.
"""

import threading
import time

from transformers import TextIteratorStreamer


def generate_with_latency(model, tokenizer, prompt: str, decoding_cfg: dict, logits_processor=None) -> dict:
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    generate_kwargs = dict(
        **inputs,
        max_new_tokens=decoding_cfg["max_new_tokens"],
        do_sample=decoding_cfg["do_sample"],
        streamer=streamer,
    )
    if logits_processor is not None:
        generate_kwargs["logits_processor"] = logits_processor

    start = time.perf_counter()
    thread = threading.Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()

    arrival_times = []
    pieces = []
    for piece in streamer:
        # TextIteratorStreamer can flush a trailing empty piece; skip it rather than
        # counting it as a token arrival, or n_tokens overstates the real token count.
        if piece == "":
            continue
        arrival_times.append(time.perf_counter())
        pieces.append(piece)
    thread.join()

    if not arrival_times:
        raise RuntimeError("generate_with_latency produced no output tokens")

    ttft_ms = (arrival_times[0] - start) * 1000
    if len(arrival_times) > 1:
        deltas_ms = [(arrival_times[i] - arrival_times[i - 1]) * 1000 for i in range(1, len(arrival_times))]
        mean_inter_token_latency_ms = sum(deltas_ms) / len(deltas_ms)
    else:
        mean_inter_token_latency_ms = 0.0
    total_ms = (arrival_times[-1] - start) * 1000

    return {
        "text": "".join(pieces),
        "ttft_ms": ttft_ms,
        "mean_inter_token_latency_ms": mean_inter_token_latency_ms,
        "total_generation_ms": total_ms,
        # Decoded-piece count, not generated-token count: BPE decoding can merge or
        # delay a token's text across streamer flushes, so this is not guaranteed to
        # equal max_new_tokens or the entropy monitor's per-step call count.
        "n_pieces": len(arrival_times),
    }
