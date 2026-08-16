import torch

from sense_neural.entropy import TokenEntropyMonitor
from sense_neural.latency import generate_with_latency

PROMPT = "The quick brown fox"
DECODING_CFG = {"do_sample": False, "max_new_tokens": 10}


def test_generate_with_latency_returns_well_formed_result(tiny_model, tiny_tokenizer):
    inputs = tiny_tokenizer(PROMPT, return_tensors="pt")
    result = generate_with_latency(tiny_model, tiny_tokenizer, inputs, DECODING_CFG)

    assert result["n_pieces"] > 0
    assert result["ttft_ms"] >= 0
    assert result["mean_inter_token_latency_ms"] >= 0
    assert result["total_generation_ms"] >= result["ttft_ms"]
    assert isinstance(result["text"], str)


def test_streaming_path_matches_non_streaming_generation(tiny_model, tiny_tokenizer):
    """The streaming path used for latency measurement must produce the same tokens
    as plain generate() — it's an instrumentation wrapper, not a different decode."""
    inputs = tiny_tokenizer(PROMPT, return_tensors="pt")
    torch.manual_seed(0)
    baseline_ids = tiny_model.generate(**inputs, max_new_tokens=10, do_sample=False)
    baseline_text = tiny_tokenizer.decode(baseline_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    torch.manual_seed(0)
    result = generate_with_latency(tiny_model, tiny_tokenizer, inputs, DECODING_CFG)

    assert result["text"] == baseline_text


def test_monitor_attached_during_latency_measurement_is_still_noop(tiny_model, tiny_tokenizer):
    inputs = tiny_tokenizer(PROMPT, return_tensors="pt")
    monitor = TokenEntropyMonitor(vocab_size=tiny_tokenizer.vocab_size)
    result = generate_with_latency(tiny_model, tiny_tokenizer, inputs, DECODING_CFG, logits_processor=[monitor])

    baseline = generate_with_latency(tiny_model, tiny_tokenizer, inputs, DECODING_CFG)

    assert result["text"] == baseline["text"]
    # One entropy value per actual generation step (max_new_tokens, absent early
    # stopping) — the ground-truth token count, unlike the streamer's piece count.
    assert len(monitor.entropies) == DECODING_CFG["max_new_tokens"]
