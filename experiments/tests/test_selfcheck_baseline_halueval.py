"""Tests for the SelfCheckGPT-NLI baseline harness.

Pure logic: the sampling-config construction and the one end-to-end path that can run
without a GPU. The scorer aggregation itself is tested in eval/tests/test_selfcheck.py.
"""

import pytest

from selfcheck_baseline_halueval import sampling_decoding_cfg

GREEDY = {"do_sample": False, "max_new_tokens": 20}


def test_sample_config_switches_to_sampling_and_sets_temperature():
    sample_cfg = sampling_decoding_cfg(GREEDY, temperature=1.0)
    assert sample_cfg["do_sample"] is True
    assert sample_cfg["temperature"] == 1.0


def test_sample_config_keeps_every_other_generation_parameter_matched():
    """Only do_sample and temperature may differ from the main answer's config — the
    sample length in particular must match, or the consistency comparison is between
    passages of different lengths."""
    sample_cfg = sampling_decoding_cfg(GREEDY, temperature=0.7)
    assert sample_cfg["max_new_tokens"] == GREEDY["max_new_tokens"]
    assert set(sample_cfg) - set(GREEDY) == {"temperature"}


def test_sample_config_does_not_mutate_the_greedy_config():
    # The greedy config is shared with every other condition in the study; mutating it
    # here would silently switch the whole run to sampling.
    before = dict(GREEDY)
    sampling_decoding_cfg(GREEDY, temperature=1.0)
    assert GREEDY == before


@pytest.mark.parametrize("temperature", [0.5, 1.0, 1.5])
def test_temperature_is_passed_through_verbatim(temperature):
    assert sampling_decoding_cfg(GREEDY, temperature)["temperature"] == temperature
