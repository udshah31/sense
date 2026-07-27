import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

TINY_MODEL = "sshleifer/tiny-gpt2"


@pytest.fixture(scope="session")
def tiny_tokenizer():
    return AutoTokenizer.from_pretrained(TINY_MODEL)


@pytest.fixture(scope="session")
def tiny_model():
    return AutoModelForCausalLM.from_pretrained(TINY_MODEL)
