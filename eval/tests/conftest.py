import pytest

from sense_eval.nli_judge import load_nli_model

NLI_HF_REPO = "cliang1453/deberta-v3-xsmall-mnli"
NLI_REVISION = "d1ca70f9ece4d8afd33015893a69df9a6e45a672"


@pytest.fixture(scope="session")
def nli_model_and_tokenizer():
    return load_nli_model(NLI_HF_REPO, NLI_REVISION)
