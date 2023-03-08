from . import huggingface
# from . import openai_api
# from . import textsynth
from . import test_model

MODEL_REGISTRY = {
    "hf-causal": huggingface.HuggingfaceCausalLM,
    # "hf-seq2seq"
    #"textsynth": textsynth.TextSynthLM,
    "dummy": test_model.DummyLM,
    # "openai"
    # "goose"
}


def get_model(model_name):
    return MODEL_REGISTRY[model_name]