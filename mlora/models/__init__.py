from .modeling_llama import LlamaForCausalLM
from .modeling_phi import PhiForCausalLM


model_dict = {
    "llama": LlamaForCausalLM,
    "phi": PhiForCausalLM,
}


def from_pretrained(llm_model, **kwargs):
    if llm_model.config.model_type in model_dict:
        return model_dict[llm_model.config.model_type].from_pretrained(llm_model, **kwargs)
    else:
        raise RuntimeError(
            f"Model {llm_model.config.model_type} not supported.")


__all__ = [
    "LlamaForCausalLM",
    "PhiForCausalLM",
    "from_pretrained",
]
