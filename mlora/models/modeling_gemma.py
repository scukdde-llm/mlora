from mlora.common.model import LLMDecoder, LLMForCausalLM
from mlora.common.feed_forward import FeedForward
from mlora.models.modeling_llama import (
    LlamaConfig,
    LLAMA_ATTENTION_CLASSES as GEMMA_ATTENTION_CLASSES,
    LlamaMLP,
    LlamaDecoderLayer,
    LlamaEmbedding,
    LlamaRMSNorm,
    LlamaSequentialWrapper,
)
from mlora.backends import get_backend
from mlora.utils import copy_parameters

from transformers.activations import ACT2FN
from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn
import transformers.models.gemma.modeling_gemma as modeling_gemma


class GemmaMLP(LlamaMLP):
    def __init__(self, w1: torch.nn.Module, w2: torch.nn.Module, w3: torch.nn.Module, args: LlamaConfig) -> None:
        super().__init__(w1, w2, w3, args)
        self.act_ = ACT2FN["gelu_pytorch_tanh"]


class GemmaForCausalLM(LLMForCausalLM):
    def __init__(self, config: LlamaConfig) -> None:
        self.config_ = config
        self.padding_idx_ = config.pad_token_id_
        self.vocab_size_ = config.vocab_size_
        self.embed_tokens_: LlamaEmbedding = None
        self.norm_: LlamaEmbedding = None
        self.lm_head_ = nn.Linear(config.dim_, config.vocab_size_, bias=False,
                                  dtype=config.dtype_, device=config.device_)
        self.layers_: List[LlamaDecoderLayer] = []

    def decoder_stack(self) -> List[LLMDecoder]:
        return self.layers_

    def sequential_module(self) -> OrderedDict:
        seq_module = OrderedDict()

        seq_module.update(
            {"embedding": LlamaSequentialWrapper(self.embed_tokens_)})
        seq_module.move_to_end("embedding")

        for index, layer in enumerate(self.layers_):
            layer_name = f"layer{index}"
            seq_module.update({layer_name: LlamaSequentialWrapper(layer)})
            seq_module.move_to_end(layer_name)

        seq_module.update(
            {"norm": LlamaSequentialWrapper(self.norm_)})
        seq_module.move_to_end("norm")

        return seq_module

    @staticmethod
    def from_pretrained(llm_model: modeling_gemma.GemmaForCausalLM,
                        attn_impl: str = "eager",
                        use_sliding_window: bool = False,
                        device: str = get_backend().device_name() + ":0"):
        assert not use_sliding_window, "Gemma model does not support SWA."
        llm_config: modeling_gemma.GemmaConfig = llm_model.config
        llm_args = LlamaConfig(
            name_or_path_=llm_config.name_or_path,
            vocab_size_=llm_config.vocab_size,
            dim_=llm_config.hidden_size,
            n_layers_=llm_config.num_hidden_layers,
            n_heads_=llm_config.num_attention_heads,
            n_kv_heads_=llm_config.num_key_value_heads,
            rms_norm_eps_=llm_config.rms_norm_eps,
            max_seq_len_=llm_config.max_position_embeddings,
            rope_theta_=llm_config.rope_theta,
            pad_token_id_=llm_config.pad_token_id,
            attn_implementation_=attn_impl,
            device_=torch.device(device),
            dtype_=llm_model.dtype,
        )

        if llm_args.pad_token_id_ is None:
            llm_args.pad_token_id_ = -1

        model = GemmaForCausalLM(llm_args)
        llm_model.requires_grad_(False)
        model.embed_tokens_ = LlamaEmbedding(
            llm_model.model.embed_tokens.weight, llm_args.pad_token_id_)
        model.norm_ = LlamaRMSNorm(
            llm_model.model.norm.weight, llm_args.rms_norm_eps_)
        copy_parameters(llm_model.lm_head, model.lm_head_)

        for idx, layer in enumerate(llm_model.model.layers):
            decoder = LlamaDecoderLayer()
            decoder.layer_id_ = idx
            decoder.self_attn_ = GEMMA_ATTENTION_CLASSES[llm_args.attn_implementation_](
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
                layer.self_attn.o_proj,
                idx,
                llm_args,
            )
            decoder.mlp_ = FeedForward(GemmaMLP(
                layer.mlp.gate_proj,
                layer.mlp.down_proj,
                layer.mlp.up_proj,
                llm_args,
            ))
            decoder.input_layernorm_ = LlamaRMSNorm(
                layer.input_layernorm.weight, llm_args.rms_norm_eps_)
            decoder.post_attention_layernorm_ = LlamaRMSNorm(
                layer.post_attention_layernorm.weight, llm_args.rms_norm_eps_)
            model.layers_.append(decoder)

        return model
