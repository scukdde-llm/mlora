from mlora.common import (
    prepare_4d_causal_attention_mask,
    Masks,
    Linear,
    FeedForward,
    MultiLoraBatchData,
    CHECKPOINT_CLASSES,
    LLMModelArgs,
    LLMAttention,
    LLMFeedForward,
    LLMDecoder,
    LLMForCausalLM,
)
from typing import Tuple, Dict, List, Optional
from collections import OrderedDict
from dataclasses import dataclass
from mlora.backends import get_backend
from mlora.utils import copy_parameters
import torch
import torch.nn as nn
import math
from torch.nn import LayerNorm


@dataclass
class ChatGLMConfig(LLMModelArgs):
    post_layer_norm_: bool = True
    rmsnorm_: bool = True
    layer_norm_eps: float = 0.0
    apply_residual_connection_post_layernorm: bool = False
    attention_dropout: float = 0.0
    fp32_residual_connection: bool = False
    kv_channels: int = 128
    multi_query_attention: bool = False
    multi_query_group_num: int = 2
    apply_query_key_layer_scaling: bool = True
    attention_softmax_in_fp32: bool = True
    original_rope: bool = True


class ChatGLMEmbedding(nn.Module):
    def __init__(self, config: ChatGLMConfig):
        super().__init__()
        self.fp32_residual_connection = config.fp32_residual_connection
        self.hidden_size = config.dim_
        # Word embeddings (parallel).
        self.embed_tokens = nn.Embedding(
            config.vocab_size_,
            config.dim_,
            config.pad_token_id_,
            dtype=config.dtype_,
            device=config.device_
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Embeddings.
        words_embeddings = self.embed_tokens(input_ids)
        embeddings = words_embeddings
        if self.fp32_residual_connection:
            embeddings = embeddings.float()
        return embeddings


class ChatGLMRMSNorm(nn.Module):
    def __init__(self, dim_, eps, device, dtype):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(dim_,
                                                     device=device, dtype=dtype))
        self.eps = eps

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        input_dtype = data.dtype
        variance = data.to(torch.float32).pow(2).mean(-1, keepdim=True)
        data = data * torch.rsqrt(variance + self.eps)
        return (self.weight * data).to(input_dtype)


class ChatGLMLayerNorm(nn.Module):
    def __init__(self, config: ChatGLMConfig):
        super().__init__()
        LayerNormFunc = ChatGLMRMSNorm if config.rmsnorm_ else LayerNorm
        self.layernorm_ = LayerNormFunc(config.dim_, config.layer_norm_eps, config.device_, config.dtype_)

    def forward(self, data: torch.tensor) -> torch.Tensor:
        return self.layernorm_.forward(data)

    def load_weight(self, weight: torch.nn.Parameter):
        self.layernorm_.weight = torch.nn.Parameter(torch.clone(weight))
        self.layernorm_.requires_grad_(False)


def split_tensor_along_last_dim(
        tensor: torch.Tensor,
        num_partitions: int,
        contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
    """Split a tensor along its last dimension.

    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.

    Returns:
        A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = tensor.size()[last_dim] // num_partitions
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, original_impl=False, device=None, dtype=None):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).to(dtype=dtype) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.dim = dim
        self.original_impl = original_impl

    def forward_impl(
            self, seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000
    ):
        """Enhanced Transformer with Rotary Position Embedding.

        Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
        transformers/rope/__init__.py. MIT License:
        https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
        """
        # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=torch.float, device=device) / n_elem))

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, dtype=torch.float, device=device)

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.outer(seq_idx, theta).float()

        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

        # this is to mimic the behaviour of complex32, else we will get different results
        if dtype in (torch.float16, torch.bfloat16, torch.int8):
            cache = cache.bfloat16() if dtype == torch.bfloat16 else cache.half()
        return cache

    def forward(self, max_seq_len, offset=0):
        return self.forward_impl(
            max_seq_len, self.dim, dtype=self.inv_freq.dtype, device=self.inv_freq.device
        )


@torch.jit.script
def apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    # x: [sq, batch, projection_size, heads_num]
    sq, np = x.size(0), x.size(2)
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:sq]
    xshaped = x.reshape(sq, -1, np, rot_dim // 2, 2)
    rope_cache = rope_cache.view(sq, -1, 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return torch.cat((x_out2, x_pass), dim=-1)


class ChatGLMCoreAttention(nn.Module):
    def __init__(self, config: ChatGLMConfig, layer_number):
        super(ChatGLMCoreAttention, self).__init__()

        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)

        projection_size = config.kv_channels * config.n_heads_

        # Per attention head and per partition values.
        self.hidden_size_per_partition = projection_size
        self.hidden_size_per_attention_head = projection_size // config.n_heads_
        self.num_attention_heads_per_partition = config.n_heads_

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff
        self.coeff = coeff

        self.attention_dropout = nn.Dropout(config.attention_dropout)

    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        # Raw attention scores

        # [b, np, sq, sk]
        output_size = (query_layer.size(1), query_layer.size(2), query_layer.size(0), key_layer.size(0))

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

        # preallocting input tensor: [b * np, sq, sk]
        matmul_input_buffer = torch.empty(
            output_size[0] * output_size[1], output_size[2], output_size[3], dtype=query_layer.dtype,
            device=query_layer.device
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query_layer.transpose(0, 1),  # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        if self.attention_softmax_in_fp32:
            attention_scores = attention_scores.float()
        if self.coeff is not None:
            attention_scores = attention_scores * self.coeff
        if attention_mask is None and attention_scores.shape[2] == attention_scores.shape[3]:
            attention_mask = torch.ones(output_size[0], 1, output_size[2], output_size[3],
                                        device=attention_scores.device, dtype=torch.bool)
            attention_mask.tril_()
            attention_mask = ~attention_mask
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask, float("-inf"))
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = attention_probs.type_as(value_layer)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_dropout(attention_probs)
        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(1), value_layer.size(2), query_layer.size(0), value_layer.size(3))
        # change view [sk, b * np, hn]
        value_layer = value_layer.view(value_layer.size(0), output_size[0] * output_size[1], -1)
        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))
        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)
        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()
        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class ChatGLMAttention(LLMAttention):
    def __init__(self, qkv_tensor: nn.Module, dense_weight: nn.Module, config: ChatGLMConfig, layer_idx) -> None:
        super().__init__()
        self.layer_id_ = max(1, layer_idx)
        self.projection_size = config.kv_channels * config.n_heads_

        # Per attention head and per partition values.
        self.hidden_size_per_attention_head = self.projection_size // config.n_heads_
        self.num_attention_heads_per_partition = config.n_heads_

        self.multi_query_attention = config.multi_query_attention
        self.qkv_hidden_size = 3 * self.projection_size
        if self.multi_query_attention:
            self.num_multi_query_groups_per_partition = config.multi_query_group_num
            self.qkv_hidden_size = (
                self.projection_size + 2 * self.hidden_size_per_attention_head * config.multi_query_group_num
            )

        # rotary_emb
        self.sequence_length = config.max_seq_len_
        rotary_dim = (
            config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels
        )
        rotary_pos_emb = RotaryEmbedding(rotary_dim // 2,
                                         original_impl=config.original_rope,
                                         device=config.device_,
                                         dtype=config.dtype_)
        self.rotary_pos_emb = rotary_pos_emb(config.max_seq_len_)

        # attetion
        self.quer_key_value: Linear = Linear(qkv_tensor, config.device_)
        self.core_attention: ChatGLMCoreAttention = ChatGLMCoreAttention(config=config, layer_number=self.layer_id_)
        self.dense_: Linear = Linear(dense_weight, config.device_)

    def state_dict(self) -> Dict[str, Linear]:
        return {
            "qkv_proj": self.quer_key_value,
            "dense": self.dense_
        }

    def forward(self,
                hidden_states: torch.Tensor,
                input_args: MultiLoraBatchData,
                attention_mask: Optional[torch.Tensor] = None):
        # hidden_states : [batch ,sequence,hidden_size] ->[batch ,sequence,qkv_hidden_size(3*projection_num*head_num)]
        mixed_x_layer = self.quer_key_value.forward(hidden_states, input_args)
        # [b ,sq,qkv_hidden_size(3*projection_num*head_num)] -> [sq, b, qkv_hidden_size(3*projection_num*head_num)]
        mixed_x_layer = mixed_x_layer.transpose(0, 1).contiguous()

        if self.multi_query_attention:
            (query_layer, key_layer, value_layer) = mixed_x_layer.split(
                [
                    self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                ],
                dim=-1,
            )
            # q:[seq,bacth,heads,per_attion]  k,v:[seq,bacth,multi_query_group,per_attion]
            query_layer = query_layer.view(
                query_layer.size()[:-1] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )
            key_layer = key_layer.view(
                key_layer.size()[:-1] + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
            value_layer = value_layer.view(
                value_layer.size()[:-1] + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
        else:
            new_tensor_shape = mixed_x_layer.size()[:-1] + (self.num_attention_heads_per_partition,
                                                            3 * self.hidden_size_per_attention_head)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [seq,bacth,head,3*per_attention] -> 3*[seq,batch,heads,per_attetion]
            (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

        batch_size, seq_length, dim_ = hidden_states.shape
        rotary_pos_emb = self.rotary_pos_emb[None, :seq_length]
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()
        query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
        key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)

        # expand the kv(group*hidden_size) -> (n_head*hidden_size)
        # kv :[sq,b,group_num,hidden_size] -> [sq, b, group, head/group, hidden_size] -> [sq, b, heads,hidden_size]
        if self.multi_query_attention:
            key_layer = key_layer.unsqueeze(-2)
            key_layer = key_layer.expand(
                -1, -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1
            )
            key_layer = key_layer.contiguous().view(
                key_layer.size()[:2] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )
            value_layer = value_layer.unsqueeze(-2)
            value_layer = value_layer.expand(
                -1, -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1
            )
            value_layer = value_layer.contiguous().view(
                value_layer.size()[:2] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )

        # ==================================
        # core attention computation
        # ==================================

        context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)

        # =================
        # context_layer [sq, b, h] -> [b, sq, h]
        # Output. [b, sq, h]
        # =================
        context_layer = context_layer.transpose(0, 1).contiguous()
        output = self.dense_(context_layer, input_args)

        return output


class ChatGLMSequentialWrapper(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.wrapper_module_ = module

    def name(self) -> str:
        return type(self.wrapper_module_).__name__

    def forward(self, input: Tuple) -> Tuple:
        module_name = self.name()

        if module_name == "ChatGLMEmbedding":
            output = self.wrapper_module_.forward(input[0])
            if input[-1].gradient_checkpoint_ != "none":
                output = output.requires_grad_(True)
            return (output,) + input[1:]
        elif module_name == "ChatGLMDecoderLayer":
            outputs = CHECKPOINT_CLASSES[input[-1].gradient_checkpoint_](
                self.wrapper_module_.forward, *input)
            if len(outputs) > 1:
                self.router_probs_ = outputs[1:]
            return (outputs[0],) + input[1:]
        elif module_name == "ChatGLMLayerNorm":
            output = self.wrapper_module_.forward(input[0])
            return (output, ) + input[1:]
        else:
            raise f"module invalid:{module_name}"


class ChatGLMMLP(LLMFeedForward):
    def __init__(self, dense_h_to_4h: nn.Module, dense_4h_to_h: nn.Module, config: ChatGLMConfig) -> None:
        super().__init__()
        self.dense_h_to_4h: Linear = Linear(dense_h_to_4h, config.device_)
        self.dense_4h_to_h: Linear = Linear(dense_4h_to_h, config.device_)

        def swiglu(x):
            x = torch.chunk(x, 2, dim=-1)
            return nn.functional.silu(x[0]) * x[1]

        self.activation_func = swiglu

    def state_dict(self) -> Dict[str, Linear]:
        return {
            "dense_h_to_4h": self.dense_h_to_4h,
            "dense_4h_to_h": self.dense_4h_to_h,
        }

    def _batch_forward(self, data: torch.Tensor, input_args: MultiLoraBatchData) -> torch.Tensor:
        # [b, s, h] -> [b, s, 4hp]
        intermediate_parallel = self.dense_h_to_4h.forward(data, input_args)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        output = self.dense_4h_to_h.forward(intermediate_parallel, input_args)

        return output

    def _lora_forward(
            self, lora_name: str, act_fn: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        if lora_name in self.dense_h_to_4h.loras_:
            hidden_states = self.dense_h_to_4h.loras_[lora_name].forward(
                self.dense_h_to_4h.base_layer_.forward(hidden_states), hidden_states
            )
        else:
            hidden_states = self.dense_h_to_4h.base_layer_.forward(hidden_states)

        hidden_states = self.activation_func(hidden_states)

        if lora_name in self.dense_4h_to_h.loras_:
            hidden_states = self.dense_4h_to_h.loras_[lora_name].forward(
                self.dense_4h_to_h.base_layer_.forward(hidden_states), hidden_states
            )
        else:
            hidden_states = self.dense_4h_to_h.base_layer_.forward(hidden_states)

        return hidden_states


class ChatGLMDecoderLayer(LLMDecoder):
    def __init__(self, attn: LLMAttention, mlp: FeedForward, config: ChatGLMConfig) -> None:
        super().__init__()
        self.layer_id_ = attn.layer_id_
        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        self.fp32_residual_connection = config.fp32_residual_connection

        # layernorm on the input data
        self.input_layernorm = ChatGLMLayerNorm(config)
        # Self attention.
        self.self_attn_: ChatGLMAttention = attn
        self.hidden_dropout = config.hidden_dropout_
        # layernorm on the attention output
        self.post_layernorm = ChatGLMLayerNorm(config)
        # mlp
        self.mlp_: FeedForward = mlp

    def state_dict(self) -> Dict[str, nn.Module]:
        linear_layers = self.self_attn_.state_dict()
        linear_layers.update(self.mlp_.state_dict())
        return linear_layers

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                input_args: MultiLoraBatchData):
        # hidden_states: [b, s, h]
        layernorm_output = self.input_layernorm(hidden_states)
        # layernrom_out = [b, s, h]
        attention_output = self.self_attn_.forward(
            hidden_states=layernorm_output,
            input_args=input_args,
            attention_mask=attention_mask
        )

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # layernorm_input [b, sq, h]
        layernorm_input = nn.functional.dropout(attention_output,
                                                p=self.hidden_dropout,
                                                training=not input_args.inference_mode_)
        layernorm_input = residual + layernorm_input

        # Layer norm post the self attention.
        layernorm_output = self.post_layernorm(layernorm_input)

        # MLP.
        mlp_output, router_logits = self.mlp_(layernorm_output, input_args)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        output = nn.functional.dropout(mlp_output, p=self.hidden_dropout, training=not input_args.inference_mode_)
        output = residual + output

        return output, *router_logits


class ChatGLMForCausalLM(LLMForCausalLM):
    def __init__(self, config: ChatGLMConfig) -> None:
        self.config_ = config
        self.padding_idx_ = config.pad_token_id_
        self.vocab_size_ = config.vocab_size_
        self.embed_tokens_ = ChatGLMEmbedding(config)
        # outputlayer
        self.lm_head_ = nn.Linear(config.dim_, config.vocab_size_, bias=False,
                                  dtype=config.dtype_, device=config.device_)
        self.layers_: List[ChatGLMDecoderLayer] = []

        if self.config_.post_layer_norm_:
            self.final_layernorm_ = ChatGLMLayerNorm(config)

    def decoder_stack(self) -> List[LLMDecoder]:
        return self.layers_

    def sequential_module(self) -> OrderedDict:
        seq_module = OrderedDict()

        seq_module.update(
            {"embedding": ChatGLMSequentialWrapper(self.embed_tokens_)})
        seq_module.move_to_end("embedding")

        for index, layer in enumerate(self.layers_):
            layer_name = f"layer{index}"
            seq_module.update({layer_name: ChatGLMSequentialWrapper(layer)})
            seq_module.move_to_end(layer_name)

        if self.config_.post_layer_norm_:
            seq_module.update(
                {"norm": ChatGLMSequentialWrapper(self.final_layernorm_)})
            seq_module.move_to_end("norm")

        return seq_module

    def causal_mask(self,
                    input_tokens: torch.Tensor,
                    additional_mask: List[Masks] = None,
                    multi_head: bool = False,
                    diagonal: int = 1) -> torch.Tensor:
        if multi_head:
            assert self.config_.attn_implementation_ == "xformers"
        else:
            assert self.config_.attn_implementation_ != "xformers"

        return prepare_4d_causal_attention_mask(input_tokens=input_tokens,
                                                n_heads=self.config_.n_heads_ if multi_head else 1,
                                                additional_mask=additional_mask, diagonal=diagonal,
                                                dtype=self.config_.dtype_, device=self.config_.device_)

    @staticmethod
    def from_pretrained(llm_model,
                        attn_impl: str = "eager",
                        use_sliding_window: bool = False,
                        device: str = get_backend().device_name() + ":0"):
        assert not use_sliding_window
        llm_config = llm_model.config
        llm_args = ChatGLMConfig(
            # llmmodelargs
            name_or_path_=llm_config._name_or_path,
            device_=torch.device(device),
            dim_=llm_config.hidden_size,
            n_heads_=llm_config.num_attention_heads,
            n_layers_=llm_config.num_layers,
            hidden_dropout_=llm_config.hidden_dropout,
            vocab_size_=llm_config.vocab_size,
            pad_token_id_=llm_config.pad_token_id,
            max_seq_len_=llm_config.seq_length,
            dtype_=llm_model.dtype,
            attn_implementation_=attn_impl,
            # chatglmargs
            apply_query_key_layer_scaling=llm_config.apply_query_key_layer_scaling,
            attention_softmax_in_fp32=llm_config.attention_softmax_in_fp32,
            fp32_residual_connection=llm_config.fp32_residual_connection,
            apply_residual_connection_post_layernorm=llm_config.apply_residual_connection_post_layernorm,
            attention_dropout=llm_config.attention_dropout,
            kv_channels=llm_config.kv_channels,
            layer_norm_eps=llm_config.layernorm_epsilon,
            multi_query_attention=llm_config.multi_query_attention,
            multi_query_group_num=llm_config.multi_query_group_num,
            original_rope=llm_config.original_rope,
            post_layer_norm_=llm_config.post_layer_norm,
            rmsnorm_=llm_config.rmsnorm
        )

        model = ChatGLMForCausalLM(llm_args)
        llm_model.requires_grad_(False)
        copy_parameters(llm_model.transformer.embedding.word_embeddings,
                        model.embed_tokens_.embed_tokens)
        copy_parameters(llm_model.transformer.output_layer,
                        model.lm_head_)
        if llm_config.post_layer_norm:
            model.final_layernorm_.load_weight(llm_model.transformer.encoder.final_layernorm.weight)

        for idx, layer in enumerate(llm_model.transformer.encoder.layers):
            decoder = ChatGLMDecoderLayer(
                ChatGLMAttention(layer.self_attention.query_key_value,
                                 layer.self_attention.dense,
                                 llm_args,
                                 idx
                                 ),
                FeedForward(ChatGLMMLP(layer.mlp.dense_h_to_4h,
                                       layer.mlp.dense_4h_to_h,
                                       llm_args
                                       )
                            ),
                llm_args
            )
            decoder.input_layernorm.load_weight(layer.input_layernorm.weight)
            decoder.post_layernorm.load_weight(layer.post_attention_layernorm.weight)
            model.layers_.append(decoder)

        return model
