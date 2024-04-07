from mlora.modelargs import MixConfig, MultiLoraBatchData
from mlora.lora_linear import get_range_tensor, Linear
from mlora.model import LLMFeedForward
from mlora.mix_lora import moe_layer_factory

from transformers.models.llama.modeling_llama import LlamaMLP as TransformersLlamaMLP
from transformers.models.phi.modeling_phi import PhiMLP as TransformersPhiMLP
from transformers.activations import ACT2FN
from typing import Dict, List, Optional
import torch


class LlamaMLP(LLMFeedForward):
    def __init__(self, mlp: TransformersLlamaMLP, device: torch.device) -> None:
        super().__init__()
        # feed forward
        self.w1_: Linear = Linear(mlp.gate_proj, device)
        self.w2_: Linear = Linear(mlp.down_proj, device)
        self.w3_: Linear = Linear(mlp.up_proj, device)
        self.act_ = ACT2FN["silu"]
        # device
        self.device_ = device

    def state_dict(self) -> Dict[str, torch.nn.Module]:
        return {
            "w1_proj": self.w1_,
            "w2_proj": self.w2_,
            "w3_proj": self.w3_,
        }

    def _batch_forward(self, data: torch.Tensor, input_args: MultiLoraBatchData) -> torch.Tensor:
        w1 = self.w1_.forward(data, input_args)
        w3 = self.w3_.forward(data, input_args)
        return self.w2_.forward(self.act_(w1) * w3, input_args)

    def _lora_forward(
            self, lora_name: str, act_fn: torch.nn.Module, data: torch.Tensor) -> torch.Tensor:
        # Applying LoRA weights to FFN weights
        if lora_name in self.w1_.loras_:
            w1 = self.w1_.loras_[lora_name].forward(
                self.w1_.base_layer_.forward(data), data)
        else:
            w1 = self.w1_.base_layer_.forward(data)

        if lora_name in self.w3_.loras_:
            w3 = self.w3_.loras_[lora_name].forward(
                self.w3_.base_layer_.forward(data), data)
        else:
            w3 = self.w3_.base_layer_.forward(data)

        act_result = act_fn(w1) * w3
        if lora_name in self.w2_.loras_:
            return self.w2_.loras_[lora_name].forward(
                self.w2_.base_layer_.forward(act_result), act_result)
        else:
            return self.w2_.base_layer_.forward(act_result)


class PhiMLP(LLMFeedForward):
    def __init__(self, mlp: TransformersPhiMLP, device: torch.device) -> None:
        super().__init__()
        # feed forward
        self.fc1_: Linear = Linear(mlp.fc1, device)
        self.fc2_: Linear = Linear(mlp.fc2, device)
        self.act_ = ACT2FN["gelu_new"]
        # device
        self.device_ = device

    def state_dict(self) -> Dict[str, torch.nn.Module]:
        return {
            "fc1_proj": self.fc1_,
            "fc2_proj": self.fc2_,
        }

    def _batch_forward(self, hidden_states: torch.Tensor, input_args: MultiLoraBatchData) -> torch.Tensor:
        hidden_states = self.fc1_.forward(hidden_states, input_args)
        hidden_states = self.act_(hidden_states)
        hidden_states = self.fc2_.forward(hidden_states, input_args)
        return hidden_states

    def _lora_forward(
            self, lora_name: str, act_fn: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        if lora_name in self.fc1_.loras_:
            hidden_states = self.fc1_.loras_[lora_name].forward(
                self.fc1_.base_layer_.forward(hidden_states), hidden_states)
        else:
            hidden_states = self.fc1_.base_layer_.forward(hidden_states)

        hidden_states = act_fn(hidden_states)

        if lora_name in self.fc2_.loras_:
            hidden_states = self.fc2_.loras_[lora_name].forward(
                self.fc2_.base_layer_.forward(hidden_states), hidden_states)
        else:
            hidden_states = self.fc2_.base_layer_.forward(hidden_states)

        return hidden_states


FeedForwardClass = {
    "llama": LlamaMLP,
    "mistral": LlamaMLP,
    "qwen2": LlamaMLP,
    "phi": PhiMLP,
}


class FeedForward(torch.nn.Module):
    def __init__(self, mlp: LLMFeedForward) -> None:
        super().__init__()
        self.mlp_: LLMFeedForward = mlp
        # device
        self.device_ = mlp.device_
        # mix of experts
        self.moes_: torch.ModuleDict = {}

    def state_dict(self) -> Dict[str, Linear]:
        return self.mlp_.state_dict()

    def forward(self, data: torch.Tensor, input_args: MultiLoraBatchData,
                router_logits: List[List] = None) -> torch.Tensor:
        if len(self.moes_) == 0:
            return self.mlp_._batch_forward(data, input_args)
        else:
            return self._mixlora_forward(data, input_args, router_logits)

    # MixLoRA
    def init_moe_weight(self, in_features: int, config: MixConfig, gate: Optional[torch.Tensor] = None):
        self.moes_[config.adapter_name] = moe_layer_factory(
            in_features, config)
        if gate is None:
            torch.nn.init.normal_(
                self.moes_[config.adapter_name].gate_.weight, mean=0.0, std=config.router_init_range_)
        else:
            with torch.no_grad():
                self.moes_[config.adapter_name].gate_.weight.copy_(gate)

    def _expert_forward_callback(self, moe_name, act_fn, expert_idx, data):
        lora_name = f"moe.{moe_name}.experts.{expert_idx}"
        return self.mlp_._lora_forward(lora_name, act_fn, data)

    def _mixlora_forward(self, data: torch.Tensor, input_args: MultiLoraBatchData,
                         router_logits: List[List] = None):
        batch_size, sequence_length, hidden_dim = data.shape
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=data.dtype, device=data.device
        )
        for idx, lora_config in enumerate(input_args.lora_batch_data_config_):
            moe_name = lora_config.adapter_name_
            start_idx = lora_config.batch_start_idx_
            end_idx = lora_config.batch_end_idx_

            if moe_name in self.moes_:
                current_hidden_states, current_router_outputs = self.moes_[
                    moe_name].forward(self._expert_forward_callback, data[start_idx:end_idx])

                if router_logits is not None and current_router_outputs is not None:
                    router_logits[idx].append(current_router_outputs)
            else:
                current_hidden_states = self.mlp_._lora_forward(
                    moe_name, self.act_, data[start_idx:end_idx])

            final_hidden_states.index_add_(0, get_range_tensor(data.device, batch_size)[
                                           start_idx:end_idx], current_hidden_states)

        return final_hidden_states


def feedforward_factory(model_type: str, mlp: torch.nn.Module, device: torch.device):
    return FeedForward(FeedForwardClass[model_type](mlp, mlp, device))
