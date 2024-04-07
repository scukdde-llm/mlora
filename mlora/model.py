from mlora.modelargs import LoraConfig, MultiLoraBatchData, LLMModelOutput

import torch

from abc import ABCMeta
from typing import Tuple, Dict, List, Optional


class LLMFeedForward(metaclass=ABCMeta):
    def state_dict(self) -> Dict[str, torch.nn.Module]:
        pass

    def _batch_forward(self, data: torch.Tensor, input_args: MultiLoraBatchData) -> torch.Tensor:
        pass

    def _lora_forward(
            self, lora_name: str, act_fn: torch.nn.Module, data: torch.Tensor) -> torch.Tensor:
        pass


class LLMOutput(metaclass=ABCMeta):
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        pass

    def loss(self,
             input_ids: torch.Tensor,
             output_logits: torch.Tensor,
             labels: List[List[int]]) -> torch.Tensor:
        pass

    def state_dict(self):
        return {}


class LLMModel(metaclass=ABCMeta):
    @classmethod
    def init_lora_layer_weight(self, config: LoraConfig, weight: Optional[Dict[str, torch.Tensor]]):
        pass

    @classmethod
    def load_adapter_weight(self, path: str, adapter_name: str = None):
        pass

    @classmethod
    def get_lora_weight_dict(self, lora_name: str) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        pass

    @classmethod
    def sequential_module(self) -> torch.nn.Sequential:
        pass

    @classmethod
    def get_generate_paramas(self) -> Dict[str, any]:
        pass

    @classmethod
    def forward(self, input: MultiLoraBatchData,
                labels: List[List[int]] = None) -> List[LLMModelOutput]:
        pass
