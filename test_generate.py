import unittest
from unittest.mock import patch, Mock
from mlora import Tokenizer
import torch
from mlora.generate import GenerateConfig, gen_outputs, logits_process, generate


class TestGenerate(unittest.TestCase):

    def setUp(self):
        self.model = Mock()
        self.tokenizer = Tokenizer("/root/lry/m_lora/mlora/llama/models_hf/7B")
        self.configs = [
            GenerateConfig(adapter_name="test_adapter", prompts=["Test prompt."])
        ]
        self.max_gen_len = 128

    def test_logits_process(self):
        probs = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
        prev_tokens = torch.tensor([[1, 2, 0, 0], [3, 0, 0, 0]])
        result = logits_process(probs, prev_tokens)
        expected_shape = (2,)
        self.assertEqual(result.shape, expected_shape)

    def test_gen_outputs(self):
        configs = [GenerateConfig(adapter_name="test_adapter", prompts=["Test prompt."])]
        self.tokenizer = Tokenizer("/root/lry/m_lora/mlora/llama/models_hf/7B")
        prompts = ["Test prompt."]
        tokens = torch.tensor([[1, 2]])
        max_gen_len = 128
        result = gen_outputs(configs, self.tokenizer, prompts, tokens, max_gen_len)
        expected_result = {'test_adapter': []}
        self.assertEqual(result, expected_result)




if __name__ == "__main__":
    unittest.main()
