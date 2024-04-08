from mlora.feed_forward import FeedForward, LLMModelArgs, Linear
from unittest.mock import MagicMock, patch
import unittest
import torch


class TestFeedForward(unittest.TestCase):
    def setUp(self):
        self.w1 = MagicMock(spec=Linear)
        self.w2 = MagicMock(spec=Linear)
        self.w3 = MagicMock(spec=Linear)
        self.args = LLMModelArgs(device_='cpu')

        self.feed_forward = FeedForward(self.w1, self.w2, self.w3, self.args)

    def test_forward_without_moes(self):
        data = torch.randn(10, 20)
        input_args = MagicMock()
        self.feed_forward.moes_ = {}

        self.w1.forward.return_value = data
        self.w2.forward.return_value = data
        self.w3.forward.return_value = data

        output = self.feed_forward.forward(data, input_args)

        self.assertIsNotNone(output)
        self.w1.forward.assert_called_once()
        self.w2.forward.assert_called_once()
        self.w3.forward.assert_called_once()

    @patch("torch.nn.functional.silu", autospec=True)
    def test_lora_forward(self, mock_silu):
        data = torch.randn(10, 20)
        act_fn = torch.nn.functional.silu
        lora_name = 'test_lora'
        lora_name_absent = 'absent_lora'

        self.w1.loras_ = {lora_name: MagicMock()}
        self.w1.base_layer_ = MagicMock()
        self.w2.loras_ = {lora_name: MagicMock()}
        self.w2.base_layer_ = MagicMock()
        # mock that not hava lora
        self.w3.loras_ = {lora_name_absent: MagicMock()}
        self.w3.base_layer_ = MagicMock()

        self.w1.loras_[lora_name].forward.return_value = data
        self.w1.base_layer_.forward.return_value = data
        self.w2.loras_[lora_name].forward.return_value = data
        self.w2.base_layer_.forward.return_value = data
        self.w3.loras_[lora_name_absent].forward.return_value = data
        self.w3.base_layer_.forward.return_value = data

        output = self.feed_forward._lora_forward(lora_name, act_fn, data)

        self.assertIsNotNone(output)
        self.assertEqual(mock_silu.call_count, 1)
        self.w1.loras_[lora_name].forward.assert_called_once()
        self.w2.loras_[lora_name].forward.assert_called_once()
        self.w3.loras_[lora_name_absent].forward.assert_not_called()
        self.w3.base_layer_.forward.assert_called_once()


# 运行测试
if __name__ == '__main__':
    unittest.main()
