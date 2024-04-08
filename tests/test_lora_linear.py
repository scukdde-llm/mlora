from mlora.lora_linear import MultiLoraBatchData, LoraConfig, Linear
from mlora.modelargs import LoraBatchDataConfig
import unittest
import torch


class TestLoraLinear(unittest.TestCase):
    def setUp(self) -> None:
        self.in_dim = 20
        self.out_dim = 30
        self.mock_weight = torch.nn.Linear(self.in_dim, self.out_dim)
        self.device = 'cpu'
        self.linear = Linear(self.mock_weight, self.device)
        self.mock_data = torch.randn(40, 20)

    def test_forward_empty_loras(self) -> None:
        result = self.linear.forward(self.mock_data, {})
        except_result = self.mock_weight.forward(self.mock_data)
        self.assertTrue(torch.equal(result, except_result))

    def test_init_lora_weight(self) -> None:
        mock_lora_config1 = LoraConfig(adapter_name="test_lora1",
                                       device=self.device,
                                       lora_r_=8,
                                       lora_alpha_=16,
                                       lora_dropout_=0.05)
        mock_lora_config2 = LoraConfig(adapter_name="test_lora2",
                                       device=self.device,
                                       lora_r_=8,
                                       lora_alpha_=16,
                                       lora_dropout_=0.05)
        self.linear.init_lora_weight(mock_lora_config1)
        self.linear.init_lora_weight(mock_lora_config2)
        self.assertEqual(self.linear.device_, torch.device(self.device))
        self.assertTrue("test_lora1" in self.linear.loras_)
        self.assertTrue("test_lora2" in self.linear.loras_)

    def test_lora_forward_valid_loras(self) -> None:
        mock_lora_config1 = LoraConfig(adapter_name="test_lora1",
                                       device=self.device,
                                       lora_r_=8,
                                       lora_alpha_=16,
                                       lora_dropout_=0.05)
        mock_data_config1 = LoraBatchDataConfig(adapter_name_="test_lora1",
                                                batch_start_idx_=0,
                                                batch_end_idx_=20)
        self.linear.init_lora_weight(mock_lora_config1)
        mock_lora_b = self.linear.loras_["test_lora1"].lora_b_.weight
        torch.nn.init.uniform(mock_lora_b)
        self.linear.loras_["test_lora1"].lora_b_.weight = mock_lora_b

        mock_mutiloraconfig = MultiLoraBatchData(lora_batch_data_config_=[mock_data_config1])

        result = self.linear.forward(self.mock_data, mock_mutiloraconfig)
        expect_result = self.mock_weight.forward(self.mock_data)

        self.assertEqual(result.shape, expect_result.shape)
        self.assertTrue(not torch.equal(result[0:20], expect_result[0:20]))
        self.assertTrue(torch.equal(result[20:60], expect_result[20:60]))


if __name__ == '__main__':
    unittest.main()
