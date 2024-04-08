from unittest.mock import MagicMock, patch
from mlora import Tokenizer
from transformers import PreTrainedTokenizer
import unittest
import sys
import os
CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]
config_path = CURRENT_DIR.rsplit('/', 1)[0]
sys.path.append(config_path)


class TestTokenizer(unittest.TestCase):
    @patch('transformers.AutoTokenizer.from_pretrained')
    def setUp(self, mock_tokenizer):
        self.tokenizer_instance = MagicMock(spec=PreTrainedTokenizer)
        self.tokenizer_instance.vocab_size = 100
        self.tokenizer_instance.bos_token_id = 1
        self.tokenizer_instance.eos_token_id = 20
        self.tokenizer_instance.pad_token_id = 21
        self.tokenizer_instance.unk_token_id = 22

        self.mock_encode_token = [21, 21, 21, 21, 21]
        self.mock_decode_str = "a, a, a, a, a"
        self.mock_mask_token = [0, 0, 0, 0, 0]
        self.tokenizer_instance.encode.return_value = self.mock_encode_token
        self.tokenizer_instance.decode.return_value = self.mock_decode_str

        mock_tokenizer.return_value = self.tokenizer_instance
        self.tokenizer = Tokenizer('xxx')

    def test_encode(self):
        result = self.tokenizer.encode('test')
        expect_result = []
        expect_result.append(self.tokenizer_instance.bos_token_id)
        expect_result.extend(self.mock_encode_token)
        self.assertEqual(result, expect_result)

    def test_decode(self):
        result = self.tokenizer.decode('test')
        except_result = self.mock_decode_str
        self.assertEqual(result, except_result)

    def test_mask_from(self):
        result = self.tokenizer.mask_from(self.mock_encode_token)
        self.assertEqual(result, self.mock_mask_token)


if __name__ == '__main__':
    unittest.main()
