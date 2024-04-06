import unittest
from typing import List, Union
from mlora.tokenizer import Tokenizer
from transformers import AutoTokenizer
class TestTokenizer(unittest.TestCase):

    def setUp(self):
        self.tokenizer = Tokenizer("/root/lry/m_lora/mlora/llama/models_hf/7B")

    def test_encode(self):
        self.assertEqual(self.tokenizer.encode("hello world", bos=True, eos=True), [1,22172,3186,2])
        self.assertEqual(self.tokenizer.encode("hello world", bos=False, eos=False), [22172, 3186])
        self.assertEqual(self.tokenizer.encode("apple", bos=False, eos=True), [26163,2])
        self.assertEqual(self.tokenizer.encode("apple", bos=True, eos=False), [1,26163])
        self.assertEqual(self.tokenizer.encode("apple"), [1,26163])

    def test_decode(self):
        self.assertEqual(self.tokenizer.decode([22172, 3186]), "hello world")
        self.assertEqual(self.tokenizer.decode([26163]), "apple")
        

    def test_mask_from(self):
        self.assertEqual(self.tokenizer.mask_from([0, 31414, 232, 328, 2005, 7]), [1, 0, 0, 0, 0,0])
        self.assertEqual(self.tokenizer.mask_from([31414, 232, 328, 2005, 7]), [0, 0, 0, 0, 0])

if __name__ == '__main__':
    unittest.main()
