from mlora.prompter import Prompter
import unittest
import sys
import os
CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]
config_path = CURRENT_DIR.rsplit('/', 1)[0]
sys.path.append(config_path)


class TestPrompter(unittest.TestCase):
    def setUp(self):
        self.prompter = Prompter(config_path + r"/template/template_demo.json")

    def test_generate_prompt_with_input(self):
        mock_test = "### Instruction:\nplease enter a sentence\n\n### Input:\nhello world!\n\n### Output:\n"
        output = self.prompter.generate_prompt("please enter a sentence", "hello world!")
        self.assertEqual(output, mock_test)

    def test_generate_prompt_without_input(self):
        mock_test = "### Instruction:\nplease enter a sentence\n\n### Output:\n"
        instruction = "please enter a sentence"
        output = self.prompter.generate_prompt(instruction)
        self.assertEqual(output, mock_test)

    def test_get_response(self):
        output = "### Output:This is the response"
        response = self.prompter.get_response(output)
        expected_response = "This is the response"
        self.assertEqual(response, expected_response)


if __name__ == "__main__":
    unittest.main()
