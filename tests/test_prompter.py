from mlora.prompter import Prompter
import unittest


class TestPrompter(unittest.TestCase):
    def setUp(self):
        template = {"description": "A demo template to experiment with.",
                    "prompt_input": "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Output:\n",
                    "prompt_no_input": "### Instruction:\n{instruction}\n\n### Output:\n",
                    "response_split": "### Output:"
                    }
        self.prompter = Prompter(template=template)

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
