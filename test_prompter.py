import unittest
from mlora import Prompter

class TestPrompter(unittest.TestCase):
    def setUp(self):
        self.prompter = Prompter("/root/lry/m_lora/mlora/template/template_demo.json")

    def test_generate_prompt_with_input(self):
        instruction = "Please share a healthy recipe"
        input_val = "Quinoa Salad"
        expected_output = "### Instruction:\nPlease share a healthy recipe\n\n### Input:\nQuinoa Salad\n\n### Output:\n"
        
        output = self.prompter.generate_prompt(instruction, input_val)
        
        self.assertEqual(output, expected_output)
    
    def test_generate_prompt_without_input(self):
        instruction = "Please share a healthy recipe"
        expected_output = "### Instruction:\nPlease share a healthy recipe\n\n### Output:\n"
        
        output = self.prompter.generate_prompt(instruction)
        
        self.assertEqual(output, expected_output)
    
    def test_get_response(self):
        output = "### Output:Here's a healthy recipe"
        expected_response = "Here's a healthy recipe"
        
        response = self.prompter.get_response(output)
        
        self.assertEqual(response, expected_response)

if __name__ == "__main__":
    unittest.main()
