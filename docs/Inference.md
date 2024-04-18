# Inference using m-LoRA Model
## Quick start
After the fine-tuning is complete, you can use the following command to quickly start the inference:
```bash
   python mlora.py --base_model xxx --config xxx --inference
```

## Introduction
The `inference.py` file explains the inference process for the MLoRA (Multi-LoRA) model and provides an interactive interface using Gradio, enabling users to interact with the model. Below are the key steps, parameters, and principles outlined in the file:

## Steps:

1. **Model Loading and Configuration**:
   - Load a pre-trained language model using the `mlora.LLMModel.from_pretrained` method.
   - Instantiate a tokenizer using `mlora.Tokenizer` to convert input text into a format acceptable by the model.
   
```bash
 model = mlora.LLMModel.from_pretrained(base_model, device=device,
                                           attn_impl="flash_attn" if flash_attn else "eager",
                                           bits=(8 if load_8bit else (
                                                 4 if load_4bit else None)),
                                         load_dtype=torch.bfloat16 if load_16bit else torch.float32)
```
   
  

2. **Model Inference**:
   - Generate a text snippet to be predicted based on the instruction and input provided by the user.
   - Perform inference using the `mlora.generator` method to generate the predicted text.

3. **Gradio Interface**:
   - Create an interactive interface using the Gradio library, consisting of text boxes, sliders, and checkboxes for users to input instructions, parameters, and other options.
   - Upon user interaction with the interface, call the respective functions to execute model inference and display the results.

## Parameters:

- `base_model`: Name or path of the base language model.
- `template`: Specify a template for generating prompts for the text to be predicted.
- `lora_weights`: Weight file for the LoRA adapter used for fine-tuning the model.
- `load_16bit`, `load_8bit`, `load_4bit`: Number of bits used during model loading.
- `flash_attn`: Boolean indicating whether flash attention is used.
- `device`: Device on which the model runs.
- `server_name`: Server address.
- `share_gradio`: Boolean indicating whether to share the Gradio interface.

## Principles:

- Utilize the MLoRA (Multi-LoRA) model for inference, which is based on the Transformer architecture and incorporates LoRA adapters to support flexible fine-tuning and adaptive tasks.
- During the inference process, generate text snippets to be predicted based on user instructions and inputs, and employ the pre-trained language model to generate the predicted text.
- Gradio provides an interactive interface, allowing users to intuitively input instructions and parameters and view real-time predictions from the model, facilitating user interaction with the model.

The primary purpose of this file is to provide a convenient interface for users to easily perform text generation tasks using the MLoRA model and adjust model behavior by tweaking parameters.


