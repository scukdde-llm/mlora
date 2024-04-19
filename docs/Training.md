# Fine-Tuning Preparation Guide
## Table contens
- [requirement](Training.md#requirement)
- [Prepare the config](Training.md#prepare-the-config)
- [Train](Training.md#train)
    - [Train using mlora.py](Training.md#train-using-mlorapy)
    - [train using launch.py](Training.md#train-using-launchpy)
- [result](Training.md#result)

## Requirement
- pre-trained LLM
    - A stable network facilitates our retrieval of model files from [huggingface.co](https://huggingface.co)
    - You can also download the relevant models through the following LLM support list.
- training dataset.


|         | Model                                                    | # Parameters       |
|---------|----------------------------------------------------------|--------------------|
| &check; | [LLaMA](https://github.com/facebookresearch/llama)       | 7B/13B/33B/65B     |
| &check; | [LLaMA-2](https://huggingface.co/meta-llama)             | 7B/13B/70B         |
| &check; | [Qwen-2](https://qwenlm.github.io)                       | 1.8B/4B/7B/14B/72B |
| &check; | [Mistral](https://mistral.ai)                            | 7B                 |
| &check; | [Gemma](https://ai.google.dev/gemma/docs)                | 2B/7B              |
| &check; | [Phi-2](https://huggingface.co/microsoft/phi-2)          | 2.7B               |

**If you have already selected a pre-trained model supported by this project and have your own training dataset, you can proceed with the training using the following steps:**
## Prepare the Config

You can use the `launch.py gen` to generate the corresponding instruction file for the config.
```
python launch.py gen \
    --template lora \
    --tasks yahma/alpaca-cleaned
```
Here are some important parameters of the gen command:
```
--template
                Specify the adapter strategy to be used, such as LORA or MixLORA.
--tasks
                Specify the type of task for fine-tuning, such as CoLA or PiQA.
--adapter_name
                Specify the adapter to be used
--file_name
                Specify the path and name of the configuration file to be output.
```

If you want to learn more about the detailed parameters for generating a config file using the `gen` , you can use the `help` command to view.

**Please note that you can use the `launch.py avail` to see the supported task types.**\
Here are examples for supported task types:
```
glue:cola
glue:mnli
...
piqa
```

## Train
**When you have selected the pre-trained model for fine-tuning, training dataset, and the configuration file, we provide you with two different ways to start training.**

### Train using launch.py


**In `launch.py`, you need to use the `train` or `run` (which executes both training and evaluation simultaneously) command to perform fine-tuning.**
```
python launch.py train / run
    --base_model yourmodelpath \
    --config yourloraconfig 
```
The following are some command parameters for `run`, with default values provided in [ ].\
You can use the `help` command to see more details.

```
--base_model       model name or path
--config           [mlora.json]          Configuration file parameters
--load_adapter     [false]              Determines whether to load an existing adapter
--cuda_device      [0]                  Sets the device
...
--attn_impl        [eager]              Sets the attention type
--dtype            [bf16]               Sets the numerical precision to fp16 or fp32
```

### Train using mlora.py
You can choose to run `mlora.py` and pass parameters to execute the training.
```
python mlora.py \
    --base_model yourmodelpath \
    --config yourloraconfig \
    --bf16
```
You can use `--help` to query more command-line parameters about the `mlora.py` file.\
The following are some parameters from the `--help`:
```
--base_model BASE_MODEL
                        Path to or name of base model
--inference           The inference mode (just for test)
--evaluate            The evaluate mode (just for test)
--disable_prompter    Disable prompter when inference
--load_adapter        Load adapter from file instead of init randomly
...
--disable_log         Disable logging.
--log_file LOG_FILE   Save log to specific file
--verbose             Show extra informations such as parameters
--overwrite           Overwrite adapter model when older one existed
```


## result
After completing the entire training based on the configuration file, **an adapter folder corresponding to the training will be generated in the working directory**. This will be loaded in subsequent inference tasks. Using adapters for fine-tuning models will better meet your expectations.

**If you want to use your trained adapter for inference and evaluation, please refer to the following documentation：**

[Inference Guide](Inference.md)\
[Evaluation Guide](Evaluation.md)