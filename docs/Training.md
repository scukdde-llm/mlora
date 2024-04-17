# Fine-tuning Preparation Guide
## table contens
- [requirement](Training.md#requirement)
- [Prepare the config](Training.md#prepare-the-config)
- [Train](Training.md#train)
    - [Train using mlora.py](Training.md#train-using-mlorapy)
    - [train using launch.py](Training.md#train-using-launchpy)
- [result](Training.md#result)

## requirement
- pre-trained LLM
    - A stable network facilitates our retrieval of model files from [huggingface.co](https://huggingface.co)
    - or you can downloaded the pre-trained LLM locally.
- clean training dataset.





**If you have already selected a pre-trained model supported by this project and have your own training set, you can proceed with the training using the following steps:**
## Prepare the Config

You can use the `launch.py gen` to generate the corresponding instruction file for the config.
```
python launch.py gen \
    --template lora \
    --tasks yahma/alpaca-cleaned
```
More command-line parameters for the `gen` are as follows:
```
--template              lora, mixlora, etc.
--tasks                 task names separate by ';'
--adapter_name          default is task name
--file_name             [mlora.json]

# The default values for the subsequent parameters will be provided by the .json with the same name in the .launcher folder.

--cutoff_len            
--save_step
--warmup_steps
--learning_rate
--loraplus_lr_ratio
--batch_size
--micro_batch_size
--test_batch_size
--num_epochs            
--use_dora
--use_rslora
--group_by_length
```
**Please note that you can use the `launch.py avail` to see the supported task types.**\
Here are examples for supported task types:
```
glue:cola
glue:mnli
glue:mrpc
glue:qnli
glue:qqp
glue:rte
glue:sst2
glue:wnli
arc-e
arc-c
boolq
obqa
piqa
```


## Train
**When you have selected the pre-trained model for fine-tuning, training dataset, and the configuration file, we provide you with two different ways to start training.**

### train using mlora.py
You can choose to run `mlora.py` and pass parameters to execute the training.
```
python mlora.py \
    --base_model yourmodelpath \
    --config yourloraconfig \
    --bf16
```
You can use `--help` to query more command-line parameters about the `mlora.py` file.
```
--base_model BASE_MODEL
                        Path to or name of base model
--inference           The inference mode (just for test)
--evaluate            The evaluate mode (just for test)
--disable_prompter    Disable prompter when inference
--load_adapter        Load adapter from file instead of init randomly
--disable_adapter     Disable the adapter modules
--attn_impl ATTN_IMPL 
                        Specify the implementation of attention
--use_swa             Use sliding window attention (requires flash
                        attention)
--fp16                Load base model in float16 precision
--bf16                Load base model in bfloat16 precision
--tf32                Use tfloat32 instead of float32 if available
--load_8bit           Load base model with 8bit quantization
--load_4bit           Load base model with 4bit quantization
--device DEVICE       Specify which GPU to be used
--config CONFIG       Path to finetune configuration
--seed SEED           Random seed in integer, default is 42
--dir DIR             Path to read or save checkpoints
--disable_log         Disable logging.
--log_file LOG_FILE   Save log to specific file
--verbose             Show extra informations such as parameters
--overwrite           Overwrite adapter model when older one existed
--debug               Enabling debugging mode
--deterministic       Use deterministic algorithms to improve the
```
### train using launch.py
If you're confused by the above parameters for mlora, \
you can also choose to use the `launch.py`, where most parameters are set to default values. 

**In `launch.py`, you need to use the `train` or `run` (which executes both training and evaluation simultaneously) command to perform fine-tuning.**
```
python launch.py train / run
    --base_model yourmodelpath \
    --config yourloraconfig 
```
The following are the command parameters for launch.py, with default values provided in [ ]

```
--base_model       model name or path
--config           [mlora.json]          Configuration file parameters
--load_adapter     [false]              Determines whether to load an existing adapter
--random_seed      [42]                 Sets the random seed
--cuda_device      [0]                  Sets the device
--log_file         [mlora.log]          Sets the log file name
--overwrite        [false]              
                    Determines whether to overwrite an existing adapter with the same name
--attn_impl        [eager]              Sets the attention type
--quantize         [none]               Quantizes parameters to 4-bit or 8-bit
--dtype            [bf16]               Sets the numerical precision to fp16 or fp32
--tf32             [false]              Determines whether to use 32-bit computation precision
```

## result
After completing the entire training based on the configuration file, **an adapter folder corresponding to the training will be generated in the working directory**. This will be loaded in subsequent inference tasks. Using adapters for fine-tuning models will better meet your expectations.
