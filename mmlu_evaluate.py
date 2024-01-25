import datasets
import logging
import mlora
import torch
import math
import fire

from typing import List

choices_map = ["A", "B", "C", "D"]


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_prompt(data_point, with_answer=True):
    p = data_point["question"]
    for idx, text in enumerate(data_point["choices"]):
        p += f"\n{choices_map[idx]}. {text}"
    if with_answer:
        p += "\nAnswer: " + choices_map[data_point["answer"]] + "\n\n"
    else:
        p += "\nAnswer: "
    return p


def prepare_data(tokenizer: mlora.Tokenizer,
                 category: str,
                 dev_data: datasets.Dataset,
                 test_data: datasets.Dataset,
                 k_shots=5,
                 max_seq_len=2048):

    sequence_lengths = []
    batch_tokens = []
    batch_labels = []
    atten_masks = []

    max_tokens_len = 0
    tokens = None
    for test_data_point in test_data:
        test_prompt = format_prompt(test_data_point, False)
        dev_prompt = "The following are multiple choice questions (with answers) about"
        dev_prompt += format_subject(category) + ".\n\n"
        k = k_shots
        for dev_data_point in dev_data:
            k -= 1
            prompt = format_prompt(dev_data_point)
            input_ids = tokenizer.encode(
                dev_prompt + prompt + test_prompt, True, False)
            if len(input_ids) <= max_seq_len:
                tokens = input_ids
                dev_prompt += prompt
            else:
                k = 0

            if k <= 0:
                break

        print(dev_prompt + test_prompt)
        max_tokens_len = max(len(tokens), max_tokens_len)
        batch_tokens.append(tokens)
        batch_labels.append(test_data_point["answer"])

    logging.info(f"Max tokens: {max_tokens_len}/{max_seq_len}")
    if max_tokens_len < max_seq_len:
        max_seq_len = math.ceil(max_tokens_len / 8) * 8
    logging.info(f"Max sequence length: {max_seq_len}")

    for tokens in batch_tokens:
        sequence_lengths.append(len(tokens))
        while len(tokens) < max_seq_len:
            tokens.append(tokenizer.pad_id_)
        atten_masks.append(tokenizer.attention_mask(tokens))

    return sequence_lengths, batch_tokens, atten_masks, batch_labels


@torch.inference_mode()
def evaluate(category: str,
             model_name: str,
             adapter_names: List[str],
             batch_size: int = 2,
             device: str = "cuda:0"):
    # prepare data
    tokenizer = mlora.Tokenizer(model_name)

    mmlu = datasets.load_dataset("cais/mmlu", category)

    sequence_lengths, batch_tokens, atten_masks, batch_labels = prepare_data(
        tokenizer, category, mmlu["dev"], mmlu["test"], 5, 2048)

    # load adapters
    model = mlora.LlamaModel.from_pretrained(
        path=model_name,
        device=device,
        bits=None,
        load_dtype=torch.bfloat16
    )

    results = {}

    for name in adapter_names:
        logging.info(f"Loading adapter {name}")
        results[name] = []
        model.load_adapter_weight(name)

    # prepare for evaluate
    sequence_lengths = torch.tensor(
        sequence_lengths, dtype=torch.long, device=model.device_)

    label_indices = [0]*len(choices_map)
    for idx, text in enumerate(choices_map):
        ids = tokenizer.encode(" " + text, False, False)
        label_indices[idx] = ids[-1]
    label_indices = torch.tensor(
        label_indices, dtype=torch.long, device=model.device_)

    start_pos = 0
    while start_pos < len(batch_tokens):
        end_pos = min(len(batch_tokens), start_pos + batch_size)
        logging.info(f"evaluation step: {start_pos}/{len(batch_tokens)}")
        bsz = end_pos - start_pos
        torch.cuda.empty_cache()
        batch_data_config = []
        batch_start_idx = 0
        for name in adapter_names:
            batch_data_config.append(mlora.LoraBatchDataConfig(
                adapter_name_=name,
                batch_start_idx_=batch_start_idx,
                batch_end_idx_=batch_start_idx + bsz,
            ))
            batch_start_idx += bsz

        input_args = mlora.MultiLoraBatchData(
            lora_batch_data_config_=batch_data_config,
            batch_tokens_=batch_tokens[start_pos:end_pos]*len(adapter_names),
            attention_masks_=atten_masks[start_pos:end_pos]*len(adapter_names),
            gradient_checkpoint_=False,
            inference_seq_pos_=0,
        )

        outputs = model.forward(input_args)

        labels = torch.tensor(
            batch_labels[start_pos:end_pos], dtype=torch.long, device=model.device_)

        for output in outputs:
            logits = output.logits
            logits = logits[torch.arange(
                bsz, device=logits.device), sequence_lengths[start_pos:end_pos]]
            logits = logits[:, label_indices]
            logits = logits.softmax(-1).argmax(-1)
            result = (logits == labels).int().tolist()
            results[output.adapter_name].extend(result)

        for name, result in results.items():
            acc = sum(result)/len(result)
            logging.info(f"    {name} accuracy: {acc}")

        start_pos = end_pos


log_handlers = [logging.StreamHandler()]
logging.basicConfig(format='[%(asctime)s] m-LoRA: %(message)s',
                    level=logging.INFO,
                    handlers=log_handlers,
                    force=True)

if __name__ == "__main__":
    fire.Fire(lambda category, model_name: evaluate(
        category, model_name, ["alpaca-lora-7b", "mixlora-7b"]))
