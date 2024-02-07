import datasets
import logging
import random
import mlora
import torch
import json
import math
import fire
import csv

from typing import List

choices_map = ["1", "2", "3", "4", "A", "B", "C", "D", "E"]
choices2id = {text: idx for idx, text in enumerate(choices_map)}


def prepare_data(tokenizer: mlora.Tokenizer,
                 data: datasets.Dataset,
                 max_seq_len=2048,
                 batch_padding=True):

    sequence_lengths = []
    batch_tokens = []
    batch_labels = []
    atten_masks = []

    max_tokens_len = 0
    tokens = None
    for data_point in data:
        prompt_str = "Question: " + data_point["question"]
        choices = data_point["choices"]
        for label, text in zip(choices["label"], choices["text"]):
            prompt_str += f" ({label}) {text}"
        prompt_str += "\nAnswer:"
        tokens = tokenizer.encode(prompt_str, bos=True, eos=False)
        max_tokens_len = max(len(tokens), max_tokens_len)
        batch_tokens.append(tokens)
        batch_labels.append(choices2id[data_point["answerKey"]])

    if batch_padding:
        logging.info(f"Max tokens: {max_tokens_len}/{max_seq_len}")
        if max_tokens_len < max_seq_len:
            max_seq_len = math.ceil(max_tokens_len / 8) * 8
        logging.info(f"Max sequence length: {max_seq_len}")

    for tokens in batch_tokens:
        if batch_padding:
            sequence_lengths.append(len(tokens) - 1)
            while len(tokens) < max_seq_len:
                tokens.append(tokenizer.pad_id_)
        else:
            sequence_lengths.append(-1)
        atten_masks.append(tokenizer.attention_mask(tokens))

    return sequence_lengths, batch_tokens, atten_masks, batch_labels


@torch.inference_mode()
def evaluate(subject: str,
             tokenizer: mlora.Tokenizer,
             model: mlora.LlamaModel,
             adapter_names: List[str],
             batch_size: int = 2,
             max_seq_len: int = 2048):
    # prepare data

    ai2_arc = datasets.load_dataset("allenai/ai2_arc", subject)

    sequence_lengths, batch_tokens, atten_masks, batch_labels = prepare_data(
        tokenizer, ai2_arc["test"], max_seq_len, batch_size > 1)

    # load adapters

    results = {}

    for name in adapter_names:
        results[name] = []

    # prepare for evaluate
    sequence_lengths = torch.tensor(
        sequence_lengths, dtype=torch.long, device=model.device_)

    label_indices = [0] * len(choices_map)
    for idx, text in enumerate(choices_map):
        ids = tokenizer.encode(text, False, False)
        label_indices[idx] = ids[-1]
    label_indices = torch.tensor(
        label_indices, dtype=torch.long, device=model.device_)

    start_pos = 0
    while start_pos < len(batch_tokens):
        end_pos = min(len(batch_tokens), start_pos + batch_size)
        logging.info(f"evaluation step: {start_pos}/{len(batch_tokens)}")
        bsz = end_pos - start_pos
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
            batch_tokens_=batch_tokens[start_pos:end_pos] * len(adapter_names),
            attention_masks_=atten_masks[start_pos:end_pos] *
            len(adapter_names),
            gradient_checkpoint_=False,
            inference_seq_pos_=-1 if batch_size > 1 else 0,
        )

        outputs = model.forward(input_args)
        avg = list(0 for _ in range(model.layers_[
                   0].ffn_.moes_["arc_mixlora"].experts_))
        for layer in model.layers_:
            for idx, val in enumerate(layer.ffn_.moes_["arc_mixlora"].profiler_):
                avg[idx] += val
        for idx, val in enumerate(avg):
            print(f"Expert {idx}, Load = {val/32}")

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
            acc = sum(result) / len(result)
            logging.info(f"    {name} accuracy: {acc}")

        start_pos = end_pos

    return results


model_dtypes = {
    "4bit": {"bits": 4, "load_dtype": torch.float32},
    "8bit": {"bits": 8, "load_dtype": torch.float32},
    "16bit": {"load_dtype": torch.bfloat16},
}


def do_evaluate(model_name: str,
                model_dtype: str,
                adapter_names: List[str],
                batch_size: int = 2,
                device: str = "cuda:0"):
    tokenizer = mlora.Tokenizer(model_name)
    model = mlora.LlamaModel.from_pretrained(
        model_name, device=device, **model_dtypes[model_dtype])
    for name in adapter_names:
        logging.info(f"Loading adapter {name}")
        model.load_adapter_weight(name)

    results = evaluate("ARC-Easy", tokenizer, model,
                       adapter_names, batch_size, model.max_seq_len_)

    for name, result in results.items():
        acc = sum(result) / len(result)
        logging.info(f"{name} accuracy: {acc}")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def main(config: str):
    setup_seed(66)
    log_handlers = [logging.StreamHandler()]
    logging.basicConfig(format='[%(asctime)s] m-LoRA: %(message)s',
                        level=logging.INFO,
                        handlers=log_handlers,
                        force=True)
    with open(config, 'r', encoding='utf8') as fp:
        config_obj = json.load(fp)
    do_evaluate(**config_obj)


if __name__ == "__main__":
    fire.Fire(main)
