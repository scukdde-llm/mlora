import mlora
import torch
import fire

instruction = "### Instruction:\nCould you provide an introduction to m-LoRA?\n\n### Output:\n"
output = "m-LoRA, short for Multi-LoRA Fine-Tune, stands as an open-source framework designed for fine-tuning Large Language Models (LLMs) using the efficient multiple LoRA/QLoRA methods."


def main(base_model: str,
         lora_weights: str = None,
         device: str = f"{mlora.get_backend().device_name()}:0"):

    tokenizer = mlora.Tokenizer(base_model)
    token_instruction = tokenizer.encode(instruction)
    token_input = tokenizer.encode(instruction + output)
    min_tokens_len = len(token_instruction)
    total_len = len(token_input)
    while len(token_instruction) < len(token_input):
        token_instruction.append(tokenizer.pad_id_)

    model = mlora.LLMModel.from_pretrained(base_model,
                                           device=device,
                                           attn_impl="eager",
                                           load_dtype=torch.bfloat16)

    inference_adapter_name = model.load_adapter_weight(
        lora_weights if lora_weights else "default")
    model.init_lora_layer_weight(mlora.LoraConfig(adapter_name="train",
                                                  lora_r_=8, lora_alpha_=16, lora_dropout_=0.05,
                                                  target_modules_={"q_proj": True, "k_proj": True, "v_proj": True, "o_proj": True}), None)

    batch_data_config = [
        mlora.LoraBatchDataConfig(
            adapter_name_=inference_adapter_name, batch_start_idx_=0, batch_end_idx_=1),
        mlora.LoraBatchDataConfig(
            adapter_name_="train", batch_start_idx_=1, batch_end_idx_=2),
    ]

    optimizer = torch.optim.AdamW(
        params=model.get_lora_weight_dict("train").values(), lr=1e-4)

    tokens = torch.tensor([token_instruction, token_input],
                          dtype=torch.int64, device=device)

    for cur_pos in range(min_tokens_len, total_len):
        attention_masks = [tokenizer.mask_from(t) for t in tokens.tolist()]

        input_data = mlora.MultiLoraBatchData(
            lora_batch_data_config_=batch_data_config,
            batch_tokens_=tokens,
            batch_labels_=tokens,
            attention_masks_=attention_masks,
            gradient_checkpoint_="recompute")

        outputs = model.forward(input_data)
        inference_output: mlora.LLMModelOutput = outputs[0]
        next_token = mlora.generator.logits_process(inference_output.logits[:, cur_pos - 1],
                                                    tokens[:0, :cur_pos])
        tokens[0, cur_pos] = next_token
        print(f"adapter: {inference_adapter_name}, output = " +
              tokenizer.decode(tokens[0, :cur_pos]))

        train_output: mlora.LLMModelOutput = outputs[1]
        print(f"adapter: train, loss: {train_output.loss}")
        train_output.loss.backward()
        optimizer.step()
        optimizer.zero_grad()


if __name__ == "__main__":
    fire.Fire(main)
