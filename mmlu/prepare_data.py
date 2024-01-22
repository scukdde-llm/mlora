import datasets
import json


def prepare_data(data, output):
    choices_map = ["A", "B", "C", "D"]
    prompt_data = []
    for data_point in data:
        choices = ""
        for idx, text in enumerate(data_point["choices"]):
            choices += f"{choices_map[idx]}: {text}\n"
        prompt = {
            "instruction": data_point["question"],
            "input": choices,
            "output": choices_map[data_point["answer"]]
        }
        prompt_data.append(prompt)

    with open(output, "w") as f:
        json.dump(prompt_data, f, indent=4)


mmlu = datasets.load_dataset("Stevross/mmlu", "us_foreign_policy")

prepare_data(mmlu["auxiliary_train"], "./mmlu_train.json")
prepare_data(mmlu["test"], "./mmlu_test.json")
