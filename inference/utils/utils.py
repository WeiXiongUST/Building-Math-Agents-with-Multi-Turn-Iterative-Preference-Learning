import json
import os
import random
from pathlib import Path
from typing import Any, Iterable, Union

import numpy as np


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()


def save_jsonl(samples, save_path):
    # ensure path
    folder = os.path.dirname(save_path)
    os.makedirs(folder, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    print("Saved to", save_path)


def lower_keys(example):
    new_example = {}
    for key, value in example.items():
        if key != key.lower():
            new_key = key.lower()
            new_example[new_key] = value
        else:
            new_example[key] = value
    return new_example


def construct_prompt(args, example):
    if args.prompt_type == "tora":
        if "gemma" in args.model_name_or_path:
            full_prompt = f"<bos><start_of_turn>user\n{example['question']}<end_of_turn>\n<start_of_turn>model\n"
        elif "mistral" in args.model_name_or_path:
            full_prompt = f"<s> [INST] {example['question']} [/INST]"
        elif "deepseek" in args.model_name_or_path:
            full_prompt = f"User: {example['question']}\nPlease integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{{}}.\n\nAssistant: "
        elif "llama3" in args.model_name_or_path:
            full_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nPlease integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{{}}.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            raise NotImplementedError(args.prompt_type + "and " + args.model_name_or_path)
    elif args.prompt_type == "cot":
        if "gemma" in args.model_name_or_path:
            full_prompt = f"<bos><start_of_turn>user\n{example['question']}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<end_of_turn>\n<start_of_turn>model\n"
        elif "mistral" in args.model_name_or_path:
            full_prompt = f"<s> [INST] {example['question']}\nPlease reason step by step, and put your final answer within \\boxed{{}}. [/INST]"
        elif "deepseek" in args.model_name_or_path:
            full_prompt = f"User: {example['question']}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\nAssistant: "
        elif "llama3" in args.model_name_or_path:
            full_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{example['question']}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            raise NotImplementedError(args.prompt_type + "and " + args.model_name_or_path)

    return full_prompt


key_map = {
    "gt": "Ground Truth",
    "pred": "Prediction",
    "gt_cot": "Reference CoT",
    "score": "Score",
}


def show_sample(sample, print_all_preds=False):
    print("==" * 20)
    for key in ["idx", "type", "level", "dataset"]:
        if key in sample:
            # capitalize
            print("{}: {}".format(key[0].upper() + key[1:], sample[key]))
    print("Question:", repr(sample["question"]))
    if "code" in sample:
        if print_all_preds:
            for code in sample["code"]:
                print("-" * 20)
                print("code:", code)
            print("Execution:", sample["report"])
        else:
            print("Solution:\n", sample["code"][0])
            print("Execution:", sample["report"][0])
    if "pred" in sample:
        print("Prediction:", repr(sample["pred"][0]))
    for key in ["gt", "score", "unit", "gt_cot"]:
        if key in sample:
            _key = key_map.get(key, key)
            print("{}: {}".format(_key, repr(sample[key])))
    print()
