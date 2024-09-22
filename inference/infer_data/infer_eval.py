"""
This scrip support is adapted from Tora project and supports multi-rounds vllm inference.
The inference is formulated as a multi-turn chat and the model should be registered as a server by scripts/register_server.sh first.
"""

import argparse
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests
from eval.evaluate import evaluate
from tqdm import tqdm
from utils.data_loader import load_data
from utils.parser import *
from utils.python_executor import PythonExecutor
from utils.utils import construct_prompt, load_jsonl, save_jsonl, set_seed
from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default="gsm8k", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="tora", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=1024, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--ports", action='append', default=[])
    parser.add_argument("--horizon", default=6, type=int) # the maximal number of tool calls
    parser.add_argument("--eval", default=False, type=bool) 

    args = parser.parse_args()
    args.top_p = 1 if args.temperature == 0 else args.top_p  # top_p must be 1 when using greedy sampling (vllm)
    return args


def prepare_data(args):
    examples = load_data(args.data_name, args.split, args.data_dir)

    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        examples = random.sample(examples, args.num_test_sample)
    elif args.num_test_sample == -1:
        args.num_test_sample = len(examples)

    # shuffle
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    # select start and end
    if args.end == -1:
        args.end = len(examples)
    examples = examples[args.start : args.end]

    # get out_file name
    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    model_name = "/".join(args.model_name_or_path.split("/")[-2:])
    out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    out_file = f"{args.output_dir}/{model_name}/{args.data_name}/{out_file_prefix}_s{args.start}_e{args.end}_{dt_string}.jsonl"
    os.makedirs(f"{args.output_dir}/{model_name}/{args.data_name}", exist_ok=True)

    # load all processed samples
    # find the files in e.g. ./output/gemma2/math/
    processed_files = [
        f
        for f in os.listdir(f"{args.output_dir}/{model_name}/{args.data_name}/")
        if f.endswith(".jsonl") and f.startswith(out_file_prefix)
    ]
    processed_samples = []
    for f in processed_files:
        processed_samples.extend(list(load_jsonl(f"{args.output_dir}/{model_name}/{args.data_name}/{f}")))

    # dedepulicate
    processed_samples = {sample["idx"]: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    total_examples = len(examples)
    # if example has been inferenced with the same seed, temperature, and model, we skip them
    examples = [example for example in examples if example["idx"] not in processed_idxs]
    print(f"Idx {args.start} - {args.end}: Remain {len(examples)}/{total_examples} samples.")
    if len(examples) == 0:
        pass
    else:
        print(examples[0])
    return examples, processed_samples, out_file


def main(args):
    ports = args.ports
    examples, processed_samples, out_file = prepare_data(args)
    # init python executor
    executor = PythonExecutor(get_answer_from_stdout=True)
    print(args.prompt_type)

    SamplingParams.seed = args.seed
    # load model and determine the number of gpus used
    if "gemma" in args.model_name_or_path:
        stop_tokens = ["<end_of_turn>", "<eos>", "```output", "<start_of_turn>"]
    elif "mistral" in args.model_name_or_path:
        stop_tokens = ["<s>", "</s>", "[INST]", "```output"]
    elif "deepseek" in args.model_name_or_path:
        stop_tokens = ["<｜end▁of▁sentence｜>", "User", "```output"]
    elif "llama3" in args.model_name_or_path:
        stop_tokens = ["<|eot_id|>", "<|start_header_id|>user", "```output"]
    else:
        raise NotImplementedError(args.prompt_type + "and " + args.model_name_or_path)
    default_args = {
        "use_beam_search": False,
        "n": 1,
        "temperature": args.temperature,
        "max_tokens": 1024,
        "seed": args.seed,
        "top_p": 1.0,
        "top_k": -1,
        "stop": stop_tokens,
    }

    def query_model(prompt, args, port):
        json = {
            **args,
            "prompt": prompt,
        }
        response = requests.post(url="http://localhost" + ":" + str(port) + "/generate", json=json)
        response_json = response.json()
        return [response_json["text"][i][len(prompt) :] for i in range(len(response_json["text"]))]

    samples = []

    for example in tqdm(examples, total=len(examples)):
        idx = example["idx"]

        # parse question and answer
        example["question"] = parse_question(example, args.data_name)
        gt_cot, gt_ans = parse_ground_truth(example, args.data_name)

        full_prompt = construct_prompt(args, example)

        sample = {"idx": idx, "question": example["question"], "gt_cot": gt_cot, "gt": gt_ans, "prompt": full_prompt}
        # add remain fields
        for key in [
            "level",
            "type",
            "unit",
            "solution_type",
            "choices",
            "solution",
            "ques_type",
            "ans_type",
            "answer_type",
            "dataset",
            "subfield",
            "filed",
            "theorem",
            "answer",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    print("dataset:", args.data_name, "samples:", len(samples))
    if len(samples) > 0:
        print("-" * 50)
        print("sample:", samples[0]["prompt"])
        print("-" * 50)

    # repeat H times
    remain_prompts = [sample["prompt"] for sample in samples for _ in range(args.n_sampling)]
    remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]
    all_gts = [sample["gt"] for sample in samples for _ in range(args.n_sampling)]

    tmp_idx = list(range(len(all_gts)))
    all_gts = dict(zip(tmp_idx, all_gts))

    end_prompts = []

    max_func_call = 1 if args.prompt_type == "cot" else args.horizon

    # start inference, measure time use
    start_time = time.time()
    print("The maxmial function call is ", max_func_call)
    for epoch in range(max_func_call):
        print("=" * 50, "Epoch", epoch)
        current_prompts = remain_prompts
        # if all the queries meet the stop criteria, break
        if len(current_prompts) == 0:
            break

        # get all outputs, each prompt is (idx, prompt_content)
        prompts = [item[1] for item in current_prompts]
        with ThreadPoolExecutor(512) as executor2:
            result = [
                executor2.submit(query_model, prompts[i], default_args, ports[i % len(ports)])
                for i in range(len(prompts))
            ]
            # use tqdm to show progress
            for _ in tqdm(as_completed(result), total=len(result)):
                pass

            outputs = [r.result()[0] for r in result]

        # print(len(outputs), len(current_prompts))

        if len(outputs) != len(current_prompts):
            raise ValueError("VLLM has some problem, the generated responsess are less than the queries.")

        # process all outputs
        remain_prompts = []
        remain_codes = []

        for (i, query), output in zip(current_prompts, outputs):
            output = output.rstrip()
            # append the y_s to the current state (history)
            query += output
            if args.prompt_type == "cot":
                # for cot, the prompt ends for one round
                end_prompts.append((i, query))
            elif "boxed" not in output and "```python" in output: #output.endswith("```"):
                # the model does not output the final answer, meanwhile, a code needs to be executed
                program = extract_program(query)
                remain_prompts.append((i, query))
                remain_codes.append(program)
            else:
                end_prompts.append((i, query))

        # execute the codes and get the results
        # note that the order of remain_codes is the same as remain_prompts
        remain_results = executor.batch_apply(remain_codes)
        for k in range(len(remain_prompts)):
            i, query = remain_prompts[k]
            res, report = remain_results[k]
            exec_result = res if res else report
            # we add the observation to the history
            if "gemma" in args.model_name_or_path:
                exec_result = f"<end_of_turn>\n<start_of_turn>user\n```output\n{exec_result}\n```<end_of_turn>\n<start_of_turn>model\n"
            elif "mistral" in args.model_name_or_path:
                exec_result = f"</s> [INST] ```output\n{exec_result}\n``` [/INST]"
            elif "deepseek" in args.model_name_or_path:
                #exec_result = f"<｜end▁of▁sentence｜>User: ```output\n{exec_result}\n```\n\nAssistant:"
                #for deepseek, we directly append the observation as the training of deepseek
                exec_result = f"\n```output\n{exec_result}\n```\n"
            elif "llama3" in args.model_name_or_path:
                exec_result = f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n```output\n{exec_result}\n```<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            else:
                raise NotImplementedError(args.prompt_type + "and " + args.model_name_or_path)

            query += exec_result

            if epoch == max_func_call - 1:
                query += "\nReach max function call limit."
            remain_prompts[k] = (i, query)

    # unsolved samples
    print("Unsolved samples:", len(remain_prompts))
    end_prompts.extend(remain_prompts)
    # sort by idx
    end_prompts = sorted(end_prompts, key=lambda x: x[0])

    if "gemma" in args.model_name_or_path:
        ans_split = "<start_of_turn>model\n"
    elif "mistral" in args.model_name_or_path:
        ans_split = "[/INST]"
    elif "deepseek" in args.model_name_or_path:
        ans_split = "\n\nAssistant:"
    else:
        raise NotImplementedError(args.prompt_type + "and " + args.model_name_or_path)

    codes = [prompt.split(ans_split)[-1].strip() for _, prompt in end_prompts]

    # extract preds, run_execute will extract the code needed to run...
    # for tora, we only extract the final answer but do not run the code
    results = [run_execute(executor, code, args.prompt_type) for code in codes]

    time_use = time.time() - start_time
    tmp_to_store = [z.split("---")[-1].strip() for _, z in end_prompts]
    # put results back to examples
    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i * args.n_sampling : (i + 1) * args.n_sampling]
        result = results[i * args.n_sampling : (i + 1) * args.n_sampling]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]
        response_tmp = tmp_to_store[i * args.n_sampling : (i + 1) * args.n_sampling]
        sample.pop("prompt")
        sample.update({"my_solu": response_tmp, "code": code, "pred": preds, "report": reports})
        all_samples.append(sample)

    # add processed samples
    all_samples.extend(processed_samples)
    save_jsonl(all_samples, out_file)

    # Evaluate the result
    if args.eval:
        result_str = evaluate(samples=all_samples, data_name=args.data_name, prompt_type=args.prompt_type, execute=True)
        result_str += f"\nTime use: {time_use:.2f}s"
        time_str = f"{int(time_use // 60)}:{int(time_use % 60):02d}"
        result_str += f"\nTime use: {time_str}"

        with open(out_file.replace(".jsonl", f"_{args.prompt_type}.metrics"), "w") as f:
            f.write(result_str)


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    main(args)
