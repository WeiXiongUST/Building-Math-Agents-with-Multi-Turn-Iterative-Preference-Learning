from datasets import load_dataset
from collections import defaultdict
import multiprocessing
from math import isclose
from typing import Union

from datasets import Dataset, concatenate_datasets

from typing import Union

from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr

ds1 = load_dataset("1231czx/7b_sft_510k_3epoch_gen_data_iter1", split="train").shuffle(seed=42)
ds2 = load_dataset("1231czx/7b_sft_510k_1epoch_gen_data_iter1", split="train").shuffle(seed=42)

ds = concatenate_datasets([ds1, ds2])  # .select(range(100000))
# ds = ds.select(range(5000))
N_pair = 1
data_comp = "1231czx/7B_iter1_dpo_N1_random_pair"
data_sft = "1231czx/7B_iter1_sft_N1"



def symbolic_equal(a, b):
    def _parse(s):
        for f in [parse_latex, parse_expr]:
            try:
                return f(s)
            except:
                pass
        return s

    a = _parse(a)
    b = _parse(b)

    try:
        if simplify(a - b) == 0:
            return True
    except:
        pass

    try:
        if isclose(N(a), N(b), rel_tol=1e-3):
            return True
    except:
        pass
    return False


def symbolic_equal_process(a, b, output_queue):
    result = symbolic_equal(a, b)
    output_queue.put(result)


def call_with_timeout(func, *args, timeout=10, **kwargs):
    output_queue = multiprocessing.Queue()
    process_args = args + (output_queue,)
    process = multiprocessing.Process(target=func, args=process_args, kwargs=kwargs)
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        # return "time out"
        return False
    return output_queue.get()


def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    is_close: bool = True,
    timeout: bool = True,
) -> bool:
    """
    Exact match of math if and only if:
    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal
    """
    try:  # 1. numerical equal
        if is_digit(prediction) and is_digit(reference):
            prediction = float(str(prediction).replace(",", ""))
            reference = float(str(reference).replace(",", ""))
            # number questions
            if include_percentage:
                gt_result = [reference / 100, reference, reference * 100]
            else:
                gt_result = [reference]
            for item in gt_result:
                try:
                    if is_close:
                        if isclose(item, prediction, rel_tol=1e-4):
                            return True
                    else:
                        if item == prediction:
                            return True
                except Exception:
                    continue
            return False
    except:
        pass
    if not prediction and prediction not in [0, False]:
        return False

    # 2. symbolic equal
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    ## deal with [], (), {}
    pred_str, ref_str = prediction, reference
    if (prediction.startswith("[") and prediction.endswith("]") and not reference.startswith("(")) or (
        prediction.startswith("(") and prediction.endswith(")") and not reference.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str == ref_str:
        return True

    ## [a, b] vs. [c, d], return a==c and b==d
    if (
        (prediction.startswith("[") and prediction.endswith("]"))
        and (reference.startswith("[") and reference.endswith("]"))
        or (prediction.startswith("(") and prediction.endswith(")"))
        and (reference.startswith("(") and reference.endswith(")"))
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(
                [math_equal(pred_parts[i], ref_parts[i], include_percentage, is_close) for i in range(len(pred_parts))]
            ):
                return True

    # symbolic equal with sympy
    if timeout:
        if call_with_timeout(symbolic_equal_process, prediction, reference):
            return True
        # elif tmp == 'time out':
        #    return "time out"
    else:
        if symbolic_equal(prediction, reference):
            return True

    return False


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_sqrt(string):
    _string = re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)
    return _string


def strip_string(string):
    string = str(string).strip()
    # linebreaks
    string = string.replace("\n", "")

    # right "."
    string = string.rstrip(".")

    # remove inverse spaces
    string = string.replace("\\!", "")
    string = string.replace("\\ ", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")

    string = string.replace("\\text", "")
    string = string.replace("x\\in", "")

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # cdot
    string = string.replace("\\cdot", "")

    # inf
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    # and
    string = string.replace("and", "")
    string = string.replace("\\mathbf", "")

    # use regex to remove \mbox{...}
    string = re.sub(r"\\mbox{.*?}", "", string)

    # quote
    string.replace("'", "")
    string.replace('"', "")

    # i, j
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    # replace a.000b where b is not number or b is end, with ab, use regex
    string = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0+$", r"\1", string)

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


import re


def check1(a, b):

    try:
        a = parse_latex(a)
        b = parse_latex(b)
        if abs(float(a) - float(b)) < 0.01:
            return True
    except:
        pass

    return False


print(check1("0.25", "\\frac{1}{4}"))


def parse_ground_truth(example):
    cnt = 0
    if example["type"] in [
        "gpt-3.5-turbo",
        "MATH_FOBAR",
        "GSM_Rephrased",
        "MATH_SV",
        "MATH_Rephrased",
        "GSM_SV",
        "GSM_FOBAR",
    ]:
        pattern = r"The answer is: (.*?)$"
        match = re.search(pattern, example["solution"])

        if match:
            gt_ans = match.group(1)
            # print("The prediction is:", prediction)
        else:
            print("No prediction found for gpt-3.5-turbo.")

        check_ans = extract_answer(example["solution"])
        if check_ans != gt_ans:
            # if math_equal(gt_ans, check_ans):
            #    pass
            if check1(gt_ans, check_ans):
                pass
            else:
                # print(example['type'], gt_ans, check_ans, "\n")
                cnt += 1
                gt_ans = "delete"
    elif example["type"] == "gsm8k":
        gt_ans = example["solution"].split("####")[-1]

    elif example["type"] == "math":
        gt_ans = extract_answer(example["solution"])

    else:
        print(example["type"])
        raise NotImplementedError()
    gt_ans = strip_string(gt_ans)
    # if gt_ans.startswith('\frac'):
    #    gt_ans = '\' + gt_ans
    return {"gt": gt_ans}


def extract_answer(pred_str):
    if "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        pred = a
    elif "he answer is" in pred_str:
        pred = pred_str.split("he answer is")[-1].strip()
    elif extract_program_output(pred_str) != "":
        # fall back to program
        pred = extract_program_output(pred_str)
    else:  # use the last number
        pattern = "-?\d*\.?\d+"
        pred = re.findall(pattern, pred_str.replace(",", ""))
        if len(pred) >= 1:
            pred = pred[-1]
        else:
            pred = ""

    # multiple line
    pred = pred.split("\n")[0]
    if pred != "" and pred[0] == ":":
        pred = pred[1:]
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]
    pred = strip_string(pred)
    return pred


from sympy.parsing.latex import parse_latex

# Example LaTeX string

z = 0
# ds = load_dataset('1231czx/7b_sft_510k_3epoch_gen_data_iter1', split='train')

# for sample in ds:
#    a, b = parse_ground_truth(sample)
#    z += b
# print(z)

ds_new = ds.map(parse_ground_truth, num_proc=32)
ds_new = ds_new.filter(lambda example: example["gt"] != "delete")

print("#######################################\n", "After delete the prompts that cannot be verified")
print(ds_new[0])
print(ds, ds_new)


###########################################

import re


def parse_conversation(example):
    # Split the data into turns based on the start_of_turn and end_of_turn markers
    data = example["my_solu"][0]
    turns = re.split(r"<start_of_turn>|<end_of_turn>", data)

    # Clean and filter out empty entries
    turns = [turn.strip() for turn in turns if turn.strip() and not turn.startswith("<eos>")]

    # Create a list to hold the parsed conversation in the desired format
    conversation = []

    # Process each turn, assigning the correct role and content
    for turn in turns:
        if turn.startswith("user\n"):
            # Extract content after the role identifier
            content = turn[5:].strip()
            conversation.append({"role": "user", "content": content})
        elif turn.startswith("model\n"):
            content = turn[6:].strip()
            conversation.append({"role": "assistant", "content": content})

    return {"messages": conversation}


ds_new = ds_new.map(parse_conversation, num_proc=32)



def is_correct(example):
    is_correct_label = False
    a = example["pred"][0]
    b = example["gt"]
    if a == b:
        is_correct_label = True
        return {"is_correct": is_correct_label}
    try:
        if abs(float(a) - float(b)) < 0.0001:
            is_correct_label = True
            return {"is_correct": is_correct_label}
    except:
        pass

    try:

        if math_equal(a, b):
            # if z == True:
            is_correct_label = True

            return {"is_correct": is_correct_label}

        # elif z == 'time out':
        #    print("time out")
        #    return {"is_correct": 'time out'}

        # signal.alarm(0)
        # except TimeoutException:
        #    pass
        # if math_equal(a, b):
        #    is_correct_label = True
    except:
        pass

    """
    try:
        if check1(a, b):
            is_correct_label = True  
            return {"is_correct": is_correct_label}
    except:
        pass
    """
    # try:
    #    if math_equal(a, b):
    #        is_correct_label = True
    # except:
    #        pass

    # print(example['type'])
    # raise NotImplementedError()
    # gt_ans = strip_string(gt_ans)
    return {"is_correct": is_correct_label}


def filter_example1(example):
    old_messages = example["messages"]

    if len(old_messages) < 4:
        return False

    if len(old_messages) % 2 != 0:
        return False

    all_mes_len = len(old_messages)
    if example["is_correct"] and "error" in old_messages[-2]["content"].lower():
        return False

    if "boxed" in old_messages[-1]["content"].lower() and "error" in old_messages[-2]["content"].lower():
        return False

    k = 0

    for mes in old_messages:
        if k % 2 != 0 and k < all_mes_len - 1:
            if "python" not in mes["content"]:
                return False
        k += 1
        if "ipython" in mes["content"].lower() and "error" in mes["content"].lower():
            return False
        if mes["content"] == "```output\nExecution error: \n```":
            return False
        if "```output\n[]" in mes["content"]:
            # print(mes['content'])
            return False

    return True


from transformers import AutoTokenizer, HfArgumentParser

# ds_new = ds_new.filter(filter_example1, num_proc=32)

print("############### Answer check")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-1.1-7b-it")



def filter_too_long_pred(example):
    try:
        if len(example["pred"][0]) > 20:
            return False
    except:
        return False
    old_messages = example["messages"]
    if len(old_messages) < 4:
        return False

    if len(old_messages) % 2 != 0:
        return False

    all_mes_len = len(old_messages)

    if "boxed" in old_messages[-1]["content"].lower() and "error" in old_messages[-2]["content"].lower():
        return False

    k = 0

    for mes in old_messages:
        if k % 2 != 0 and k < all_mes_len - 1:
            if "python" not in mes["content"]:
                return False
        k += 1
        if "ipython" in mes["content"].lower() and "error" in mes["content"].lower():
            return False
        if mes["content"] == "```output\nExecution error: \n```":
            return False
        if "```output\n[]" in mes["content"]:
            # print(mes['content'])
            return False

    z = len(tokenizer.apply_chat_template(old_messages, tokenize=True))
    if z > 2048:
        return False

    # b = len(tokenizer.apply_chat_template(ds_test[i]['rejected'], tokenize=True))

    return True


ds_new = ds_new.filter(filter_too_long_pred, num_proc=32)

ds_new = ds_new.map(is_correct, num_proc=32)


ds_new = ds_new.filter(filter_example1, num_proc=32)

ds_win = ds_new.filter(lambda example: example["is_correct"] == True)
ds_lose = ds_new.filter(lambda example: example["is_correct"] == False)

print(len(ds_win) + len(ds_lose), len(ds_new))


# print("I have win ", len(ds_win), " and lose ", len(ds_lose))
def filter_win(example):
    old_messages = example["messages"]

    if "error" in old_messages[-2]["content"].lower():
        return False
    if "none" in old_messages[-2]["content"].lower():
        return False

    return True


print("I have win ", len(ds_win), " and lose ", len(ds_lose))
ds_win = ds_win.filter(filter_win, num_proc=32)
print("I have win ", len(ds_win), " and lose ", len(ds_lose))


win_ret = defaultdict(list)
lose_ret = defaultdict(list)


for sample in ds_win:
    idx = sample["idx"]
    win_ret[idx].append(sample)


for sample in ds_lose:
    idx = sample["idx"]
    lose_ret[idx].append(sample)


new_win_ret = defaultdict(list)

cnt_win = 0

for key, value in win_ret.items():
    if len(win_ret[key]) == 0:
        continue
    j = key
    all_samples = win_ret[j]
    all_texts = []
    new_samples = []
    for ins in all_samples:
        if ins["messages"][1]["content"] in all_texts:
            continue
        all_texts.append(ins["messages"][1]["content"])
        new_samples.append(ins)
        cnt_win += 1

    new_win_ret[j].extend(new_samples)


cnt_lose = 0
new_lose_ret = defaultdict(list)


for key, value in lose_ret.items():
    if len(lose_ret[key]) == 0:
        continue
    j = key
    all_samples = lose_ret[j]
    all_texts = []
    new_samples = []
    for ins in all_samples:
        if ins["messages"][1]["content"] in all_texts:
            continue
        all_texts.append(ins["messages"][1]["content"])
        new_samples.append(ins)
        cnt_lose += 1

    new_lose_ret[j].extend(new_samples)

print("Before get final pairs, I have win and lose", cnt_win, cnt_lose)
import random

import numpy as np

all_comp = []
all_sft = []
all_keys = list(new_lose_ret.keys()) + list(new_win_ret.keys())
all_keys = list(set(all_keys))
import itertools

"""
for ins in new_win_ret[0]:
    print(is_correct(ins), ins['pred'][0])

for ins in new_lose_ret[0]:
    print(is_correct(ins), ins['pred'][0], ins)
    print(check1(ins['gt'], ins['pred'][0]))
    print(check1('\\frac{1}{4}', '0.25'))
"""
cnt_comp = 0
# N = 1
for j in all_keys:
    if len(new_lose_ret[j]) > 0 and len(new_win_ret[j]) > 0:
        cnt_comp += N_pair
    else:
        continue
    all_pos = new_win_ret[j]
    all_neg = new_lose_ret[j]
    random.shuffle(all_pos)
    random.shuffle(all_neg)
    if len(all_pos) > N_pair and len(all_neg) > N_pair:
        for k in range(N_pair):
            all_comp.append(
                {
                    "gt": all_pos[k]["gt"],
                    "rej": all_neg[k]["pred"],
                    "chosen": all_pos[k]["messages"],
                    "rejected": all_neg[k]["messages"],
                }
            )
            all_sft.append({"messages": all_pos[k]["messages"]})
        continue

    combinations = list(itertools.product(list(range(len(all_pos))), list(range(len(all_neg)))))

    random.shuffle(combinations)
    for k in range(np.min([len(combinations), N_pair])):
        all_comp.append(
            {
                "gt": all_pos[combinations[k][0]]["gt"],
                "chosen": all_pos[combinations[k][0]]["messages"],
                "rejected": all_neg[combinations[k][1]]["messages"],
                "rej": all_neg[combinations[k][1]]["pred"],
            }
        )
        all_sft.append({"messages": all_pos[combinations[k][0]]["messages"]})


# print(all_comp[0])
output_eval_dataset = {}
output_eval_dataset["type"] = "text_only"
output_eval_dataset["instances"] = all_comp
print("I collect ", len(all_comp), "samples", len(all_sft))

import json

with open("tmp_comp.json", "w", encoding="utf8") as f:
    json.dump(output_eval_dataset, f, ensure_ascii=False)

ds_comp = load_dataset("json", data_files="tmp_comp.json", split="train", field="instances")
ds_comp.push_to_hub(data_comp)

output_eval_dataset = {}
output_eval_dataset["type"] = "text_only"
output_eval_dataset["instances"] = all_sft
print("I collect ", len(all_comp), "samples", len(all_sft))

import json

with open("tmp_sft.json", "w", encoding="utf8") as f:
    json.dump(output_eval_dataset, f, ensure_ascii=False)

ds_sft = load_dataset("json", data_files="tmp_sft.json", split="train", field="instances")
ds_sft.push_to_hub(data_sft)

# ds_new = ds_new.remove_columns("idx")

# def add_index(example, idx):
#    # Add the current index to the example under a new field 'index'
#    example['idx'] = idx
#    return example

# ds_new = ds_new.map(add_index, with_indices=True)
# ds_new.push_to_hub("1231czx/prompts_80K_with_original_MATH_GSM8K_iter1")

###################################

N_pair = 3
data_comp = "1231czx/7B_iter1_dpo_N3_random_pair"
data_sft = "1231czx/7B_iter1_sft_N3"

all_comp = []
all_sft = []
all_keys = list(new_lose_ret.keys()) + list(new_win_ret.keys())
all_keys = list(set(all_keys))
import itertools

"""
for ins in new_win_ret[0]:
    print(is_correct(ins), ins['pred'][0])

for ins in new_lose_ret[0]:
    print(is_correct(ins), ins['pred'][0], ins)
    print(check1(ins['gt'], ins['pred'][0]))
    print(check1('\\frac{1}{4}', '0.25'))
"""
cnt_comp = 0
# N = 1
for j in all_keys:
    if len(new_lose_ret[j]) > 0 and len(new_win_ret[j]) > 0:
        cnt_comp += N_pair
    else:
        continue
    all_pos = new_win_ret[j]
    all_neg = new_lose_ret[j]
    random.shuffle(all_pos)
    random.shuffle(all_neg)
    if len(all_pos) > N_pair and len(all_neg) > N_pair:
        for k in range(N_pair):
            all_comp.append(
                {
                    "gt": all_pos[k]["gt"],
                    "rej": all_neg[k]["pred"],
                    "chosen": all_pos[k]["messages"],
                    "rejected": all_neg[k]["messages"],
                }
            )
            all_sft.append({"messages": all_pos[k]["messages"]})
        continue

    combinations = list(itertools.product(list(range(len(all_pos))), list(range(len(all_neg)))))

    random.shuffle(combinations)
    for k in range(np.min([len(combinations), N_pair])):
        all_comp.append(
            {
                "gt": all_pos[combinations[k][0]]["gt"],
                "chosen": all_pos[combinations[k][0]]["messages"],
                "rejected": all_neg[combinations[k][1]]["messages"],
                "rej": all_neg[combinations[k][1]]["pred"],
            }
        )
        all_sft.append({"messages": all_pos[combinations[k][0]]["messages"]})


# print(all_comp[0])
output_eval_dataset = {}
output_eval_dataset["type"] = "text_only"
output_eval_dataset["instances"] = all_comp
print("I collect ", len(all_comp), "samples", len(all_sft))

import json

with open("tmp_comp.json", "w", encoding="utf8") as f:
    json.dump(output_eval_dataset, f, ensure_ascii=False)

ds_comp = load_dataset("json", data_files="tmp_comp.json", split="train", field="instances")
ds_comp.push_to_hub(data_comp)

output_eval_dataset = {}
output_eval_dataset["type"] = "text_only"
output_eval_dataset["instances"] = all_sft
print("I collect ", len(all_comp), "samples", len(all_sft))

import json

with open("tmp_sft.json", "w", encoding="utf8") as f:
    json.dump(output_eval_dataset, f, ensure_ascii=False)

ds_sft = load_dataset("json", data_files="tmp_sft.json", split="train", field="instances")
ds_sft.push_to_hub(data_sft)


##########################

N_pair = 8
data_comp = "1231czx/7B_iter1_dpo_N8_random_pair"
data_sft = "1231czx/7B_iter1_sft_N8"

all_comp = []
all_sft = []
all_keys = list(new_lose_ret.keys()) + list(new_win_ret.keys())
all_keys = list(set(all_keys))
import itertools

"""
for ins in new_win_ret[0]:
    print(is_correct(ins), ins['pred'][0])

for ins in new_lose_ret[0]:
    print(is_correct(ins), ins['pred'][0], ins)
    print(check1(ins['gt'], ins['pred'][0]))
    print(check1('\\frac{1}{4}', '0.25'))
"""
cnt_comp = 0
# N = 1
for j in all_keys:
    if len(new_lose_ret[j]) > 0 and len(new_win_ret[j]) > 0:
        cnt_comp += N_pair
    else:
        continue
    all_pos = new_win_ret[j]
    all_neg = new_lose_ret[j]
    random.shuffle(all_pos)
    random.shuffle(all_neg)
    if len(all_pos) > N_pair and len(all_neg) > N_pair:
        for k in range(N_pair):
            all_comp.append(
                {
                    "gt": all_pos[k]["gt"],
                    "rej": all_neg[k]["pred"],
                    "chosen": all_pos[k]["messages"],
                    "rejected": all_neg[k]["messages"],
                }
            )
            all_sft.append({"messages": all_pos[k]["messages"]})
        continue

    combinations = list(itertools.product(list(range(len(all_pos))), list(range(len(all_neg)))))

    random.shuffle(combinations)
    for k in range(np.min([len(combinations), N_pair])):
        all_comp.append(
            {
                "gt": all_pos[combinations[k][0]]["gt"],
                "chosen": all_pos[combinations[k][0]]["messages"],
                "rejected": all_neg[combinations[k][1]]["messages"],
                "rej": all_neg[combinations[k][1]]["pred"],
            }
        )
        all_sft.append({"messages": all_pos[combinations[k][0]]["messages"]})


# print(all_comp[0])
output_eval_dataset = {}
output_eval_dataset["type"] = "text_only"
output_eval_dataset["instances"] = all_comp
print("I collect ", len(all_comp), "samples", len(all_sft))

import json

with open("tmp_comp.json", "w", encoding="utf8") as f:
    json.dump(output_eval_dataset, f, ensure_ascii=False)

ds_comp = load_dataset("json", data_files="tmp_comp.json", split="train", field="instances")
ds_comp.push_to_hub(data_comp)

output_eval_dataset = {}
output_eval_dataset["type"] = "text_only"
output_eval_dataset["instances"] = all_sft
print("I collect ", len(all_comp), "samples", len(all_sft))

import json

with open("tmp_sft.json", "w", encoding="utf8") as f:
    json.dump(output_eval_dataset, f, ensure_ascii=False)

ds_sft = load_dataset("json", data_files="tmp_sft.json", split="train", field="instances")
ds_sft.push_to_hub(data_sft)
