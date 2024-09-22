from typing import Union
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import AutoTokenizer, HfArgumentParser
import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="Process some strings.")

parser.add_argument("--data_dir", type=str, help="The dataset address", default=None)
parser.add_argument("--output_dir", type=str, help="The output address", default=None)
parser.add_argument("--model_name", type=str, help="The model used to collect the data", default='gemma')

# Parse command line arguments
args = parser.parse_args()


# Step 1: load dataset
if ".json" in args.data_dir:
    ds = load_dataset("json", data_files=args.data_dir, split='train', field='instances').shuffle(seed=42)
else:
    ds = load_dataset(args.data_dir, split="train").shuffle(seed=42)

# You may want to merge different datasets...
#ds = concatenate_datasets([ds1, ds2])

# Step 2: we split the trajectory into the standard multi-turn format

def parse_conversation(example, model_name='gemma'):
    # Split the data into turns based on the start_of_turn and end_of_turn markers
    
    if 'gemma' in model_name:
        data = example["my_solu"][0]
        turns = re.split(r"<start_of_turn>|<end_of_turn>", data)
    elif 'mistral' in model_name:
        data = example["my_solu"][0].replace("</s>", "").replace("<s>", "")
        turns = re.split(r"\[INST\]|\[/INST\]", data)
    else:
        raise NotImplementedError(model_name)
    
    # Clean and filter out empty entries
    turns = [turn.strip() for turn in turns if turn.strip() and not turn.startswith("<eos>")]
    
    # Create a list to hold the parsed conversation in the desired format
    conversation = []

    if 'gemma' in model_name:
        for turn in turns:
            if turn.startswith("user\n"):
                # Extract content after the role identifier
                content = turn[5:].strip()
                conversation.append({"role": "user", "content": content})
            elif turn.startswith("model\n"):
                content = turn[6:].strip()
                conversation.append({"role": "assistant", "content": content}
        
    elif 'mistral' in model_name:
        j = 0
        for turn in turns:
            if j % 2 == 0:
                content = turn.strip()
                conversation.append({"role": "user", "content": content})
                j += 1
            else:
                content = turn.strip()
                conversation.append({"role": "assistant", "content": content}
                j += 1
    else:
        raise NotImplementedError(model_name)
        

    return {"messages": conversation}


ds_new = ds.map(parse_conversation, num_proc=32)

# Step 3: we filter the examples which are with ood rounds, make mistake in the second last round but still give a guess of the result


def filter_example1(example):
    old_messages = example["messages"]

    if len(old_messages) < 4:
        return False

    if len(old_messages) % 2 != 0:
        return False

    if "boxed" in old_messages[-1]["content"].lower() and "error" in old_messages[-2]["content"].lower():
        return False

    for mes in old_messages:
        # the code interpreter destroy the conda environment from time to time, we delete the samples collected when the env is wrong
        if "ipython" in mes["content"].lower() and "error" in mes["content"].lower():
            return False
        if "```output\n[]" in mes["content"]:
            return False

        if "traitlets" in mes['content'] and 'error' in mes['content'].lower():
            return False
        if "sympy.core.numbers" in mes['content'] and 'error' in mes['content'].lower():
            return False
        if 'sympy.tensor.tensor' in mes['content'] and 'error' in mes['content']:
            return False
        if 'minlex() got an' in mes['content']:
            return False
        if 'No module named' in mes['content'] and 'sympy.' in mes['content']:
            return False
        if 'object is not subscriptable' in mes['content'].lower():
            return False
    
        # We delete the samples that reach max function call
        # it does not influence the final result but can significantly accelerate the training process
        if 'Reach max function call' in mes['content']:
            return False

    return True

ds_new = ds_new.filter(filter_example1, num_proc=32)


# Step 4: we delete the samples that are too long
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

def filter_too_long_pred(example):
    z = len(tokenizer.apply_chat_template(example["messages"], tokenize=True))
    if z > 2048:
        return False
    return True

ds_new = ds_new.filter(filter_too_long_pred, num_proc=32)


# Step 5: output the filtered dataset

# we delete the columns that are unnecessary
columns_to_keep = ["idx", "gt", "level", "type", "messages", "pred"]
ds_new = ds_new.remove_columns([col for col in dataset.column_names if col not in columns_to_keep])


if ".json" in args.output_dir:
    all_data = [sample for sample in ds_new]
    output_eval_dataset = {}
    output_eval_dataset["type"] = "text_only"
    output_eval_dataset["instances"] = all_data
    print("I collect ", len(all_data), "samples")
    with open(args.output_dir, "w", encoding="utf8") as f:
        json.dump(output_eval_dataset, f, ensure_ascii=False)
else:
    ds_new.push_to_hub(args.output_dir)

    
