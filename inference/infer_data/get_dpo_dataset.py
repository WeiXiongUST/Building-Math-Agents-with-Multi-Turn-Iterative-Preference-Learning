from datasets import load_dataset, Dataset

import random
import json

# Define the file path and output dir
data_files = 'here.jsonl'
output_dir = "huggingface/dataset_name"

# Load the dataset
ds0 = load_dataset('json', data_files='here.jsonl', split='train')

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

# we first transform the data into standard format
ds1 = ds0.map(parse_conversation, num_proc=32)


def filter_example(example):
    old_messages = example["messages"]

    if len(old_messages) < 4:
        return False

    if len(old_messages) % 2 != 0:
        return False

    all_mes_len = len(old_messages)
    # if the model makes mistake but predict the correct answer
    if "error" in old_messages[-2]["content"].lower():
        return False
    if "boxed" in old_messages[-1]["content"].lower() and "error" in old_messages[-2]["content"].lower():
        return False

    k = 0

    for mes in old_messages:
        if k % 2 != 0 and k < all_mes_len - 1:
            if "python" not in mes["content"]:
                return False
        k += 1
        # env error
        if "ipython" in mes["content"].lower() and "error" in mes["content"].lower():
            return False

    return True
    
ds2 = ds1.filter(filter_example, num_proc=32)


# Function to de-duplicate and group entries
def deduplicate_and_group(dataset):
    unique_entries = {}
    for entry in dataset:
        idx = entry['idx']
        solu = entry['my_solu'][0]  # tuples are hashable and can be used as dictionary keys
        if idx not in unique_entries:
            unique_entries[idx] = {}
        if solu not in unique_entries[idx]:
            unique_entries[idx][solu] = entry
    return unique_entries

# Group by 'idx' and de-duplicate by 'my_solu'
grouped_data = deduplicate_and_group(ds2)

# Select one sample with scores [True] and one with scores [False]
def select_samples(groups):
    selected_pairs = []
    for idx, solutions in groups.items():
        true_samples = [sol for sol in solutions.values() if sol['score'][0] == True]
        false_samples = [sol for sol in solutions.values() if sol['score'][0] == False]
        if true_samples and false_samples:
            selected_true = random.choice(true_samples)
            selected_false = random.choice(false_samples)
            selected_pairs.append((selected_true, selected_false))
    return selected_pairs

# Apply the selection function
selected_samples = select_samples(grouped_data)


# get the dpo dataset
all_samples = []
for pair in selected_samples:
    all_samples.append(
        {
            "gt": pair[0]["gt"],
            "chosen": pair[0]["messages"],
            "rejected": pair[1]["messages"],}
    )

dict_data = {
    "rejected": [d['rejected'] for d in all_samples],
    "chosen": [d['chosen'] for d in all_samples],
    "gt": [d['gt'] for d in all_samples],
}

final_ds = Dataset.from_dict(dict_data)
final_ds.push_to_hub(output_dir)
