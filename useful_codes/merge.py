# This script finds all the files ending with jsonl and merge them into one file 
import os
from datasets import load_dataset
import json

# The folders to load data
all_folder_path = [
        './7b_sft3epoch_gen_data/iter1',
        './7b_sft1epoch_gen_data/iter1'
]

output_dir='all_math.json'

all_data = []
for folder_path in all_folder_path:
    jsonl_files = [folder_path + '/' + f for f in os.listdir(folder_path) if f.endswith('.jsonl')]
    for dir_ in jsonl_files:
        ds_test = load_dataset('json', data_files=dir_, split='train')
        for sample in ds_test:
            all_data.append(sample)

output_eval_dataset = {}
output_eval_dataset["type"] = "text_only"
output_eval_dataset["instances"] = all_data
print("I collect ", len(all_data), "samples")

with open(output_dir, "w", encoding="utf8") as f:
    json.dump(output_eval_dataset, f, ensure_ascii=False)
     
# You can also upload the dataset to the huggingface, which is particularly useful when collecting data with multiple machines
"""
output_dir = "xxx"
dict_data = {
    "idx": [d['idx'] for d in all_data],
    "gt": [d['gt'] for d in all_data],
    "level": [d['level'] for d in all_data],
    "type": [d['type'] for d in all_data],
    "my_solu": [d['my_solu'] for d in all_data],
    "pred": [d['pred'] for d in all_data],
}

dataset = Dataset.from_dict(dict_data)
DatasetDict({'train': dataset}).push_to_hub(output_dir)
"""
