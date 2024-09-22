# The evaluator is adapted from the ToRA project
# https://github.com/microsoft/ToRA
# ToRA authors: Zhibin Gou and Zhihong Shao and Yeyun Gong and yelong shen and Yujiu Yang and Minlie Huang and Nan Duan and Weizhu Chen

import argparse
import numpy as np
from tqdm import tqdm
from pebble import ProcessPool
from concurrent.futures import TimeoutError

from eval.grader import *
from utils.parser import *
from utils.utils import load_jsonl
from utils.python_executor import PythonExecutor
from datasets import load_dataset, Dataset

def evaluate(data_name, output_dir=None):
    if ".json" in data_name:
        ds = load_dataset("json", data_files=data_name, split='train', field='instances').shuffle(seed=42)
    else:
        ds = load_dataset(data_name, split="train").shuffle(seed=42)

    samples = [sample for sample in ds]
    params = [(idx, pred, sample['gt']) for idx, sample in enumerate(samples) for pred in sample['pred']]

    scores = []
    timeout_cnt = 0 

    with ProcessPool() as pool:
        future = pool.map(math_equal_process, params, timeout=10)
        iterator = future.result()
        with tqdm(total=len(samples), desc="Evaluate") as progress_bar:
            while True:
                try:
                    result = next(iterator)
                    scores.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(error)
                    scores.append(False)
                    timeout_cnt += 1
                except Exception as error:
                    print(error.traceback)
                    exit()
                progress_bar.update(1) 

    idx = 0
    score_mat = []
    for sample in samples:
        sample['score'] = scores[idx: idx+len(sample['pred'])]
        assert len(sample['score']) == len(sample['pred'])
    
  if ".json" in args.output_dir:
      all_data = [sample for sample in samples]
      output_eval_dataset = {}
      output_eval_dataset["type"] = "text_only"
      output_eval_dataset["instances"] = all_data
      print("I collect ", len(all_data), "samples")
      with open(output_dir, "w", encoding="utf8") as f:
          json.dump(output_eval_dataset, f, ensure_ascii=False)
  else:
      all_data = [sample for sample in samples]
      dict_data = {
          "idx": [d['idx'] for d in all_data],
          "gt": [d['gt'] for d in all_data],
          "level": [d['level'] for d in all_data],
          "type": [d['type'] for d in all_data],
          "messages": [d['messages'] for d in all_data],
          "pred": [d['pred'] for d in all_data],
          "score": [d['score'] for d in all_data],
      }

      dataset = Dataset.from_dict(dict_data)
      DatasetDict({'train': dataset}).push_to_hub(output_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="math")
    parser.add_argument("--output_dir", type=str, default=None, required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    evaluate(data_name=args.data_name, output_dir=args.output_dir)
