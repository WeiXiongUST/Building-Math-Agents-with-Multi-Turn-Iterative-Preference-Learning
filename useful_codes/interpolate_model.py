import torch
#!/usr/bin/env python
# coding=utf-8
# We provide a script to merge different checkpoints of the model
# See a detailed study in Mitigating the Alignment Tax of RLHF, https://arxiv.org/abs/2309.06256

import json
import os
import sys
from transformers import HfArgumentParser, AutoModelForCausalLM
import argparse

parser = argparse.ArgumentParser(description="merge model checkpoints")
parser.add_argument("--base_model", type=str, required=True, help="dir of base model")
parser.add_argument("--new_model", type=str, required=True, help="dir of new model")
parser.add_argument("--output_dir", type=str, required=True, help="output dir")
parser.add_argument("--ratio", type=float, required=True, help="the ratio of the new model")

args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)
print(f"Base model: {args.base_model}")
print(f"New model: {args.new_model}")
print(f"Output directory: {args.output_dir}")

new_dir = args.new_model
base_dir = args.base_model
weight_ensamble_save_path = args.output_dir
weight_ensamble_ratios = args.ratio

# Get the paths and ratios of weight-ensamble models.
# args.ratio * new_model + (1 - args.ratio) * base_model

weight_ensamble_names_paths = [new_dir, base_dir]
weight_ensamble_ratios.append(1 - weight_ensamble_ratios[0])
assert len(weight_ensamble_ratios) == 2, 'Only 2 merge is supported.'
print('Model Paths:', weight_ensamble_names_paths)
print('Model Ratio:', weight_ensamble_ratios)

base_model = None
backend_models = []
for model_path in weight_ensamble_names_paths:
    #model_args.model_name_or_path = model_path
    print('loading:', model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)#, torch_dtype=torch.bfloat16)
    backend_models.append(model.to('cpu'))
    if base_model is None:
        base_model = model
    print('Finish load:', model_path)
base_backend_model = backend_models[0]
print('Finish load All:', base_backend_model)

updated_state_dict = {}
for key in base_backend_model.state_dict():
    ensambled_state_dicts = [ratio * backend_model.state_dict()[key] for backend_model, ratio in zip(backend_models, weight_ensamble_ratios)]
    updated_state_dict[key] = sum(ensambled_state_dicts)

base_backend_model.load_state_dict(updated_state_dict)
base_model.save_pretrained(weight_ensamble_save_path)
