import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = 'model_dir'
output_name = 'output_name'
tokenizer_name = name

model = AutoModelForCausalLM.from_pretrained(
    name,
    torch_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

model.push_to_hub(output_name)
tokenizer.push_to_hub(output_name)
