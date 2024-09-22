import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = 'mistralai/Mistral-7B-v0.3'
tokenizer_name = name

model = AutoModelForCausalLM.from_pretrained(
    name,
    torch_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.config.pad_token_id = tokenizer.pad_token_id

model.resize_token_embeddings(len(tokenizer))

model.save_pretrained("output_dir")
tokenizer.save_pretrained("output_dir")
