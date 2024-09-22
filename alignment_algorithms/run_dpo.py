import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import Dataset, load_dataset
from dpo import PreferenceTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters, i.e., the KL penalty in the paper
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    model_name_or_path: Optional[str] = field(
        default="RLHF4MATH/Gemma-7B-it-SFT3epoch",
        metadata={"help": "the location of the model name or path"},
    )
    ref_model: Optional[str] = field(
        default="RLHF4MATH/Gemma-7B-it-SFT3epoch",
        metadata={"help": "the location of the SFT model name or path"},
    )
    train_dir: Optional[str] = field(
        default="RLHF4MATH/Gemma-7B-1.1-it-iter1-random-pairs",
        metadata={"help": "the location of the dataset name or path"},
    )
    eval_dir: Optional[str] = field(
        default="RLHF4MATH/Gemma-7B-1.1-it-iter1-random-pairs",
        metadata={"help": "the location of the evalset name or path"},
    )
    learning_rate: Optional[float] = field(default=4e-7, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=50, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.01, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=1, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    eos_padding: Optional[bool] = field(default=True, metadata={"help": "whether to pad with eos token"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    margin_scale: Optional[float] = field(default=1.0, metadata={"help": "the margin scale"})

    max_prompt_length: Optional[int] = field(default=1000, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=2048, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=4000, metadata={"help": "max number of training steps"})
    num_train_epochs: Optional[int] = field(default=2, metadata={"help": "max number of training epochs"})
    logging_steps: Optional[int] = field(default=2, metadata={"help": "the logging frequency"})
    save_strategy: Optional[str] = field(default="steps", metadata={"help": "the saving strategy"})
    save_steps: Optional[int] = field(default=25, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=300, metadata={"help": "the evaluation frequency"})
    run_name: Optional[str] = field(default="mdpo_iter1_gemma7b_lr4e7_bz32", metadata={"help": "the run name"})
    loss_type: Optional[str] = field(default="sigmoid", metadata={"help": "the loss type"})
    output_dir: Optional[str] = field(
        default="./mdpo_iter1_gemma7b_lr4e7_bz32", metadata={"help": "the output directory"}
    )
    log_freq: Optional[int] = field(default=2, metadata={"help": "the logging frequency"})

    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})

    max_training_samples: Optional[int] = field(default=-1, metadata={"help": "the maximum sample size"})

    choose_type: Optional[str] = field(default="max_random", metadata={"help": "the choose type"})

    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    eot_token: Optional[str] = field(default="", metadata={"help": "the end of text token"})
    mask_prompt: Optional[bool] = field(default=False, metadata={"help": "mask prompt"})
    len_penalty: Optional[float] = field(default=0, metadata={"help": "the length penalty"})

    masking_user_turn: Optional[bool] = field(default=True, metadata={"help": "mask user turn"})
    nll_coefficient: Optional[float] = field(default=0, metadata={"help": "the coefficeint of NLL loss"})


def prepare_data(
    tokenizer,
    data_dir: str = "xxx",
    sanity_check: bool = False,
    cache_dir: str = None,
    num_proc=24,
    margin_scale=1,
    choose_type="random",
    eot_token="",
    length_penalty=0,
) -> Dataset:
    """Prepare the dataset for DPO training. The input datasets are supposed to be in the standard format with keys chosen and rejected.
    The margin is not used currently and may be activated later for future research.

    [ { "content": "If a 40-foot tree is casting a 10-foot shadow, and Andrea is casting a 15-inch shadow at the same time, how tall is Andrea in inches?", "role": "user" },
    { "content": "The shadow of the tree is 10 feet which is 120 inches.\nSo let's set *tree height* = 40 feet = 40 * 12 inches\n*tree shadow* = 10 feet = 120 inches\n*Andrea's shadow* = 15 inches\nFrom the similar triangles, we can find Andrea's height.\n```python\ntree_height = 40 * 12 # tree is 40 feet which is 40 * 12 inches\ntree_shadow = 10 * 12 # tree shadow is 10 feet = 120 inches\nandrea_shadow = 15 # Andrea's shadow is 15 inches\n\n# Find Andrea's height using similar triangles\nandrea_height = andrea_shadow * (tree_height / tree_shadow)\nandrea_height\n```", "role": "assistant" },
    { "content": "```output\n60.0\n```", "role": "user" },
    { "content": "So Andrea is $\\boxed{60}$ inches tall.", "role": "assistant" } ]
    """
    ds = load_dataset(data_dir, split="train")
    ds = ds.shuffle(seed=42)
    print(ds)

    pos = []
    neg = []
    prompts = []
    margin = []
    for sample in ds:
        chosen = sample["chosen"]
        rejected = sample["rejected"]
        prompt = tokenizer.apply_chat_template([chosen[0]], tokenize=False, add_generation_prompt=True)
        prompt2 = tokenizer.apply_chat_template([rejected[0]], tokenize=False, add_generation_prompt=True)
        if prompt != prompt2:
            continue

        # assert prompt == prompt2
        chosen_str = tokenizer.apply_chat_template(chosen, tokenize=False).replace(prompt, "")
        rejected_str = tokenizer.apply_chat_template(rejected, tokenize=False).replace(prompt, "")
        prompts.append(prompt)
        pos.append(chosen_str)
        neg.append(rejected_str)
        margin.append(0.5) # not used so far
    dataset = Dataset.from_dict({"prompt": prompts, "chosen": pos, "rejected": neg, "margin": margin})
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 100)))

    return dataset


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # 1. load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        use_flash_attention_2=True,
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    if script_args.ref_model:
        ref_name = script_args.ref_model
    else:
        ref_name = script_args.model_name_or_path

    model_ref = AutoModelForCausalLM.from_pretrained(
        ref_name,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)

    # 2. Load the paired dataset
    train_dataset = prepare_data(
        tokenizer,
        data_dir=script_args.train_dir,
        margin_scale=script_args.margin_scale,
        sanity_check=script_args.sanity_check,
        choose_type=script_args.choose_type,
        eot_token=script_args.eot_token,
        length_penalty=script_args.len_penalty,
    )
    print(train_dataset)
    print(train_dataset[0])
    if script_args.max_training_samples > 0:
        train_dataset = train_dataset.select(range(script_args.max_training_samples))

    # 3. Load evaluation dataset
    eval_dataset = prepare_data(
        tokenizer,
        data_dir=script_args.eval_dir,
        sanity_check=True,
        margin_scale=script_args.margin_scale,
        eot_token=script_args.eot_token,
    )

    # 4. initialize training arguments:

    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        # max_steps=script_args.max_steps,
        num_train_epochs=script_args.num_train_epochs,
        save_strategy=script_args.save_strategy,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        evaluation_strategy="steps",
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        # report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        # optim=script_args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name=script_args.run_name,
        save_only_model=True,
    )
    print(training_args)

    # 5. initialize the DPO trainer

    dpo_trainer = PreferenceTrainer(
        model,
        model_ref,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        loss_type=script_args.loss_type,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
        mask_prompt=script_args.mask_prompt,
        len_penalty=script_args.len_penalty,
        nll_coefficient=script_args.nll_coefficient,
        masking_user_turn=script_args.masking_user_turn,
    )
    print("begin to train")

    # 6. train
    dpo_trainer.train()
    dpo_trainer.save_model(script_args.output_dir)

    # 7. save
    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)
