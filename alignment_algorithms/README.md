# DPO/KTO Training with Multi-turn Data

The implementation of DPO and KTO are adapted from open-source packages [TRL](https://github.com/huggingface/trl) and [RLHFlow](https://github.com/RLHFlow/Online-RLHF). Comparedto the original DPO/KTO, we only need to modify the mask of the samples to mask out all the external tokens. The current implementation supports Gemma and Mistral model. You can read the "get_new_mask" function in dpo_trainer or kto_trainer to get the idea and easily implement for other LLMs.


## 1 Installation instructions

**Note that the numpy version should be `numpy<2.0`.  `Numpy 2.0` will encounter unexpected issues!!!**


Before starting, please make sure your linux machine has nvidia-cuda-toolkit installed. See SFT part for the guidance. 


**Training Environment**

```sh
conda create -n alignment_train python=3.10.9
conda activate alignment_train

git clone https://github.com/huggingface/alignment-handbook.git
cd ./alignment-handbook/
git checkout d17fd7cd3b71c6a7bf7af34d8dc73135bb7ea8e9
pip3 install torch==2.1.2 torchvision torchaudio
python -m pip install .
pip install flash-attn==2.6.3
pip install accelerate==0.33.0

pip install huggingface-hub==0.24.7
pip install wandb
wandb login
huggingface-cli login
```

## 2 Hakcing the DPO Trainer and KTO Trainer

### 2.1 Hack DPO Trainer

The code is based on RLHFlow/Online-RLHF but we need to hack the trainer to implement some additional functions. We highlight the modified part with ############## MODIFICATION.

```sh
# Step 1: find the original DPO trainer
cd anaconda3/envs/alignment_train/lib/python3.10/site-packages/trl/trainer/

# Step 2: delete the old one
rm dpo_trainer.py

# Step 3: use the modified one in this repo. The following command need to be modified to use the correct address 
mv dpo_train/dpo_trainer.py anaconda3/envs/alignment_train/lib/python3.10/site-packages/trl/trainer/dpo_trainer.py
```

### 2.2 Hack KTO Trainer

The code is based on RLHFlow/Online-RLHF but we need to hack the KTO trainer to implement some additional functions. We highlight the modified part with ############## MODIFICATION.

```sh
# Step 1: find the original DPO trainer
cd anaconda3/envs/alignment_train/lib/python3.10/site-packages/trl/trainer/

# Step 2: delete the old one
rm kto_trainer.py

# Step 3: use the modified one in this repo. The following command need to be modified to use the correct address 
mv kto_train/kto_trainer.py anaconda3/envs/alignment_train/lib/python3.10/site-packages/trl/trainer/kto_trainer.py

# Step 4: modify the KTO config according to your GPU resource.
vim ./trl/trainer/kto_config.py
max_length: Optional[int] = 2048
max_prompt_length: Optional[int] = 1024
max_completion_length: Optional[int] = 2048
```

### 2.3 Fix Import Error

For transformers > 4.38.2, you will encounter an import issue related to the following function in anaconda3/envs/alignment_train/lib/python3.10/site-packages/trl/core.py. You can comment on the import from transformers and copy and paste the following hard code version in core.py.

```python
def top_k_top_p_filtering(
    logits: torch.FloatTensor,
    top_k: int = 0
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.FloatTensor:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.

    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        top_k (`int`, *optional*, defaults to 0):
            If > 0, only keep the top k tokens with highest probability (top-k filtering)
        top_p (`float`, *optional*, defaults to 1.0):
            If < 1.0, only keep the top tokens with cumulative probability >= top_p (nucleus filtering). Nucleus
            filtering is described in Holtzman et al. (https://huggingface.co/papers/1904.09751)
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimumber of tokens we keep per batch example in the output.

    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """

    if top_k > 0:
        logits = TopKLogitsWarper(top_k=top_k, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep)(
            None, logits
        )

    if 0 <= top_p <= 1.0:
        logits = TopPLogitsWarper(top_p=top_p, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep)(
            None, logits
        )

    return logits
```


## 3 Running the Code

### 3.1 DPO
Running the code before modify num_processes: 8 in ./training_configs/zero2_pf.yaml, the number 8 means that you will use 8 GPUs. Also modify the parameters, models, and datasets provided in run_dpo.py.

```shell
accelerate launch --config_file ./training_configs/zero2_pf.yaml run_dpo.py ./training_configs/training.yaml

```

### 3.2 KTO 

```shell
bash run_kto.sh
```

If you encounter out-of-memory issue. Running the code with Gemma-7b-it with zero3_pf.yaml. You can also reduce the max length of the data.



