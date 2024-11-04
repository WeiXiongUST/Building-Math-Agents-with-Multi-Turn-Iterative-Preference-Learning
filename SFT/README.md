# SFT for Multi-round Chat

We conduct supervised fine-tuning here for multi-turn chat. The data is supposed to be processed into the standard format (see [RLHF4MATH/SFT_510K](https://huggingface.co/datasets/RLHF4MATH/SFT_510K)) for an example). 

We formulate the problem into a multi-turn chat format where the ``user'' prompts the assistant (LLM) to solve a math problem and provides the external feedback. An example is as follows.

- **User**: Nine adults went to a play with seven children. Adult tickets are $11 each and children's tickets are $7 each. How many dollars more did the adults' tickets cost in total than the children's tickets in total?"
- **Assistant**: Let's solve this problem using Python code.\n```python\nadult_price = 11\nchild_price = 7\nadult_count = 9\nchild_count = 7\nadult_ticket_total = adult_price * adult_count\nchild_ticket_total = child_price * child_count\nadult_ticket_total - child_ticket_total\n```
- **User**: ```output\n50\n```
- **Assistant**: Thus the adult's ticket total was \\boxed{50} dollars more than the children's tickets total.

We will mask all user turns so that they do not contribute to the gradient. 


## Installation instructions

Before starting, please make sure your linux machine has [nvidia-cuda-toolkit](https://developer.nvidia.com/cuda-toolkit) installed. 

```shell
sudo apt install nvidia-driver-530 # (530 for 12.1+)
wget https://repo.anaconda.com/archive/Anaconda3-2023.07-0-Linux-x86_64.sh
bash Anaconda3-2023.07-0-Linux-x86_64.sh # install conda
conda install nvidia/label/cuda-12.2.0::cuda-nvcc
```

Now we set up the python environment.

```shell
conda create -n sft python=3.10.9
conda activate sft

# The test cuda version is 12.1, 12.2. You may need to update the torch version based on your cuda version...
pip3 install torch==2.1.2 torchvision torchaudio
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.7/flash_attn-2.5.7+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

## Get axolotl for general model
git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl
git checkout 55cc214c767741e83ee7b346e5e13e6c03b7b9fa
pip install -e .

## Get FastChat
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip install -e .

pip install deepspeed

# You also need to install wandb to record the training and log in with the huggingface accout to access Gemma.

pip install wandb
wandb login

huggingface-cli login
```
## Running the Code

Running the code with Gemma.

```shell
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun --nproc_per_node 8 --master_port 20001 -m axolotl.cli.train gemma-7b-it.yaml
```

You can also modify the learning rate, batch size, output_path.. with either command or modify the ScriptArguments in the gemma-7b-it.yml

If you encounter out-of-memory issue. Running the code with Gemma-7b-it with deepspeed stage 3 and gradient checkpoint (set in the config).

```shell
CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node 4 --master_port 20001 -m axolotl.cli.train gemma-2b-it.yml --deepspeed ./sft_configs/sft_ds3.json
```

**REMARK: note that with deepspeed stage 3, the final mode saving does not work normally. We set the store strategy as epoch so we can store a normal model just before we finish the training for one epoch. If you modify the store stragety, you should set the save_every_steps as the total number of training steps - 1 so that the trainer will save a model for you just before finishing the training.**


Finally, for the models without an official padding token (like Mistral), you may need to set the padding token by ../useful_codes/prepare_model.py first.

