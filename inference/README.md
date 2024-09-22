## Inference for MATH Problem with External Python Intepreter 

In this repo, we implement the MATH problem solving with external python tool using the VLLM to accelerate infernece. The codebase is largely based on the Tora project.


## 1 Installation instructions

**Note that the numpy version should be `numpy<2.0`.  `Numpy 2.0` will encounter unexpected issues!!!**


Before starting, please make sure your linux machine has nvidia-cuda-toolkit installed. See SFT part for the guidance. 


```sh
conda create -n infer python=3.10.9
conda activate infer
# You may check nvcc -V , if no nvcc exists, you may run the following code
# conda install nvidia/label/cuda-12.2.0::cuda-nvcc

pip install datasets

# The following code is tested for CUDA12.0-12.2 and CUDA12.6
# To develop llama-3, mistral, gemma-1, 1.1, 2, deepseek you can consider the following vllm version
pip install vllm==0.5.4

pip install accelerate==0.33.0
pip install deepspeed==0.14.5
pip install numpy==1.26.4 #Note that the numpy version should be `numpy<2.0`.  `Numpy 2.0` will encounter unexpected issues!!!


huggingface-cli login
pip install pebble
pip install timeout_decorator
pip install ipython
pip install sympy==1.12
pip install antlr4-python3-runtime==4.11 # The versions of sympy and antlr4 cannot be modified!!!!!
```

## 2 The General Process of Inference

The current codes are implemented specially for Gemma, Mistral, deepseek, and Llama3 (mainly in terms of the prompt format). You should include gemma, mistral, deepseek, or llama in the model name so that the code can specify the prompt format.


**Step 1** To start with, we prepare a prompt (problem) into the following Gemma format:

```python
prompt = "<bos><start_of_turn>user\nEvaluate $\\left\\lceil3\\left(6-\\frac12\\right)\\right\\rceil$.<end_of_turn>\n<start_of_turn>model"
```

**Step 2** The model received the prompt and generate some reasoning step and/or write some code.

```python
response = "To evaluate the expression, we will use Python's sympy library.\npython\\nfrom sympy import ceiling, Rational\\n\\n# Evaluate the expression\\nexpression_result = ceiling(3 * (6 - Rational(1, 2)))\\n\\nexpression_result\\n"
```

After getting the response, we first detect whether the model outputs the final results by \\boxed{FINAL RESULT}. If not, we continue to detect whether the model outputs a python code needs to be executed by "\`\`\`python SOME CODE \`\`\`". If yes, we run the code and get the result.

In this example, the execution result is "17". Therefore, we obtain the new prompt by adding the observation into the history:


```python
prompt2 = "<bos><start_of_turn>user\nEvaluate $\\left\\lceil3\\left(6-\\frac12\\right)\\right\\rceil$.<end_of_turn>\n<start_of_turn>model\nTo evaluate the expression, we will use Python's sympy library.\npython\\nfrom sympy import ceiling, Rational\\n\\n# Evaluate the expression\\nexpression_result = ceiling(3 * (6 - Rational(1, 2)))\\n\\nexpression_result\\n<end_of_turn>\n<start_of_turn>user\noutput\\n17\\n<end_of_turn>\n<start_of_turn>model\n"
```

A new step begins and we stop either the model outputs the final answer or reaches the maximal number of tool calls. 

## 3 Running the Generation Code

To run the generation, the first step is to register the model as a server so that we can query it. You should first modify the scripts/register_server.sh according to your GPU setup. Then, run the following command with your model.

```sh
bash scripts/register_server.sh google/gemma-1.1-7b-it
```

Then, you can run the scripts/infer_eval.sh, which will generate trajectories with the model and evaluate the output.

```sh
bash scripts/infer_eval.sh gemma_7b
```

The iter_infer_to_collect_data.sh additionally takes a for loop to iteratively generate trajectories. The model name should contain mistral, gemma, llama, or deepseek so that the code can specify the prompt format.


## 4 Annotate Data

```sh
python -um infer_data.annotate_data --data_name gsm8k --prompt_type tora --file_path ./collect_data/gemma_7b/gsm8k/train_tora_7473_seed1_t0.0_s0_e7473_09-22_16-22.jsonl --output_dir test_output.jsonl
```
