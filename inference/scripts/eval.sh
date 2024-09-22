
if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_path>"
    exit 1
fi
MODEL_NAME_OR_PATH=$1

# DATA_LIST = ['math', 'gsm8k', 'gsm-hard', 'svamp', 'tabmwp', 'asdiv', 'mawps']

DATA_NAME="gsm8k"

OUTPUT_DIR="./output1"

SPLIT="test"
PROMPT_TYPE="tora"
NUM_TEST_SAMPLE=-1


CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
python -um infer_data.infer_eval \
--model_name_or_path ${MODEL_NAME_OR_PATH} \
--data_name ${DATA_NAME} \
--output_dir ${OUTPUT_DIR} \
--split ${SPLIT} \
--prompt_type ${PROMPT_TYPE} \
--num_test_sample ${NUM_TEST_SAMPLE} \
--seed 1 \
--temperature 0 \
--n_sampling 1 \
--top_p 1 \
--start 0 \
--end -1 \
--horizon 4 \
--ports "8000" \
--ports "8001" \
--ports "8002" \
--ports "8003" \
--ports "8004" \
--ports "8005" \
--ports "8006" \
--ports "8007" \
--eval eval \



DATA_NAME="math"

CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
python -um infer_data.infer_eval \
--model_name_or_path ${MODEL_NAME_OR_PATH} \
--data_name ${DATA_NAME} \
--output_dir ${OUTPUT_DIR} \
--split ${SPLIT} \
--prompt_type ${PROMPT_TYPE} \
--num_test_sample ${NUM_TEST_SAMPLE} \
--seed 1 \
--temperature 0 \
--n_sampling 1 \
--top_p 1 \
--start 0 \
--end -1 \
--horizon 4 \
--ports "8000" \
--ports "8001" \
--ports "8002" \
--ports "8003" \
--ports "8004" \
--ports "8005" \
--ports "8006" \
--ports "8007" \
--eval eval \     
