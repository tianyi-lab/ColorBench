#!/bin/bash

ROOT_DIR="PATH/TO/ROOT_DIR"
RESULT_DIR="PATH/TO/RESULT_DIR"
GEMINI_API_KEY="xxxx"
GPT4O_API_KEY="xxxx"

python3 model/llava_ov.py --modeltype="llava_ov_1b" --root_dir="${ROOT_DIR}" --save_dir="${RESULT_DIR}" --datatype="dataset"

python3 model/llava_next.py --modeltype="llava_16_v_7b" --root_dir="${ROOT_DIR}" --save_dir="${RESULT_DIR}" --datatype="dataset"

python3 model/internvl.py --modeltype="internvl2_1b" --root_dir="${ROOT_DIR}" --save_dir="${RESULT_DIR}" --datatype="dataset"

python3 model/eaglex5.py --modeltype="eaglex5_7b" --root_dir="${ROOT_DIR}" --save_dir="${RESULT_DIR}" --datatype="dataset"

python3 model/cambrian1.py --modeltype="cambrian_3b" --root_dir="${ROOT_DIR}" --save_dir="${RESULT_DIR}" --datatype="dataset"

python3 model/qwen.py --modeltype="qwen25_3b" --root_dir="${ROOT_DIR}" --save_dir="${RESULT_DIR}" --datatype="dataset"

python3 model/gpt4o_infer.py --root_dir="${ROOT_DIR}" --save_dir="${RESULT_DIR}" --datatype="dataset" --api_key="${GEMINI_API_KEY}" --use_cot

python3 model/gemini_infer.py --root_dir="${ROOT_DIR}" --save_dir="${RESULT_DIR}" --datatype="dataset" --api_key="${GPT4O_API_KEY}" --use_cot

