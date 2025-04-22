import sys
import os

sys.path.insert(0, os.getcwd())
from utils.path_utils import *

CACHE_DIR = set_root_folder()

os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
os.environ["HF_MODULES_CACHE"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

from utils.infer_utils import *
from llava.model.builder import load_pretrained_model
from llava.mm_utils import (process_images, tokenizer_image_token, )
from llava.constants import (IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, )
from llava.conversation import conv_templates
from PIL import Image
from typing import Optional
import copy
import torch
import traceback
import warnings
import json
from tqdm import tqdm
from argparse import ArgumentParser

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(53)

conv_template = "qwen_1_5"
base_prompt = "USER: You'll be given an image, an instruction and some options. You have to select the correct one. Do not explain your reasoning. Answer with only the letter that corresponds to the correct option.\n"
cot_prompt = "USER: You'll be given an image, an instruction and some options. You have to select the correct one. \nThink step by step before answering. Then conclude with the letter that corresponds to the correct option. Make sure the option letter is in the parentheses like (X). Do not include ( or ) in the response except for the answer.\n"


def load_models(model_path: str, device: str, load_quantized: bool = False, ):

    model_name = "llava_qwen"
    device_map = "auto"
    tokenizer, model, image_processor, max_length = load_pretrained_model(model_path, None, model_name, device_map=device_map, torch_dtype="bfloat16", )
    model.eval()

    # find option-related tokens:
    vocab = tokenizer.get_vocab()
    tokens_with_ids, tokens_cluster = find_token_mappings(vocab)

    return model, tokenizer, image_processor, tokens_with_ids, tokens_cluster


def load_image(datatype: str, data: Dict, device: str, model, image_processor):
    if datatype != 'json':
        image = data[f"image"].convert("RGB")
    else:
        image = Image.open(data[f"img_path"]).convert("RGB")

    image_sizes = [image.size]  # type: ignore
    image = process_images([image], image_processor, model.config)
    image = [img.to(dtype=torch.bfloat16, device=device) for img in image]
    return image, image_sizes


def prepare_prompt(d_prompt: str, m_method: Optional[str]=None):
    if m_method is None:
        prompt = base_prompt + d_prompt
    else:
        # chain of thoughts
        prompt = cot_prompt + d_prompt

    prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return prompt


def prepare_model_input(prompt: str, tokenizer, device: str):
    input_ids = (tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device))
    return input_ids


def process_output(generation_output, tokenizer, update_ans_ids: bool = False, model_path='llava_ov_7b', m_method=None):
    # replied answer
    outputs = generation_output.sequences[0].detach().cpu()
    decode_res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    model_answer = decode_res[0]
    if model_path != 'llava_ov_72b':
        model_answer = model_answer.split("ASSISTANT: ")[-1][0].strip().lower()
    else:
        model_answer = model_answer.split("ASSISTANT: ")[-1].strip().lower()
    if model_answer.lower() not in ('a', 'b', 'c', 'd', 'e'):
        model_answer = ''.join(decode_res)
        if m_method is None:
            model_answer = unify_ans(model_answer)
    logits = generation_output.scores[0][0]  # shape: |V|
    return model_answer, logits


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="ROOT_DIR")
    parser.add_argument("--save_dir", type=str, default="SAVE_DIR")
    parser.add_argument("--modeltype", type=str, default="llava_ov_1b")
    parser.add_argument("--datatype", type=str, default="dataset")
    parser.add_argument("--load_quantized", type=bool, default=True)
    args = parser.parse_args()

    root_dir = args.root_dir
    save_dir = args.save_dir
    load_quantized = args.load_quantized
    modeltype = args.modeltype
    datatype = args.datatype
    m_method = None  # None for fast-thinking, 'CoT' for slow-thinking

    model_path = "lmms-lab/llava-onevision-qwen2-7b-ov"
    if modeltype == 'llava_ov_7b':
        model_path = "lmms-lab/llava-onevision-qwen2-7b-ov"
    elif modeltype == 'llava_ov_1b':
        model_path = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
    elif modeltype == 'llava_ov_72b':
        model_path = "lmms-lab/llava-onevision-qwen2-72b-ov-sft"
    print(f"Evaluating model: {model_path}")
    os.makedirs(save_dir, exist_ok=True)

    #############################
    # load model & tokenizer
    model, tokenizer, image_processor, tokens_with_ids, tokens_cluster = load_models(model_path=model_path, device=device, load_quantized=load_quantized)

    #############################
    # load data
    eval_dataset = load_data(data_type=datatype, root_dir=root_dir, )

    #############################
    # Start inference
    dict_result = dict()
    for i, data in enumerate(tqdm(eval_dataset)):
        try:
            # load image
            image, image_sizes = load_image(datatype=datatype, data=data, device=device, model=model, image_processor=image_processor)

            # prepare prompt
            prompt = prepare_prompt(d_prompt=data["prompt"], m_method=m_method)

            # tokenize input
            input_ids = prepare_model_input(prompt=prompt, tokenizer=tokenizer, device=device)

            with torch.no_grad():
                generation_output = model.generate(input_ids, images=image, image_sizes=image_sizes, do_sample=False, temperature=0, max_new_tokens=1024, return_dict_in_generate=True, output_scores=True, )
            
            ####################
            # Process answer
            model_answer, logits = process_output(generation_output, tokenizer, model_path=model_path, m_method=m_method)

            # calculate probs within options
            probs, logits_options, dict_option_prob = calculate_probs(logits=logits, list_options=data['choices'], tokens_with_ids=tokens_with_ids, tokens_cluster=tokens_cluster)

            dict_result[i] = copy.deepcopy(data)
            if 'image' in dict_result[i]:
                del dict_result[i]['image']
                
            dict_result[i]["model_ans"] = model_answer

        except Exception as e:
            print(e)
            print("skipping", i)
            torch.cuda.empty_cache()
            traceback.print_exc()
            sys.exit(-1)

    # save results to json
    write_file = os.path.join(save_dir, f"{modeltype}.json")
    print(f"write to file {write_file}")
    with open(write_file, "w") as f:
        json.dump(dict_result, f, indent=4)
