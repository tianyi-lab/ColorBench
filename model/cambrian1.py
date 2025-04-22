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
from cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from cambrian.conversation import conv_templates
from cambrian.model.builder import load_pretrained_model
from cambrian.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

import traceback
from PIL import Image
from typing import Optional
import torch
import warnings
import re
import copy
import json
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(53)

base_prompt = "You'll be given an image, an instruction and some options. You have to select the correct one. Do not explain your reasoning. Answer with only the letter that corresponds to the correct option. Do not repeat the entire answer. Answer with the option's letter from the given choices directly.\n"
cot_prompt = "USER: You'll be given an image, an instruction and some options. You have to select the correct one. \nThink step by step before answering. Then conclude with the letter that corresponds to the correct option. Make sure the option letter is in the parentheses like (X). Do not include ( or ) in the response except for the answer.\n"


def load_models(model_path: str, device: str, load_quantized: bool = False, ):
    model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

    # find option-related tokens:
    vocab = tokenizer.get_vocab()
    tokens_with_ids, tokens_cluster = find_token_mappings(vocab)

    return model, image_processor, tokenizer, tokens_with_ids, tokens_cluster


def load_image(datatype: str, data: Dict, image_processor, model_config):
    if datatype != 'json':
        image = data[f"image"].convert("RGB")
    else:
        image = Image.open(data[f"img_path"]).convert("RGB")

    image_size = [image.size]
    image_tensor = process_images([image], image_processor, model_config)
    return image_size, image_tensor



def prepare_prompt(d_prompt: str, model_config, conv_mode: str, m_method: Optional[str]=None):
    if m_method is None:
        prompt = base_prompt + d_prompt
    else:
        # chain of thoughts
        prompt = cot_prompt + d_prompt

    if model_config.mm_use_im_start_end:
        prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + d_prompt
    else:
        prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt 

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return prompt


def prepare_model_input(prompt: str, tokenizer, device: str):
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
    return input_ids


def process_output(generation_output, tokenizer, input_ids, update_ans_ids: bool = False):
    # replied answer
    outputs = generation_output.sequences[0].detach().cpu()
    model_answer = tokenizer.decode(outputs, skip_special_tokens=True)
    logits = generation_output.scores[0][0]  # shape: |V|
    return model_answer, logits


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="ROOT_DIR")
    parser.add_argument("--save_dir", type=str, default="SAVE_DIR")
    parser.add_argument("--modeltype", type=str, default="cambrian_3b")
    parser.add_argument("--datatype", type=str, default="dataset")
    parser.add_argument("--load_quantized", type=bool, default=True)
    args = parser.parse_args()

    root_dir = args.root_dir
    save_dir = args.save_dir
    load_quantized = args.load_quantized
    modeltype = args.modeltype
    datatype = args.datatype
    m_method = None  # None for fast-thinking, 'CoT' for slow-thinking

    # defind model
    model_path = "nyu-visionx/cambrian-8b"
    conv_mode = "llama_3" 
    if modeltype =='cambrian_3b':
        model_path = "nyu-visionx/cambrian-phi3-3b"
        conv_mode = "phi3"
    elif modeltype == 'cambrian_8b':
        model_path = "nyu-visionx/cambrian-8b"
        conv_mode = "llama_3" 
    elif modeltype =='cambrian_13b':
        model_path = "nyu-visionx/cambrian-13b"
        conv_mode = "vicuna_v1"
    elif modeltype == 'cambrian_34b':
        model_path = "nyu-visionx/cambrian-34b"
        conv_mode = "chatml_direct"
    print(f"Evaluating model: {model_path}")
    os.makedirs(save_dir, exist_ok=True)

    #############################
    # load model & tokenizer
    model, image_processor, tokenizer, tokens_with_ids, tokens_cluster = load_models(model_path=model_path, device=device, load_quantized=load_quantized)
    
    #############################
    # load data
    eval_dataset = load_data(data_type=datatype, root_dir=root_dir, )

    #############################
    # Start inference
    dict_result = dict()
    for i, data in enumerate(tqdm(eval_dataset)):
        try:
            # load image
            image_size, image = load_image(datatype=datatype, data=data, image_processor=image_processor, model_config=model.config)

            # prepare prompt
            prompt = prepare_prompt(d_prompt=data["prompt"], model_config=model.config, conv_mode=conv_mode, m_method=m_method)

            # tokenize input
            input_ids = prepare_model_input(prompt=prompt, tokenizer=tokenizer, device=device)

            # inference
            with torch.no_grad():
                generation_output = model.generate(input_ids, images=image, image_sizes=image_size, num_beams=1, do_sample=False, temperature=0, max_new_tokens=1, return_dict_in_generate=True, output_scores=True, )

            ####################
            # Process answer
            model_answer, logits = process_output(generation_output, tokenizer, input_ids=input_ids, update_ans_ids=True)

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
