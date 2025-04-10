import sys
import os

sys.path.insert(0, os.getcwd())
from utils.path_utils import *

CACHE_DIR, ROOT_DIR, _ = set_root_folder()

os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
os.environ["HF_MODULES_CACHE"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

from utils.infer_utils import *
from eagle import conversation as conversation_lib
from eagle.constants import DEFAULT_IMAGE_TOKEN
from eagle.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from eagle.conversation import conv_templates, SeparatorStyle
from eagle.model.builder import load_pretrained_model
from eagle.utils import disable_torch_init
from eagle.mm_utils import tokenizer_image_token, get_model_name_from_path, process_images, KeywordsStoppingCriteria
from transformers import TextIteratorStreamer
from threading import Thread

import traceback
from PIL import Image
import torch
import warnings
import json
import copy
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(53)

base_prompt = "You'll be given an image, an instruction and some options. You have to select the correct one. Do not explain your reasoning. Answer with only the letter that corresponds to the correct option. Do not repeat the entire answer. \n"
cot_prompt = "USER: You'll be given an image, an instruction and some options. You have to select the correct one. \nThink step by step before answering. Then conclude with the letter that corresponds to the correct option. Make sure the option letter is in the parentheses like (X). Do not include ( or ) in the response except for the answer. \n"


def load_models(model_path: str, device: str, load_quantized: bool = False, ):
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, False, False)

    # find option-related tokens:
    vocab = tokenizer.get_vocab()
    tokens_with_ids, tokens_cluster = find_token_mappings(vocab)

    return model, tokenizer, image_processor, context_len, tokens_with_ids, tokens_cluster


def load_image(datatype: str, data: Dict, ):
    if datatype != 'json':
        image = data[f"image"].convert("RGB")
    else:
        image = Image.open(data[f"img_path"]).convert("RGB")

    return image


def prepare_prompt(d_prompt: str, model, conv_mode: str, m_method: Optional[str]=None):
    if m_method is None:
        input_prompt = base_prompt + d_prompt
    else:
        # chain of thoughts
        input_prompt = cot_prompt + d_prompt
        
    if model.config.mm_use_im_start_end:
        input_prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + input_prompt
    else:
        input_prompt = DEFAULT_IMAGE_TOKEN + '\n' + input_prompt

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], input_prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return prompt


def prepare_model_input(prompt: str, image, model, tokenizer, image_processor, device: str):
    image_tensor = process_images([image], image_processor, model.config)[0]
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

    input_ids = input_ids.to(device=device, non_blocking=True)
    image_tensor = image_tensor.to(dtype=torch.float16, device=device, non_blocking=True)

    return input_ids, image_tensor


def process_output(generation_output, tokenizer, update_ans_ids: bool = False, m_method=None):
    # replied answer
    outputs = generation_output.sequences[0].detach().cpu()
    decode_res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    model_answer = decode_res[0]
    logits = generation_output.scores[0][0]  # shape: |V|
    if model_answer.lower() not in ('a', 'b', 'c', 'd', 'e') and len(decode_res) > 1:
        model_answer = ''.join(decode_res)
        if ' ' not in model_answer:
            model_answer = ' '.join(decode_res)
        if m_method is None:
            model_answer = unify_ans(model_answer)
    return model_answer, logits


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="ROOT_DIR")
    parser.add_argument("--save_dir", type=str, default="SAVE_DIR")
    parser.add_argument("--modeltype", type=str, default="eaglex5_7b")
    parser.add_argument("--datatype", type=str, default="dataset")
    parser.add_argument("--load_quantized", type=bool, default=True)
    args = parser.parse_args()

    root_dir = args.root_dir
    save_dir = args.save_dir
    load_quantized = args.load_quantized
    modeltype = args.modeltype
    datatype = args.datatype
    m_method = None  # None for fast-thinking, 'CoT' for slow-thinking

    model_path = "NVEagle/Eagle-X5-7B"
    if modeltype == 'eaglex5_7b':
        model_path = "NVEagle/Eagle-X5-7B"
        conv_mode = "vicuna_v1"
    elif modeltype == 'eaglex4_8b':
        model_path = "NVEagle/Eagle-X4-8B-Plus"
        conv_mode = "llama3"
    elif modeltype == 'eaglex4_13b':
        model_path = "NVEagle/Eagle-X4-13B-Plus"
        conv_mode = "vicuna_v1"
    elif modeltype == 'eaglex5_34b':
        model_path = "NVEagle/Eagle-X5-34B-Plus"
        conv_mode = "yi_34b_chatml_direct"
    print(f"Evaluating model: {model_path}")
    os.makedirs(save_dir, exist_ok=True)

    #############################
    # load model & tokenizer
    model, tokenizer, image_processor, context_len, tokens_with_ids, tokens_cluster = load_models(model_path=model_path, device=device, load_quantized=load_quantized)

    #############################
    # load data
    eval_dataset = load_data(data_type=datatype, root_dir=root_dir, )

    #############################
    # Start inference
    dict_result = dict()
    for i, data in enumerate(tqdm(eval_dataset)):
        try:
            # load image
            image = load_image(datatype=datatype, data=data, )

            # prepare prompt
            prompt = prepare_prompt(d_prompt=data["prompt"], model=model, conv_mode=conv_mode, m_method=m_method)

            # tokenize input
            input_ids, image_tensor = prepare_model_input(image=image, prompt=prompt, model=model, image_processor=image_processor, tokenizer=tokenizer, device=device)

            # inference
            with torch.no_grad():
                generation_output = model.generate(input_ids.unsqueeze(0), images=image_tensor.unsqueeze(0), image_sizes=[image.size], min_length=1, do_sample=False, use_cache=True, temperature=0, max_new_tokens=1, return_dict_in_generate=True, output_scores=True, )

            ####################
            # Process answer
            model_answer, logits = process_output(generation_output, tokenizer, update_ans_ids=True)

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
