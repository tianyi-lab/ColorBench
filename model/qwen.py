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
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import traceback
from PIL import Image
import torch
import warnings
import copy
import json
from tqdm import tqdm
from argparse import ArgumentParser
from io import BytesIO
from typing import Optional
import base64
warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(53)

base_prompt = "You'll be given an image, an instruction and some options. You have to select the correct one. Do not explain your reasoning. Answer with only the letter that corresponds to the correct option. Do not repeat the entire answer. \n"
cot_prompt = "USER: You'll be given an image, an instruction and some options. You have to select the correct one. \nThink step by step before answering. Then conclude with the letter that corresponds to the correct option. Make sure the option letter is in the parentheses like (X). Do not include ( or ) in the response except for the answer.\n"


def load_models(model_path: str, ):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto", )
    processor = AutoProcessor.from_pretrained(model_path)

    # find option-related tokens:
    tokenizer = processor.tokenizer
    vocab = tokenizer.get_vocab()
    tokens_with_ids, tokens_cluster = find_token_mappings(vocab)

    return model, processor, tokens_with_ids, tokens_cluster


def load_image(datatype: str, data: Dict, ):
    if datatype != 'json':
        image = data[f"image"].convert("RGB")
    else:
        image = Image.open(data[f"img_path"]).convert("RGB")
    
    buffered = BytesIO()
    image.save(buffered, format="JPEG") 
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def prepare_prompt(d_prompt: str, image, processor, m_method: Optional[str]=None):
    if m_method is None:
        prompt = base_prompt + d_prompt
    else:
        # chain of thoughts
        prompt = cot_prompt + d_prompt
    conversation = [{"role": "user", "content": [{"type": "image", "image": "data:image;base64,"+image, }, {"type": "text", "text": prompt}, ],} ]
    
    # Preparation for inference
    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    return prompt, conversation


def prepare_model_input(prompt: str, messages, processor, ):
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[prompt], images=image_inputs, padding=True, return_tensors="pt", )
    inputs = inputs.to("cuda")
    return inputs


def process_output(generation_output, processor, input_ids, update_ans_ids: bool = False):
    # replied answer
    outputs = generation_output.sequences.detach().cpu()
    outputs_strimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, outputs)]
    model_answer = processor.tokenizer.batch_decode(outputs_strimmed, skip_special_tokens=True)[0]
    logits = generation_output.scores[0][0]  # shape: |V|
    return model_answer, logits


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="ROOT_DIR")
    parser.add_argument("--save_dir", type=str, default="SAVE_DIR")
    parser.add_argument("--modeltype", type=str, default="qwen25_3b")
    parser.add_argument("--datatype", type=str, default="dataset")
    parser.add_argument("--load_quantized", type=bool, default=True)
    args = parser.parse_args()
    
    root_dir = args.root_dir
    save_dir = args.save_dir
    load_quantized = args.load_quantized
    modeltype = args.modeltype
    datatype = args.datatype
    m_method = None  # None for fast-thinking

    model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    if modeltype == 'qwen25_7b':
        model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    elif modeltype == 'qwen25_3b':
        model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    elif modeltype == 'qwen25_72b':
        model_path = "Qwen/Qwen2.5-VL-72B-Instruct"
        # model_path = "Qwen/Qwen2.5-VL-72B-Instruct-AWQ"
    print(f"Evaluating model: {model_path}")
    os.makedirs(save_dir, exist_ok=True)

    #############################
    # load model & tokenizer
    model, processor, tokens_with_ids, tokens_cluster = load_models(model_path=model_path, )

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
            prompt, conversation = prepare_prompt(d_prompt=data["prompt"], image=image, processor=processor, m_method=m_method)
            # tokenize input
            inputs = prepare_model_input(prompt=prompt, messages=conversation, processor=processor, )

            # inference
            with torch.no_grad():
                generation_output = model.generate(**inputs, min_length=1, do_sample=False, temperature=0, max_new_tokens=2048, return_dict_in_generate=True, output_scores=True, )

            ####################
            # Process answer
            model_answer, logits = process_output(generation_output, processor, input_ids=inputs.input_ids, update_ans_ids=True)

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
