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
import torch
from PIL import Image
import json
import copy
import traceback
from tqdm import tqdm
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from argparse import ArgumentParser
import warnings

warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(53)
base_prompt = "USER: You'll be given an image, an instruction and some options. You have to select the correct one. Do not explain your reasoning. Answer with only the letter that corresponds to the correct option. \n"
cot_prompt = "USER: You'll be given an image, an instruction and some options. You have to select the correct one. Only one option is correct. \nThink step by step before answering. Then conclude with the letter that corresponds to the correct option. Make sure the option letter is in the parentheses like (X). Do not include ( or ) in the response except for the answer.\n"


def load_models(model_path: str, device: str, load_quantized: bool = False, ):
    model = LlavaNextForConditionalGeneration.from_pretrained(model_path, low_cpu_mem_usage=True, device_map="auto", torch_dtype=torch.float16, ).eval()

    processor = LlavaNextProcessor.from_pretrained(model_path, cache_dir=CACHE_DIR)
    processor.tokenizer.padding_side = "left"
    tokenizer = processor.tokenizer

    # find option-related tokens:
    vocab = tokenizer.get_vocab()
    tokens_with_ids, tokens_cluster = find_token_mappings(vocab)
    return model, tokenizer, processor, tokens_with_ids, tokens_cluster


def load_image(datatype: str, data: Dict, ):
    if datatype != 'json':
        image = data[f"image"].convert("RGB")
    else:
        image = Image.open(data[f"img_path"]).convert("RGB")

    return image


def prepare_prompt(d_prompt: str, processor, m_method: Optional[str]=None):
    if m_method is None:
        prompt = base_prompt + d_prompt
    else:
        # chain of thoughts
        prompt = cot_prompt + d_prompt

    conversation = [
        {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    return prompt


def prepare_model_input(prompt: str, image, processor, device: str):
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    return inputs


def process_output(generation_output, tokenizer, processor, input_ids, m_method=None):
    # replied answer
    outputs = generation_output.sequences[0].detach().cpu()
    model_answer = processor.decode(outputs, skip_special_tokens=True)  # output: str
    model_answer = model_answer.split("ASSISTANT: ")[-1].replace('[/INST]', '').strip()
    if len(model_answer) > 1:
        outputs_new = outputs[input_ids.shape[1]:]
        model_answer = processor.decode(outputs_new, skip_special_tokens=True)  # output: str

    if model_answer.lower() not in ('a', 'b', 'c', 'd', 'e'):
        if m_method is None:
            model_answer = unify_ans(model_answer)
    logits = generation_output.scores[0][0]  # shape: |V|

    return model_answer, logits


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="ROOT_DIR")
    parser.add_argument("--save_dir", type=str, default="SAVE_DIR")
    parser.add_argument("--modeltype", type=str, default="llava_16_v_7b")
    parser.add_argument("--datatype", type=str, default="dataset")
    parser.add_argument("--load_quantized", type=bool, default=True)
    args = parser.parse_args()

    root_dir = args.root_dir
    save_dir = args.save_dir
    load_quantized = args.load_quantized
    modeltype = args.modeltype
    datatype = args.datatype
    m_method = None  # None for fast-thinking, 'CoT' for slow-thinking

    # Load the model and processor
    model_path = 'llava-hf/llava-v1.6-vicuna-7b-hf'
    if modeltype == "llava_16_v_7b":
        model_path = 'llava-hf/llava-v1.6-vicuna-7b-hf'
    if modeltype == "llava_16_m_7b":
        model_path = 'llava-hf/llava-v1.6-mistral-7b-hf'
    elif modeltype == "llava_16_13b":
        model_path = 'llava-hf/llava-v1.6-vicuna-13b-hf'
    elif modeltype == "llava_16_34b":
        model_path = 'llava-hf/llava-v1.6-34b-hf'
    elif modeltype == "llava_16_72b":
        model_path = 'llava-hf/llava-next-72b-hf'

    print(f"Evaluating model: {model_path}")
    os.makedirs(save_dir, exist_ok=True)

    #############################
    # load model & tokenizer
    model, tokenizer, processor, tokens_with_ids, tokens_cluster = load_models(model_path=model_path, device=device, load_quantized=load_quantized)
    
    #############################
    # load data
    eval_dataset = load_data(data_type=datatype, root_dir=root_dir, )

    #############################
    # Start inference
    dict_result = {}
    for i, data in tqdm(enumerate(eval_dataset), total=len(eval_dataset)):
        try:

            # load image
            image = load_image(datatype=datatype, data=data, )

            # prepare prompt
            prompt = prepare_prompt(d_prompt=data["prompt"], processor=processor, m_method=m_method)

            # tokenize input
            inputs = prepare_model_input(prompt=prompt, image=image, processor=processor, device=model.device)

            with torch.no_grad():
                generation_output = model.generate(**inputs, do_sample=False, min_length=1, max_new_tokens=1024, return_dict_in_generate=True, output_scores=True, )

            ####################
            # Process answer
            model_answer, logits = process_output(generation_output, tokenizer, processor, input_ids=inputs['input_ids'], m_method=m_method)
            
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

    # Save results
    write_file = os.path.join(save_dir, f"{modeltype}.json")
    print(f"write to file {write_file}")
    with open(write_file, "w") as f:
        json.dump(dict_result, f, indent=4)
