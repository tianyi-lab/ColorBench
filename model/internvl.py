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
from transformers import AutoTokenizer, AutoModel
import traceback
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import warnings
import json
import re
import math
import copy
from tqdm import tqdm
from argparse import ArgumentParser

warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(53)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
generation_config = dict(max_new_tokens=1024, do_sample=False)
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
base_prompt = "Answer with only the letter that corresponds to the correct option. Do not repeat the entire answer. Do not explain your reasoning."
cot_prompt = "Think step by step before answering. Then conclude with the letter that corresponds to the correct option. Make sure the option letter is in the parentheses like (X). Do not include ( or ) in the response except for the answer.\n"


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img), T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
                        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size, (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size,
               ((i // (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_internvl_image(image_cv2=None, image_file=None, input_size=448, max_num=12):
    if image_file is not None:
        image = Image.open(image_file).convert('RGB')
    if image_cv2 is not None:
        image = image_cv2

    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def load_image(datatype: str, data: Dict, ):
    if datatype != 'json':
        image = data[f"image"].convert("RGB")
        pixel_values = load_internvl_image(image_cv2=data[f"image"].convert("RGB"), max_num=12).to(
            torch.bfloat16).cuda()
    else:
        image = Image.open(data[f"img_path"]).convert("RGB")
        pixel_values = load_internvl_image(image_file=data[f"img_path"], max_num=12).to(torch.bfloat16).cuda()
    return image, pixel_values


def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80, 
        'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


def load_models(model_path: str, device: str, load_quantized: bool = False, ):
    if model_path not in ('OpenGVLab/InternVL2-Llama3-76B', "OpenGVLab/InternVL2_5-78B"):
        model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, load_in_8bit=load_quantized, low_cpu_mem_usage=True, use_flash_attn=True, trust_remote_code=True, cache_dir=CACHE_DIR).to(device).eval()
    else:
        device_map = split_model(model_path.split('/')[-1])

        model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device_map, load_in_8bit=True,  low_cpu_mem_usage=True, use_flash_attn=True, trust_remote_code=True, cache_dir=CACHE_DIR).eval()
        
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_context_token_id = img_context_token_id

    # find option-related tokens:
    vocab = tokenizer.get_vocab()
    tokens_with_ids, tokens_cluster = find_token_mappings(vocab)

    return model, tokenizer, tokens_with_ids, tokens_cluster


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="ROOT_DIR")
    parser.add_argument("--save_dir", type=str, default="SAVE_DIR")
    parser.add_argument("--modeltype", type=str, default="internvl2_1b")
    parser.add_argument("--datatype", type=str, default="dataset")
    parser.add_argument("--load_quantized", type=bool, default=False)
    args = parser.parse_args()
    
    root_dir = args.root_dir
    save_dir = args.save_dir
    load_quantized = args.load_quantized
    modeltype = args.modeltype
    datatype = args.datatype
    m_method = None  # None for fast-thinking, 'CoT' for slow-thinking

    model_path = "OpenGVLab/InternVL2-8B"
    if modeltype == 'internvl2_1b':
        model_path = "OpenGVLab/InternVL2-1B"
    elif modeltype == 'internvl2_2b':
        model_path = "OpenGVLab/InternVL2-2B"
    elif modeltype == 'internvl2_4b':
        model_path = "OpenGVLab/InternVL2-4B"
    elif modeltype == 'internvl2_8b':
        model_path = "OpenGVLab/InternVL2-8B"
    elif modeltype == 'internvl2_26b':
        model_path = "OpenGVLab/InternVL2-26B"
    elif modeltype == 'internvl2_40b':
        model_path = "OpenGVLab/InternVL2-40B"
    elif modeltype == 'internvl2_72b':
        model_path = "OpenGVLab/InternVL2-Llama3-76B"
    elif modeltype == 'internvl25_1b':
        model_path = "OpenGVLab/InternVL2_5-1B"
    elif modeltype == 'internvl25_2b':
        model_path = "OpenGVLab/InternVL2_5-2B"
    elif modeltype == 'internvl25_4b':
        model_path = "OpenGVLab/InternVL2_5-4B"
    elif modeltype == 'internvl25_8b':
        model_path = "OpenGVLab/InternVL2_5-8B"
    elif modeltype == 'internvl25_26b':
        model_path = "OpenGVLab/InternVL2_5-26B"
    elif modeltype == 'internvl25_38b':
        model_path = "OpenGVLab/InternVL2_5-38B"
    elif modeltype == 'internvl25_72b':
        model_path = "OpenGVLab/InternVL2_5-78B"
    print(f"Evaluating model: {model_path}")
    os.makedirs(save_dir, exist_ok=True)

    #############################
    # load model & tokenizer
    model, tokenizer, tokens_with_ids, tokens_cluster = load_models(model_path=model_path, device=device, load_quantized=load_quantized)
    
    #############################
    # load data
    eval_dataset = load_data(data_type=datatype, root_dir=root_dir, )

    #############################
    # Start inference
    dict_result = dict()
    for i, data in enumerate(tqdm(eval_dataset)):
        try:
            # load image
            image, pixel_values = load_image(datatype=datatype, data=data, )

            if m_method is None:
                prompt = base_prompt + data["prompt"]
            else:
                # chain of thoughts
                prompt = cot_prompt + data["prompt"]

            model_answer, _ = model.chat(tokenizer, pixel_values, prompt, generation_config, history=None, return_history=True)
            model_answer = unify_ans(model_answer)

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
