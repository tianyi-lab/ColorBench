import json
import re
import torch
import copy
from typing import List, Dict, Tuple, Optional
import torch.nn.functional as F
from .path_utils import *

_, ROOT_FOLDER, _ = set_root_folder()

from datasets import load_dataset


# Define a function to check the token format
def is_valid_token(token, ):
    # Check if the token matches the required format
    if len(token) == 1 and token.upper() in "ABCDEF":  # Single letter like 'a' or 'A'
        return True, token.upper()
    if len(token) == 2 and token.startswith(" ") and token[1].upper() in "ABCDEF":  # ' a' or ' A'
        return True, token[1].upper()
    if len(token) == 2 and token.startswith("_") and token[1].upper() in "ABCDEF":  # ' a' or ' A'
        return True, token[1].upper()
    if len(token) == 2 and token.startswith("▁") and token[1].upper() in "ABCDEF":  # ' a' or ' A'
        return True, token[1].upper()
    if len(token) == 2 and token.startswith("(") and token[1].upper() in "ABCDEF":  # ' a' or ' A'
        return True, token[1].upper()
    if len(token) == 3 and token.startswith("(") and token.endswith(")") and token[1].upper() in "ABCDEF":  # '(A)'
        return True, token[1].upper()
    return False, None


def find_token_mappings(vocab_dict):
    # find option-related tokens:
    tokens_with_ids = dict()
    tokens_cluster = dict()
    for token, token_id in vocab_dict.items():
        res, token_format = is_valid_token(token)
        if res:
            tokens_with_ids[token] = token_id
            if token_format not in tokens_cluster:
                tokens_cluster[token_format] = []
            tokens_cluster[token_format].append(token)

    return tokens_with_ids, tokens_cluster


def check_ans(pr_ans: str, gt_ans: str):
    if pr_ans.strip().lower() == gt_ans.strip().lower():
        return True
    else:
        return False


def load_data(data_type: str, root_dir: str='', ):
    # load data
    if data_type not in ('json',):
        # hf dataset
        eval_dataset = load_dataset("umd-zhou-lab/ColorBench", split='test')
    else:
        # json
        with open(f"{root_dir}/all_data.json", 'r') as f:
            eval_dataset = json.load(f)

        # change image file path
        for i, data in enumerate(eval_dataset):
            img_path = os.path.join(root_dir, data['filename'])
            data['img_path'] = img_path
    return eval_dataset


def calculate_probs(logits, list_options: List, tokens_with_ids: Dict, tokens_cluster: Dict):
    # calculate probs within options
    logits = logits.detach().cpu()
    options = [f"{chr(65 + opt_i)}" for opt_i, item in enumerate(list_options)]

    # Initialize a dictionary to store aggregated logits for each option
    aggregated_logits = {}
    dict_option_prob = {}

    for option in options:
        # Get all related formats of the option from tokens_cluster
        related_tokens = tokens_cluster.get(option, [])

        # Sum the logits of all formats of the option
        aggregated_logit = sum(logits[tokens_with_ids[token]] for token in related_tokens if token in tokens_with_ids)
        aggregated_logits[option] = aggregated_logit
        for token in related_tokens:
            dict_option_prob[token] = logits[tokens_with_ids[token]].detach().cpu().numpy().item()

    # Convert aggregated logits to a tensor
    logits_options = torch.tensor([aggregated_logits[option] for option in options])
    probs = F.softmax(logits_options, dim=0, ).detach().cpu().numpy()

    return probs.tolist(), logits_options.tolist(), dict_option_prob


def unify_ans(answer: str, ):

    formated_answer = answer.replace('(', '').replace(')', '').lower()
    if formated_answer not in ('a', 'b', 'c', 'd', 'e', 'f'):
        # find the option letter
        match = re.search(r"\((a|b|c|d|e|f)\)", answer.lower())
        if match: 
            formated_answer = match.group(0).replace('(', '').replace(')', '').lower()

    if formated_answer not in ('a', 'b', 'c', 'd', 'e', 'f'):
        # find the option letter
        match = re.search(r"([a-z])\) \d+", answer.lower())
        if match: 
            formated_answer = match.group(1)
    
    return formated_answer


def check_answer(model_ans, gt_ans, ):
    gt_ans = gt_ans.replace('(', '').replace(')', '').lower()
    model_ans = unify_ans(model_ans, )

    if model_ans == gt_ans:
        return True, model_ans
    else:
        return False, model_ans


def extract_letter_cot(answer):
    """Extracts the letter choice from an answer that's in parentheses like (X)."""
    # Look for the last occurrence of a pattern like (A), (B), etc.
    matches = re.findall(r"\(([A-Za-z])\)", answer.strip())
    return matches[-1].lower() if matches else ""  # Return the last letter found in parentheses, converted to uppercase


def extract_letter(answer):
    """Extracts the last letter choice from an answer, ensuring it is uppercase (e.g., '(a)' → 'A', 'Selected: (c)' → 'C')."""
    matches = re.findall(r"[A-Za-z]", answer.strip())  # Find all letters (uppercase & lowercase)
    return matches[-1].lower() if matches else ""  # Return the last letter found, converted to uppercase


def parse_res(model_ans, options, gt_ans):
    str_opt = [str(item).lower() for item in options if item != '']
    check_res, model_ans_new = check_answer(model_ans.strip(), gt_ans)
    find_res = True
    if model_ans_new.lower() not in ('a', 'b', 'c', 'd', 'e'):
        if len(model_ans_new.lower().split(' ')) == 2 and model_ans_new.lower().split(' ')[0] in ('a', 'b', 'c', 'd', 'e'):
            model_ans_new = model_ans_new.lower().split(' ')[0]
        elif model_ans_new in options:
            ans_id = options.index(model_ans_new)
            model_ans_new = chr(65 + ans_id)
        elif len([item for item in str_opt if item in model_ans_new or item.replace(' ', '') in model_ans_new]) == 1:
            ans_id = [item for item in str_opt if item in model_ans_new or item.replace(' ', '') in model_ans_new][0]
            ans_id = str_opt.index(ans_id)
            model_ans_new = chr(65 + ans_id)   
        else:
            find_res = False
    return model_ans_new, check_res, find_res