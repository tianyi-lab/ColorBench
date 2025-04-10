import os
import sys
sys.path.insert(0, os.getcwd())

from utils.infer_utils import *
import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from argparse import ArgumentParser


def process_PandR(list_models: List, result_dir: str, list_tasks_PR: List, ):

    dict_cate_task = dict()
    dict_model_cnt = dict()
    list_acc = []
    for model_name in list_models:
        dict_model_cnt[model_name] = dict()  # save acc for each task
        json_path = os.path.join(result_dir, f"{model_name}.json")
        with open(json_path, 'r') as f:
            dict_res = json.load(f)

        for sample_idx, img_meta in dict_res.items():
            category = img_meta['type']
            if category == 'Robustness':
                continue
            
            task = img_meta['task']
            if task not in dict_cate_task:
                dict_cate_task[task] = category
            
            # stores correct / incorrect cnt for each task
            if task not in dict_model_cnt[model_name]:
                dict_model_cnt[model_name][task] = [0, 0]  # [correct, incorrect]

            options = img_meta['choices']
            gt_ans = img_meta['answer'].replace('(', '').replace(')', '').lower()
            model_ans = img_meta['model_ans']

            if 'cot' in model_name:  # for gpt / gemini cot
                model_ans_new = extract_letter_cot(model_ans)
                if_correct = (model_ans_new == gt_ans)
            elif 'gpt' in model_name or 'gemini' in model_name:  # for gpt / gemini
                model_ans_new = extract_letter(model_ans)
                if_correct = (model_ans_new == gt_ans)
            else:  # for open sourced model
                model_ans_new, if_correct, find_res = parse_res(model_ans=model_ans, options=options, gt_ans=gt_ans)

            if if_correct:
                # correct
                dict_model_cnt[model_name][task][0] += 1
            else:
                # incorrect
                dict_model_cnt[model_name][task][1] += 1

        # print model acc:
        dict_acc = {key: [sum(item), item[0], item[0]/sum(item)] for key, item in dict_model_cnt[model_name].items() if sum(item) > 0}
        for task, (sum_cnt, cor_cnt, acc) in dict_acc.items():
            list_acc.append([model_name, task, sum_cnt, cor_cnt, np.round(acc, 6)])

    dict_formated = {key: [0]*len(list_tasks_PR) for key in list_models}
    dict_formated_cor = {key: [0]*len(list_tasks_PR) for key in list_models}
    dict_formated_sum = {key: [0]*len(list_tasks_PR) for key in list_models}

    for item_meta in list_acc:
        model_name, task, sum_cnt, cor_cnt, acc = item_meta
        t_idx = list_tasks_PR.index(task)
        dict_formated[model_name][t_idx] = acc
        dict_formated_cor[model_name][t_idx] = cor_cnt
        dict_formated_sum[model_name][t_idx] = sum_cnt

    # calculate perception / reasoning / overall acc
    for model_name in dict_formated_sum.keys():
        percept_cnt = [0, 0]
        reasoning_cnt = [0, 0]
        overall_cnt = [0, 0]

        for task_id, task in enumerate(list_tasks_PR):
            category = dict_cate_task[task]
            cor_cnt = dict_formated_cor[model_name][task_id]
            all_cnt = dict_formated_sum[model_name][task_id]
            overall_cnt[0] += cor_cnt
            overall_cnt[1] += all_cnt
            if category.lower() == 'perception':
                percept_cnt[0] += cor_cnt
                percept_cnt[1] += all_cnt
            if category.lower() == 'reasoning':
                reasoning_cnt[0] += cor_cnt
                reasoning_cnt[1] += all_cnt
        
        try:
            perception_acc = np.round(percept_cnt[0]/percept_cnt[1], 6)
            reasoning_acc = np.round(reasoning_cnt[0]/reasoning_cnt[1], 6)
            overall_acc = np.round(overall_cnt[0]/overall_cnt[1], 6)
        except:
            perception_acc, reasoning_acc, overall_acc = 0, 0, 0
        dict_formated[model_name].extend([perception_acc, reasoning_acc, overall_acc])
    return dict_formated


def process_robustness(list_models: List, result_dir: str, dict_formated: Dict, ):
    # Count robustness
    dict_id_newres = dict()
    dict_model_cnt = dict()
    list_rob = []
    for model_name in list_models:
        dict_model_cnt[model_name] = [0, 0]  # save cnt for each model
        dict_id_newres[model_name] = dict()  # save cnt for each model
        dict_correct = dict()
        json_path = os.path.join(result_dir, f"{model_name}.json")
        with open(json_path, 'r') as f:
            dict_res = json.load(f)

        for sample_idx, img_meta in dict_res.items():
            category = img_meta['type']
            if category != 'Robustness':
                continue
            
            if_ori = False
            img_name = img_meta["filename"].split('/')[-1].split('.')[0]
            if '_' not in img_name:
                # original image
                if_ori = True
                img_id = int(img_name)
            else:
                # recolored image
                img_id = int(img_name.split('_')[0])

            options = img_meta['choices']
            gt_ans = img_meta['answer'].replace('(', '').replace(')', '').lower()
            model_ans = img_meta['model_ans']

            if img_id not in dict_id_newres[model_name]:
                dict_id_newres[model_name][img_id] = [gt_ans, None, []]  # [gt answer, result for original image, list of results for recolored image]
                dict_correct[img_id] = [False, []]  # [correct / not for original image, list of bool]

            if 'cot' in model_name:  # for gpt / gemini cot
                model_ans_new = extract_letter_cot(model_ans)
                if_correct = (model_ans_new == gt_ans)
            elif 'gpt' in model_name or 'gemini' in model_name:  # for gpt / gemini
                model_ans_new = extract_letter(model_ans)
                if_correct = (model_ans_new == gt_ans)
            else:  # for open sourced model
                model_ans_new, if_correct, find_res = parse_res(model_ans=model_ans, options=options, gt_ans=gt_ans)

            if if_ori: 
                dict_id_newres[model_name][img_id][1] = model_ans_new
                dict_correct[img_id][0] = if_correct
            else:
                dict_id_newres[model_name][img_id][2].append(model_ans_new)
                dict_correct[img_id][1].append(if_correct)

        # cnt robust answers
        for img_id, list_res in dict_id_newres[model_name].items():
            gt_ans, ori_ans, list_new_ans = list_res
            ori_bool, list_new_bool = dict_correct[img_id]
            if ori_bool and False not in list_new_bool:
                # robust
                dict_model_cnt[model_name][0] += 1
            else:
                # not
                dict_model_cnt[model_name][1] += 1

    # print model robust:
    dict_robust = {key: [sum(item), item[0], item[0]/sum(item)] for key, item in dict_model_cnt.items() if sum(item) > 0}
    for model_name, (sum_cnt, rob_cnt, robustness) in dict_robust.items():
        list_rob.append([model_name, sum_cnt, rob_cnt, np.round(robustness, 6)])

    for item_meta in list_rob:
        model_name, sum_cnt, rob_cnt, robustness = item_meta
        dict_formated[model_name].append(robustness)

    return dict_formated


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--result_dir", type=str, default="RESULT_DIR")
    parser.add_argument("--save_dir", type=str, default="SAVE_DIR")
    args = parser.parse_args()
    
    result_dir = args.result_dir
    save_dir = args.save_dir

    list_tasks_PR = ["Color Recognition", "Color Extraction", "Object Recognition", "Color Proportion", "Color Comparison", "Color Counting", "Object Counting", "Color Illusion", "Color Mimicry", "Color Blindness",]

    # Load model inference results
    list_jsons = os.listdir(result_dir)
    list_models = [item.split('.')[0] for item in list_jsons if 'json' in item]

    # Count acc for each task
    dict_formated = process_PandR(list_models=list_models, result_dir=result_dir, list_tasks_PR=list_tasks_PR)
    dict_formated = process_robustness(list_models=list_models, result_dir=result_dir, dict_formated=dict_formated, )

    # Save to csv
    df_result = pd.DataFrame(dict_formated,).T
    df_result.columns = list_tasks_PR + ['Perception Acc', 'Reasoning Acc', 'Overall Acc'] + ['Color Robustness']
    df_result = df_result.reset_index().rename(columns={'index': 'model_type',})
    df_result.to_csv(os.path.join(save_dir, 'inference.csv'), index=False)
