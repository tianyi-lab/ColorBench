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
from google import genai
import json
import argparse
import os
import time
import copy
import re
from tqdm import tqdm
from PIL import Image


# Function to load the image
def load_image(datatype: str, data):
    """Loads an image file using PIL."""
    if datatype != 'json':
        image = data[f"image"].convert("RGB")
    else:
        image = Image.open(data[f"img_path"]).convert("RGB")

    return image


def ask_gemini_about_image(api_key, datatype: str, data, use_cot=False, max_retries=3):
    """Sends an image and a question to Gemini for visual question answering with error handling."""
    image = load_image(datatype=datatype, data=data, )
    if image is None:
        return "Error: Image file missing or invalid"

    client = genai.Client(api_key=api_key)
    model = "gemini-2.0-flash"  # Using the flash version for faster response

    # Modify the question based on whether CoT is enabled
    if use_cot:
        # Chain of Thought prompt
        modified_question = data['prompt'] + "\nThink step by step before answering. Then conclude with the letter that corresponds to the correct option. Make sure the option letter is in the parentheses like (X)."
    else:
        # Original direct prompt
        modified_question = data['prompt'] + "\nAnswer with only the letter that corresponds to the correct option. Do not repeat the entire answer. Do not explain your reasoning."

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[modified_question, image]
            )
            return response.text.strip()
        
        except Exception as e:
            print(f"Error on attempt {attempt+1}: {e}")
            time.sleep(2)  # Wait before retrying

    return "Error: Failed after multiple attempts"


def process_json(api_key, datatype, eval_dataset, save_dir, use_cot=False):
    """Processes a JSON file to get Gemini answers and compute accuracy."""
        
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    if use_cot:
        modeltype = 'gemini_cot'
    else:
        modeltype = 'gemini'
    
    dict_result = dict()
    for i, data in enumerate(tqdm(eval_dataset)):
            
        # Process with CoT or without based on the parameter
        if use_cot:
            print(f"Processing ID: {i} with CoT")
        
            gemini_answer = ask_gemini_about_image(api_key, datatype, data, use_cot=True)
            print(f"Gemini CoT Answer: {gemini_answer}")
        else:
            print(f"Processing ID: {i} with CoT")
            gemini_answer = ask_gemini_about_image(api_key, datatype, data, use_cot=False)
            print(f"Gemini Answer: {gemini_answer}")

        dict_result[i] = copy.deepcopy(data)
        if 'image' in dict_result[i]:
            del dict_result[i]['image']
            
        dict_result[i]["model_ans"] = gemini_answer

    # Save the updated JSON file
    write_file = os.path.join(save_dir, f"{modeltype}.json")
    print(f"write to file {write_file}")
    with open(write_file, "w") as file:
        json.dump(data, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images and questions using Gemini.")
    
    parser.add_argument("--root_dir", type=str, default="ROOT_DIR")
    parser.add_argument("--save_dir", type=str, default="SAVE_DIR")
    parser.add_argument("--datatype", type=str, default="dataset")
    parser.add_argument("--api_key", type=str, default='', help="Gemini API key")
    parser.add_argument("--use_cot", action="store_true", help="Use Chain of Thought reasoning")

    args = parser.parse_args()

    root_dir = args.root_dir
    save_dir = args.save_dir
    datatype = args.datatype
    api_key = args.api_key
    use_cot = args.use_cot

    #############################
    # load data
    eval_dataset = load_data(data_type=datatype, root_dir=root_dir, )

    #############################
    # Start inference
    process_json(api_key, datatype, eval_dataset, save_dir, use_cot)
    