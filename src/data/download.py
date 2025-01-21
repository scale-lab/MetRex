"""Download vgen and RTL-Coder datasets.
"""
import os
import re
import json
import tqdm
import argparse
import requests

from datasets import load_dataset


RTL_CODER_URL = "https://raw.githubusercontent.com/hkust-zhiyao/RTL-Coder/main/dataset/Resyn27k.json"

def clean_comments(verilog):
    patterns = [
        r'(?:\/\/[^\n]*\n){2,}|\/\*[\s\S]*?\*\/',
        r'(?m)^\s*`timescale\s+.*?$|^\s*`default_nettype\s+.*?$|\(\*.*?\*\)',
        r"^\s*`(?:ifndef|define|endif)\s+.*?SKY130.*?$|^\s*`(celldefine|endcelldefine)\s*$",
        r'r"^\s*supply[01]\s+\w+\s*;\s*$"'
    ]

    cleaned_code = verilog
    for pattern in patterns: 
        cleaned_code = re.sub(pattern, '', cleaned_code, flags=re.MULTILINE)
    return cleaned_code


def download_vgen(output_dir):
    
    dataset = load_dataset(
        "shailja/Verilog_GitHub", 
        streaming=True,
        split="train"
    )

    count = 0
    for entry in tqdm.tqdm(dataset):
        name = f"vgen_{count}.v"
        count += 1 
        verilog = entry['text']
        verilog = clean_comments(verilog)
        f = open(os.path.join(output_dir, name), "w")
        f.write(verilog)
        f.close()
       


def download_rtlcoder(output_dir):

    response = requests.get(RTL_CODER_URL)

    json_file = os.path.join(output_dir, "Resyn27k.json")
    if response.status_code == 200:
        with open(json_file, "w", encoding="utf-8") as file:
            file.write(response.text)
        print(f"JSON file downloaded and saved as {json_file}.")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")
        
    des_data = []
    with open(json_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            des_data.append(data)

    responses = []
    for instruction in tqdm.tqdm(des_data): 
        response = instruction['Response']
        responses.append(response)

    for i, module in enumerate(tqdm.tqdm(responses)): 
        verilog_module = module[0]
        f = open(os.path.join(output_dir, f"{i}.v"), "w")
        f.write(verilog_module)
        f.close()   
         
         

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, help="Path to output_dir", default="data/dataset/raw")
    args = parser.parse_args()

    output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)

    download_vgen(output_dir)
    download_rtlcoder(output_dir)
    

if __name__ == "__main__":
    main()