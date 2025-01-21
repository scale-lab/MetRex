"""Preporcess the metrex dataset to create instruction dataset of input verilog designs and output metric reasoning pairs.
"""

import os
import yaml
import argparse
from yacs.config import CfgNode as ConfigurationNode

from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.model import Model
from config.config import combine_cfgs
from utils import get_logger
from plot import plot_data_lengths 

SYSTEM_PROMPT = "Your task is to estimate {metric} for RTL designs in {tech} technology node."
INSTRUCTION = "Estimate {metric} for the given RTL design. Reason about the number and type of gates that would be present after synthesis. \n # RTL: \n {rtl}"


def create_prompt(
    sample: dict, 
    tokenizer: AutoTokenizer, 
    model: Model, 
    metric: str = 'area', 
    test: bool = False, 
    add_sys_prompt: bool = True,
) -> str:
    
    m = metric
    if metric == 'static_power':
        m = 'static power'
    
    if add_sys_prompt: 
        system_prompt = SYSTEM_PROMPT.format(
            metric=m,
            tech="Skywater 130nm"
        )
    else:
        system_prompt = ""

    user_prompt = INSTRUCTION.format(
        metric=m,
        rtl=sample['RTL']
    )
 
    if test:
        assistant_answer = ""
    else: 
        assistant_answer =  sample['synth'] + "\n" + sample[metric]
   
    chat_template = model.get_chat_template(
        system_prompt=system_prompt, 
        user_prompt=user_prompt, 
        assistant_answer=assistant_answer
    )

    full_prompt = tokenizer.apply_chat_template(
        chat_template, 
        tokenize=False, 
        add_generation_prompt=False
    )
    
    return full_prompt


def preprocess(
    config: ConfigurationNode, 
    tokenizer: AutoTokenizer, 
    model: AutoModelForCausalLM, 
    filter_max_seq: bool = True, 
    add_sys_prompt: bool = True,
    shuffle: bool = True,
) -> DatasetDict:
    train_split, val_split, test_split = config.DATASET.TRAIN_VAL_TEST_SPLIT 

    access_token = os.getenv('HF_ACCESS_TOKEN')
    dataset = load_dataset("scale-lab/MetRex", split="train", token=access_token)
    
    if shuffle: 
        dataset = dataset.shuffle(seed=config.seed)  

    # tokenize the dataset
    text_column = [
        create_prompt(
            sample=data_point, 
            tokenizer=tokenizer, 
            model=model, 
            metric=config.DATASET.METRIC,
            add_sys_prompt=add_sys_prompt
        ) for data_point in dataset]
    
    dataset = dataset.add_column("prompt", text_column)
    dataset = dataset.map(lambda samples: tokenizer(samples["prompt"]))
    
    if filter_max_seq:
        print("Dataset Length before dropping >= max_seq_length: ", len(dataset))
        dataset = dataset.filter(lambda example: len(example['input_ids']) <= config.train.max_seq_length)
        print("Dataset Length after dropping >= max_seq_length: ", len(dataset))

    train_testvalid = dataset.train_test_split(
        train_size=train_split, 
        test_size=test_split+val_split, 
        shuffle=shuffle, 
        seed=config.seed
    )
        
    test_size = test_split / (test_split + val_split)
    if test_size != 0: 
        test_valid = train_testvalid['test'].train_test_split(
            test_size=test_size, 
            shuffle=shuffle, 
            seed=config.seed
        )
        dataset = DatasetDict({
            'train': train_testvalid['train'],
            'test': test_valid['test'],
            'valid': test_valid['train']}
        )
        dataset["test"] = dataset["test"].remove_columns(["prompt", "input_ids"])
        updated_test_prompt = [create_prompt(
            data_point, 
            tokenizer, 
            model, 
            metric=config.DATASET.METRIC, 
            test=True,
            add_sys_prompt=add_sys_prompt
            ) for data_point in dataset['test']]
        
        target_column = [data_point['synth'] + "\n" + data_point[config.DATASET.METRIC] for data_point in dataset['test']]

        dataset['test'] = dataset['test'].add_column("target", target_column)
        dataset["test"] =  dataset["test"].add_column("prompt", updated_test_prompt)
        dataset["test"] = dataset["test"].map(lambda samples: tokenizer(samples["prompt"]))
    
    else:
        dataset = DatasetDict({
            'train': train_testvalid['train'],
            'test': [],
            'valid': train_testvalid['test']}
        )
    
    
    if config.DATASET.TEST_SET:
        test_dataset_dict =  DatasetDict({
            'test1': [],
            'test2': [],
            'test3': []}
        )
        test_set = load_dataset('json', data_files=config.DATASET.TEST_SET)['train'] 
        for level in ['test1', 'test2', 'test3']:
            prompts = [create_prompt(
                data_point, 
                tokenizer,  
                model,  
                metric=config.DATASET.METRIC, 
                test=True,
                add_sys_prompt=add_sys_prompt
                ) for data_point in test_set[level][0]]
            
            targets = [data_point['synth'] + "\n" + data_point[config.DATASET.METRIC] for data_point in test_set[level][0]]
            dataset_test = Dataset.from_dict({"target": targets, "prompt": prompts})
            test_dataset_dict[level] = dataset_test

        dataset['test'] = test_dataset_dict
    
    return dataset 



def load_test_dataset(
    json: str, 
    config: ConfigurationNode, 
    tokenizer: AutoTokenizer, 
    model: Model,
    add_sys_prompt: bool = True
) -> DatasetDict:
    test_dataset_dict =  DatasetDict({
        'test1': [],
        'test2': [],
        'test3': []}
    )
    
    test_set = load_dataset('json', data_files=json)['train'] 
    
    for level in ['test1', 'test2', 'test3']:
        prompts = [create_prompt(
            data_point, 
            tokenizer,  
            model,  
            metric=config.DATASET.METRIC, 
            test=True,
            add_sys_prompt=add_sys_prompt
        ) for data_point in test_set[level][0]]
        
        targets = [data_point['synth'] + "\n" + data_point[config.DATASET.METRIC] for data_point in test_set[level][0]]
        rtls = [data_point['RTL']  for data_point in test_set[level][0]]
        filenames = [data_point['file'] for data_point in test_set[level][0]]

        dataset_test = Dataset.from_dict({
            "target": targets, 
            "prompt": prompts, 
            "RTL": rtls, 
            "file": filenames
        })

        test_dataset_dict[level] = dataset_test


    return test_dataset_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to config file used for finetuning", required=False, default="config/experiments/llama3_r64.yaml")
    parser.add_argument('--output_dir', type=str, help="Path to output directory ", required=False, default="./output/sft")
    
    args = parser.parse_args()

    config_file = args.config 
    output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)

    logger = get_logger(os.path.join(output_dir, "preprocess.log"))
    
    config = combine_cfgs(config_file)
    shuffle = config.DATASET.SHUFFLE 
    model_name = config.LLM.NAME 
    
    with open(os.path.join(output_dir, "config.yaml"), 'w') as f:
        yaml.dump(config, f)

    # load tokenizer
    model = Model(model_name=model_name)
    model_id = model.model_instruct_ids[model_name]

    access_token = os.getenv('HF_ACCESS_TOKEN')

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        token=access_token, 
        use_fast=True, 
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.padding_side = "left"

    dataset = preprocess(
        config, 
        tokenizer, 
        model, 
        filter_max_seq=True, 
        add_sys_prompt=config.DATASET.ADD_SYS_PROMPT,
        shuffle=shuffle
    )    
   
    plot_data_lengths(dataset["train"], dataset["valid"], os.path.join(output_dir, "data_lengths_filtered.png"))

    logger.info(f"CYAN: Training Prompt: {dataset['train']['prompt'][0]}")
    logger.info(f"MAGENTA: Validation Prompt: {dataset['valid']['prompt'][0]}")
    logger.info(f"YELLOW: Test Prompt: {dataset['test']['test1']['prompt'][0]}")


if __name__ == "__main__":
    main()