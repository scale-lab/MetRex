"""Evaluate the finetuned model on the test dataset 
"""
import os
import sys 
import csv
import argparse 

import torch
import numpy as np 

from peft import  PeftModel
from transformers import set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from model.model import Model, test_model
from model.metrics import compute_mre, compute_acc_at_k_majority
from model.preprocess import  load_test_dataset
from config.config import combine_cfgs
from utils import get_target, get_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to config file used for finetuning", required=True)
    parser.add_argument('--adapter', type=str, help="Path to finetuned LoRA adapter or name of the HF adapter.", required=True)
    parser.add_argument('--metric', type=str, help="Metric (area/delay/power)", required=True)
    parser.add_argument('--num_samples', type=int, help="Number of samples to generate for the LLM per problem (k)", default=10)
    parser.add_argument('--temperature', type=float, help="Temperature value for LLM generation", default=0.4)
    parser.add_argument('--eval', type=str, help="Path evaluation dataset", required=False)
    parser.add_argument('--output_dir', type=str, help="Path to output directory", required=True)
    args = parser.parse_args()

    config_file = args.config
    num_samples = args.num_samples
    temperature = args.temperature
    metric = args.metric
    adapter = args.adapter 
    eval_path = args.eval 
    output_dir = args.output_dir

    access_token = os.getenv('HF_ACCESS_TOKEN')
    
    if not access_token:
        logger.info(f"RED: Please set the HF_ACCESS_TOKEN!")
        sys.exit(0)

    os.makedirs(output_dir, exist_ok=True)
    
    config = combine_cfgs(config_file)

    model_name = config.LLM.NAME 
    logfile = os.path.join(output_dir, f"{model_name}")
    logger = get_logger(output=logfile)
    
    set_seed(config.seed)
    
    if metric == 'power': 
        metric = 'static_power'

    config.DATASET.METRIC = metric 
    
    logger.info(f"CYAN: Temperature set to {temperature}")
    logger.info(f"CYAN: Testing Checkpoint {adapter}")
    logger.info(f"CYAN: Metric set to {metric}")

    base_model = Model.model_instruct_ids[model_name]    
    
    bnb_4bit_compute_dtype = "float16"
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.bnb.load_in_4bit, 
        bnb_4bit_quant_type=config.bnb.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype, 
        bnb_4bit_use_double_quant=config.bnb.bnb_4bit_use_double_quant, 
    )

    # Load base model
    base_model_reload = AutoModelForCausalLM.from_pretrained(
        base_model,
        token=access_token,
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    # merge base model with the finetuned adapter
    model = PeftModel.from_pretrained(base_model_reload, adapter)
    
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, 
        token=access_token, 
        trust_remote_code=True
    )

    tokenizer.pad_token = tokenizer.eos_token # end-of-sequence token
    tokenizer.padding_side = "right" # pad to have max seq length
        
    model.eval()
    model.config.use_cache = False
    
    test_dataset = load_test_dataset(
        json=eval_path, 
        config=config,
        tokenizer=tokenizer, 
        model=Model(model_name, load_model=False)
    )

    answers_bylevel = {}
    
    all_answers = []
    all_targets = []
    mre_bylevel  = {'test1': [], 'test2': [], 'test3': []}
    pred_bylevel = {'test1': [], 'test2': [], 'test3': []}
    ground_truth_bylevel = {'test1': [], 'test2': [], 'test3': []}
    
    for level in test_dataset.keys():
        targets = [get_target({metric: data_point['target']}, metrics=[metric]) for data_point in test_dataset[level]]
        target_thoughts = [data_point['target'] for data_point in test_dataset[level]]
        all_targets.extend(targets)
        
        answers = test_model(
            model=model, 
            tokenizer=tokenizer, 
            dataset=test_dataset[level], 
            temperature=config.LLM.TEMP,
            batch_size=config.test.batch_size, 
            num_samples=num_samples
        )
      
        answers_bylevel[level] = answers 
        
        assert len(answers) == len(test_dataset[level])*num_samples
        
        errors = []
        for i in range(0, len(answers), num_samples):
            # measure MRE for the top answer 
            out = answers[i]
            tar = targets[int(i/num_samples)]
            target_thought = target_thoughts[int(i/num_samples)]
            pred = get_target({metric: out}, metrics=[metric])
            
            error = compute_mre(pred[metric],  tar[metric])  
            errors.append(error)
            
            logger.info(f"YELLOW: LLM Answer is {out}")
            logger.info("GREEN:---------------Target---------------")
            logger.info(f"GREEN: {target_thought}")
            logger.info("------------------------------------")
            logger.info(f"YELLOW: Prediction is {pred} Target is {tar}")
            logger.info(f"RED: MRE Error is {error}")

        logger.info(f"MAGENTA: Average MRE Error is {np.mean(errors)}")

        # reshape answers to list of lists (num_problems * num_samples)
        answers_2d = [answers[i * num_samples:(i + 1) * num_samples] for i in range(len(test_dataset[level]))]
        all_answers.extend(answers_2d)
        
        logger.info(f"Length of answers=(Level set size): {len(answers_2d)} Length of one answer (num_samples): {len(answers_2d[0])}")
    
        for data_point in answers_2d:
            logger.info("---------------------------")
            for ans in data_point: 
                logger.info(f"BLUE: Answer in datapoint: {ans}")
                logger.info(f"BLUE: Target is {get_target({metric: ans}, metrics=[metric])}")
        
            logger.info("BLUE: ---------------------------")

      
        for t in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]: 
            acc_at_1, acc_at_5, acc_at_10  = compute_acc_at_k_majority(
                answers={metric: answers_2d}, 
                ground_truth=targets, 
                k=[1, 5, 10],
                tolerance=t
            )
            
            logger.info(f"YELLOW: Level {level}, acc@1 t={t}, {acc_at_1} ")
            logger.info(f"YELLOW: Level {level}, acc@5 t={t}, {acc_at_5}")
            logger.info(f"YELLOW: Level {level}, acc@10 t={t}, {acc_at_10}")

        predictions = [get_target({metric: data_point[0]}, metrics=[metric]) for data_point in answers_2d]
        predictions = [d[metric] for d in predictions]
  
    
    for t in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]: 
        acc_at_1, acc_at_5, acc_at_10  = compute_acc_at_k_majority(
            answers={metric: all_answers}, 
            ground_truth=all_targets, 
            k=[1, 5, 10],
            tolerance=t,
            function=np.median
        )
        
        logger.info(f"YELLOW: Overall median-acc@1 t={t}, {acc_at_1} ")
        logger.info(f"YELLOW: Overall median-acc@5 t={t}, {acc_at_5}")
        logger.info(f"YELLOW: Overall median-acc@10 t={t}, {acc_at_10}")

    csv_path = os.path.join(output_dir, f"{config.LLM.NAME}_{metric}_n={num_samples}_metrics.csv")
    with open(csv_path,  'w') as file:
        csvwriter = csv.writer(file)
        csvwriter.writerow(['level', 'design', 'pred', 'ground truth', 'mre'])
        for level in ['test1', 'test2', 'test3']: 
            for i in range(len(pred_bylevel[level])): 
                csvwriter.writerow([level, test_dataset[level][i]['file'], pred_bylevel[level][i], ground_truth_bylevel[level][i],  mre_bylevel[level][i][0]])



if __name__ == "__main__":
    main()