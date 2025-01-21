"""Finetuning configuration
"""
import sys 

from pathlib import Path
from yacs.config import CfgNode as ConfigurationNode
from accelerate import Accelerator

__C = ConfigurationNode()
__C.name = "exp01"
__C.seed = 42
__C.multi_gpu = False 
__C.device_map ='cuda'

# training data
__C.DATASET = ConfigurationNode()
__C.DATASET.TEST_SET = "data/dataset/verilog_eval_icl.json"
__C.DATASET.SHUFFLE = True
__C.DATASET.ADD_SYS_PROMPT = True
__C.DATASET.METRIC = 'area'
__C.DATASET.TRAIN_VAL_TEST_SPLIT = [0.8, 0.1, 0.1]

# LLM configs
__C.LLM = ConfigurationNode()
__C.LLM.NAME = 'mistral'   
__C.LLM.TEMP = 0.1   

__C.tokenizer = ConfigurationNode()
__C.tokenizer.padding_side = 'right' 
__C.tokenizer.use_fast = True

# Bits and Bytes Config 
__C.bnb = ConfigurationNode()
__C.bnb.load_in_4bit = True 
__C.bnb.load_in_8bit = False 
__C.bnb.bnb_4bit_quant_type = "nf4" 
__C.bnb.bnb_4bit_use_double_quant = False 

# LoRA config 
__C.lora = ConfigurationNode()
__C.lora.lora_r = 64
__C.lora.lora_alpha = 32
__C.lora.lora_dropout = 0.1
__C.lora.use_dora = False 

# Training Config 
__C.train = ConfigurationNode()
__C.train.epochs = 10
__C.train.max_steps = 1
__C.train.learning_rate = 2e-4
__C.train.batch_size = 8
__C.train.max_seq_length = 2048
__C.train.warmup_steps = 2
__C.train.logging_steps = 1
__C.train.eval_steps = 100
__C.train.save_steps = 100
__C.train.evaluation_strategy = "no"
__C.train.gradient_accumulation_steps = 1
__C.train.gradient_checkpointing = False
__C.train.dataloader_num_workers = 0
__C.train.report_to = "none"
__C.train.run_name = "LLM4PPA"
__C.train.logging_strategy = "steps"
__C.train.save_strategy = "steps"
__C.train.load_best_model_at_end = True
__C.train.lr_scheduler_type = 'constant'

__C.test = ConfigurationNode()
__C.test.batch_size = 8
__C.test.num_samples = 10

def get_cfg_defaults():
    return __C.clone()

def combine_cfgs(path_cfg_data: Path=None, path_cfg_override: Path=None):
    if path_cfg_data is not None:
        path_cfg_data=Path(path_cfg_data)
        if not path_cfg_data.exists():
            print(f"[ERROR]: {path_cfg_data} doesn't exist.")
            sys.exit()
            
    if path_cfg_override is not None:
        path_cfg_override=Path(path_cfg_override)

    cfg_base = get_cfg_defaults()

    if path_cfg_data is not None and path_cfg_data.exists():
        cfg_base.merge_from_file(path_cfg_data.absolute())

    if path_cfg_override is not None and path_cfg_override.exists():
        cfg_base.merge_from_file(path_cfg_override.absolute())

    return cfg_base


def main():
    cfg_path = "experiments/gemma2b.yaml"
    cfg  = combine_cfgs(cfg_path) 
    print(cfg)


if __name__ == "__main__":
    main()