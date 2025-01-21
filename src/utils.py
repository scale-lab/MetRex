import re 
import sys
import argparse
import random
import torch
import logging
import numpy as np

from colorama import Fore, Style, init
from pyverilog.vparser.parser import parse

init(autoreset=True)

class ColorFormatter(logging.Formatter):
    FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    FORMATS = {
        logging.DEBUG: Fore.CYAN + FORMAT + Style.RESET_ALL,
        logging.INFO: Fore.GREEN + FORMAT + Style.RESET_ALL,
        logging.WARNING: Fore.YELLOW + FORMAT + Style.RESET_ALL,
        logging.ERROR: Fore.RED + FORMAT + Style.RESET_ALL,
        logging.CRITICAL: Fore.MAGENTA + FORMAT + Style.RESET_ALL
    }

    def format(self, record):
        if record.levelno == logging.INFO and ':' in record.msg:
            # Extract color from the message
            color, message = record.msg.split(':', 1)
            color_attr = getattr(Fore, color.upper(), Fore.WHITE)
            record.msg = message.strip()
            log_fmt = color_attr + self.FORMAT + Style.RESET_ALL
        else:
            log_fmt = self.FORMATS.get(record.levelno, self.FORMAT)
        
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)



def get_logger(output):
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=output, level=logging.INFO, filemode='w')
    logging.getLogger("httpx").setLevel(logging.WARNING)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColorFormatter())

    file_handler = logging.FileHandler(output)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(ColorFormatter())

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger



def parse_metrics(answer, metrics=['area', 'delay', 'static_power']):
    patterns = {
        'area': r"total area is\s+(\d+(\.\d+)?)",
        'static_power': r"total static power is\s+(\d+(\.\d+)?)",
        'delay': r"total delay is\s+(\d+(\.\d+)?)"
    }
    patterns2 = {
        'area': r"total area is\s+(\d+(\.\d+)?)",
        'delay': r"the total delay for this design is\s+(\d+(\.\d+)?)",
        'static_power': r"total static.*?power is\s+(\d+(\.\d+)?)"
    }
    patterns3 = {
        'area': r"total area is\s+(\d+(\.\d+)?)",
        'delay': r"the total delay for this design is\s+(\d+(\.\d+)?)",
        'static_power': r"total static\s+(\d+(\.\d+)?)"
    }
    patterns4 = {
        'area': r"Total Area is\s+(\d+(\.\d+)?)",
        'delay': r"Total Delay is\s+(\d+(\.\d+)?)",
        'static_power': r"Total Static power is\s+(\d+(\.\d+)?)"
    }
    patterns5 = {
        'area': r"Total Area is\s+(\d+(\.\d+)?)",
        'delay': r"delay is\s+(\d+(\.\d+)?)",
        'static_power': r"Total Static power is\s+(\d+(\.\d+)?)"
    }
    patterns6 = {
        'area': r"\s+(\d+(\.\d+)?)",
        'delay': r"\s+(\d+(\.\d+)?)",
        'static_power': r"\s+(\d+(\.\d+)?)"
    }
    pattern7 = {
        'area': r"(\d+\.\d+)(?!.*\d+\.\d+)",
        'delay': r"(\d+\.\d+)(?!.*\d+\.\d+)",
        'static_power': r"(\d+\.\d+)(?!.*\d+\.\d+)"

    }
    patterns8 = {
        'area': r"(\d+(\.\d+)?)",
        'delay': r"(\d+(\.\d+)?)",
        'static_power': r"(\d+(\.\d+)?)"
    }
    pattern_equation = r'=\s*([\d.]+)'

    target = {}
    for m in metrics:
        matches = re.findall(patterns[m], answer[m])
        matches2 = re.findall(pattern_equation, answer[m])
        matches3 = re.findall(patterns2[m], answer[m])
        matches4 = re.findall(patterns3[m], answer[m])
        matches5 = re.findall(patterns4[m], answer[m])
        matches6 = re.findall(patterns5[m], answer[m])
        matches7 = re.findall(patterns6[m], answer[m])
        matches8 = re.findall(pattern7[m], answer[m])
        matches9 = re.findall(patterns8[m], answer[m])

        if matches:
            target[m] = float(matches[0][0])
        elif matches3:
            target[m] = float(matches3[0][0])
        elif matches4:
            target[m] = float(matches4[0][0])
        elif matches5:
            target[m] = float(matches5[0][0])
        elif matches6:
            target[m] = float(matches6[0][0])
        elif matches8:
            target[m] = float(matches8[-1])
        elif matches2:
            target[m] =  float(matches2[-1])    
        elif matches9:
            target[m] = float(matches9[-1][0])
        elif matches7:
            target[m] = float(matches7[0][0])
        else:
            target[m] = None 
            
    return target


def parse_answer_mistral(answer):
    pattern = r'\[\/INST\]((?:(?!\[INST\])[\s\S])*$)'
    match = re.findall(pattern, answer, re.DOTALL)  
    if match:
        result = match[-1]
    else:
        result = None
    return result


def parse_answer_llama(answer):
    pattern = r'(?:.*assistant)([\s\S]*)'
    match = re.findall(pattern, answer, re.DOTALL)  
    if match:
        result = match[-1]
    else:
        result = None
    return result


def parse_answer(answer, model_name, llama_models, mistral_models):
    if model_name in mistral_models:
        parse_fn = parse_answer_mistral
    elif model_name in llama_models:
        parse_fn = parse_answer_llama
    return parse_fn(answer)



def get_target(sample, metrics=['area', 'delay', 'static_power']):
    patterns = {
        'area': r"total area is\s+(\d+(\.\d+)?)",
        'static_power': r"total static power is\s+(\d+(\.\d+)?)",
        'delay': r"total delay is\s+(\d+(\.\d+)?)"
    }
    patterns2 = {
        'area': r"total area is\s+(\d+(\.\d+)?)",
        'delay': r"the total delay for this design is\s+(\d+(\.\d+)?)",
        'static_power': r"total static.*?power is\s+(\d+(\.\d+)?)"
    }
    patterns3 = {
        'area': r"total area is\s+(\d+(\.\d+)?)",
        'delay': r"the total delay for this design is\s+(\d+(\.\d+)?)",
        'static_power': r"total static\s+(\d+(\.\d+)?)"
    }
    
    pattern_equation = r'=\s*([\d.]+)'

    target = {}
    for m in metrics:
        matches = re.findall(patterns[m], sample[m])
        matches2 = re.findall(pattern_equation, sample[m])
        matches3 = re.findall(patterns2[m], sample[m])
        matches4 = re.findall(patterns3[m], sample[m])
        if matches:
            target[m] = float(matches[0][0])
        elif matches2:
            target[m] =  float(matches2[0][0])    
        elif matches3:
            target[m] = float(matches3[0][0])
        elif matches4:
            target[m] = float(matches4[0][0])
        else:
            target[m] = None 
            
    return target



def get_top_module(design):
    top_module = None 
    try: 
        ast, directives = parse([design]) 
        for definition in ast.description.definitions:
            def_type = type(definition).__name__
            if def_type == "ModuleDef":
                top_module = definition.name 
    except: 
        print("[Error] Pyverilog Failed at extracting the top module name")
        module_pattern = re.compile(r'^\s*module\s+([a-zA-Z_][a-zA-Z0-9_]*)')
        with open(design, 'r') as file:
            for line in file:
                match = module_pattern.search(line)
                if match:
                    top_module = match.group(1)
                    break 

    if top_module is None: 
        print("[Error] Failed at extracting the top module name!!")
        
    print("Top Module: ", top_module)
    
    return top_module 


def has_clk_signal(input_file):
    has_clk = False
    clk_name = None
    with open(input_file, 'r') as f:
        line = f.readline()
        while line:
            tokens = re.split('[ ,()_]', line.strip().strip(';').strip())
            if ('input' in tokens and ('clk' in tokens or 'CLK' in tokens or 'clock' in tokens)):
                has_clk = True
                tokens_2 = re.split('[ ,()]', line.strip().strip(';').strip())
                for token in tokens_2:
                    if ('clk' in token or 'CLK' in token):
                        clk_name = token
                break     
            line = f.readline()       
    return has_clk, clk_name


def has_clk_signal_netlist(input_netlist):
    has_clk = False
    clock_name = None 
    with open(input_netlist, 'r') as f:
        netlsit = f.read()

    clock_regex = r"\.CLK\(([^)]+)\)"
    match = re.search(clock_regex, netlsit)
    
    if match:
        clock_name = match.group(1)
        has_clk = True 
        
    return has_clk, clock_name


def print_trainable_parameters(model, logger):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(f"CYAN: trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")


def set_seed(seed, n_gpu=1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
