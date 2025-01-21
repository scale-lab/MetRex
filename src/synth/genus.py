"""Runs Cadence Genus to synthesize RTL designs & report area, delay, and power. 
"""

import os
import sys
import csv 
import glob
import tqdm
import time
import shutil
import argparse 
import subprocess
import numpy as np 

from utils import get_top_module, has_clk_signal, get_logger
from genus_parser import parse_delay, parse_area, parse_power


def genus_synth(design, liberty, output_dir, logger):
    metrics = {
        'area': -1, 
        'gate_count': -1,
        'comb_cells': -1,
        'seq_cells': -1,
        'delay': -1, 
        'static_power': -1, 
        'switching_power': -1, 
        'time': -1
    }
     
    area_dir = os.path.join(output_dir, "area_analysis")
    gate_dir = os.path.join(output_dir, "gate_analysis")
    delay_dir = os.path.join(output_dir, "delay_analysis")
    power_dir = os.path.join(output_dir, "power_analysis")
    verilog_dir = os.path.join(output_dir, "synth")
    log_dir =  os.path.join(output_dir, "log")
   
    for directory in [area_dir, gate_dir, delay_dir, power_dir, verilog_dir, log_dir]:
        os.makedirs(directory, exist_ok=True)
       
    has_clk, clk_name = has_clk_signal(design)

    file_name = os.path.basename(design)
    output_verilog = os.path.join(verilog_dir, file_name)
    area_report = os.path.join(area_dir, f"{file_name}.txt")
    gate_report = os.path.join(gate_dir, f"{file_name}.txt")
    timing_report = os.path.join(delay_dir, f"{file_name}.txt")
    power_report = os.path.join(power_dir, f"{file_name}.txt")

    if has_clk:
        sdc_timing = f"""
            create_clock -name {clk_name} -period 0 [get_ports {clk_name}]; 
            set_input_delay -clock {clk_name} 0 [remove_from_collection [all_inputs] [get_ports {clk_name}]]; 
            set_output_delay -clock {clk_name} 0 [all_outputs];
        """
    else: 
        sdc_timing = f"""
            create_clock -name clk -period 0;
            set_input_delay -clock clk 0 [all_inputs]; 
            set_output_delay -clock clk 0 [all_outputs]; 
        """
          
    genus_script = f'''
        read_libs {liberty};
        read_hdl -sv {design};
        elaborate;
        flatten;
        synthesize -to_generic -effort high;
        syn_map;
        syn_opt; 
        report_area > {area_report};
        report_gates > {gate_report};
        write_hdl -mapped > {output_verilog};
        {sdc_timing}
        report_timing > {timing_report};
        report_power -by_libcell > {power_report};
        exit;
    '''
    
    start = time.time()

    genus_command = [
        'genus', 
        '-abort_on_error', 
        '-execute', 
        genus_script, 
        '-log', 
        '/dev/null'
    ]

    try:
        synth_log = open(os.path.join(log_dir, f"{file_name}_synth.log"), 'w')
        synth_err_log = open(os.path.join(log_dir, f"{file_name}_synth_err.log"), 'w')
        subprocess.call(genus_command, stdout=synth_log, stderr=synth_err_log)
        synth_log.close()
        synth_err_log.close()
    except subprocess.CalledProcessError as e:
        synth_log.close()
        synth_err_log.close()
        logger.info(f"RED: Genus execution failed: {e}")
        return metrics

    end = time.time()
    metrics['time'] = end-start 

    # Get area, day, and power numbers
    if os.path.isfile(gate_report): 
        cells, metrics['area'] = parse_area(gate_report)
        metrics['gate_count']= sum(cell['count'] for cell in cells.values())
        metrics['comb_cells'] = sum(cell['count'] for name, cell in cells.items() if 'DF' not in name)
        metrics['seq_cells'] = metrics['gate_count'] - metrics['comb_cells']
    
    if os.path.isfile(timing_report): 
        metrics['delay'] = parse_delay(timing_report)
    
    if os.path.isfile(power_report): 
        power = parse_power(power_report)
        metrics['static_power'] = power['leakage']
        metrics['switching_power'] = power['switching']

    return metrics
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help="Path to RTL data dir", default="data/dataset/clean")
    parser.add_argument('--liberty', type=str, help="Liberty file to use for synthesis", required=True)
    parser.add_argument('--force',  action='store_const', const=True, help="Force write existing run", default=False)
    parser.add_argument('--output_dir', type=str, help="Path to output data dir", required=True)
    parser.add_argument('--start', type=float, help="start marker", default=0.0)
    parser.add_argument('--end', type=float, help="end marker", default=1.0)
    
    args = parser.parse_args()

    data = args.data 
    liberty = args.liberty
    force = args.force
    output_dir = args.output_dir 
    start = args.start 
    end = args.end 

    logger = get_logger("genus.log")
    
    if not os.path.isdir(data):
        logger.info(f"RED: Path {data} doesn't exist!!")
        sys.exit(1)

    if force and os.path.exists(output_dir): 
        shutil.rmtree(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)

    # init_csv 
    csv_path = os.path.join(output_dir, "data.csv")
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Design', 'Gate Count', 'Sequential Cells' , 'Combinational Cells', 'Area', 'Delay', 'Static Power', 'Switching Power', 'Time'])

    input_designs = sorted(glob.glob(os.path.join(data, "*.v"), recursive=False)) 
    input_designs.sort(key=lambda f: int(''.join(filter(str.isdigit, f)) or 0))

    synthesized_designs = sorted(glob.glob(os.path.join(os.path.join(output_dir, "synth"), "*.v"), recursive=False)) 
    synthesized_designs.sort(key=lambda f: int(''.join(filter(str.isdigit, f)) or 0))

    num_designs = len(input_designs)
    st = int(start*num_designs)
    end = int(end*num_designs)
    logger.info(f"CYAN: Running from {start} to {end}")

    input_designs = input_designs[st:end]

    # filter out already synthesized designs 
    synthesized_filenames = [os.path.basename(file) for file in synthesized_designs]
    
    design_list = [design for design in input_designs 
                if os.path.basename(design) not in synthesized_filenames]

    for design in tqdm.tqdm(design_list):
        logger.info(f"BLUE: Running {design}")
       
        metrics = genus_synth(
            design=design,
            liberty=liberty,
            output_dir=output_dir,
            logger=logger
        )
        
        file_name = os.path.basename(design)
        row = [file_name, 
            metrics['gate_count'], 
            metrics['seq_cells'],
            metrics['comb_cells'], 
            metrics['area'],
            metrics['delay'], 
            metrics['static_power'],
            metrics['switching_power'], 
            metrics['time']
        ]
        
        logger.info(f"Synthesis Results, {row}")
        
        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row) 


if __name__ == '__main__':
    main()
    
  