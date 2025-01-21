"""Creates the chain of thoughts prompt from yosys & opensta logs 
"""

import os 
import glob
import json
import tqdm
import argparse 

from liberty.types import *
from liberty.parser import parse_liberty

from cot import create_synth_thought, create_area_cot, \
                create_delay_cot, create_static_power_cot,\
                get_bycell, sort, write_json
from yosys_parsers import parse_area, parse_delay, parse_power
from utils import  get_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--synth', type=str, help="Path to synthesis output dir", required=True)
    parser.add_argument('--verilog', type=str, help="Path to verilog data dir", required=True)
    parser.add_argument('--liberty', type=str, help="Liberty file used for synthesis", required=True)
    parser.add_argument('--output_dir', type=str, help="Path to output directory", default=".")

    args = parser.parse_args()

    synth = args.synth
    verilog_dir = args.verilog
    liberty_file = args.liberty
    output_dir = args.output_dir 

    logger = get_logger('yosys_cot.log')

    library = parse_liberty(open(liberty_file).read())

    output_json = os.path.join(output_dir, "cot.json")
   
    area_reports = sort(os.path.join(synth, 'area_analysis', '*.txt'))
    delay_reports = sort(os.path.join(synth, 'delay_analysis', '*.txt'))
    power_reports = sort(os.path.join(synth, 'power_analysis', '*.txt'))

    # Extract the basename to identify common files
    common_files = set(os.path.basename(f) for f in area_reports) & \
               set(os.path.basename(f) for f in delay_reports) & \
               set(os.path.basename(f) for f in power_reports) 

    common_files = set(os.path.basename(f) for f in common_files)

    # Filter the original lists to only include common files
    area_reports = [f for f in area_reports if os.path.basename(f) in common_files]
    delay_reports = [f for f in delay_reports if os.path.basename(f) in common_files]
    power_reports = [f for f in power_reports if os.path.basename(f) in common_files]

    thoughts = []
    for area_report, delay_report, power_report in tqdm.tqdm(zip(area_reports, delay_reports, power_reports)):

        filename, _ =  os.path.splitext(os.path.basename(area_report))
        
        if not os.path.exists(os.path.join(verilog_dir, filename)):
            logger.info(f"YELLOW: Verilog for {filename} not found")
            continue
    
        with open(os.path.join(verilog_dir, filename), 'r') as file:
            rtl = file.read()
                
        try: 
            cells, area = parse_area(area_report)
            delay = 1
            stages, delay = parse_delay(delay_report)
            power = parse_power(power_report)
            
            if any(metric == -1 for metric in [area, delay, power]) or len(stages) == 0:
                logger.info(f"YELLOW: Skippin metric = -1 {area_report}")
                continue 

            # get cells breakdown 
            metrics_bycell = get_bycell(cells, library)
            if len(metrics_bycell) == 0:
                logger.info(f"YELLOW: Skipping design {area_report} because cell not found {area_report}")
                continue 


            synth_thought = create_synth_thought(metrics_bycell)

            area_prompt = create_area_cot(
                metrics_bycell, 
                golden_area=area, 
            )
            delay_prompt = create_delay_cot(
                stages, 
                total_delay=delay,
                scale=1
            )
            static_power_prompt = create_static_power_cot(
                metrics_bycell, 
                power['static'], 
            )
            
            logger.info("CYAN: ----------Synthesis Though --------")
            logger.info(f"MAGENTA: {synth_thought}")
            logger.info("CYAN: ----------Delay Thought-------------")
            logger.info(f"MAGENTA: {delay_prompt}")
            logger.info("CYAN: ----------Area -------------")
            logger.info(f"MAGENTA: {area_prompt}")
            logger.info("CYAN: ----------Static Power -------------")
            logger.info(f"MAGENTA: {static_power_prompt}")

        except AssertionError as e:
            logger.info(f"YELLOW: Skipping design {area_report}, there is mismatch between CoT prompt and final metrics!!!")
            continue 
        
        thought = {
            'file': filename, 
            'RTL': rtl, 
            'synth': synth_thought, 
            'area': area_prompt, 
            'delay': delay_prompt,
            'static_power': static_power_prompt
        }
       
        thoughts.append(thought)
        
    write_json(thoughts, output_json)
 

if __name__ == '__main__':
    main()