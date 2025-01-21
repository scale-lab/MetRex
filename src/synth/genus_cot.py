"""Creates the chain of thoughts prompt from genus logs 
"""
import os 
import json
import tqdm
import argparse 
import regex as re
import numpy as np 

from liberty.parser import parse_liberty

from utils import get_logger
from cot import write_json, sort, get_bycell
from cot import create_synth_thought, create_delay_cot, create_static_power_cot, create_area_cot


def parse_area(area_report):
    cells = {}
    total_area = -1 
   
    with open(area_report, 'r') as file:
        data = file.read()
        
    pattern = re.compile(r"^(\S+)\s+(\d+)\s+([\d\.]+)", re.MULTILINE)
    matches = pattern.findall(data)
    
    area_sum = 0.0 
    for gate_type, instances, area in matches: 
        if gate_type == 'total':
            total_area = float(area) 
            break 
        area_sum += float(area) 
        
        if float(instances) != 0: 
            cell_area = float(area) / float(instances)
        else:
            cell_area = 0.0
        cells[gate_type] = {'count': float(instances), 'area': cell_area}
    
    if total_area != -1: 
        assert np.isclose(area_sum, total_area, rtol=0.0001)
   
    return cells, total_area      


def parse_power(power_report):
    cells = {}
    total_power = {}
    ref_sum = {
        'leakage': 0.0, 
        'internal': 0.0, 
        'switching': 0.0, 
        'total': 0.0
    }
    with open(power_report, 'r') as file:
        data = file.read()
    
    if not data: 
        ref_sum = {
            'leakage': -1, 
            'internal':-1, 
            'switching': -1,
            'total': -1
        } 
        return cells, ref_sum
    
    power_pattern = re.compile(r"(\d+\.\d+e[-+]\d+)\s+(\d+\.\d+e[-+]\d+)\s+(\d+\.\d+e[-+]\d+)\s+(\d+\.\d+e[-+]\d+)\s+(\S+)")
    matches = power_pattern.findall(data)

    for match in matches:
        leakage, internal, switching, total, gate_type = match
        if gate_type == 'Total':
            total_power = {
                'leakage': float(leakage), 
                'internal': float(internal), 
                'switching': float(switching), 
                'total': float(total)
            }
        else: 
            cells[gate_type] = {
                'leakage': float(leakage), 
                'internal': float(internal), 
                'switching': float(switching), 
                'total': float(total)
            }
            ref_sum['leakage'] += float(leakage)
            ref_sum['internal'] += float(internal)
            ref_sum['switching'] += float(switching)
            ref_sum['total'] += float(total)

    for p_type in ref_sum.keys():
        assert np.isclose(ref_sum[p_type], total_power[p_type], rtol=0.0001)
        
    return cells, total_power 


def parse_delay(delay_report):
    stages = {}
    total_delay = 0.0 
    ref_sum = 0.0 
    
    with open(delay_report, 'r') as file:
        data = file.read()
    
    if 'No paths found' in data:
        return stages, -1 
    
    pattern = re.compile(r"^\s+(\S+)\s+.*?\s+\S+\s+\S+\s+(\S+)\s+(\d+)\s+(\d+\.\d+)\s+\d+\s+(\d+)\s+\d+\s+.*", re.MULTILINE)
    matches = pattern.findall(data)
    keywords = ["(arrival)", "(port)"]
    for match in matches:
        timing_point, stage, fanout, load, delay = match
        if any(keyword in stage for keyword in keywords):
            continue
        stages[timing_point] = {
            'cell': stage, 
            'fanout': float(fanout), 
            'cap': float(load), 
            'delay': float(delay)
        }
        ref_sum += float(delay)
        
    data_path_pattern = re.compile(r'Data Path:-\s+(\d+)')
    match = data_path_pattern.search(data)
    if match:
        total_delay = float(match.group(1))
    
    assert np.isclose(ref_sum, total_delay, rtol=0.06)

    return stages, total_delay 
    
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help="Path to dataset dir", default="genus_synth")
    parser.add_argument('--verilog', type=str, help="Path to verilog data dir", default="dataset/clean")
    parser.add_argument('--liberty', type=str, help="Liberty file used for synthesis", default="/home/manar/LLM4PPA/ppa/tsmc65.lib")
    parser.add_argument('--output_dir', type=str, help="Path to output directory", default=".")

    args = parser.parse_args()

    data = args.data    
    verilog_dir = args.verilog
    liberty_file = args.liberty
    output_dir = args.output_dir 

    logger = get_logger('genus_cot.log')

    # Read and parse a library.
    library = parse_liberty(open(liberty_file).read())
 
    # loop over synthesis logs 
    area_reports = sort(os.path.join(data, 'area_analysis', '*.txt'))
    gate_reports = sort(os.path.join(data, 'gate_analysis', '*.txt'))
    delay_reports = sort(os.path.join(data, 'delay_analysis', '*.txt'))
    power_reports = sort(os.path.join(data, 'power_analysis', '*.txt'))

    # Extract the basename to identify common files
    common_files = set(os.path.basename(f) for f in area_reports) & \
               set(os.path.basename(f) for f in gate_reports) & \
               set(os.path.basename(f) for f in delay_reports) & \
               set(os.path.basename(f) for f in power_reports) 

    # Filter the original lists to only include common files
    area_reports = [f for f in area_reports if os.path.basename(f) in common_files]
    gate_reports = [f for f in gate_reports if os.path.basename(f) in common_files]
    delay_reports = [f for f in delay_reports if os.path.basename(f) in common_files]
    power_reports = [f for f in power_reports if os.path.basename(f) in common_files]

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
    for area_report, gate_report, delay_report, power_report in tqdm.tqdm(zip(area_reports, gate_reports, delay_reports, power_reports)):
       
        filename, _ =  os.path.splitext(os.path.basename(area_report))
        if not os.path.exists(os.path.join(verilog_dir, filename)):
            logger.info(f"[RED]: Verilog for {filename} not found")
            continue
        
        with open(os.path.join(verilog_dir, filename), 'r') as file:
            rtl = file.read()

        cells, total_area = parse_area(gate_report)
        stages, total_delay = parse_delay(delay_report)
        power, total_power = parse_power(power_report)
        
        if any(metric == -1 for metric in [total_area, total_delay, total_power['total']]):
            continue 

        # get cell information from the liberty such as area, static power, input/output capacitence 
        metrics_bycell = get_bycell(cells, library)
        if len(metrics_bycell) == 0:
            continue 

        synth_thought = create_synth_thought(metrics_bycell)
        
        area_prompt = create_area_cot(
            metrics_bycell, 
            golden_area=total_area, 
        )

        if total_delay == 0.0: 
            delay_prompt = "There is no datapath in this design. Thus, the total delay is 0.0"
        else: 
            delay_prompt = create_delay_cot(
                stages, 
                total_delay=total_delay, 
            )

        power_prompt = create_static_power_cot(
            metrics_bycell, 
            total_power['leakage'], 
            scale=1e3, 
            enable_assert=False
        )
        
        thought = {
            'file': filename, 
            'RTL': rtl,
            'synth': synth_thought, 
            'area': area_prompt, 
            'delay': delay_prompt, 
            'static_power': power_prompt
        }
        thoughts.append(thought)
        
    output_json = os.path.join(output_dir, "cot.json")
    write_json(thoughts, output_json)
    


if __name__ == '__main__':
    main()