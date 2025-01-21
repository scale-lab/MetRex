"""Runs yosys to synthesize RTL designs & report area. 
   Runs OpenSTA to report delay, and power numbers.
"""
import os 
import re
import csv
import sys
import glob
import time
import tqdm
import shutil
import argparse
import subprocess 

from utils import get_logger, has_clk_signal, get_top_module
from sdc import SDC_CONSTRAINTS_CLOCK, SDC_DRIVER, SDC_CONSTRAINTS_WO_CLOCK
from yosys_parsers import parse_power, parse_delay, get_delay, parse_area


def yosys_synth(design, liberty, target_clock_period, output_dir, logger):
    
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
    sdc_dir = os.path.join(output_dir, "sdc")
    log_dir =  os.path.join(output_dir, "log")
   
    for directory in [area_dir, gate_dir, delay_dir, power_dir, verilog_dir, sdc_dir, log_dir]:
        os.makedirs(directory, exist_ok=True)
        
    has_clk, clk_port = has_clk_signal(design)

    file_name = os.path.basename(design)
    output_verilog = os.path.join(verilog_dir, file_name)
    sdc_file = os.path.join(sdc_dir, f"{file_name[:-2]}.sdc")
    area_report = os.path.join(area_dir, f"{file_name}.txt")
    timing_report = os.path.join(delay_dir, f"{file_name}.txt")
    power_report = os.path.join(power_dir, f"{file_name}.txt")

    tri_buf_map = f"yosys_maps/tribuff_map.v"
    latch_map = f"yosys_maps/latch_map.v"
    
    if has_clk: 
        synth_sdc_script = SDC_CONSTRAINTS_CLOCK.format(
            clock_name="clk", 
            clock_port=clk_port, 
            clock_period=target_clock_period, 
            max_fanout=10, 
        )
        delay_sdc = SDC_CONSTRAINTS_CLOCK.format(
            clock_name="clk",
            clock_port=clk_port,
            clock_period=0,
            max_fanout=10
        )
        power_sdc = SDC_CONSTRAINTS_CLOCK.format(
            has_clk, 
            clock_name="myClock", 
            clock_port=clk_port, 
            clock_period=1, 
            max_fanout=10, 
            driving_cell="sky130_fd_sc_hd__inv_2", 
            driving_cell_pin="Y"
        )
    else:
        synth_sdc_script = SDC_CONSTRAINTS_WO_CLOCK.format(
            clock_name="clk", 
            clock_period=target_clock_period, 
            max_fanout=10, 
        )
        delay_sdc = SDC_CONSTRAINTS_WO_CLOCK.format(
            clock_name="clk",
            clock_period=0,
            max_fanout=10
        )
        power_sdc = SDC_CONSTRAINTS_WO_CLOCK.format(
            has_clk, 
            clock_name="myClock", 
            clock_period=1, 
            max_fanout=10, 
        )
        
    synth_sdc_script += SDC_DRIVER.format(
        driving_cell="sky130_fd_sc_hd__inv_2",
        driving_cell_pin="Y"
    )
    
    delay_sdc += SDC_DRIVER.format(
        driving_cell="sky130_fd_sc_hd__inv_2",
        driving_cell_pin="Y"
    )
    
    power_sdc += SDC_DRIVER.format(
        driving_cell="sky130_fd_sc_hd__inv_2",
        driving_cell_pin="Y"
    )
     
    with open(sdc_file, "a") as file: 
        file.write(synth_sdc_script)

    yosys_script = f"""
        # Read the liberty file
        read_liberty -lib -ignore_miss_dir -setattr blackbox {liberty}
        
        # Read the design
        read_verilog -sv {design}
                
        # Attempt to automatically determine the top module
        hierarchy -check -auto-top

        # Convert the design to a generic netlist
        synth -auto-top -flatten
        
        # map tri-state buffers
        techmap -map {tri_buf_map}

        # map latches
        techmap -map {latch_map}

        # mapping flip-flops
        dfflibmap -liberty {liberty}

        # abc optimizations
        abc -liberty {liberty}

        # Technology mapping of constant hi- and/or lo-drivers
        hilomap -singleton \
        -hicell sky130_fd_sc_hd__conb_1 HI \
        -locell sky130_fd_sc_hd__conb_1 LO

        # write synthesized netlist
        opt_clean -purge
       
        # report area 
        tee -o {area_report} stat -liberty {liberty} 
        
        write_verilog -noattr -noexpr -nohex -nodec -defparam  {output_verilog}
    """

    start = time.time()

    yosys_command = ['yosys']
    try:
        synth_log = open(os.path.join(log_dir, f"{file_name}_synth.log"), 'w')
        synth_err_log = open(os.path.join(log_dir, f"{file_name}_synth_err.log"), 'w')
        process = subprocess.run(
            yosys_command, 
            input=yosys_script, 
            timeout=60*4,
            universal_newlines=True,
            shell=True, 
            stdout=synth_log, 
            stderr=synth_err_log
        )
        synth_log.close()
        synth_err_log.close()
    except subprocess.TimeoutExpired:
        synth_log.close()
        synth_err_log.close()
        logger.info(f"RED: Yosys Timeout, design: {design}.")
        return metrics
    except subprocess.CalledProcessError as e:
        synth_log.close()
        synth_err_log.close()
        logger.info(f"RED: Yosys execution failed: {e}")
        return metrics

    correct = (process.returncode == 0)

    # report timing and power
    top_module = None 
    if os.path.isfile(output_verilog): 
        top_module = get_top_module(output_verilog)
    
    if top_module is None: 
        return metrics

    opensta_script = f"""
        read_liberty {liberty}
        read_verilog {output_verilog}
        link_design {top_module}
        {delay_sdc};
        report_checks -sort_by_slack -path_delay max -fields {{slew cap input nets fanout}} -format full_clock_expanded -group_count 1 > {timing_report}
        {power_sdc}
        set_power_activity -global -activity 0.000000001
        report_power > {power_report}
    """
    opensta_command = ['sta']

    # Run STA
    try: 
        sta_log = open(os.path.join(log_dir, f"{file_name}_sta.log"), 'w')
        sta_err_log = open(os.path.join(log_dir, f"{file_name}_sta_err.log"), 'w')
        process = subprocess.run(
            opensta_command, 
            input=opensta_script, 
            timeout=60*4,
            universal_newlines=True, 
            shell=True, 
            stdout=sta_log,
            stderr=sta_err_log
        )
        sta_log.close()
        sta_err_log.close()
    except subprocess.TimeoutExpired:
        sta_log.close()
        sta_err_log.close()
        logger.info(f"RED: OpenSTA Timeout, design: {design}.")
        return metrics
    except subprocess.CalledProcessError as e:
        sta_log.close()
        sta_err_log.close()
        logger.info(f"RED: OpenSTA execution failed: {e}")
        return metrics

    end = time.time()
    
    metrics['time'] = end-start 
    
    if os.path.isfile(area_report):
        cells, metrics['area']= parse_area(area_report)
        metrics['gate_count']= sum(cell['count'] for cell in cells.values())
        metrics['comb_cells'] = sum(cell['count'] for name, cell in cells.items() if 'df' not in name)
        metrics['seq_cells'] = metrics['gate_count'] - metrics['comb_cells']
        
    if os.path.isfile(timing_report):
        metrics['delay']=get_delay(timing_report) 
    
    if os.path.isfile(power_report):
        power=parse_power(power_report)
        metrics['static_power'] = power['static']
        metrics['switching_power'] = power['switching']

    return metrics

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help="Path to RTL data directory", default="data/dataset/clean")
    parser.add_argument('--liberty', type=str, help="Liberty file to use for synthesis", required=True)
    parser.add_argument('--force',  action='store_const', const=True, help="Force write existing run", default=False)
    parser.add_argument('--target_clock_period', type=int, help="Target clock period in ns", default=20)
    parser.add_argument('--output_dir', type=str, help="Path to output data dir", required=True)
    parser.add_argument('--start', type=float, help="Start Index ", default=0.0)
    parser.add_argument('--end', type=float, help="End Index", default=1.0)

    args = parser.parse_args()

    data = args.data 
    liberty = args.liberty
    force = args.force
    output_dir = args.output_dir 
    start = args.start 
    end = args.end 
    target_clock_period = args.target_clock_period

    logger = get_logger('yosys.log')

    if not os.path.isdir(data):
        logger.info(f"RED: Path {data} doesn't exist!!")
        sys.exit(1)

    if force and os.path.exists(output_dir): 
        shutil.rmtree(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, "data.csv")
    headers = ['Design', 'Gate Count', 'Sequential Cells' , 'Combinational Cells', \
        'Area', 'Delay', 'Static Power', 'Switching Power', 'Time']
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

    input_designs = sorted(glob.glob(os.path.join(data, "*.v"), recursive=False)) 
    input_designs.sort(key=lambda f: int(''.join(filter(str.isdigit, f)) or 0))

    synthesized_designs = sorted(glob.glob(os.path.join(os.path.join(output_dir, "synth"), "*.v"), recursive=False)) 
    synthesized_designs.sort(key=lambda f: int(''.join(filter(str.isdigit, f)) or 0))

    num_designs = len(input_designs)
    start = int(start*num_designs)
    end = int(end*num_designs)
    logger.info(f"CYAN: Running from {start} to {end}")

    input_designs = input_designs[start:end]

    # filter out already synthesized designs 
    synthesized_filenames = [os.path.basename(file) for file in synthesized_designs]
    
    design_list = [design for design in input_designs 
            if os.path.basename(design) not in synthesized_filenames]
    
    for design in tqdm.tqdm(design_list):
        logger.info(f"BLUE: Running {design}")
        
        metrics = yosys_synth(
            design=design, 
            liberty=liberty, 
            target_clock_period=target_clock_period, 
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