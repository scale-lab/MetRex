"""Parse information from yosys and opensta logs
"""
import re 
import numpy as np


def parse_area(report):

    cells = dict()
    chip_area = -1 

    with open(report, 'r') as file:
        area_analysis = file.read()

    chip_area_pattern = re.compile(r"Chip area for module '\\?\\?([^']+)'?: ([\d.]+)")    
    area_match = chip_area_pattern.search(area_analysis)

    if area_match: 
        chip_area = float(area_match.group(2))
 
    cell_pattern = re.compile(r'\s*([a-zA-Z0-9_]+)\s+(\d+)')
    matches = cell_pattern.findall(area_analysis)
    
    for name, count in matches:
        cells[name] = {'count': int(count)} 

    return cells, chip_area  


def parse_delay(report):
    total_delay = -1
    stages = dict()

    with open(report, 'r') as file:
        delay_analysis = file.read()

    if 'No paths found.' in delay_analysis or not delay_analysis:
        return stages, total_delay
    
    data_arrival_pattern = re.compile(r"([\d.-]+)\s+data arrival time")
    match = data_arrival_pattern.search(delay_analysis)
    if match:
        total_delay = float(match.group(1))
    
    name_pattern = r"(_\d+_).*\(([^)]+)\)"
    fanout_load_delay_pattern = r"^\s*(\d+)\s+([\d\.]+)\s+(-?INF|[\d\.]+)\s+([\d\.]+)\s+[\d\.]+\s+[v\^]\s+(.*)"

    KEYWORDS = [
        "(rise edge)", 
        "clock network delay (ideal)", 
        "(fall edge)",
        "input external delay", 
        "(in)",
        "(out)", 
        "(inout)",
        "recovery", 
        "clock reconvergence pessimism",
        "output external delay"
    ]

    stages = dict()
    delay_sum = 0.0
    for match in re.finditer(fanout_load_delay_pattern, delay_analysis, re.MULTILINE):
        fanout, capacitance, slew, delay, description = match.groups()
        
        if any(keyword in description for keyword in KEYWORDS):
            continue
        
        name_match = re.search(name_pattern, description)
        instance_name = name_match.group(1)
        cell_name = name_match.group(2)
       
        if slew == "-INF":
            slew = 0 
            
        stages[instance_name] = {
            'delay': float(delay), 
            'fanout': int(fanout), 
            'cell': cell_name, 
            'slew': float(slew), 
            'cap': float(capacitance)
        }
        delay_sum += float(delay)
    
    delay_sum = round(delay_sum, 2)

    assert np.isclose(delay_sum, total_delay, rtol=0.15)
    
    return stages, total_delay 


def get_delay(filename):
    data_arrival_pattern = re.compile(r"([\d.-]+)\s+data arrival time")

    with open(filename, 'r') as file:
        file_content = file.read()
        match = data_arrival_pattern.search(file_content)
        if match:
            return float(match.group(1))
    return -1


def parse_power(report):
    power = {
        'internal': -1, 
        'switching': -1,
        'static': -1, 
        'total': -1
    }
    with open(report, 'r') as file:
        power_analysis = file.read()

    pattern = r"^Total\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)"
    match = re.search(pattern, power_analysis, re.MULTILINE)
    if match:
        power['internal'] = float(match.group(1))
        power['switching'] = float(match.group(2))
        power['static'] = float(match.group(3))
        power['total'] = float(match.group(4))

        total_power = power['internal']+power['static']+power['switching']
        np.isclose(power['total'], total_power, rtol=0.0001)
        
        return power
    else:
        return power



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
