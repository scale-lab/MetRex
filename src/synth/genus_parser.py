import re 
import numpy as np

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

def parse_delay(report):
    delay = None 
    with open(report, 'r') as file:
        file_content = file.read()
        slack_pattern = re.compile(r"Slack:\s*=\s*(-?\d+)")
        match = slack_pattern.search(file_content)
        if match:
            delay = -1*float(match.group(1))
    return delay 


def parse_power(report):
    power = {'leakage': None, 'internal': None, 'switching': None, 'total': None} 
    with open(report, 'r') as file:
        file_content = file.read()
        power_pattern = re.compile(r'(\d+\.\d+e[-+]\d+)\s+(\d+\.\d+e[-+]\d+)\s+(\d+\.\d+e[-+]\d+)\s+(\d+\.\d+e[-+]\d+)\s+Total\s+N')
        matches = power_pattern.findall(file_content)
        if matches:
            for i, power_type in enumerate(['leakage', 'internal', 'switching', 'total']): 
                power[power_type] = float(matches[-1][i])

            total_power = power['leakage'] + power['internal'] + power['switching']
            assert np.isclose(power['total'], total_power, rtol=0.0001)

    return power 