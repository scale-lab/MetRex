import json 
import glob
import numpy as np 

from liberty.types import *
from liberty.parser import parse_liberty


def create_synth_thought(cells):
    thought = "After synthesis, this design has "
    for name, count, _, _ , _ in cells: 
        thought += f"{count} {name}, "

    thought = thought[:-2] + '. '
    return thought


def create_area_cot(cells, golden_area):
    area_sum = 0.0
    area_bycell = ""
    equation = ""
    for name, count, area, _, _ in cells: 
        area = round(area, 2)
        area_bycell += f"Area of {name} is {area}. "
        equation += f"{count}*{area} + "
        area_sum += int(count) * float(area)

    area_sum = round(area_sum, 2)

    result = eval(equation[:-2])
    equation = f"{equation[:-2]} = {str(area_sum)}"

    thought = f"{area_bycell} \nIn total, we can compute {equation}"
    thought += f"\nThus, the total area is {area_sum}"
    
    assert np.isclose([area_sum, result], golden_area, rtol=0.01).all()

    return thought


def create_delay_cot(critical_path, total_delay, scale=1):
    delay_sum = 0.0
    stage_delay = ""
    equation = ""
    thought = "The critical path goes through the following gates: "

    for _, stage in critical_path.items(): 
        stage['delay'] = round(stage['delay']*scale, 2)
        stage['cap'] = round(stage['cap']*scale, 2)
            
        name = stage['cell'].replace("sky130_fd_sc_hd__", "", 1)  
        thought += f"{name} with a fanout of {stage['fanout']} and load capacitence of {stage['cap']}, "
        stage_delay += f"Delay of {name} is {stage['delay']}, "
        equation += f"{stage['delay']} + "
        delay_sum += stage['delay'] 

    
    result = eval(equation[:-2])

    delay_sum = round(delay_sum, 2)

    equation = f"{equation[:-2]} = {delay_sum}"

    thought = f"{thought[:-2]}. {stage_delay}. In total, we can compute {equation}"
    thought += f"\nThus, the total delay is {delay_sum}"

    assert np.isclose([delay_sum, result], total_delay*scale, rtol=0.15).all()
    
    return thought 


def create_static_power_cot(cells, total_power, scale=1e3,  enable_assert=True):
    static_power_sum = 0
    thought = ""
    power_equation = ""
    for name, count, _, static_power, _ in cells: 
        # conver leakage power unit from 1nW to picowatt 
        static_power = round(static_power*scale, 2)

        thought += f"Leakage power of {name} is {static_power}. "
        power_equation += f"{count}*{static_power} + "
        static_power_sum += int(count) * float(static_power)

    static_power_sum = round(static_power_sum, 2)

    result = eval(power_equation[:-2])

    # convert total power from Watts to Picowatt 
    total_power = total_power * 1e12

    power_equation = f"{power_equation[:-2]} = {static_power_sum}"
    thought += f"\nIn total, we can compute {power_equation}.\nThus, the total static power is {static_power_sum}."

    if enable_assert: 
        assert np.isclose([static_power_sum, result], total_power, rtol=0.1).all()

    return thought 


def get_bycell(cells, library):
    metrics = []
    for name, value in cells.items(): 
        if name == 'scopeinfo':
            continue 
        
        try: # if cell is in liberty file
            pin_cap = {}
            cell = select_cell(library, name)
           
            for pin in cell.get_groups('pin'):
                pin_name = pin.args[0]
                cap = pin['capacitance']
                pin_cap[pin_name] = cap 

            for attribute in cell.attributes:
                if attribute.name == 'area':
                    cell_area = attribute.value
                if attribute.name == 'cell_leakage_power':
                    leakage_power = attribute.value
           
            name = name.replace("sky130_fd_sc_hd__", "", 1)  
            metrics.append((name, value['count'], cell_area, leakage_power, pin_cap))
        except: 
            has_blackbox = True 
            break 

    return metrics 


def write_json(thoughts, json_file):
    data = {"data": thoughts}
    with open(json_file, 'w') as file:
        json.dump(data, file, indent=4)


def sort(path):
    files = sorted(glob.glob(path, recursive=False))
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f)) or 0))
    return files 

