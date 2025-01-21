"""
Design Checker: 
    - Makes sure design is syntax-error free and synthesizable
"""

import os
import sys
import tqdm
import glob
import shutil
import argparse
import subprocess 
import statistics 

import pyverilog
from pyverilog.vparser.parser import parse

from MetRex.src.utils import get_logger
from typing import List

class Checker:
    
    @staticmethod
    def iverilog_check(design: str, logger):
        compile_command = ['iverilog', design, '-o', f'{os.path.join("/tmp", os.path.basename(design))}_2.out']
        try:
            compile_result = subprocess.run(
                compile_command, 
                timeout=60*4, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
        except subprocess.TimeoutExpired:
            logger.info(f"RED: Iverilog Timeout, design {design}.")
            return False, "", ""
        except Exception as e:
            logger.info(f"RED: Iverilog error {e}, design {design}")
            return False, compile_result.stderr, ""
        
        errors = compile_result.stderr
      
        # Filter out warnings in stdout and stderr
        warnings = [line for line in compile_result.stdout.decode('utf-8').split('\n') if "warning" in line] + \
           [line for line in compile_result.stderr.decode('utf-8').split('\n') if "warning" in line]
        warnings = ("\n".join(warnings))
                
        correct = (compile_result.returncode == 0)
        
        return correct, errors, warnings

    @staticmethod
    def yosys_check(design: str, logger):
        yosys_script = f"""
        # Read the design
        read_verilog {design}

        # Attempt to automatically determine the top module
        hierarchy -check -auto-top

        # Convert the design to a generic netlist
        synth -auto-top -flatten;
        """

        yosys_command = ['yosys']

        try:
            process = subprocess.run(
                yosys_command,  
                timeout=60*7, 
                input=yosys_script, 
                text=True, 
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        except subprocess.TimeoutExpired:
            logger.info(f"RED: Yosys Timeout, design {design}.")
            return False, "", ""
        except subprocess.CalledProcessError as e:
            logger.info(f"RED: Yosys execution failed {e}")
            return False, "", ""

        correct = (process.returncode == 0)

        # Filter out warnings in stdout and stderr
        warnings =  [line for line in process.stdout.split('\n') if "Warning" in line] + \
                    [line for line in process.stderr.split('\n') if "Warning" in line]
        
        # ignore memory warnings
        for i, warn in enumerate(warnings):
            if "Replacing memory" in warn:
                del warnings[i]
  
        for i, warn in enumerate(warnings):
            if "unique messages" in warn:
                del warnings[i]
                          
        warnings =  ("\n".join(warnings))
        
        errors = ""
        if not correct:
            errors = process.stderr
    
        return correct, errors, warnings


    @staticmethod
    def genus_check(design: str, liberty: str, logger):
        genus_script = f"""
        read_hdl -sv {design};
        
        # Check for errors after read_hdl
        if {{[catch {{read_hdl -sv {design}}} result]}} {{
            # If there is an error, print the error message and exit with a custom exit code
            exit 1
        }}
        
        read_libs {liberty};
        
        elaborate;
        
        # Check for errors after elaboration
        if {{[catch {{elaborate}} result]}} {{
            # If there is an error during elaboration, print the error and exit
            exit 2
        }} 
        
        synthesize -to_generic -effort high;
        
        # Check for errors after generic synthesis
        if {{[catch {{ synthesize -to_generic -effort high;}} result]}} {{
            # If there is an error during elaboration, print the error and exit
            exit 3
        }} 
        
        # Exit successfully if all operations complete without errors
        exit 0
        """ 
        
        genus_command = ['genus', '-no_gui', '-abort_on_error', '-log', '/dev/null']
        
        try:
            process = subprocess.run(
                genus_command, 
                input=genus_script, 
                timeout=60*4, 
                universal_newlines=True,
                shell=True, 
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        except subprocess.TimeoutExpired:
            logger.info(f"Genus Timeout, design {design}.")
            return False, "", ""
        except subprocess.CalledProcessError as e:
            logger.info(f"Genus execution failed {e}")
            return False, "", ""
        except Exception as e: 
            logger.info(f"An exception happened {e}")
            return False, "", ""

        correct = (process.returncode == 0)

        # Filter out warnings in stdout and stderr
        warnings = [line for line in process.stdout.split('\n') if "Warning" in line] +\
                [line for line in process.stderr.split('\n') if "Warning" in line]
        warnings = ("\n".join(warnings))
        
        errors = ""
        if not correct:
            # Filter for errors in stdout and stderr
            errors = [line for line in process.stdout.split('\n') if "Error" in line] +\
                [line for line in process.stderr.split('\n') if "Error" in line]
            errors =  ("\n".join(errors))
       
        return correct, errors, warnings


    @staticmethod 
    def count_submodules(design: str) -> int:
        count = 0
        ast, _ = parse([design]) 
        for definition in ast.description.definitions:
            def_type = type(definition).__name__
            
            if def_type == "ModuleDef":
                count += 1
        return count 


    @staticmethod
    def check(input_dir: str, tool: str, liberty: str, logger):
        design_list = glob.glob(os.path.join(input_dir, "*.v"), recursive=False)
        design_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f)) or 0))
        
        valid_count = 0 
        invalid_count = 0
        for design in tqdm.tqdm(design_list): 
            if "iverilog" in tool: 
                status, errors, warnings = Checker.iverilog_check(
                    design=design, 
                    logger=logger
                )

            if "yosys" in tool: 
                status, errors, warnings= Checker.yosys_check(
                    design=design,
                    logger=logger
                )
            
            if "genus" in tool:
                if not liberty:
                    logger.info("RED: [ERROR] you must specify liberty for genus check.")
                    sys.exit(1)
                    
                status, errors, warnings = Checker.genus_check(
                    design=design,
                    liberty=liberty,
                    logger=logger
                )


            if not status: 
                invalid_count += 1
                logger.info(f"RED: Design {design} is not synthesizable or has syntax errors.")
                logger.info(f"YELLOW: Warnings (if any): {warnings}")
                logger.info(f"RED: Errors (if any): {errors}")
            else:
                valid_count += 1
                logger.info(f"GREEN: Design {design} is correct")

        logger.info(f"GREEN: Valid Count is {valid_count}")
        logger.info(f"RED: Invalid Count is {invalid_count}")

    def get_check(check: str):
        if check == "iverilog":
            check_fn = Checker.iverilog_check
        elif check == "yosys":
            check_fn = Checker.yosys_check 
        elif check == "genus":
            check_fn = Checker.genus_check
        else:
            print(f"[Error]: Invalid {check} type!")
            sys.exit(0)
            
        return check_fn 
    
    @staticmethod
    def check_submodule_count(input_dir: str, fixed_dir: str, logger) -> None:
        raw_design_list = glob.glob(os.path.join(input_dir, "*.v"), recursive=False)
        fixed_design_list = glob.glob(os.path.join(fixed_dir, "*.v"), recursive=False)

        for design in tqdm.tqdm(fixed_design_list): 
            file_name = os.path.basename(design)
            raw_design = os.path.join(input_dir, file_name)
            
            count1 = Checker.count_submodules(design)
            count2 = Checker.count_submodules(raw_design)

            if count1 == count2: 
                logger.info(f"GREEN: Design {design} is valid")
                logger.info(f"GREEN: Design {raw_design} is valid")

                # if it isn't in fixed directory move there 
                if not os.path.isfile(os.path.join("dataset/fixed", file_name)):
                    shutil.move(design, "dataset/fixed")
                    path = os.path.join("dataset/fixed", file_name)
                    logger.info(f"YELLOW: Moving Design {design} to fixed directory {path}")
                else:
                    os.remove(design)
            else:
                logger.info(f"RED: Design {design} is NOT valid")
                os.remove(design)

    @staticmethod
    def contains_clock(design: str) -> bool:
        keywords = ["clk", "clock", "CLK", "@(posedge", "@(negedge"]
        with open(design, 'r') as file:
            contents = file.read()
            for keyword in keywords:
                if keyword in contents:
                    return True
        return False

    @staticmethod
    def count_lines(design: str) -> int:
        with open(design, 'r') as fp:
            num_lines = len(fp.readlines())
        return num_lines 
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help="Path to RTL data directory", required=True)
    parser.add_argument('--liberty', type=str, help='Path to the liberty file, required only for genus', required=False, default=None)
    parser.add_argument('--tool', nargs='+', type=str, help="Tools name to use for checking the design (genus/yosys/iverilog)", default=["yosys"])

    args = parser.parse_args()

    data = args.data 
    liberty = args.liberty
    tool = args.tool 

    logger = get_logger('checker.log')

    Checker.check(
        input_dir=data, 
        tool=tool,
        liberty=liberty,
        logger=logger 
    )
    
    design_list = glob.glob(os.path.join(data, "*.v"), recursive=False)
    design_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f)) or 0))

    # count number of lines
    line_count = []
    for design in tqdm.tqdm(design_list): 
        num_lines = Checker.count_lines(design)
        line_count.append(num_lines)
    
    if line_count: 
        logger.info(f"WHITE: Line Count Stats, Min {min(line_count)}, Median {statistics.median(line_count)} Max, {max(line_count)}")
    
        
if __name__ == "__main__":
    main()
