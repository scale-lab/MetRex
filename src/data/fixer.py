"""
Automatically Fix RTL syntax and synthesis errors using LLM
"""

import os
import re
import sys
import time
import glob
import tqdm
import shutil
import openai
import argparse

from openai import OpenAI
import google.generativeai as genai

from checker import Checker
from utils import get_logger

class Fixer:
    
    def __init__(self, input_dir, output_dir, logger):
        self.input_dir = input_dir 
        self.output_dir = output_dir
        self.logger = logger
        self.tmp_dir = os.path.join(self.output_dir, "tmp")
        self.tmp_dir2 = os.path.join(self.input_dir, "tmp")
        self.not_fixed_dir = os.path.join(self.input_dir, "tmp_not_fixed_gemini")

        os.makedirs(self.output_dir, exist_ok=True)        
        os.makedirs(self.tmp_dir, exist_ok=True)
        os.makedirs(self.tmp_dir2, exist_ok=True)
        os.makedirs(self.not_fixed_dir, exist_ok=True)

        self.prompt = lambda msg: f"Please act as a professional Verilog designer. \
                        Your task is to fix the synthesis erros and warnings in this verilog design: {msg['rtl']} \n \
                        error: {msg['error']}. \n \
                        warnings: {msg['warning']}. \n \
                        Please give full RTL implementation."

        self.gemini_prompt = lambda msg: f"Please fix this RTL: {msg['rtl']} \n \
                        error: {msg['error']}. \n \
                        warnings: {msg['warning']}."
                
    def fix(self, llm="gpt", check=Checker.iverilog_check, max_iters=5):
        design_list = sorted(glob.glob(os.path.join(self.input_dir, "*.v"), recursive=False)) 
        fixed_list = glob.glob(os.path.join(self.output_dir, "*.v"), recursive=False)

        # filter out already fixed designs 
        fixed_filenames = [os.path.basename(file) for file in fixed_list]
        design_list = [design for design in design_list 
                    if os.path.basename(design) not in fixed_filenames]

        design_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f)) or 0))

        for design in tqdm.tqdm(design_list):
            if not os.path.exists(design):
                self.logger.info(f"RED: File doesn't exist {design}.")
                continue 

            self.logger.info(f"Current Design: {design}")
            correct, errors, warnings = check(
                design=design, 
                logger=self.logger
            )
            
            if not correct: 
                self.logger.info(f"RED: Errors {errors}")
                self.logger.info(f"YELLOW: Warnings {warnings}")
                    
            count = 0
            current_design = design 
            
            with open(current_design, 'r') as file:
                fixed_rtl = file.read()
                    
            while (not correct or warnings) and (count <= max_iters): 
                with open(current_design, 'r') as file:
                    rtl = file.read()
                
                if llm == "gpt":
                    if len(warnings) > 5:
                        warnings = warnings[0:5]
                        
                    prompt = self.prompt({
                        'rtl': rtl, 
                        'error': errors, 
                        'warning': warnings}
                    )
                    status, response = self.call_gpt(prompt)
                    fixed_rtl = response[0]['text']
                elif llm == "gemini":
                    prompt = self.gemini_prompt({
                        'rtl': rtl, 
                        'error': errors,
                        'warning': warnings
                    })
                    status, response = self.call_gemini(prompt)
                    fixed_rtl = response
                else:
                    self.logger.info(f"RED: [Error] Invalid LLM type {llm}")
                    sys.exit(0)
                
                if not status:
                    break 
                
                tmp_path = os.path.join(self.tmp_dir, f"tmp_{count}.v")
                self.write_design(fixed_rtl, tmp_path)
                current_design = tmp_path

                correct, errors, warnings = check(
                    design=current_design,
                    logger=self.logger 
                )
                self.logger.info(f"RED: Errors {errors}")
                self.logger.info(f"YELLOW: Warnings {warnings}") 
                count += 1 
                
                # if fixer == "gpt":
                #     time.sleep(21)

            if correct and not warnings:             
                file_name = os.path.basename(design)
                self.write_design(fixed_rtl, os.path.join(self.output_dir, f"{file_name}"))
                shutil.move(design, os.path.join(self.tmp_dir2, file_name))
                self.logger.info(f"GREEN: Design {os.path.join(self.tmp_dir2, design)} is fixed")
            else:
                file_name = os.path.basename(design)
                shutil.move(design, os.path.join(self.not_fixed_dir, file_name))
                self.logger.info(f"RED: Design {os.path.join(self.not_fixed_dir, design)} is  NOT fixed")

    
    def call_gpt(self, prompt, model='gpt-3.5-turbo', temperature=0.7):
        api_key = os.environ.get('OPENAI_KEY')
        if not api_key:
            raise EnvironmentError("Environment variable 'OPENAI_KEY' is not set.")
       
        client = OpenAI(api_key=api_key)

        p_message = [
            {'role': 'system', 'content': 'Act as a Professional Verilog code fixer.'},
            {'role': 'user', 'content': prompt}
        ]
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=p_message,
                temperature=temperature,
            )
            ans = response.choices[0].message.content
            pattern = r"(module\s+\w+\s*(#\s*\(\s*[\s\S]*?\)\s*)?\(\s*[\s\S]*?\)\s*;[\s\S]*?endmodule)"
            matches = re.findall(pattern, ans)
            
            self.logger.info(f"WHITE: gpt answer: \n {ans} ")
            
            if len(matches) == 0:
                self.logger.info(f"RED: GPT response invalid: {ans}")
                return False, ""
            
            verilog_code = ""
            for match in matches:
                for m in match:
                    verilog_code += f"\n{m}"
                    break 
            
            dic = {'text': verilog_code, 'finish_reason': response.choices[0].finish_reason}
            
        except openai.OpenAIError as e:
            self.logger.info(f"RED: OpenAIError {e}.")
            dic = {'text': '', 'finish_reason': f"error {e}"}
            sys.exit(0)
            return False, [dic]

        return True, [dic] 
    

    def call_gemini(self, prompt):
        GOOGLE_API_KEY = os.environ.get('GEMINI_KEY')
        if not GOOGLE_API_KEY:
            raise EnvironmentError("Environment variable 'GEMINI_KEY' is not set.")
       
        genai.configure(api_key=GOOGLE_API_KEY) 
        model = genai.GenerativeModel('gemini-pro')

        try:
            response = model.generate_content(prompt)
            pattern = r"(module\s+\w+\s*(#\s*\(\s*[\s\S]*?\)\s*)?\(\s*[\s\S]*?\)\s*;[\s\S]*?endmodule)"
            matches = re.findall(pattern, response.text)
            if len(matches) == 0:
                self.logger.info(f"RED: Gemini response invalid, {response.text}")
                return False, ""
        except Exception as e: 
            self.logger.info(f"RED: [Error] an exception has occured, {e}")
            return False, ""

        # Add matches together 
        verilog_code = ""
        for match in matches:
            for m in match:
                verilog_code += f"\n{m}"
                break 
            
        return True, verilog_code 
        
    def write_design(self, design, save_path):
        f = open(save_path, "w")
        f.write(design)
        f.close()
         
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', type=str, help="LLM Model to use (i.e gpt or gemini)", default="gpt")
    parser.add_argument('--max_iters', type=int, help="Path to output dir", default=5)
    parser.add_argument('--check', type=str, help="Tool used for checking syntax/synthesis errors", default="yosys")
    parser.add_argument('--input_dir', type=str, help="Path to RTL data to be fixed", default="data/dataset/leftover")
    parser.add_argument('--output_dir', type=str, help="Path to output dir", default="data/dataset/fixed")

    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    llm = args.llm 
    max_iters = args.max_iters
    check = args.check
    
    logger = get_logger('fixer.log')
    check_fn = Checker.get_check(check)
    
    fixer = Fixer(
        input_dir=input_dir, 
        output_dir=output_dir,
        logger=logger
    ) 

    fixer.fix(
        llm, 
        max_iters=max_iters, 
        check=check_fn
    )
    


if __name__ == "__main__":
    main()