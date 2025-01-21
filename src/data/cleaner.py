"""
Seperate the Dataset into Clean and unclean designs
"""

import os
import glob
import tqdm
import shutil
import argparse

from checker import Checker
from MetRex.src.utils import get_logger

class Cleaner:

    def __init__(self, data_path, clean_dir, leftover_dir, liberty, logger):
        self.data_path = data_path
        self.clean_dir = clean_dir
        self.leftover_dir = leftover_dir
        self.liberty = liberty
        self.logger = logger 
        
        os.makedirs(self.clean_dir, exist_ok=True)
        os.makedirs(self.leftover_dir, exist_ok=True)
        
        
    def clean(self, check='genus'):
        design_list = glob.glob(os.path.join(self.data_path, "*.v"), recursive=False)
        cleaned_list = glob.glob(os.path.join(self.clean_dir, "*.v"), recursive=False)
        leftover_list = glob.glob(os.path.join(self.leftover_dir, "*.v"), recursive=False)

        cleaned_filenames = [os.path.basename(file) for file in cleaned_list]
        leftover_filenames = [os.path.basename(file) for file in leftover_list]
       
        # filter out already cleaned designs 
        design_list = [design for design in design_list 
                if os.path.basename(design) not in cleaned_filenames \
                and os.path.basename(design) not in leftover_filenames]
                    
        design_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f)) or 0))

        for design in tqdm.tqdm(design_list): 
            
            self.logger.info(f"WHITE: Current Design {design}")
            
            if check == "genus":
                status, errors, warnings = Checker.genus_check(
                    design=design, 
                    liberty=self.liberty,
                    logger=self.logger
                )
                
            if check == "iverilog":
                status, errors, warnings = Checker.iverilog_check(
                    design=design,
                    logger=self.logger
                )
            
            if check == "yosys":
                status, errors, warnings = Checker.yosys_check(
                    design=design,
                    logger=self.logger
                )
           
            if status and not warnings: 
                shutil.copy(design, self.clean_dir)
            elif len(errors) != 0 or len(warnings) != 0:
                self.logger.info(f"RED: Errors {errors}")
                self.logger.info(f"YELLOW: Warnings {warnings}")  
                shutil.copy(design, self.leftover_dir)
             
    def count(self):
        counter = lambda dir: len(glob.glob(os.path.join(dir, "*.v"), recursive=False))
        clean_count = counter(self.clean_dir)
        incorrect_count = counter(self.leftover_dir)
        return clean_count, incorrect_count

        
    def clean_data(self):
        design_list = glob.glob(os.path.join(self.data_path, "*.v"), recursive=False)
        cleaned_list = glob.glob(os.path.join(self.clean_dir, "*.v"), recursive=False)
        leftover_list = glob.glob(os.path.join(self.leftover_dir, "*.v"), recursive=False)

        cleaned_filenames = [os.path.basename(file) for file in cleaned_list]
        leftover_filenames = [os.path.basename(file) for file in leftover_list]

        # filter already cleaned designs and delete them 
        design_list = [design for design in design_list 
                    if os.path.basename(design) in cleaned_filenames or os.path.basename(design) in leftover_filenames]
        
        for design in design_list:
            self.logger.info(f"YELLOW: Removing Design {design}")
            os.remove(design)
                   

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help="Path to RTL data directory", default="data/dataset/raw/verilog")
    parser.add_argument('--check', type=str, help="Tool to use to check the design syntax.", default="yosys")
    parser.add_argument('--liberty', type=str, help='Path to the liberty file', required=False, default=None)
    parser.add_argument('--clean_dir', type=str, help="Path to output clean directory", default="data/dataset/clean")
    parser.add_argument('--leftover_dir', type=str, help="Path to output leftover dir", default="data/dataset/leftover")

    args = parser.parse_args()
    
    data_dir = args.data 
    check = args.check
    liberty = args.liberty 
    clean_dir = args.clean_dir
    leftover_dir = args.leftover_dir
    
    os.makedirs(clean_dir, exist_ok=True)
    output = os.path.join('cleaner.log')

    logger = get_logger(output=output)
    
    cleaner = Cleaner(
        data_path=data_dir, 
        clean_dir=clean_dir,
        leftover_dir=leftover_dir,
        liberty=liberty,
        logger=logger
    )   
    cleaner.clean(check=check)
    
    clean_count, incorrect_count = cleaner.count()

    logger.info(f"GREEN: Number of clean designs {clean_count}")
    logger.info(f"RED: Number of erroneous designs {incorrect_count}")


if __name__ == "__main__":
    main() 

