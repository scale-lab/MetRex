"""
Remove exact duplciates from the RTL dataset 
"""

import os
import re
import sys
import glob
import tqdm
import argparse

from MetRex.src.utils import get_logger  

def deduplicate(data_dir, logger):
    design_list = glob.glob(os.path.join(data_dir, "*.v"), recursive=False)
    design_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f)) or 0))
     
    seen = []
    seen_filenmes = []
    
    count = 0
    for design in tqdm.tqdm(design_list):
        with open(design, 'r+') as file:
            rtl = file.read()
            if rtl in seen: 
                seen_idx = seen.index(rtl)
                logger.info(f"YELLOW: {design} is a duplicate of {seen_filenmes[seen_idx]}. Removing it!")
                os.remove(design)
                count += 1 
            seen.append(rtl)
            seen_filenmes.append(design)
            
    logger.info(f"YELLOW: Duplicate Count is {count}") 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help="Path to Verilog directory", default="data/dataset/raw/verilog")

    args = parser.parse_args()
    
    logger = get_logger('deduplicate.log')
    
    deduplicate(args.data, logger)


    
    
if __name__ == "__main__":
    main()