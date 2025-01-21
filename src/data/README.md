# Data Cleaning Scripts 

## Prerequisites 

- [Yosys](https://github.com/YosysHQ/yosys) 
- [Icarus Verilog](https://github.com/steveicarus/iverilog) 
- Cadence Genus (If you want to use it as your synthesis tool)

## Download Datasets 

Run the following command to download the [RTLCoder](https://github.com/hkust-zhiyao/RTL-Coder/blob/main/dataset/Resyn27k.json) and [Verigen](https://huggingface.co/datasets/shailja/Verilog_GitHub) dataset and write both datasets to the output_dir: 

```
python data/download.py --rtlcoder data/dataset/raw/Resyn27k.json --output_dir data/dataset/raw/verilog 
```

Then run deduplicate to remove duplicates from the raw Verilog dataset: 
```
python data/deduplicate.py --data data/dataset/raw/verilog
```

## Seperate the Dataset into Clean and Unclean designs

Then, we need to separete the dataset into clean and unclean designs (to be fixed later): 

```
python data/cleaner.py --data dataset/raw/verilog --clean_dir dataset/clean --leftover_dir dataset/leftover --check yosys
```


## Fix the Erroneous Designs Using GPT/Gemini

First, set your Gemini or OpenAI keys depending on which LLM you want to use for fixing the designs: 

```
export GEMINI_KEY=<your-gemini-key>
export OPENAI_KEY=<your-openai-key>
```

Then, run the auto-fix flow on the unclean designs using your GPT/Gemini:  

```
python data/fixer.py --input_dir --input_dir data/dataset/leftover --output_dir data/dataset/clean --llm gpt --check yosys
```

## Check Syntax and Synthesis Errors of the Clean Set

Make sure that your clean set is synthesizable and error free: 

```
python data/checker.py --data data/dataset/clean --check yosys 
```
