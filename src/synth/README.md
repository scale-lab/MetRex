# Synthesis Scripts 

## Prerequisites 
- [Yosys](https://github.com/YosysHQ/yosys) 
- [Icarus Verilog](https://github.com/steveicarus/iverilog) 
- [Pyverilog](https://github.com/PyHDI/Pyverilog) 

### Synthesize Clean Designs with Yosys 

```
python synth/yosys.py --data dataset/clean --liberty <path-to-liberty-file> --output_dir dataset/yosys_synth 
```

### Synthesize Clean Designs with Cadence Genus 

```
python synth/genus.py --data dataset/clean --liberty <path-to-liberty-file> --output_dir dataset/genus_synth 
```

## Create Dataset from the Synthesized Designs 

For yosys: 

```
python synth/yosys_cot.py --synth yosys_synth --liberty <path-to-liberty-file> --verilog dataset/clean 
```

For Genus: 

```
python synth/genus_cot.py --data genus_synth --liberty <path-to-liberty-file> --verilog dataset/clean 
```
