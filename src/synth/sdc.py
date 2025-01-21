
SDC_CONSTRAINTS_CLOCK= """
    create_clock -name {clock_name} -period {clock_period} [get_ports {clock_port}]; 
    set clk_input [get_port {clock_port}]
    set clk_indx [lsearch [all_inputs] $clk_input]
    set all_inputs_wo_clk [lreplace [all_inputs] $clk_indx $clk_indx ""]
    set_input_delay -clock {clock_name} 0 $all_inputs_wo_clk
    set_output_delay -clock {clock_name} 0 [all_outputs]
    set_max_fanout {max_fanout} [current_design]
"""

SDC_CONSTRAINTS_WO_CLOCK = """
create_clock -name {clock_name} -period {clock_period};
set_input_delay -clock {clock_name} 0 [all_inputs]; 
set_output_delay -clock {clock_name} 0 [all_outputs]; 
set_max_fanout {max_fanout} [current_design];
"""

SDC_DRIVER = """
set_driving_cell -lib_cell {driving_cell} -pin {driving_cell_pin} $all_inputs_wo_clk
set_driving_cell -lib_cell {driving_cell} -pin {driving_cell_pin} $clk_input
"""