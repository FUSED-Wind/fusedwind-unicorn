
import numpy as np
from fusedwind.fused_dakota import Independent_Variable_Dakota_Params
from fusedwind.fused_util import Split_Vector
from fusedwind.fused_spline import SplineModule_PiecewiseLinear
from fusedwind.fused_wind import create_interface, set_input, set_output, create_variable

# Some basic variables
######################
parameter_name = 'params'
cp_grid = np.array([0.0, 0.5, 1.0])
soln_grid = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

# Lets drop in the parameter input
##################################

# The parameters
param = Independent_Variable_Dakota_Params('temp.in.2', parameter_name, object_name_in='param_object')

# Split the parameter vector to it's different parts
split = Split_Vector(parameter_name, object_name_in='split_object')
split.add_output_split('chord_cp', 3)
split.add_output_split('twist_cp', 3, 6)
split.add_output_split('thickness_cp', 3)
split.connect(param)

# Now lets drop in some splines
###############################

# The chord spline
chord_spline = SplineModule_PiecewiseLinear(cp_grid, var_name_in='chord_cp', object_name_in='chord_spline_object')
chord_solution = chord_spline.get_spline_solution(soln_grid, var_name_in='chord', spline_solution_name_in='chord_spline_solution_object')
chord_spline.connect(split)

# The twist spline
twist_spline = SplineModule_PiecewiseLinear(cp_grid, var_name_in='twist_cp', object_name_in='twist_spline_object')
twist_solution = twist_spline.get_spline_solution(soln_grid, var_name_in='twist', spline_solution_name_in='twist_spline_solution_object')
twist_spline.connect(split)

# The thickness spline
thickness_spline = SplineModule_PiecewiseLinear(cp_grid, var_name_in='thickness_cp', object_name_in='thickness_spline_object')
thickness_solution = thickness_spline.get_spline_solution(soln_grid, var_name_in='thickness', spline_solution_name_in='thickness_spline_solution_object')
thickness_spline.connect(split)

# Retrieve the output and print it
##################################

# Retrieve the output
chord_output = chord_solution.get_output_value()
twist_output = twist_solution.get_output_value()
thickness_output = thickness_solution.get_output_value()

# print the output
print('chord_output:', chord_output)
print('twist_output:', twist_output)
print('thickness_output:', thickness_output)

