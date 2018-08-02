
# FUSED wrapper
from fusedwind.fused_openmdao import FUSED_Component, FUSED_Group, FUSED_add, FUSED_print, \
                                     FUSED_Problem, FUSED_setup, FUSED_run, FUSED_OpenMDAOBase

from fusedwind.examples.fused_dummy_example import A, B, C, dummy_work_flow_objects

import numpy as np

# This is the structure of the work flow, basically the top object takes input from the 2 objects below it:
#
#     A
#    B C
#   D E F
#  G H I J
#

def run_total_openMDAO():

    output_name_list = [ 'G.G_data', 'H.H_data', 'I.I_data', 'J.J_data', 'D.sum', 'E.sum', 'F.sum', 'B.sum', 'C.sum', 'A.sum' ]

    root = FUSED_Group()
    for obj in dummy_work_flow_objects:
        FUSED_add(root, obj.object_name, FUSED_Component(obj))

    prob = FUSED_Problem(root)

    FUSED_OpenMDAOBase.setup_splits(root)

    FUSED_setup(prob)

    FUSED_run(prob)

    print("output")
    FUSED_print(root)

    for output_name in output_name_list:
        print('OUTPUT',output_name,':',prob[output_name])

    print('Note that the final sum-value should be 28 and 44')

def run_partial_openMDAO():

    output_name_list = [ 'G.G_data', 'H.H_data', 'I.I_data', 'J.J_data', 'B.E>sum', 'B.B>sum', 'C.sum', 'A.sum' ]

    root = FUSED_Group()
    split_obj = [A, B, C]
    for obj in dummy_work_flow_objects:
        if obj in split_obj or obj.is_independent_variable():
            FUSED_add(root, obj.object_name, FUSED_Component(obj))

    prob = FUSED_Problem(root)

    FUSED_OpenMDAOBase.setup_splits(root)

    FUSED_setup(prob)

    FUSED_run(prob)

    print("output")
    FUSED_print(root)

    for output_name in output_name_list:
        print('OUTPUT',output_name,':',prob[output_name])
    
    print('Note that the final sum-value should be 28 and 44')

if __name__ == '__main__':

    print('import and run one of the following functions: run_partial_openMDAO run_total_openMDAO')

