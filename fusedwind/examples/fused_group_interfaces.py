
# FUSED wrapper
from fusedwind.fused_wind import FUSED_Object, FUSED_Group, Independent_Variable, get_execution_order

import numpy as np

class Dummy_Simulation(object):

    def __init__(self):
        pass

    def calculate_stuff(self, data1, data2):
        return np.array([np.sum(data1),np.sum(data2)])

class FUSED_Dummy_Simulation(FUSED_Object):

    def __init__(self, object_name_in='unnamed_dummy_object'):
        super(FUSED_Dummy_Simulation, self).__init__(object_name_in)

        self.model = Dummy_Simulation()

    def _build_interface(self):

        self.add_input('data1', shape=2)
        self.add_input('data2', shape=2)
        self.add_output('sum', shape=2)

    def compute(self, inputs, outputs):

        outputs['sum']=self.model.calculate_stuff(inputs['data1'],inputs['data2'])

# This is the structure of the work flow, basically the top object takes input from the 2 objects below it:
#
#     A
#    B C
#   D E F
#

A = FUSED_Dummy_Simulation(object_name_in='A')
B = FUSED_Dummy_Simulation(object_name_in='B')
C = FUSED_Dummy_Simulation(object_name_in='C')
D = Independent_Variable(np.array([1,2]),'D_data', object_name_in='D')
E = Independent_Variable(np.array([3,4]),'E_data', object_name_in='E')
F = Independent_Variable(np.array([5,6]),'F_data', object_name_in='F')

print('MIMC Find based on un-filled connections of B and C, note, no connections have been defined')
print('MIMC Expects:', "(['B__data1', 'C__data1', 'B__data2', 'C__data2'], ['B__sum', 'C__sum'])")
grp = FUSED_Group([B, C])
grp.add_input_interface_from_connections(use_set_connections=False)
grp.add_output_interface_from_connections(use_set_connections=False)
print('MIMC Answers:', (list(grp.get_interface()['input']),list(grp.get_interface()['output'])))

A.connect(B, 'data1', 'sum')
A.connect(C, 'data2', 'sum')

# def add_input_interface_from_objects(self, object_list = None):
# def add_output_interface_from_objects(self, object_list = None):
print('MIMC Find based on objects B for input and object A for the output')
print('MIMC Expects:', "(['data1', 'data2'], ['sum'])")
grp = FUSED_Group([A, B, C])
grp.add_input_interface_from_objects([B])
grp.add_output_interface_from_objects([A])
print('MIMC Answers:', (list(grp.get_interface()['input']),list(grp.get_interface()['output'])))
print('MIMC Find based on objects B and C for input and objects A and B for the output')
print('MIMC Expects:', "(['B__data1', 'C__data1', 'B__data2', 'C__data2'], ['A__sum', 'B__sum'])")
grp = FUSED_Group([A, B, C])
grp.add_input_interface_from_objects([B, C])
grp.add_output_interface_from_objects([A, B])
print('MIMC Answers:', (list(grp.get_interface()['input']),list(grp.get_interface()['output'])))

B.connect(D, 'data1', 'D_data')
C.connect(E, 'data1', 'E_data')

print('MIMC Find based on connections where A is connected and data1 is connected')
print('MIMC Expects:', "(['B__data1', 'C__data1'], ['B__sum', 'C__sum'])")
grp = FUSED_Group([B, C])
grp.add_input_interface_from_connections()
grp.add_output_interface_from_connections()
print('MIMC Answers:', (list(grp.get_interface()['input']),list(grp.get_interface()['output'])))

print('MIMC Find based on unconnected variables where A is connected and data1 is connected')
print('MIMC Expects:', "(['B__data2', 'C__data2'], [])")
grp = FUSED_Group([B, C])
grp.add_input_interface_from_connections(use_set_connections=False)
grp.add_output_interface_from_connections(use_set_connections=False)
print('MIMC Answers:', (list(grp.get_interface()['input']),list(grp.get_interface()['output'])))

B.connect(E, 'data2', 'E_data')
C.connect(F, 'data2', 'F_data')

print('MIMC Find based on independent variables with all objects connected and all objects contained in the group')
print('MIMC Expects:', "(['D_data', 'E_data', 'F_data'], ['sum'])")
grp = FUSED_Group([A, B, C, D, E, F])
grp.add_input_interface_from_independent_variables()
grp.add_output_interface_from_objects([A])
print('MIMC Answers:', (list(grp.get_interface()['input']),list(grp.get_interface()['output'])))

# def set_input_interface_from_independent_variables(self, var_name, indep_var = None):
# def set_input_variable(self, var_name, obj_dict = None, dest_list = None):
# def set_output_variable(self, var_name, obj = None, local_output_name = None):

print('MIMC Setting the input based on independent variables, then the output based on all the data')
print('MIMC Expects:', "(['D_data', 'E_data', 'F_input_value'], ['group_output'])")
grp = FUSED_Group([A, B, C, D, E, F])
grp.set_input_interface_from_independent_variables('D_data')
grp.set_input_interface_from_independent_variables('E_data', E)
grp.set_input_interface_from_independent_variables('F_input_value', F)
grp.set_output_variable( 'group_output', A, 'sum')
print('MIMC Answers:', (list(grp.get_interface()['input']),list(grp.get_interface()['output'])))

print('MIMC Setting the input based on independent variables, then the output based on all the data')
print('MIMC Expects:', "(['my_B_data1', 'all_data_2', 'data1'], ['group_output'])")
grp = FUSED_Group([A, B, C, D, E, F])
grp.set_input_variable('my_B_data1', obj_dict={B:['data1']})
grp.set_input_variable('all_data_2', dest_list=['data2'])
grp.add_input_interface_from_objects([C])
grp.set_output_variable( 'group_output', A, 'sum')
print('MIMC Answers:', (list(grp.get_interface()['input']),list(grp.get_interface()['output'])))

dummy_work_flow_objects = get_execution_order([A, B, C, D, E, F])

if __name__ == '__main__':

    print('Answer should be 10 and 18')
    print(A.get_output_value())


