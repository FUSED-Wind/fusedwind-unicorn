
# FUSED wrapper
from fusedwind.fused_wind import FUSED_Object, Independent_Variable, get_execution_order

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
#  G H I J
#

A = FUSED_Dummy_Simulation(object_name_in='A')
B = FUSED_Dummy_Simulation(object_name_in='B')
C = FUSED_Dummy_Simulation(object_name_in='C')
D = FUSED_Dummy_Simulation(object_name_in='D')
E = FUSED_Dummy_Simulation(object_name_in='E')
F = FUSED_Dummy_Simulation(object_name_in='F')
G = Independent_Variable(np.array([1,2]),'G_data', object_name_in='G')
H = Independent_Variable(np.array([3,4]),'H_data', object_name_in='H')
I = Independent_Variable(np.array([5,6]),'I_data', object_name_in='I')
J = Independent_Variable(np.array([7,8]),'J_data', object_name_in='J')

A.connect(B, 'data1', 'sum')
A.connect(C, 'data2', 'sum')
B.connect(D, 'data1', 'sum')
B.connect(E, 'data2', 'sum')
C.connect(E, 'data1', 'sum')
C.connect(F, 'data2', 'sum')
D.connect(G, 'data1', 'G_data')
D.connect(H, 'data2', 'H_data')
E.connect(H, 'data1', 'H_data')
E.connect(I, 'data2', 'I_data')
F.connect(I, 'data1', 'I_data')
F.connect(J, 'data2', 'J_data')

dummy_work_flow_objects = get_execution_order([A, B, C, D, E, F, G, H, I, J])

if __name__ == '__main__':

    print('Answer should be 28 and 44')
    print(A.get_output_value())

