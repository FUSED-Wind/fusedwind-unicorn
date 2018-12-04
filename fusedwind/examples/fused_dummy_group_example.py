
# FUSED wrapper
from fusedwind.fused_wind import FUSED_Object, FUSED_Group, Independent_Variable, get_execution_order, print_interface

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
# Non-Grouped configuration:
# --------------------------
#
#         A
#        B C
#       D E F
#      G H I J
#     K L M N O
#    P Q R S T U
#
# Grouped configuration:
# ----------------------
#
#               A
#            +-----+
#        /-\ | B C | |-\
#       / D \ \ E / / F \
#      / G H \ \-/ / I J \
#      +-----+     +-----+
#       K L     M     N O
#      P Q   R     S   T U

def get_non_grouped_work_flow_raw():

    object_dict={}

    A = FUSED_Dummy_Simulation(object_name_in='A') # 176, 240
    # -------------------------------------------------------
    B = FUSED_Dummy_Simulation(object_name_in='B') #  72, 104
    C = FUSED_Dummy_Simulation(object_name_in='C') # 104, 136
    # -------------------------------------------------------
    D = FUSED_Dummy_Simulation(object_name_in='D') #  28,  44
    E = FUSED_Dummy_Simulation(object_name_in='E') #  44,  60
    F = FUSED_Dummy_Simulation(object_name_in='F') #  60,  76
    # -------------------------------------------------------
    G = FUSED_Dummy_Simulation(object_name_in='G') #  10,  18
    H = FUSED_Dummy_Simulation(object_name_in='H') #  18,  26
    I = FUSED_Dummy_Simulation(object_name_in='I') #  26,  34
    J = FUSED_Dummy_Simulation(object_name_in='J') #  34,  42
    # -------------------------------------------------------
    K = FUSED_Dummy_Simulation(object_name_in='K') #   3,   7
    L = FUSED_Dummy_Simulation(object_name_in='L') #   7,  11
    M = FUSED_Dummy_Simulation(object_name_in='M') #  11,  15
    N = FUSED_Dummy_Simulation(object_name_in='N') #  15,  19
    O = FUSED_Dummy_Simulation(object_name_in='O') #  19,  23
    # -------------------------------------------------------
    P = Independent_Variable(np.array([ 1, 2]),'P_data', object_name_in='P')
    Q = Independent_Variable(np.array([ 3, 4]),'Q_data', object_name_in='Q')
    R = Independent_Variable(np.array([ 5, 6]),'R_data', object_name_in='R')
    S = Independent_Variable(np.array([ 7, 8]),'S_data', object_name_in='S')
    T = Independent_Variable(np.array([ 9,10]),'T_data', object_name_in='T')
    U = Independent_Variable(np.array([11,12]),'U_data', object_name_in='U')

    A.connect(B, 'data1', 'sum')
    A.connect(C, 'data2', 'sum')
    B.connect(D, 'data1', 'sum')
    B.connect(E, 'data2', 'sum')
    C.connect(E, 'data1', 'sum')
    C.connect(F, 'data2', 'sum')
    D.connect(G, 'data1', 'sum')
    D.connect(H, 'data2', 'sum')
    E.connect(H, 'data1', 'sum')
    E.connect(I, 'data2', 'sum')
    F.connect(I, 'data1', 'sum')
    F.connect(J, 'data2', 'sum')
    G.connect(K, 'data1', 'sum')
    G.connect(L, 'data2', 'sum')
    H.connect(L, 'data1', 'sum')
    H.connect(M, 'data2', 'sum')
    I.connect(M, 'data1', 'sum')
    I.connect(N, 'data2', 'sum')
    J.connect(N, 'data1', 'sum')
    J.connect(O, 'data2', 'sum')
    K.connect(P, 'data1', 'P_data')
    K.connect(Q, 'data2', 'Q_data')
    L.connect(Q, 'data1', 'Q_data')
    L.connect(R, 'data2', 'R_data')
    M.connect(R, 'data1', 'R_data')
    M.connect(S, 'data2', 'S_data')
    N.connect(S, 'data1', 'S_data')
    N.connect(T, 'data2', 'T_data')
    O.connect(T, 'data1', 'T_data')
    O.connect(U, 'data2', 'U_data')

    object_dict['A']=A
    object_dict['B']=B
    object_dict['C']=C
    object_dict['D']=D
    object_dict['E']=E
    object_dict['F']=F
    object_dict['G']=G
    object_dict['H']=H
    object_dict['I']=I
    object_dict['J']=J
    object_dict['K']=K
    object_dict['L']=L
    object_dict['M']=M
    object_dict['N']=N
    object_dict['O']=O
    object_dict['P']=P
    object_dict['Q']=Q
    object_dict['R']=R
    object_dict['S']=S
    object_dict['T']=T
    object_dict['U']=U

    object_list = [ A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U]

    object_list = get_execution_order(object_list)

    return object_dict, object_list

def get_non_grouped_work_flow_in_group():

    object_dict, object_list = get_non_grouped_work_flow_raw()

    return FUSED_Group(object_list, [object_dict['A']])

def get_group_1():

    # Group 1
    #########
    # -------------------------------------------------------
    B = FUSED_Dummy_Simulation(object_name_in='B') #  72, 104
    C = FUSED_Dummy_Simulation(object_name_in='C') # 104, 136
    # -------------------------------------------------------
    D_in_1 = Independent_Variable(np.array([28,44]),'D_in_1_data', object_name_in='D_in_1')
    E = FUSED_Dummy_Simulation(object_name_in='E') #  44,  60
    F_in_1 = Independent_Variable(np.array([60,76]),'F_in_1_data', object_name_in='F_in_1')
    # -------------------------------------------------------
    H_in_1 = Independent_Variable(np.array([18,26]),'H_in_1_data', object_name_in='H_in_1')
    I_in_1 = Independent_Variable(np.array([26,34]),'I_in_1_data', object_name_in='I_in_1')
    # group 1 connections
    B.connect(D_in_1, 'data1', 'D_in_1_data')
    B.connect(E, 'data2', 'sum')
    C.connect(E, 'data1', 'sum')
    C.connect(F_in_1, 'data2', 'F_in_1_data')
    E.connect(H_in_1, 'data1', 'H_in_1_data')
    E.connect(I_in_1, 'data2', 'I_in_1_data')
    # group 1 creation
    group_1 = FUSED_Group([B,C,D_in_1,E,F_in_1,H_in_1,I_in_1],[B, C])

    return group_1

def get_group_2():

    # Group 2
    #########
    D = FUSED_Dummy_Simulation(object_name_in='D') #  28,  44
    # -------------------------------------------------------
    G = FUSED_Dummy_Simulation(object_name_in='G') #  10,  18
    H = FUSED_Dummy_Simulation(object_name_in='H') #  18,  26
    # -------------------------------------------------------
    K_in_1 = Independent_Variable(np.array([ 3, 7]),'K_in_2_data', object_name_in='K_in_2')
    L_in_1 = Independent_Variable(np.array([ 7,11]),'L_in_2_data', object_name_in='L_in_2')
    M_in_1 = Independent_Variable(np.array([11,15]),'M_in_2_data', object_name_in='M_in_2')
    # group 2 connections
    D.connect(G, 'data1', 'sum')
    D.connect(H, 'data2', 'sum')
    G.connect(K_in_1, 'data1', 'K_in_2_data')
    G.connect(L_in_1, 'data2', 'L_in_2_data')
    H.connect(L_in_1, 'data1', 'L_in_2_data')
    H.connect(M_in_1, 'data2', 'M_in_2_data')
    # group 2 creation
    group_2 = FUSED_Group([D,G,H,K_in_1,L_in_1,M_in_1],[D,H])

    return group_2

def get_group_3():

    # Group 3
    #########
    F = FUSED_Dummy_Simulation(object_name_in='F') #  60,  76
    # -------------------------------------------------------
    I = FUSED_Dummy_Simulation(object_name_in='I') #  26,  34
    J = FUSED_Dummy_Simulation(object_name_in='J') #  34,  42
    # -------------------------------------------------------
    M_in_1 = Independent_Variable(np.array([11,15]),'M_in_3_data', object_name_in='M_in_3')
    N_in_1 = Independent_Variable(np.array([15,19]),'N_in_3_data', object_name_in='N_in_3')
    O_in_1 = Independent_Variable(np.array([19,23]),'O_in_3_data', object_name_in='O_in_3')
    # group 2 connections
    F.connect(I, 'data1', 'sum')
    F.connect(J, 'data2', 'sum')
    I.connect(M_in_1, 'data1', 'M_in_3_data')
    I.connect(N_in_1, 'data2', 'N_in_3_data')
    J.connect(N_in_1, 'data1', 'N_in_3_data')
    J.connect(O_in_1, 'data2', 'O_in_3_data')
    # group 3 creation
    group_3 = FUSED_Group([F,I,J,M_in_1,N_in_1,O_in_1],[F,I])

    return group_3

def get_grouped_work_flow_raw():

#
# Grouped configuration:
# ----------------------
#
#               A
#            +-----+
#        /-\ | B C | /-\
#       / D \ \ E / / F \
#      / G H \ \-/ / I J \
#      +-----+     +-----+
#       K L     M     N O
#      P Q   R     S   T U

    object_dict={}

    # Top level object
    A = FUSED_Dummy_Simulation(object_name_in='A') # 176, 240
    object_dict['A']=A

    # group 1
    group_1 = get_group_1()
    object_dict['group_1']=group_1

    # group 2
    group_2 = get_group_2()
    object_dict['group_2']=group_2

    # group 3
    group_3 = get_group_3()
    object_dict['group_3']=group_3

    # Bottom level objects
    # -------------------------------------------------------
    K = FUSED_Dummy_Simulation(object_name_in='K') #   3,   7
    L = FUSED_Dummy_Simulation(object_name_in='L') #   7,  11
    M = FUSED_Dummy_Simulation(object_name_in='M') #  11,  15
    N = FUSED_Dummy_Simulation(object_name_in='N') #  15,  19
    O = FUSED_Dummy_Simulation(object_name_in='O') #  19,  23
    object_dict['K']=K
    object_dict['L']=L
    object_dict['M']=M
    object_dict['N']=N
    object_dict['O']=O

    # Indeps
    # -------------------------------------------------------
    P = Independent_Variable(np.array([ 1, 2]),'P_data', object_name_in='P')
    Q = Independent_Variable(np.array([ 3, 4]),'Q_data', object_name_in='Q')
    R = Independent_Variable(np.array([ 5, 6]),'R_data', object_name_in='R')
    S = Independent_Variable(np.array([ 7, 8]),'S_data', object_name_in='S')
    T = Independent_Variable(np.array([ 9,10]),'T_data', object_name_in='T')
    U = Independent_Variable(np.array([11,12]),'U_data', object_name_in='U')
    object_dict['P']=P
    object_dict['Q']=Q
    object_dict['R']=R
    object_dict['S']=S
    object_dict['T']=T
    object_dict['U']=U

    A.connect(group_1, 'data1', 'B__sum')
    A.connect(group_1, 'data2', 'C__sum')

    group_1.connect(group_2, 'D_in_1_data', 'D__sum')
    group_1.connect(group_2, 'H_in_1_data', 'H__sum')
    group_1.connect(group_3, 'F_in_1_data', 'F__sum')
    group_1.connect(group_3, 'I_in_1_data', 'I__sum')

    group_2.connect(K, 'K_in_2_data', 'sum')
    group_2.connect(L, 'L_in_2_data', 'sum')
    group_2.connect(M, 'M_in_2_data', 'sum')

    group_3.connect(M, 'M_in_3_data', 'sum')
    group_3.connect(N, 'N_in_3_data', 'sum')
    group_3.connect(O, 'O_in_3_data', 'sum')

    K.connect(P, 'data1', 'P_data')
    K.connect(Q, 'data2', 'Q_data')
    L.connect(Q, 'data1', 'Q_data')
    L.connect(R, 'data2', 'R_data')
    M.connect(R, 'data1', 'R_data')
    M.connect(S, 'data2', 'S_data')
    N.connect(S, 'data1', 'S_data')
    N.connect(T, 'data2', 'T_data')
    O.connect(T, 'data1', 'T_data')
    O.connect(U, 'data2', 'U_data')

    return object_dict

def get_grouped_work_flow_in_group():

    object_dict = get_grouped_work_flow_raw()

    return FUSED_Group(object_dict.values(), [object_dict['A']])

if __name__ == '__main__':

    print('Answer should be 176 and 240')
    object_dict, object_list = get_non_grouped_work_flow_raw()
    print(object_dict['A'].get_output_value())

    group_1 = get_group_1()
    group_2 = get_group_2()
    group_3 = get_group_3()

    print('\ngroup 1:')
    print('==================')
    print('interface:')
    print('------------------')
    print_interface(group_1.get_interface())
    print('output:')
    print('Answer should be [72, 104] for B and [104, 136] for C')
    print('------------------')
    print(group_1.get_output_value())

    print('\ngroup 2:')
    print('==================')
    print('interface:')
    print('------------------')
    print_interface(group_2.get_interface())
    print('output:')
    print('Answer should be [28, 44] for D and [18, 26] for H')
    print('------------------')
    print(group_2.get_output_value())

    print('\ngroup 3:')
    print('==================')
    print('interface:')
    print('------------------')
    print_interface(group_3.get_interface())
    print('output:')
    print('Answer should be [26, 34] for I and [60, 76] for F')
    print('------------------')
    print(group_3.get_output_value(),'\n')

    print('Answer should be 176 and 240')
    object_dict = get_grouped_work_flow_raw()
    print(object_dict['A'].get_output_value())

    flat_group = get_non_grouped_work_flow_in_group()
    nest_group = get_grouped_work_flow_in_group()

    print('\nflat group:')
    print('==================')
    print('interface:')
    print('------------------')
    print_interface(flat_group.get_interface())
    print('output:')
    print('Answer should be 176 and 240')
    print('------------------')
    print(flat_group.get_output_value())

    print('\nnest group:')
    print('==================')
    print('interface:')
    print('------------------')
    nest_group.get_interface()
    print_interface(nest_group.get_interface())
    print('output:')
    print('Answer should be 176 and 240')
    print('------------------')
    print(nest_group.get_output_value())

