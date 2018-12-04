
# FUSED wrapper
from fusedwind.fused_wind import FUSED_Object, FUSED_Group, Independent_Variable, get_execution_order, print_interface

import numpy as np

class Dummy_Add_One(object):

    def __init__(self):
        super(Dummy_Add_One, self).__init__()

    def calculate_stuff(self, data):
        return data+1

class FUSED_Dummy_Add_One(FUSED_Object):

    def __init__(self, object_name_in='unnamed_dummy_object'):
        super(FUSED_Dummy_Add_One, self).__init__(object_name_in)

        self.model = Dummy_Add_One()

    def _build_interface(self):

        self.add_input('data', val=0.0)
        self.add_output('soln', val=0.0)

    def compute(self, inputs, outputs):

        outputs['soln']=self.model.calculate_stuff(inputs['data'])

class Dummy_Add_All(object):

    def __init__(self, cnt):
        super(Dummy_Add_All, self).__init__()
        self.cnt = cnt

    def calculate_stuff(self, *args):
        retval = 0.0
        for arg in args:
            retval+=arg
        return retval

class FUSED_Dummy_Add_All(FUSED_Object):

    def __init__(self, cnt, object_name_in='unnamed_dummy_object'):
        super(FUSED_Dummy_Add_All, self).__init__(object_name_in)

        self.model = Dummy_Add_All(cnt)

    def _build_interface(self):

        for i in range(0, self.model.cnt):
            self.add_input('data%d'%(i), val=0.0)
        self.add_output('sum', val=0.0)

    def compute(self, inputs, outputs):

        args = []
        for i in range(0, self.model.cnt):
            k = 'data%d'%(i)
            val = inputs[k]
            args.append(val)
        outputs['sum']=self.model.calculate_stuff(*args)

def get_work_flow(cnt=1, stage_cnt=1):

    # This is the object dictionary
    object_dict = {}

    # Lets drop in my Independent_Variable
    input_object = Independent_Variable(0.0, 'input_data', object_name_in='input_object')
    object_dict['input_object'] = input_object

    # lets add a serial operation
    serial_add_one = FUSED_Dummy_Add_One('serial_add_one')
    object_dict['serial_add_one'] = serial_add_one
    serial_add_one.connect(input_object, 'data', 'input_data')

    # Now we create th parallel part of the work flow
    group_list = []
    for i in range(0, cnt):
        group_objects = []
        last_obj = serial_add_one
        for j in range(0, stage_cnt):
            k = 'parallel_add_one_track_%d_stage_%d'
            obj = FUSED_Dummy_Add_One(object_name_in=k)
            object_dict[k]=obj
            group_objects.append(obj)
            obj.connect(last_obj, 'data', 'soln')
            last_obj = obj
        grp = FUSED_Group(group_objects, [last_obj])
        group_list.append(grp)

    # Lets create the case runner
    case_runner = FUSED_MPI_ObjectCases(group_list)

