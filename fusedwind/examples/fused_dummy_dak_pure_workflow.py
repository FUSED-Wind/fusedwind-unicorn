# This is a dummy workflow that is used to test dakota based work flow. It merely sums 9 scalars to produce a scalar

from fusedwind.fused_wind import FUSED_Object, Independent_Variable, get_execution_order

class Dummy_Dakota_Simulation(object):

    def __init__(self):
        super(Dummy_Dakota_Simulation, self).__init__()

    def calculate_stuff(self, data1, data2, data3, data4, data5, data6, data7, data8, data9):
        return data1 + data2 + data3 + data4 + data5 + data6 + data7 + data8 + data9

class FUSED_Dummy_Dakota_Simulation(FUSED_Object):

    def __init__(self, object_name_in='unnamed_dummy_object', state_version_in=None):
        super(FUSED_Dummy_Dakota_Simulation, self).__init__(object_name_in, state_version_in)

        self.model = Dummy_Dakota_Simulation()

    def _build_interface(self):

        self.add_input('data1', val=0.0)
        self.add_input('data2', val=0.0)
        self.add_input('data3', val=0.0)
        self.add_input('data4', val=0.0)
        self.add_input('data5', val=0.0)
        self.add_input('data6', val=0.0)
        self.add_input('data7', val=0.0)
        self.add_input('data8', val=0.0)
        self.add_input('data9', val=0.0)
        self.add_output('sum', val=0.0)

    def compute(self, inputs, outputs):

        outputs['sum']=self.model.calculate_stuff(\
                inputs['data1'],\
                inputs['data2'],\
                inputs['data3'],\
                inputs['data4'],\
                inputs['data5'],\
                inputs['data6'],\
                inputs['data7'],\
                inputs['data8'],\
                inputs['data9'])

indep_var1 = Independent_Variable(1.0, 'data1', object_name_in='indep_var1')
indep_var2 = Independent_Variable(2.0, 'data2', object_name_in='indep_var2')
indep_var3 = Independent_Variable(3.0, 'data3', object_name_in='indep_var3')
indep_var4 = Independent_Variable(4.0, 'data4', object_name_in='indep_var4')
indep_var5 = Independent_Variable(5.0, 'data5', object_name_in='indep_var5')
indep_var6 = Independent_Variable(6.0, 'data6', object_name_in='indep_var6')
indep_var7 = Independent_Variable(7.0, 'data7', object_name_in='indep_var7')
indep_var8 = Independent_Variable(8.0, 'data8', object_name_in='indep_var8')
indep_var9 = Independent_Variable(9.0, 'data9', object_name_in='indep_var9')

indep_list = [\
        indep_var1, \
        indep_var2, \
        indep_var3, \
        indep_var4, \
        indep_var5, \
        indep_var6, \
        indep_var7, \
        indep_var8, \
        indep_var9]

sim_object = FUSED_Dummy_Dakota_Simulation('dummy_dakota_simulation')

sim_object.connect(indep_list)

dummy_dakota_work_flow_objects = [\
        indep_var1, \
        indep_var2, \
        indep_var3, \
        indep_var4, \
        indep_var5, \
        indep_var6, \
        indep_var7, \
        indep_var8, \
        indep_var9, \
        sim_object]

dummy_dakota_work_flow_objects = get_execution_order(dummy_dakota_work_flow_objects)

dakota_output_list = [('sum', sim_object, 'sum')]

if __name__ == '__main__':

    print('Answer should be 45')
    print(sim_object.get_output_value())

