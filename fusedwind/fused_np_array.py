import numpy as np
try: from mpi4py import MPI
except: MPI_Loaded = False; print('mpi4py could not be loaded')
else: MPI_Loaded = True
from fusedwind.fused_wind import Independent_Variable
import pdb
from random import randint

class Independent_Variable_np_Array(Independent_Variable):
    
    def __init__(self, np_array_in = None, var_name_in='unnamed_variable', var_meta_in=None, object_name_in='unnamed_object', state_version_in=None):
        Independent_Variable.__init__(self, None, var_name_in, var_meta_in, object_name_in, state_version_in)

        self.np_array = None
        if not np_array_in is None:
                self.read_np_array(np_array_in)
        
    def read_np_array(self, np_array_in=None):
        
        if not np_array_in is None:
            self.np_array = np_array_in
        else:
            raise Exception('No numpy array given')

        self.set_data(np_array)

class np_Array_Reader(object):

    def __init__(self, np_array_in=None):
        super(np_Array_Reader, self).__init__()

        self.param_data = None
        self.np_array = None
        self.iv_list = []
        self.vector_size = 0

        if not np_array_in is None:
            self.read_np_array(np_array_in)

    def add_independent_variable(self, indep_var, var_size = None):
        
        #Looking at outputs to know the size of data.
        ifc = indep_var.get_interface()
        #Searching for size of data:
        for name, meta in ifc['output'].items():
            if 'val' in meta:
                val = meta['val']
                if isinstance(val, np.ndarray):
                    var_size = meta['val'].size
                else:
                    var_size = 1
            elif 'shape' in meta:
                var_size = 11
                for val in meta['shape'].shape:
                    var_size*=val
            # Variable size can't be determined
            if var_size is None or var_size<=0:
                raise ValueError('The size of the data in the independent variable must be specified somehow')
            self.iv_list.append((indep_var, self.vector_size, self.vector_size+var_size))

            #Adding data
            if not self.np_array is None and self.param_data.size>=(self.vector_size+var_size):
                if var_size>1:
                    indep_var.set_data(self.param_data[self.vector_size:self.vector_size+var_size])
                else:
                    indep_var.set_data(self.param_data[self.vector_size])

            #The expected size of the vector:
            self.vector_size+=var_size

    def read_np_array(self, np_array_in=None):
        if not np_array_in is None:
            self.np_array = np_array_in
        else:
            raise Exception('No numpy array given')

        n_var = self.np_array.size
        self.param_data = self.np_array

        for indep_var_data in self.iv_list:
            indep_var = indep_var_data[0]
            start = indep_var_data[1]
            end = indep_var_data[2]
            var_size = end-start
            if var_size>1:
                indep_var.set_data(self.param_data[start:end])
            else:
                indep_var.set_data(self.param_data[start])

class np_Array_Results_Writer(object):
    
    def __init__(self):
        super(np_Array_Results_Writer, self).__init__()
        if MPI_Loaded is True:
            self.comm=MPI.COMM_WORLD

        self.output_dict = {}
        self.output_list = []

    def add_output(self, output_name, output_object, output_tag = None):
        
        if output_tag is None:
            output_tag = output_name

        if not output_object in self.output_dict:
            self.output_dict[output_object]=None

        self.output_list.append((output_object, output_name, output_tag))

    def write_results(self, result_file = None, number_of_runs = 1):
        for obj in self.output_dict.keys():
            self.output_dict[obj]=obj.get_output_value()

        #CMOS: peace of code made to catch objects without output defined:
        no_output = True
        for ifc in self.output_list:
            try: switch = len(ifc[0].interface['output'].keys())==0
            except: print('CMOS: Could not test if output is present....')
            else:
                if not len(ifc[0].interface['output'].keys())==0:
                    no_output = False

        # ------- For now just save it to different files: -------
        result = []
        if no_output:
            return result

        for obj, local_name, name in self.output_list:
            result.append(self.output_dict[obj][local_name])

        return result

class np_Array_Work_Flow(object):

    def __init__(self, input_object_list, output_list):
        self.reader = np_Array_Reader()
        for in_obj in input_object_list:
            self.reader.add_independent_variable(in_obj)

        self.writer = np_Array_Results_Writer()
        for output_name, output_object, name in output_list:
            self.writer.add_output(output_name, output_object, name)

    def execute(self, np_array_in, result_file=None, number_of_runs=None):
        self.reader.read_np_array(np_array_in)
        result = self.writer.write_results(result_file, number_of_runs)
        return result

class np_Array_Job(object):

    def __init__(self, work_flow, np_array, result_file=None, number_of_runs=None):

        self.work_flow = work_flow
        self.np_array = np_array
        self.result_file = result_file
        self.number_of_runs = number_of_runs

    def execute(self):
        result = self.work_flow.execute(self.np_array, self.result_file, self.number_of_runs)
        return {'result_file':self.result_file,'np_array':self.np_array,'result':result}
