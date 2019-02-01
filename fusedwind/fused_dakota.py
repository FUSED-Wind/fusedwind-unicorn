
import numpy as np
from fusedwind.fused_wind import Independent_Variable

class Independent_Variable_Dakota_Params(Independent_Variable):

    def __init__(self, param_file_in=None, var_name_in='unnamed_variable', var_meta_in=None, object_name_in='unnamed_object', state_version_in=None): 
        from site import print_trace_now; print_trace_now()
        Independent_Variable.__init__(self, None, var_name_in, var_meta_in, object_name_in, state_version_in)

        self.param_file = None
        if not param_file_in is None:
            self.read_file(param_file_in)

    def read_file(self, param_file_in=None):
        from site import print_trace_now; print_trace_now()

        # Save the parameters file
        if not param_file_in is None:
            self.param_file = param_file_in
        if self.param_file is None:
            raise Exception('No parameters file has been specified')

        # Open the parameters file
        my_file = open(self.param_file, 'r')

        # Extract the number of parameters
        line = my_file.readline()
        words = line.split()
        n_var = int(words[0])

        # Read in the data
        my_param_data = np.zeros(n_var)
        for I in range(0, n_var):
            line = my_file.readline()
            words = line.split()
            my_param_data[I] = float(words[0])

        # Set my own data structure
        self.set_data(my_param_data)

# This will read dakota parameter files
class Dakota_Parameter_File_Reader(object):

    def __init__(self, param_file_in=None):
        from site import print_trace_now; print_trace_now()

        super(Dakota_Parameter_File_Reader, self).__init__()

        self.param_data = None
        self.param_file = None
        self.iv_list = []
        self.vector_size = 0

        if not param_file_in is None:
            self.read_file(param_file_in)

    def add_independent_variable(self, indep_var, var_size = None):
        from site import print_trace_now; print_trace_now()

        # need to look at outputs to know the size of the data
        ifc = indep_var.get_interface()
        # search through each output variable to get the size of the data
        for name, meta in ifc['output'].items():
            # Try to retrieve the variable size
            if 'val' in meta:
                val = meta['val']
                # if array then we need array size
                if isinstance(val, np.ndarray):
                    var_size = meta['val'].size
                # else we have a scalar
                else:
                    var_size=1
            if 'shape' in meta:
                var_size = 1
                for val in meta['shape'].shape:
                    var_size*=val
            # Nothing can be done if now variable size is given
            if var_size is None or var_size<=0:
                raise ValueError('The size of the data in the independent variable must be specified some how')
            self.iv_list.append((indep_var, self.vector_size, self.vector_size+var_size))
            # if there is data to add, then just add it
            if not self.param_file is None and self.param_data.size>=(self.vector_size+var_size):
                if var_size>1:
                    indep_var.set_data(self.param_data[self.vector_size:self.vector_size+var_size])
                else:
                    indep_var.set_data(self.param_data[self.vector_size])
            # record the expected size of the vector now
            self.vector_size+=var_size

    def read_file(self, param_file_in=None):
        from site import print_trace_now; print_trace_now()

        # Save the parameters file
        if not param_file_in is None:
            self.param_file = param_file_in
        if self.param_file is None:
            raise Exception('No parameters file has been specified')

        # Open the parameters file
        my_file = open(self.param_file, 'r')

        # Extract the number of parameters
        line = my_file.readline()
        words = line.split()
        n_var = int(words[0])

        # Read in the data
        self.param_data = np.zeros(n_var)
        for I in range(0, n_var):
            line = my_file.readline()
            words = line.split()
            self.param_data[I] = float(words[0])
        my_file.close()

        # copy the data to the indep vars
        for indep_var_data in self.iv_list:
            indep_var = indep_var_data[0]
            start = indep_var_data[1]
            end = indep_var_data[2]
            var_size = end-start
            if var_size>1:
                indep_var.set_data(self.param_data[start:end])
            else:
                indep_var.set_data(self.param_data[start])

# This will write dakota results
class Dakota_Results_File_Writer(object):

    def __init__(self):
        from site import print_trace_now; print_trace_now()
        super(Dakota_Results_File_Writer, self).__init__()

        self.output_dict = {}
        self.output_list = []

    def add_output(self, output_name, output_object, dakota_output_tag = None):
        from site import print_trace_now; print_trace_now()
        
        if dakota_output_tag is None:
            dakota_output_tag = output_name

        if not output_object in self.output_dict:
            self.output_dict[output_object]=None

        self.output_list.append((output_object, output_name, dakota_output_tag))

    def write_results(self, results_file_name = None):
        from site import print_trace_now; print_trace_now()

        my_file = open(results_file_name, 'w')

        # retrieve the data
        for obj in self.output_dict.keys():
            self.output_dict[obj]=obj.get_output_value()

        # write the data
        for obj, local_name, dakota_name in self.output_list:
            my_file.write(str(self.output_dict[obj][local_name])+' '+dakota_name+'\n')

        my_file.close()

# This is a work-flow that will read dakota parameter files, then write the results
class Dakota_Work_Flow(object):

    def __init__(self, input_object_list, output_list):
        from site import print_trace_now; print_trace_now()

        self.reader = Dakota_Parameter_File_Reader()
        for in_obj in input_object_list:
            self.reader.add_independent_variable(in_obj)

        self.writer = Dakota_Results_File_Writer()
        for output_name, output_object, dakota_name in output_list:
            self.writer.add_output(output_name, output_object, dakota_name)

    def execute(self, param_file, result_file):
        from site import print_trace_now; print_trace_now()

        self.reader.read_file(param_file)
        self.writer.write_results(result_file)

# This is one instance of a dakota job that can be ran in parallel
class Dakota_Job(object):

    def __init__(self, work_flow, param_file, result_file):
        from site import print_trace_now; print_trace_now()

        self.work_flow = work_flow
        self.param_file = param_file
        self.result_file = result_file

    def execute(self):
        from site import print_trace_now; print_trace_now()
        self.work_flow.execute(self.param_file, self.result_file)

