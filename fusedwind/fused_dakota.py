
import numpy as np
from fusedwind.fused_wind import Independent_Variable

class Independent_Variable_Dakota_Params(Independent_Variable):

    def __init__(self, param_file_in=None, var_name_in='unnamed_variable', var_meta_in=None, object_name_in='unnamed_object', state_version_in=None): 

        Independent_Variable.__init__(self, None, var_name_in, var_meta_in, object_name_in, state_version_in)
        if not param_file_in is None:
            self.read_file(param_file_in)

    def read_file(self, param_file_in=None):

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


