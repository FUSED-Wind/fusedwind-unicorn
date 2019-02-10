import h5py
import numpy as np
import os
from fusedwind.fused_wind import FUSED_Object, Independent_Variable, get_execution_order
import time

#CMOS: Mpi import for debugging:
from mpi4py import MPI


#Class to contain DOEs
#On disc it is represented by a hdf5 file. In RAM it is represented by this object containing numpy arrays and lists in a dictionary called data

# MIMC
# Move away from an input output classification at a low-level

class FUSED_Data_Set(object):
    def __init__(self, object_name_in = 'Unnamed_DOE_object'):
        self.name = object_name_in
        self.job_count = 0
        self.input_collumns = 0
        self.output_collumns = 0
        self.data = dict()
#        self.data['inputs'] = dict()
#        self.data['outputs'] = dict()
        self.collumn_list = []
        self.data_types = dict()
        self.input_indep_var_list = []
        self.output_list = []
        self.result_up2date = []

    def save(self,*args):
        print('Saving requires the user to choose between .save_hdf5 which saves the input and output data only and .save_pickle which saves the object. Use the corresponding loading. Notice that the only thing lost in hdf5 process is other objects like independent variables and outputs')
    
    def save_hdf5(self, hdf5_file=None):
        if hdf5_file is None:
            hdf5_file=self.name+'.hdf5'
        if os.path.isfile(hdf5_file):
            print('File exists already')
            old_number = 1
            while True:
                if not os.path.isfile('old_{}_{}'.format(old_number,hdf5_file)):
                    print('Moving {} to old_{}_{}'.format(hdf5_file,old_number,hdf5_file))
                    os.rename(hdf5_file,'old_{}_{}'.format(old_number,hdf5_file))
                    break
                old_number +=1
        print('Saving DOE as {}'.format(hdf5_file))        
        f = h5py.File(hdf5_file)
        for key in self.data.keys():
            f['data/'+key+'/values'] = self.data[key]['values']
            f['data/'+key+'/status'] = self.data[key]['status']

        stringSet = f.create_dataset('stringSet', (100,), dtype=h5py.special_dtype(vlen=str))
        stringSet.attrs["name"] = self.name

        f['job_count'] = self.job_count
        f.close()

    def save_pickle(self,destination=None):
        pass

    #Load hdf5
    def load_hdf5(self, hdf5_file):
        if not os.path.isfile(hdf5_file):
            raise Exception('The file does not exist')

        f = h5py.File(hdf5_file)
        stringSet = f['stringSet']

        self.name = stringSet.attrs['name']
        self.job_count = int(np.array(f['job_count']))
        for key in f['data'].keys():
            #converting from hdf5 format:
            self.data[key] = {}
            self.data[key]['values'] = np.array(f['data/'+key+'/values'])
            self.data[key]['status'] = np.array(f['data/'+key+'/status'])

    # MIMC
    # Maybe a low-level I/O
    #
    #    set_data(name , data):
    #          # It will create a new field if needed, otherwise over-write the data
    #    get_data(name)
    #          if not name in data:
    #              raise error
    #    has_data(name)
    #          if not name in data:
    #              raise error
    #          returns a vector of bools on whether the data is real
    #   
    # Another data flag is FAILED
    # Another meta field is data size
    #
    # So in the last three functions there is some inconsistency in the granularity,
    #    add_input takes a full colum where the others want all the columns at a given row
    #    maybe a better approach is to have something where colum and row selection is an argument and none assumes that all (or something like that)
    #

    def set_data(self, data, name=None):
        #Data set object 2.0 need the data name to avoid any default connections in the data_set.
        if name is None:
            raise Exception('No name of the data provided')
        
        #If the data_set_object is empty it is initiated:
        if len(self.data.keys()) is 0:
            self.job_count = len(data)
            print('Data set initiated with length {}'.format(self.job_count))
        
        #Does the data already exists? This is not nescesarily a problem:
        if name in self.data.keys():
            print('Data name {} already exists in data set and will be overwritten'.format(name))
        
        #Is the data the correct length?
        if not len(data) is self.job_count:
            raise Exception('Data length {} is not corresponding to the existing job_count {}. Create a new data set object for two lengths of data'.format(len(data),self.job_count))

        #Setting the data and meta data:
        self.data[name] = {}
        self.data[name]['values'] =  data
        #The data point status is default 0, 1 if the data is set and up to date and 2 if it is failed More can be added in a costumized version of the object.
        self.data[name]['status'] = np.ones(len(data),dtype=int)
        self.collumn_list.append(name)
        
    #In 2.0 there is no distinction between input and data. Thus it is possible to add empty data set for output concerns.
    def add_empty_data(self, name=None):
        if name is None:
            raise Exception('No name provided')

        if name in self.data.keys():
            raise Exception('Data already exists with the name {}. Remove the data before initiating empty data row'.format(name))

        if self.job_count is None:
            raise Exception('The data_set has no length yet. This should be set manually or by providing an input before an empty set can be initiated')

        #2.0 still only handles floats as model outputs:
        self.data[name] = {}
        self.data[name]['values'] = np.empty(self.job_count)
        self.data[name]['status'] = np.zeros(self.job_count,dtype=int)
        self.collumn_list.append(name)
        
    def get_output(self,job_id):
        output = dict()
        for output_tag, output_obj, output_name in self.output_list:
            if self.data[output_name]['status'][job_id] == 1:
                output[output_name] = self.data[output_name]['values'][job_id]
            else:
                output[output_name] = []
        return output

    def set_output(self,job_id,output):
        for output_name in output:
            if self.data[output_name]['status'][job_id] == 1:
                print('!!! WARNING !!! updating result with status 1 name: {}, job_id: {}'.format(output_name,job_id))
            self.data[output_name]['values'][job_id] = output[output_name]
            self.data[output_name]['status'][job_id] = 1
    
    #If the DOE should be able to push and pull results directly from a workflow the communication is like in other fusedwind cases using independent variables. And object_tags combined with fused_objects.
    def add_indep_var(self,indep_var, data_set_var_name=None):
        if data_set_var_name is None:
            data_set_var_name = indep_var.name

        self.input_indep_var_list.append((indep_var,data_set_var_name))
        # MIMC
        #
        # Short-term solution is to have the user give the name of the data set variable to assign the iv
        #    def add_indep_var(self, indep_var, data_set_var_name = None):
        #         if data_set_var_name is None:
        #              data_set_var_name = indep_var.name
        #         if data_set_var_name not in data:
        #              raise error
        #
        # Long-term solution ...
        #    The input/output interface is standardized across all objects, so one can use the connect function
        #

    def add_output(self, output_tag, output_obj, output_name):
        # MIMC
        #
        # Long-term solution ...
        #    The input/output interface is standardized across all objects, so one can use the connect function
        #
        self.output_list.append([output_tag, output_obj, output_name])
        if output_name not in self.data.keys():
            self.add_empty_data(output_name)
            print('Empty data collumn {} initiated'.format(output_name))
        else:
            print('Data collumn of name {} already exists. Check that this is not an error'.format(output_name))

    #This method returns a list of job-objects which can be executed in mpi. jobrange is an array of two numbers.Start and finish job.
    def get_job_list(self,job_range=[]):

        # MIMC
        #    As we move away from a input/output classification, we need to track each column seperately to see if it has been set. Then the job list is generated on a per column or a combination of columns.
        #

        job_list = []
        if len(job_range) is 0:
            job_range = [0,self.job_count]
        elif job_range[1]>job_count or job_range[0]<0:
            raise Exception('The jobrange is beyond the current available DOE')

        for n in range(job_range[0],job_range[1]):
            already_run = 'True'
            for name in self.data.keys():
                if not self.data[name]['status'][n] == 1:
                    already_run = 'False'
            if not already_run is 'True':
                job_list.append(data_set_job(self,n))

        return job_list

    #Method to push data to the independent variables and pull outputs.
    def write_output(self,job_id):
        self.push_input(job_id)
        self.pull_output(job_id)
        
    #Method to determine the type of input. Should be expanded as new types are tested in the object.
    def type(self,inp):
        #
        # MIMC Maybe this is a helpful internal method? If so preceed with _type in the name
        #
        typestr = str(type(inp))
        if "class 'numpy." in typestr:
            return np.array([1])
        if "class 'list" in typestr:
            return np.array([2])

        print('The type of data is not yet included')
    
    #Returning a dictionary of three numpy arrays. input,output and result_up2date. It only returns variables and outputs that are already in numpy array format. If other data is needed the .data dictionary of the object should be consulted directly.
    def get_numpy_array(self,collumn_list,return_status='False'):
        np_array = []
        status_array = []
        if isinstance(collumn_list,list):
            for name in collumn_list:
                if not name in self.data:
                    raise Exception('Name {} is not found in data set'.format(name))
                if np_array == []:
                    np_array = [self.data[name]['values']]
                    status_array = [self.data[name]['status']]
                else:
                    np_array = np.concatenate((np_array,[self.data[name]['values']]),axis=0)
                    status_array = np.concatenate((np_array,[self.data[name]['status']]),axis=0)

        elif isinstance(collumn_list,str):
            name = collumn_list
            if not name in self.data:
                raise Exception('Name {} is not found in data set'.format(name))

            np_array = self.data[name]['values']
            status_array = self.data[name]['status']
        
        else:
            raise Exception('{} is not a supportet type in get_numpy_array'.format(type(collumn_list)))

        if not return_status is 'False':
            return np_array, status_array
        else:
            return np_array

    #Pushing input to the independent variables:
    def push_input(self,job_id):
        #If the inputs are not named the standard inputs are used. Notice that this might connect the inputs wrongly and thus it is recommended to name the inputs.
        default_input_used = 0
        for indep, name in self.input_indep_var_list:
            if name in self.data.keys():
                if not self.data[name]['status'][job_id] == 1:
                    raise Exception('Data flag is not 1 for name: {}, job_id: {}'.format(name,job_id))
                indep.set_data(self.data[name]['values'][job_id])
            else:
                raise Exception('Independent variable {} could not be populated from the data. If the data shouldn\'t be changed it shouldn\'t be provided to the dataset.'.format(indep.name))

    def pull_output(self,job_id=None):
        for output_tag, output_obj, output_name in self.output_list:
            if not self.data[output_name]['status'][job_id] == 1:
                comm = MPI.COMM_WORLD
                rank = comm.Get_rank()
                self.data[output_name]['values'][job_id] = output_obj[output_tag]
                self.data[output_name]['status'][job_id] = 1

class data_set_job(object):
    def __init__(self,data_set,job_id):
        self.data_set = data_set
        self.job_id = job_id

    def execute(self):
        return self.data_set.write_output(self.job_id)

    def get_output(self):
        return self.data_set.get_output(self.job_id)

    def set_output(self,output):
        self.data_set.set_output(self.job_id,output)

# MIMC
#
#class FUSED_Data_Set(object):
#    def __init__(self, object_name_in = 'Unnamed_DOE_object'):
#
#    # FILE IO
##################
#
#    def save(self,*args):
#    def save_hdf5(self, hdf5_file=None):
#    def save_pickle(self,destination=None):
#    def load_hdf5(self, hdf5_file):
#
#    # Low-level IO
##################
#
#    def add_input(self, inp, name=None):                          -> set_input
#    def get_output(self,job_id):
#    def set_output(self,job_id,output):
#
#    # Connecting to a work flow
###################################
#
#    def add_indep_var(self,indep_var):                            -> connect_indep_var
#    def add_output(self, output_tag, output_obj, output_name):    -> connect_output_obj
#    def get_job_list(self,job_range=[]):
#    def write_output(self,job_id):                                -> execute_push_pull
#    def push_input(self,job_id):
#    def pull_output(self,job_id=None):
#
#    # Converting to formats useful for surrogates
####################################################
#
#    def get_numpy_array(self):
#
#    # Internal methods for this class
####################################################
#
#    def type(self,inp):
#        if "class 'numpy." in typestr:
#        if "class 'list" in typestr:
#
#
#class data_set_job(object):
#    def __init__(self,data_set,job_id):
#    def execute(self):
#    def get_output(self):
#    def set_output(self,output):
