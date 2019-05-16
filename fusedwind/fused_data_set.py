try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    mpi_size = comm.Get_size()
except:
    MPI = None
    rank = 0
    mpi_size = 1
    comm = None

import numpy as np
import os
from fusedwind.fused_wind import FUSED_Object
import time

#Class to contain DOEs
#On disc it is represented by a hdf5 file. In RAM it is represented by this object containing numpy arrays and lists in a dictionary called data


class FUSED_Data_Set(object):
    """
    The Fused data set contains data in a dictionary. It is facilitating connections in the fused-wind environment to a DOE. It consists of the capabilities:

    class FUSED_Data_Set(object):
    def __init__(self, object_name_in = 'Unnamed_DOE_object'):

    # FILE IO
    #################

    def save_hdf5(self, hdf5_file=None):
    def load_hdf5(self, hdf5_file):

    # Low-level IO
    #################

    def declare_variable(self, name, dtype=None):
    def get_data(self, name=None, job_id=None, return_status=False):
    def set_data(self, data, name, job_id=None, dtype=None):
    def set_status(self,name,status,job_id=None):
    def has_updated_data(self,name,job_id=None,status_flag=1):

    # Connecting to a work flow
    ##################################

    def connect_indep_var(self,indep_var,data_set_var_name=None):
    def connect_output_obj(self, output_tag, output_obj, output_name):
    def get_job_list(self,job_range=[],names=None,status=None):
    def execute_push_pull(self,job_id):
    def push_input(self,job_id):
    def pull_output(self,job_id):

    # Converting to formats useful for external packages
    ###################################################

    def get_numpy_array(self,column_list,return_status='False'):

    # For now a list of jobs can be given and executed in mpi. This feature might be removed in near future updates.
    class data_set_job(object):
        def execute(self):
        def get_output(self):
        def set_output(self,output):
    """
    def __init__(self, job_count_in=0, object_name_in = 'Unnamed_Data_Set_object'):
        self.name = object_name_in
        self.job_count = job_count_in
        self.data = dict()
        self.column_list = []
        self.input_indep_var_list = []
        self.output_list = []
    
    #save_hdf5 since independent variables and outputs are pytho objects they cannot be saved to hdf5 in a smooth way. Thus the function save_pickle is meant to save everything and used in a python environment.
    def save_hdf5(self, hdf5_file=None):
        import h5py
        if hdf5_file is None:
            hdf5_file=self.name+'.hdf5'
        if os.path.isfile(hdf5_file):
            print('File exists already'.format(hdf5_file))
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
            f['data/'+key+'/status'] = self.data[key]['is_set']

        stringSet = f.create_dataset('stringSet', (100,), dtype=h5py.special_dtype(vlen=str))
        stringSet.attrs["name"] = self.name

        f['job_count'] = self.job_count
        f.close()

    #load_hdf5
    def load_hdf5(self, hdf5_file):
        import h5py
        #if not os.path.isfile(hdf5_file):
        #    raise Exception('The file does not exist')

        #f = h5py.File(hdf5_file)
        #stringSet = f['stringSet']

        #self.name = stringSet.attrs['name']
        #self.job_count = int(np.array(f['job_count']))
        #for key in f['data'].keys():
        #    #converting from hdf5 format:
        #    self.data[key] = {}
        #    self.data[key]['values'] = np.array(f['data/'+key+'/values'])
        #    self.data[key]['is_set'] = np.array(f['data/'+key+'/status'])

        if rank==0:
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
                self.data[key]['is_set'] = np.array(f['data/'+key+'/status'])
                self.column_list.append(key)
            key_list = list(f['data'].keys())
        else:
            key_list = None
            self.name = None
            self.job_count = None
        if not comm is None:
            self.name = comm.bcast(self.name, root=0)
            self.job_count = comm.bcast(self.job_count, root=0)
            key_list = comm.bcast(key_list, root=0)
            for key in key_list:
                if rank!=0:
                    self.data[key] = {}
                    self.data[key]['values'] = None
                    self.data[key]['is_set'] = None
                    self.column_list.append(key)

                self.data[key]['values'] = comm.bcast(self.data[key]['values'], root=0)
                self.data[key]['is_set'] = comm.bcast(self.data[key]['is_set'], root=0)

#    def save_pickle(self,pickle_name=None):
#        import pickle
#        if pickle_name is None:
#           pickle_name = self.name+'.p'
#
#        if os.path.isfile(pickle_name):
#            print('File {} exists already'.format(pickle_name))
#            old_number = 1
#            while True:
#                if not os.path.isfile('old_{}_{}'.format(old_number,pickle_name)):
#                    print('Moving {} to old_{}_{}'.format(pickle_name,old_number,pickle_name))
#                    os.rename(pickle_name,'old_{}_{}'.format(old_number,pickle_name))
#                    break
#                old_number +=1
#
#        #Finding comm objects:
#    
#        print('Saving DOE as {}'.format(pickle_name))
#
#        pickle.dump(self,open(pickle_name,"wb"))
#
#    def load_pickle(self,pickle_name):
#        if not os.path.isfile(pickle_name):
#            raise Exception('The file {} does not exist'.format(pickle_name))
#            
#        DOE = pickle.load( open(pickle_name), "rb")
#        print('DOE from {} open'.format(pickle_name))
#        try:
#            self.name = DOE.name
#            self.job_count = DOE.job_count
#            self.data = DOE.data
#            self.column_list = DOE.column_list
#            self.input_indep_var_list = DOE.indput_indep_var_list
#            self.output_list = DOE.output_list
#        except:
#            raise Exception('load of DOE failed. Check that the pickle is saved correctly')

    def get_labels(self):
        return self.data.keys()

    def get_data(self, name=None, job_id=None, return_status=False):
        #Check if the data is requested for several names:
        single_name = 0
        if type(name) is str:
            name = [name]
            single_name = True
        elif name is None:
            name = self.column_list
            print('Returning data for all names')
        elif not type(name) is list:
            raise Exception('The data set get_data method expects string or list but got {}'.format(type(name)))

        #Do the same with job_id:
        if job_id is None:
            job_id = range(self.job_count)
        elif not type(job_id) is list:
            try:
                job_id = [int(job_id)]
            except:
                raise Exception('The job_ids should be integer or list of integers')
            
        #Now gathering the requested data:            
        values_out = []
        status_out = []
        for n in name:
            values_out.append([self.data[n]['values'][id] for id in job_id])
            if return_status is True:
               status_out.append([self.data[n]['is_set'][id] for id in job_id])

        #If only one column is requested the data is compiled to a single list:
        if single_name:
            values_out = values_out[0]
            if return_status is True:
                status_out = status_out[0]
        
        #Returning the data:
        if return_status is True:
            return(values_out,status_out)
        else:
            return(values_out)


    #set_data data and adds it to the data set. A job_id can be given if only parts of a data column should be altered.
    def set_data(self, data, name, job_id=None, dtype=None, verbose=True):
        #Determining the numpy datatype:
        if dtype is None:
            try:
                dtype = data.dtype
            except:
                raise Exception('The data type could not be defined from the data alone. Provide a dtype or give data as numpy array with defined dtype')

        #If the data_set_object is empty it is initiated:
        if len(self.data.keys()) is 0:
            self.job_count = len(data)
            if verbose:
                print('Data set initiated with length {}'.format(self.job_count))

        #Does the data already exist? This is not nescesarily a problem:
        if name in self.column_list:
            print('Writing to existing data name {}.'.format(name))
        else:
            print('Declaring new data name {}.'.format(name))
            self.declare_variable(name,dtype)

        #If the job_id is None the entire column should be set:
        if job_id is None:
            job_id = range(self.job_count)
        elif not type(job_id) == list:
            job_id = [job_id]

        #Is the data the correct length?
        if not len(data) == len(job_id):
            raise Exception('Data length {} is not corresponding to job_id count {}.'.format(len(data)),format(len(job_id)))
        
        #Setting the data and meta data:
        for i, id in enumerate(job_id):
            self.data[name]['values'][id] =  data[i]
            #The data point status is default 0, 1 if the data is set and up to date and 2 if it is failed More can be added in a costumized version of the object.
            self.data[name]['is_set'][id] =  True

    #Sets the status flag. Automatically if output is pulled the flag is set to 1.
    def set_status(self,name,status,job_id=None):
        """Set status flag for data. Automatically if output is pulled the flag is set to 1."""
        status = int(status)
        if job_id is None:
            for id in self.data[name]['is_set']:
                id=status
        else:
            self.data[name]['is_set'][job_id] = status

    #Checks whether the entire column or a single job_id has updated data.
    def has_updated_data(self,name,job_id=None,status_flag=1):
        out = True
        if name not in self.data.keys():
            raise Exception('Name not in dataset.')

        if job_id is None:
            for id in self.data[name]['is_set']:
                if not id == status_flag:
                    out = True
        else:
            if not self.data[name]['is_set'][job_id] == status_flag:
                out = False
        return out
    
    #Iniate a variable column in the data set with name:
    def declare_variable(self, name, dtype=None):
        if name in self.data.keys():
            raise Exception('Data already exists with the name {}. Remove the data before initiating empty data row'.format(name))

        if self.job_count is None:
            raise Exception('The data_set has no length yet. This should be set manually or by providing an input before an empty set can be initiated')
        
        #Setting the data:
        self.data[name] = {}
        self.data[name]['values'] = np.empty(self.job_count,dtype=dtype)
        self.data[name]['is_set'] = np.zeros(self.job_count,dtype=bool)

        self.column_list.append(name)
      
    #If the DOE should be able to push and pull results directly from a workflow the communication is like in other fusedwind cases using independent variables. And object_tags combined with fused_objects.
    def connect_indep_var(self, indep_var, data_set_var_name=None):
        if data_set_var_name is None:
            data_set_var_name = indep_var.name

        self.input_indep_var_list.append((indep_var,data_set_var_name))

    #Function to add a fusedwind output to the dataset. It connects the output to the corresponding data column
    def connect_output_obj(self, output_tag, output_obj, output_name):
        self.output_list.append((output_tag, output_obj, output_name))
        if output_name not in self.data.keys():
            self.declare_variable(output_name)
            #print('Empty data column {} initiated'.format(output_name))
        else:
            print('WARNING:! Data column of name {} already exists. Retaining original data'.format(output_name))

    #This method returns a list of job-objects which can be executed in mpi.
    #job_range is an array of two numbers.Start and finish job.
    #names is a list of the names in the data_set to consider.
    #status is the status flag to return. eks. 2. If status is None it returns all jobs that are not flagged 1.
    def get_job_list(self,job_range=[],names=None,status=None):
        job_list = []
        if len(job_range) is 0:
            job_range = range(0,self.job_count)
        elif len(job_range) == 2:
            if job_range[1]>job_count or job_range[0]<0:
                raise Exception('The jobrange is beyond the current available DOE')
            print('Returning relevant jobs between id {} and {}'.format(job_range[0],job_range[1]))
            job_range = range(job_range[0],job_range[1])

        if names is None:
            names = self.data.keys()

        if status is None:
            for n in job_range:
                already_run = True
                for name in names:
                    if not self.data[name]['is_set'][n] == True:
                        already_run = False
                if not already_run is True:
                    job_list.append(data_set_job(self,n))
        else:
            for n in job_range:
                return_job = False
                for name in names:
                    if self.data[name]['is_set'][n] == status:
                        return_job = True
                if return_job is True:
                    job_list.append(data_set_job(self,n))

        return job_list

    #Method to push data to the independent variables and pull outputs.
    def execute_push_pull(self,job_id):
        self.push_input(job_id)
        self.pull_output(job_id)

    #Returning numpy arrays. column_list and job_id are lists of the data to be returned. The return_status flag returns another numpy array i.e. the "is-set" variables at the data-points.
    #If data requested is different dtypes the function breaks.
    def get_numpy_array(self,column_list=None,return_status=False,job_id=None):
        np_array = []
        status_array = []
        if column_list is None:
            column_list = self.column_list

        if job_id is None:
            job_id = list(range(self.job_count))

        if not type(job_id) == list:
            job_id = [job_id]

        row_cnt = len(job_id)
        if isinstance(column_list,list):
            col_cnt = len(column_list)
            np_array = np.zeros((row_cnt, col_cnt))
            status_array = np.zeros((row_cnt, col_cnt), dtype=bool)
            for at_col, name in enumerate(column_list):
                if not name in self.data:
                    raise Exception('Name {} is not found in data set'.format(name))
                current_values = np.array([self.data[name]['values'][id] for id in job_id])
                current_status = np.array([self.data[name]['is_set'][id] for id in job_id])
                np_array[:,at_col] = current_values
                status_array[:,at_col] = current_status
            print('MIMC This is a dirty HACK because some scripts only work when transposed')
            np_array = np.transpose(np_array)
            status_array = np.transpose(status_array)

        elif isinstance(column_list,str):
            name = column_list
            if not name in self.data:
                raise Exception('Name {} is not found in data set'.format(name))

            print('MIMC This is a dirty HACK because some scripts only work when transposed')
            np_array = np.reshape(np.array([self.data[name]['values'][id] for id in job_id]),(1,row_cnt))
            status_array = np.reshape(np.array([self.data[name]['is_set'][id] for id in job_id], dtype=bool),(1,row_cnt))

        else:
            raise Exception('{} is not a supportet type in get_numpy_array'.format(type(column_list)))

        if not return_status is False:
            return np_array, status_array
        else:
            return np_array

    #Pushing input to the independent variables:
    def push_input(self,job_id):
        #If the inputs are not named the standard inputs are used. Notice that this might connect the inputs wrongly and thus it is recommended to name the inputs.
        for indep, name in self.input_indep_var_list:
            if name in self.data.keys():
                if not self.data[name]['is_set'][job_id] == True:
                    raise Exception('Data flag is not 1 for name: {}, job_id: {}'.format(name,job_id))
                indep.set_data(self.data[name]['values'][job_id])
            else:
                raise Exception('Independent variable {} could not be populated from the data. If the data shouldn\'t be changed it shouldn\'t be provided to the dataset.'.format(indep.name))

    def pull_output(self,job_id):
        has_updated = set()
        for output_tag, output_obj, output_name in self.output_list:
            if not self.data[output_name]['is_set'][job_id] == True:
                if not output_obj in has_updated:
                    output_obj.update_output_data()
                    has_updated.add(output_obj)
                if output_obj.succeeded:
                    self.data[output_name]['values'][job_id] = output_obj[output_tag]
                    self.data[output_name]['is_set'][job_id] = True

    def print_bounds(self,names=None):
        if names == None:
            names = self.collumn_list
        #Find longest name:
        string_length = str(len(max(names,key=len)))
        row_format = "{0:>"+string_length+"}{1:.4}{2:4}{3:4}{4:4}"
        print(row_format.format('Name',' ','Min',' ','Max'))
        row_format = "{0:>"+string_length+"}{1:4}{2:2E}{3:4}{4:2E}"
        for name in names:
            array = self.get_numpy_array(name)
            print(row_format.format(name,'',np.min(array),'',np.max(array)))

    def split_data_set(self,index):
        data1 = self.get_numpy_array(self.column_list,job_id=range(index))
        data2 = self.get_numpy_array(self.column_list,job_id=range(index,self.job_count))
        
        data_set1 = FUSED_Data_Set()
        data_set2 = FUSED_Data_Set()

        for index,name in enumerate(self.column_list):
            data_set1.set_data(data1[index],name)
            data_set2.set_data(data2[index],name)

        return data_set1, data_set2
        #data1 = self.get_numpy_array(self.

class data_set_job(object):
    def __init__(self,data_set,job_id):
        self.data_set = data_set
        self.job_id = job_id

    def execute(self):
        return self.data_set.execute_job(self.job_id)

    def get_output(self):
        return self.data_set.get_output(self.job_id)

    def set_output(self,output):
        self.data_set.set_output(self.job_id,output)
