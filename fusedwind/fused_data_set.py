import h5py
import numpy as np
import os
from fusedwind.fused_wind import FUSED_Object, Independent_Variable, get_execution_order
import time

#Class to contain DOEs
#On disc it is represented by a hdf5 file. In RAM it is represented by this object containing numpy arrays and lists in a dictionary called data

class FUSED_Data_Set(object):
    def __init__(self, object_name_in = 'Unnamed_DOE_object'):
        self.name = object_name_in
        self.job_count = 0
        self.input_collumns = 0
        self.output_collumns = 0
        self.data = dict()
        self.data['inputs'] = dict()
        self.data['outputs'] = dict()
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

        for key in self.data['inputs'].keys():
            f['data/inputs/'+key] = self.data['inputs'][key]
            f['data_types/'+key] = self.data_types[key]

        for key in self.data['outputs'].keys():
            f['data/outputs/'+key] = self.data['outputs'][key]
            f['data_types/'+key] = self.data_types[key]

        stringSet = f.create_dataset('stringSet', (100,), dtype=h5py.special_dtype(vlen=str))
        stringSet.attrs["name"] = self.name

        f['job_count'] = self.job_count
        f['result_up2date'] = self.result_up2date
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
        self.result_up2date = np.array(f['result_up2date'])
        self.data_types=dict(f['data_types'])

        for key in f['data/outputs'].keys():
            #converting from hdf5 format:
            self.data_types[key] = np.array(self.data_types[key])
            #Testing file format (1 is numpy array, 2 is list...)
            if int(self.data_types[key][0]) is 1:
                self.data['outputs'][key] = np.array(f['data/outputs/'+key])
            else:
                raise Exception('Only numpy array data type is supported at the moment')

        for key in f['data/inputs'].keys():
            #converting from hdf5 format:
            self.data_types[key] = np.array(self.data_types[key])
            #Testing file format (1 is numpy array, 2 is list...)
            if int(self.data_types[key][0]) is 1:
                self.data['inputs'][key] = np.array(f['data/inputs/'+key])
            else:
                raise Exception('Only numpy array data type is supported at the moment')

    def add_input(self, inp=None, name=None):
        if inp is None:
            raise Exception('The input is not defined')

        if self.job_count is 0:
            self.job_count = len(inp)

        elif not len(inp) is self.job_count:
            raise Exception('The input is not the correct size ({}). The input size is measured to {}'.format(self.job_count,len(inp)))
        
        if name is None:
            name = 'input_{}'.format(self.input_collumns)
        elif name in self.data['inputs'].keys():
            print('The data name already exists. To avoid deletion of data the name is changed.')
            name = '{}_{}'.format(name,self.input_collumns)

        self.data['inputs'][name] = inp
        self.data_types[name] = self.type(inp)
        self.input_collumns += 1
        self.result_up2date = np.zeros(self.job_count)
    
    #If the DOE should be able to push and pull results directly from a workflow the communication is like in other fusedwind cases using independent variables. And object_tags combined with fused_objects.
    def add_indep_var(self,indep_var):
        self.input_indep_var_list.append(indep_var)

    def add_output(self, output_tag, output_obj, output_name):
        self.output_list.append([output_tag, output_obj, output_name])
        self.data['outputs'][output_name] = np.zeros(self.job_count)
        self.result_up2date = np.zeros(self.job_count)

    #This method returns a list of job-objects which can be executed in mpi. jobrange is an array of two numbers.Start and finish job.
    def get_job_list(self,job_range=[]):
        job_list = []
        if len(job_range) is 0:
            job_range = [0,self.job_count]
        elif job_range[1]>job_count or job_range[0]<0:
            raise Exception('The jobrange is beyond the current available DOE')

        for n in range(job_range[0],job_range[1]):
            if int(self.result_up2date[n]) is 0:
                job_list.append(data_set_job(self,n))

        return job_list

    #Method to push data to the independent variables and pull outputs.
    def write_output(self,job_id):
        self.push_input(job_id)
        self.pull_output(job_id)
        self.result_up2date[job_id] = 1
        
    #Method to determine the type of input. Should be expanded as new types are tested in the object.
    def type(self,inp):
        typestr = str(type(inp))
        if "class 'numpy." in typestr:
            return np.array([1])
        if "class 'list" in typestr:
            return np.array([2])

        print('The type of data is not yet included')
    
    #Returning a dictionary of three numpy arrays. input,output and result_up2date. It only returns variables and outputs that are already in numpy array format. If other data is needed the .data dictionary of the object should be consulted directly.
    def get_numpy_array(self):
        outpt = np.array([])
        for key in self.data['outputs'].keys():
            if 'numpy' in str(type(self.data['outputs'][key])):
                if outpt.size is 0:
                    outpt = np.vstack(self.data['outputs'][key])
                else:
                    outpt = np.concatenate([outpt,np.vstack(self.data['outputs'][key])],axis=1)

        inpt = np.array([])
        for key in self.data['inputs'].keys():
            if 'numpy' in str(type(self.data['inputs'][key])):
                if inpt.size is 0:
                    inpt = np.vstack(self.data['inputs'][key])
                else:
                    inpt = np.concatenate([inpt,np.vstack(self.data['inputs'][key])],axis=1)

        return {'inputs': inpt, 'outputs':outpt, 'result_up2date':self.result_up2date}

    #Pushing input to the independent variables:
    def push_input(self,job_id):
        #If the inputs are not named the standard inputs are used. Notice that this might connect the inputs wrongly and thus it is recommended to name the inputs.
        default_input_used = 0
        for indep in self.input_indep_var_list:
            if indep.name in self.data['inputs']:
                indep.set_data(self.data['inputs'][indep.name][job_id])
            elif 'input_{}'.format(default_input_used) in self.data['inputs']:
                indep.set_data(self.data['inputs']['input_{}'.format(default_input_used)][job_id])
                default_input_used += 1
                print('!!WARNING!! Default input name input_{} is used instead of {}. This is not recommended!!'.format(default_input_used,indep.name))
                time.sleep(0.5)
            else:
                raise Exception('Input can not be written! {} is not found in input data names and all {} default named inputs are used'.format(indep.name,default_input_used))

    def pull_output(self,job_id=None):
        for output_tag, output_obj, output_name in self.output_list:
            self.data['outputs'][output_name][job_id] = output_obj.get_output_value()[output_tag]
            if not output_name in self.data_types:
                self.data_types[output_name] = self.type(self.data['outputs'][output_name][job_id])
                
        self.result_up2date[job_id] = 1

    def get_output(self,job_id=None):
        if job_id is None:
            return self.get_numpy_array()
        else:
            output = dict()
            for output_tag, output_obj, output_name in self.output_list:
                output[output_name] = self.data['outputs'][output_name][job_id]

            return output

    def set_output(self,job_id,output):
        if int(self.result_up2date[job_id]) is 1:
            print('!!!!!! Warning: Overwriting updated result!!!!!!')
        for output_name in output:
            self.data['outputs'][output_name][job_id] = output[output_name]
            if not output_name in self.data_types:
                self.data_types[output_name] = self.type(output[output_name])
        self.result_up2date[job_id] = 1

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
