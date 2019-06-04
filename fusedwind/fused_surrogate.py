from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn import linear_model
from sklearn.externals import joblib
from fusedwind.fused_wind import FUSED_Object, Independent_Variable, get_execution_order
import numpy as np
from copy import copy
import pickle
import os
import matplotlib.pyplot as plt
# CMOS Thoughts on the surrogate capabilities:
#   The surrogate object should when finished be able to connect to independent variables and outputs. The "Model" part is what the power user defines. This should take a numpy array(matrix) and spit out predictions(vector) in a function .get_prediction(np_input) further more it should include a field(list) calles .input_names which allows the surrogate object to know where to connect which inputs. Notice that the order of this list is essential!
# The model might have an output name. Otherwise this is simply 'prediction'
# Example of very simple surrogate:

#class simple_surrogate(object):
#    def __init__(self):
#        self.input_names = ['x','y']
#        self.output_name = ['prediction']
#
#    def get_prediction(np_input):
#        return [sum(np_input,axis=1)]


class FUSED_Surrogate(FUSED_Object):
    def __init__(self, object_name_in='unnamed_surrogate_object',state_version_in=None,model_in=None,input_names=[],output_name=None):
        super(FUSED_Surrogate, self).__init__(object_name_in, state_version_in)

        self.model = None
        self.model_input_names = []
        self.model_output_names = []

        self.input_names = input_names
        self.output_name = output_name

        if not model_in is None:
            self.set_model(model_in)

    def _build_interface(self):
        if self.model is None:
            print('No model connected yet. The interface is still empty')
        else:
            #Adding the outputs that exist in the model to the interface:
            for var in self.output_names:
                if not var in self.model_output_names:
                    print('output name %s not found in model'%var,flush=True)
                else:
                    self.add_output(var)

            #Testing that all the inputs for the model is supplied:
            for var in self.model_input_names:
                if not var in self.input_names:
                    raise Exception('Input name %s not supplied by the fused object. Make sure that all inputs are given to the model'%var)
                else:        
                    self.add_input(var)
    
    #Class to set the model. The model_obj is described above.
    def set_model(self,model_obj):
        self.model = model_obj
        if not hasattr(self.model,'input_names'):
            raise Exception('The model has no attribute .input_names. Add this to the model object to couple it to fused_wind')
        self.model_input_names = self.model.input_names
        if hasattr(self.model,'output_names'):
            self.model_output_names = self.model.output_names
        else:
            print('The model has no outputname. The output name of the fused-object is by default \'prediction\'')
        if self.input_names == []:
            self.input_names = self.model.input_names

        if self.output_name == None:
            self.output_name = self.model.output_name
        
        self.output_names = ['%s_prediction'%self.output_name,'%s_sigma'%self.output_name]

    #Fused wind way of getting output:
    def compute(self,inputs,outputs):
        if self.model is None:
            raise Exception('No model has been connected')

        np_array = np.empty((1,len(inputs)))
        for n, var in enumerate(self.model_input_names):
            if not var in inputs:
                raise Exception('The name %s was not found in the input variables. Syncronize the connected independent variables and the model.input_names to ensure that the model is connect correctly'%var)
            np_array[0,n] = inputs[var]
        prediction = self.model.get_prediction(np_array)
        for n, var in enumerate(self.model_output_names):
            outputs[var] = prediction[n]

## ----------- The rest of the classes are CMOS customized surrogates. They can be used as default, but are a bit complicated to modify and NOT seen as an integrated part of fused wind.
#Single fidelity surrogate object:
class Single_Fidelity_Surrogate(object):

    def __init__(self, input=None, output=None, dataset=None, input_names=None, output_name=None, include_kriging=True,include_LARS=True):
        self.input = input
        self.output = output

        self.include_kriging = include_kriging
        self.include_LARS = include_LARS

        self.input_names = []
        self.input_names.extend(input_names)
        self.output_name = output_name
        if output_name == None:
            self.output_names = ['prediction','sigma']
        else:
            self.output_names = ['%s_prediction'%output_name,'%s_sigma'%output_name]
        
        if self.include_LARS:
            self.linear_model = LARS_model(input,output)
            if self.include_kriging:
                linear_remainder = self.output-self.linear_model.get_prediction(input)
                self.GP_model = Kriging_Model(input,linear_remainder)
        elif self.include_kriging:
            self.GP_model = Kriging_Model(input,output)
        else:
            raise Exception('either LARS or Kriging should be included in the model')
        
    def get_prediction(self,input,return_std=True):
        #There is no speed deficit by taking the standard deviation of the prediction as well:
        prediction = 0
        gp_sigma = 0

        if self.include_LARS:
            prediction = self.linear_model.get_prediction(input)

        if self.include_kriging:
            gp_prediction, gp_sigma = self.GP_model.get_prediction(input,return_std=True)
            prediction = prediction+gp_prediction
        
        if return_std:
            return prediction,gp_sigma
        else:
            return prediction

    def do_LOO(self, extra_array_input=None, extra_array_output=None):
        error_array = do_LOO(self, extra_array_input, extra_array_output)
        return error_array

    #Print a variogram of the data:
    def print_variogram(self,file_location=False,n_lags=30):
        try:
            from skgstat import Variogram
        except:
            print('!ERROR! not able to import skgstat.Variogram for variogram printing')
        
        if self.include_kriging:
            coordinates = self.GP_model.X
        elif self.include_LARS:
            coordinates = self.linear_model.X
        else:
            coordinates = self.input

        V = Variogram(coordinates=coordinates,values=np.transpose(self.output)[0],n_lags=n_lags)
        plt = V.plot()
        plt.suptitle('%s variogram'%self.output_name)
        if not file_location is False:
            plt.savefig(file_location)

import sklearn.linear_model as skl_lin
import sklearn.preprocessing as skl_pre
import sklearn.gaussian_process as skl_gp
import pylab as py 

class LARS_model(object):

    def __init__(self,input,output):

        #Test the special case where

        #Adjustment parameters:
        self.p = 4
        self.q = 0.6

        self.train_input = input
        self.train_output = np.transpose(output)[0]

        self.n_vars = len(self.train_input[0,:])

        self.std_obj = skl_pre.StandardScaler()
        self.X = self.std_obj.fit_transform(self.train_input)
        
        self.X_feat, self.poly_names, self.poly_names_all = self._get_features(self.X,ret_names=True)
        self.std_poly_obj = skl_pre.StandardScaler().fit(self.X_feat)
        self.X_feat = self._get_scale_feat(self.X)

        #Fitting the polynomial surrogate:
        folds = 10 # Cross validation folds
        self.reg_poly = skl_lin.LassoLarsCV(cv=folds,normalize=False).fit(self.X_feat,self.train_output) 

    def get_prediction(self,input):
        std_input = self.std_obj.transform(input)
        std_input_feat = self._get_scale_feat(std_input)
        predict = self.reg_poly.predict(std_input_feat)
        return np.transpose([predict])

    def _get_features(self,X,ret_names=False):
        poly_feat_obj = skl_pre.PolynomialFeatures(degree=self.p,include_bias=False)
        X_feat_all = poly_feat_obj.fit_transform(X)

        poly_names_all = poly_feat_obj.get_feature_names()
        X_feat = []
        poly_names = []
        for feat, name in zip(X_feat_all.T,poly_names_all): #Looping over features
            if py.norm(self._name2polydeg(name),self.q) <= self.p:
                X_feat.append(feat)
                poly_names.append(name)
        X_feat = py.array(X_feat).T
        if ret_names:
            return X_feat, poly_names, poly_names_all
        else:
            return X_feat

    def _get_scale_feat(self,X):
        X_feat = self._get_features(X)
        return self.std_poly_obj.transform(X_feat)

    def _name2polydeg(self,names):
        degs =[0]*self.n_vars #initial deg list
        for name in names.split(" "):
            name = name.replace("x","")
            name = name.split("^")
            [ind,deg] = name if len(name) > 1 else [name[0],"1"]
            degs[int(ind)] = int(deg)
        return degs

    #Adding methods to pring the model parameters of the LARS prediction:
    def _name2name(self,names,input_names=[]):
        if not len(input_names) == self.n_vars:
            print('Correct input names not provided')
            return names
        else:
            if isinstance(names,list) or isinstance(names,py.ndarray):
                out = []
                for name in names:
                    for number,input_name in enumerate(input_names):
                        name = name.replace('x%i'%number,input_name)
                out.append(name)
            else:
                for number,input_name in enumerate(input_names):
                    names = names.replace('x%i'%number,input_name)
                out = names
                return out                   

    def print_model_params(self,number_params=5,input_names=[]):
        ind_max_coef = py.argsort(abs(self.reg_poly.coef_))[::-1]
        print('The %d features with the largest coefficients:'%number_params)
        for i_max_coef in ind_max_coef[:number_params]:
            print(self._name2name(self.poly_names[i_max_coef],input_names),'(val: %1.12e)'%self.reg_poly.coef_[i_max_coef])
        print("Non-zero coefficients: %d, out of %d coef"%(len(self.reg_poly.coef_[self.reg_poly.coef_ != 0.0]),len(self.reg_poly.coef_)))


class Kriging_Model(object):

    def __init__(self,input,output):
        self.train_input = input
        self.output = np.transpose(output)[0]
        self.normalizer = np.amax(np.abs(self.output))

        self.train_output = np.divide(self.output,self.normalizer)
        #Number of variables:
        self.n_vars = len(self.train_input[0,:])
        
        #The std_obj is scaling the input matrix to match the sklearn standard. Print variogram for graphical interpretation.
        self.std_obj = skl_pre.StandardScaler()
        self.X = self.std_obj.fit_transform(self.train_input)
        
        #Creating kernel !!This is a tunable point!!
        RBF_kernel = skl_gp.kernels.ConstantKernel(1,(1e-6,1e20))*skl_gp.kernels.RBF(length_scale=[1]*self.n_vars,length_scale_bounds=[(1e-20,10)]*self.n_vars)
        self.GP = skl_gp.GaussianProcessRegressor(kernel=RBF_kernel,alpha=4e-4,n_restarts_optimizer=9,normalize_y=True).fit(self.X,self.train_output)

    def get_prediction(self, input, return_std=True):
        input = self.std_obj.transform(input)
        if return_std:
            prediction, sigma = self.GP.predict(input, return_std)
            sigma = np.transpose([sigma])
        else:
            prediction = self.GP.predict(input, return_std)
            sigma = []
        prediction = np.transpose([prediction*self.normalizer])

        return prediction, sigma 

def do_LOO(self, extra_array_input=None, extra_array_output=None):
    '''
    This function creates a Leave One Out error calculation on the input/output data of the model.
    The function takes an extra data array of test-cases.
    '''
    full_input = self.input
    full_output = self.output
    error_array = []

    #Doing leave one out test:
    for ind in range(len(self.input[:,0])):
        test_input = np.array([full_input[ind]])
        test_output = full_output[ind]

        self.input = np.delete(full_input,ind,0)
        self.output = np.delete(full_output,ind)

        self.build_model()
        predicted_output = self.get_prediction(test_input)[0]

        error_array.append(np.abs(predicted_output-test_output))

    self.input = full_input
    self.output = full_output
    self.build_model()

    #Testing error on extra points:
    if not extra_array_input is None:
        for ind, test_input in enumerate(extra_array_input):
            TI = np.array([test_input])
            TO = extra_array_output[ind]
            predicted_output = self.get_prediction(TI)[0]
            error_array.append(np.abs(predicted_output-TO))

    return np.mean(error_array)

def Create_Group_Of_Surrogates_On_Dataset(data_set,input_column_names,output_column_names,linear_model=None,GP_model=None, include_kriging=True, include_LARS=True, model_list=[],output_data_to_use=[]):
    from fusedwind.fused_wind import FUSED_Group

    #Get input data:
    input_array = np.transpose(data_set.get_numpy_array(input_column_names))
    surrogate_list = []

    #We need a surrogate for each output_collumn:
    for index,output_name in enumerate(output_column_names):
        current_lin_model = copy(linear_model)
        current_GP_model = copy(GP_model)

        output_array, status_array = data_set.get_numpy_array(output_name,return_status=True)

        output_array = np.transpose(output_array)
        status_array = np.transpose(status_array)
        
        #Has data indexes been given:
        if not output_data_to_use == []:
            output_array = output_array[output_data_to_use[index]]
            status_array = status_array[output_data_to_use[index]]

        good_output = []
        good_input = []

        for input,output,status in zip(input_array,output_array,status_array):
            if status == True:
                good_output.append(output)
                good_input.append(input)
                
        good_output = np.array(good_output)
        good_input = np.array(good_input)

        if not model_list==[]:
            surrogate_model = model_list[index]
        else:
            surrogate_model = Single_Fidelity_Surrogate(good_input,good_output,input_names=input_column_names,output_name=output_name, include_kriging=include_kriging,include_LARS=include_LARS)

        fused_object = FUSED_Surrogate(model_in=surrogate_model)
        surrogate_list.append(fused_object)
    
    group = FUSED_Group(surrogate_list)
    group.add_input_interface_from_objects(surrogate_list,merge_by_input_name=True)
    group.add_output_interface_from_objects(surrogate_list)
    return group

#A method to get much faster sampling from a group of surrogates than to push and pull:
def get_matrix_prediction_from_group(surrogate_group, input, return_std=True, return_LARS_predictions=False):
    #Get object list:
    object_list = [] #List of the surrogate objects
    group_output_list = [] #List of the output keys from the group
    method_output_list = [] #List of output keys from this method
    output = [] #Nummeric output from this method

    #Extract the objects from the group:
    for key in surrogate_group.get_all_object_keys():
        object_list.append(surrogate_group.get_object(key))

    #Extract the output keys from the group:
    for key in surrogate_group.get_interface()['output'].keys():
        group_output_list.append(key)

    #Going through objects and get outputs:
    for object in object_list:
        output_name = object.output_name #The fused surrogate has an output_name
        object_used = False #Is the object relevant?
        avail_std = False #Does the object have std avail?

        #Checking for the common output naming types - if the output_name isn't in the group outputs it is not used:
        if output_name in group_output_list:
            method_output_list.append(output_name)
            object_used = True
        elif output_name+'_prediction':
            method_output_list.append(output_name+'_prediction')
            object_used = True

        if output_name+'_sigma' in group_output_list:
            if return_std:
                method_output_list.append(output_name+'_sigma')
            avail_std = True

        if return_LARS_predictions:
            method_output_list.append(output_name+'_linear_model')
        
        if object_used:
            prediction = object.model.get_prediction(input,return_std=avail_std)
            #If linear model should be returned and  we have a linear model:
            if return_LARS_predictions and 'linear_model' in object.model.__dict__.keys():
                LARS_prediction = object.model.linear_model.get_prediction(input)
            #Otherwise we set i to 0:
            elif return_LARS_predictions:
                LARS_prediction = 0
            
            #Do we have std_error in the surrogate? Then return the std error:
            if avail_std:
                if return_std:
                    prediction = [prediction[0],prediction[1]]
                else:
                    prediction = [prediction[0]]
            else:
                prediction = [prediction]
                print('WARNING matrix prediction of group is not tested without available standard deviation yet.',flush=True)
            
            if return_LARS_predictions:
                prediction.append(LARS_prediction)
            
#            import pdb;pdb.set_trace()
            output.append(prediction)
#            output.extend(prediction)

    output = np.squeeze(np.concatenate(output,axis=0))
    
    return output,method_output_list

#Method to do a LOO analysis on a data set. It uses the standard building method to build surrogates.
def do_LOO_on_data_set(data_set,input_columns,output_columns,step_size=10,job_ids=[],include_kriging=True,include_LARS=True,add_LARS=False,target_ids=[]):
    from fusedwind.fused_data_set import FUSED_Data_Set

    #Extracting data arrays:
    input = np.array(data_set.get_numpy_array(input_columns))
    output = np.array(data_set.get_numpy_array(output_columns))
    #List for errors:
    LOO_error_list = []
    LOO_LARS_error_list = []

    print('Conducting LOO analysis')
    if target_ids == []:
        indexes = range(0,data_set.job_count,step_size)
    else:
        indexes = target_ids

    #Looping through a range of indexes to leave out:
    for index in range(0,data_set.job_count,step_size):
        print('LOO %i of %i'%(index,data_set.job_count))
        current_data_set = FUSED_Data_Set()

        #The current data:
        current_benchmark_input = input[:,index]
        current_benchmark_output = output[:,index]
        current_input = np.delete(input,index,1)
        current_output = np.delete(output,index,1)

        #Creating the data set to build a surrogate group from:
        for data,name in zip(np.ndarray.tolist(current_input)+np.ndarray.tolist(current_output),input_columns+output_columns):
            current_data_set.set_data(np.array(data),name,verbose=False)

        #Creating the surrogate:
        surrogate_group = Create_Group_Of_Surrogates_On_Dataset(current_data_set,input_columns,output_columns,include_LARS=include_LARS,include_kriging=include_kriging)

        #Getting prediction with LARS included if asked for.
        prediction,names = get_matrix_prediction_from_group(surrogate_group,[current_benchmark_input],return_std=False,return_LARS_predictions=add_LARS)

        full_prediction = []
        linear_prediction = []

        #Making sure that we have lists:
        if not (type(prediction)==list or 'array' in str(type(prediction))):
            prediction = [prediction]

        if not (type(names)==list or 'array' in str(type(names))):
            names = [names]

        for pred, name in zip(prediction,names):
            if name.endswith('_prediction'):
                full_prediction.append(pred)
            elif name.endswith('_linear_model'):
                linear_prediction.append(pred)

        #Calculating the error at the current point:
        error = np.abs(np.subtract(current_benchmark_output,full_prediction))
        LOO_error_list.append(error)

        if add_LARS:
            error = np.abs(np.subtract(current_benchmark_output,linear_prediction))
            LOO_LARS_error_list.append(error)
    if add_LARS:
        return LOO_error_list, LOO_LARS_error_list
    else:
        return LOO_error_list

def plot_center_fit(surrogate_group, center_point=[], input_names=[], output_names=[], file_base_name='plot_center_fit',window_fraction_plot=1,save_hdf5=False):

    if center_point == []:
        raise Exception('provide center point for the input')

    resolution = 100

    n_vars = len(input_names)
    
    basic_input_array = np.array([center_point]*resolution)
    
    #The data is saved in a dict:
    data_dict = {}
    data_dict['n_vars'] = n_vars
    data_dict['input_names'] = input_names
    data_dict['output_names'] = output_names
    data_dict['center_point'] = center_point

    if not os.path.isfile(file_base_name+'.pkl'):
        for obj,output_name in zip(surrogate_group.system_objects,output_names):
            obj_dict = {}
            obj_dict['meta'] = {\
                    'Include_LARS':obj.model.include_LARS,\
                    'Include_Kriging':obj.model.include_kriging}

            print(output_name + ' : ' + obj.output_name)
            input = np.transpose(obj.model.input)
            output = obj.model.output
            for index,name  in enumerate(input_names):
                input_dict = {}
                line_bound = [np.amin(input[index]),np.amax(input[index])]
                #Creating the input data for the 
                line = np.linspace(*line_bound,100)
                current_input = copy(basic_input_array)
                current_input[:,index] = line

                #First we just sort it for maximum distance:
                window_width = np.amax(np.transpose(input),axis=0)-np.amin(np.transpose(input),axis=0)
                window_width = np.delete(window_width,index)

                center_benchmark = np.delete(center_point,index)

                max_distance = np.zeros(len(output))

                for index_2,point in enumerate(np.transpose(input)):
                    point = np.delete(point,index)
                    distance = np.divide(np.abs(np.subtract(point,center_benchmark)),window_width)
                    max_distance[index_2] = np.amax(distance)
     
                #Applying max distance criterion:
                good_index = np.where(max_distance<window_fraction_plot)
                data_input = [input[index][i] for i in good_index]
                data_output = [output[i] for i in good_index]
                max_distance_good = [max_distance[i] for i in good_index]

                #Now we can get_the surrogate_output:
                current_surrogate_output = obj.model.get_prediction(current_input)[0]
                input_dict['main'] = (line,current_surrogate_output)

                legend = ['Output']

                if obj.model.include_LARS:
                    LARS_output = obj.model.linear_model.get_prediction(current_input)
                    input_dict['LARS'] = (line,LARS_output)
                    legend.append('LARS')

                if obj.model.include_kriging:
                    kriging_output = obj.model.GP_model.get_prediction(current_input)[0]
                    input_dict['Kriging'] = (line,np.add(LARS_output,kriging_output))
                    legend.append('Kriging')
                
                input_dict['scatter'] = (data_input,data_output,max_distance_good)
                obj_dict[name] = input_dict

                #Going through the input data and find out whether to plot it or not 
            data_dict[output_name] = obj_dict
            data_dict['base_name'] = file_base_name

        with open(file_base_name+'.pkl','wb') as file:
            pickle.dump(data_dict,file)

        display_center_fit(data_dict=data_dict)
    else:
        display_center_fit(pickle_file=file_base_name+'.pkl')

#Method to display center fit plots generated and parsed as pickle or dict from the plot method above.
def display_center_fit(data_dict={},pickle_file=None):
    #Check whether the data is pickle or dict:
    if data_dict == {}:
        with open(pickle_file,'rb') as file:
            data_dict = pickle.load(file)

    n_vars = data_dict['n_vars']
    file_base_name = data_dict['base_name']

    for output_name in data_dict['output_names']:
        #Initiate figure - each output should be a figure and each dimension should be a subplot
        plt.figure(figsize=(10,15)) #The size of the figure is arbitrary and set to something that fitted a screen sort of...
        obj_dict = data_dict[output_name] 
        print(output_name)
        for index,name  in enumerate(data_dict['input_names']):
            input_dict = obj_dict[name]

            #Now we can get_the surrogate_output:
            plt.subplot(n_vars,1,index+1)
            plt.plot(*input_dict['main'])
            legend = ['Output']

            if obj_dict['meta']['Include_LARS']:
                plt.plot(*input_dict['LARS'])
                legend.append('LARS')

            if obj_dict['meta']['Include_Kriging']:
                plt.plot(*input_dict['Kriging'])
                legend.append('Kriging')

            scatter_size = len(input_dict['scatter'][0][0])
            if not scatter_size == 0:
                sc = plt.scatter(input_dict['scatter'][0],input_dict['scatter'][1],c=input_dict['scatter'][2])
                plt.colorbar(sc,label='Max distance')
                plt.clim(0,np.amax(input_dict['scatter'][2])+0.1)
            plt.xlabel(name)
            plt.ylabel(str.join('_',output_name.split('.')[-1].split('_')[-4:]))
            if index == 0:
                plt.legend(legend,loc='right')
                plt.xlim(np.amin(input_dict['main'][0]),np.amax(input_dict['main'][0])+0.2*(np.amax(input_dict['main'][0])-np.amin(input_dict['main'][0])))
            plt.grid(True)

        plt.tight_layout(pad=0.4,w_pad=0.2,h_pad=0.2)
        plt.subplots_adjust(top=0.95)
        plt.suptitle(output_name)
        plt.savefig(file_base_name+'_'+output_name+'.png')
        plt.clf
        plt.close()
