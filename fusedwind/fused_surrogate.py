from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn import linear_model
from sklearn.externals import joblib
from fusedwind.fused_wind import FUSED_Object, Independent_Variable, get_execution_order
import numpy as np
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
        folds = 5 # Cross validation folds
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

class Kriging_Model(object):

    def __init__(self,input,output):
        self.train_input = input
        self.train_output = np.transpose(output)[0]
        
        #Number of variables:
        self.n_vars = len(self.train_input[0,:])
        
        #The std_obj is scaling the input matrix to match the sklearn standard. Print variogram for graphical interpretation.
        self.std_obj = skl_pre.StandardScaler()
        self.X = self.std_obj.fit_transform(self.train_input)
        
        #Creating kernel !!This is a tunable point!!
        RBF_kernel = skl_gp.kernels.ConstantKernel(1,(1e-6,1e20))*skl_gp.kernels.RBF(length_scale=[1]*self.n_vars,length_scale_bounds=[(1e-20,10)]*self.n_vars)
        print('MIMC Note that below this line, you can set the n_restarts to larger values to get better surrogates')
        self.GP = skl_gp.GaussianProcessRegressor(kernel=RBF_kernel,alpha=4e-4,n_restarts_optimizer=9,normalize_y=True).fit(self.X,self.train_output)

    def get_prediction(self, input, return_std=True):
        input = self.std_obj.transform(input)
        if return_std:
            prediction, sigma = self.GP.predict(input, return_std)
            sigma = np.transpose([sigma])
        else:
            prediction = self.GP.predict(input, return_std)
            sigma = []
        prediction = np.transpose([prediction])

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

def Create_Group_Of_Surrogates_On_Dataset(data_set,input_collumn_names,output_collumn_names,linear_model=None,GP_model=None, include_kriging=True, include_LARS=True, model_list=[]):

    from fusedwind.fused_wind import FUSED_Group
    from copy import copy

    #Get input data:
    input_array = np.transpose(data_set.get_numpy_array(input_collumn_names))
    surrogate_list = []

    #We need a surrogate for each output_collumn:
    for index,output in enumerate(output_collumn_names):
        current_lin_model = copy(linear_model)
        current_GP_model = copy(GP_model)

        output_array = np.transpose(data_set.get_numpy_array(output))
        if not model_list==[]:
            surrogate_model = model_list[index]
        else:
            surrogate_model = Single_Fidelity_Surrogate(input_array,output_array,input_names=input_collumn_names,output_name = output, include_kriging=include_kriging,include_LARS=include_LARS)
        fused_object = FUSED_Surrogate(model_in=surrogate_model)
        surrogate_list.append(fused_object)
    
    group = FUSED_Group(surrogate_list)
    group.add_input_interface_from_objects(surrogate_list,merge_by_input_name=True)
    group.add_output_interface_from_objects(surrogate_list)
    return group

#A method to get much faster sampling from a group of surrogates than to push and pull:
def get_matrix_prediction_from_group(surrogate_group, input, return_std=True):
    #Get object list:
    object_list = []
    group_output_list = []
    method_output_list = []
    output = []

    for key in surrogate_group.get_all_object_keys():
        object_list.append(surrogate_group.get_object(key))
    for key in surrogate_group.get_interface()['output'].keys():
        group_output_list.append(key)

    for object in object_list:
        output_name = object.output_name
        object_used = False
        avail_std = False

        #Checking for the common output naming types:
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
        
        if object_used:
            prediction = object.model.get_prediction(input,return_std=avail_std)
            if avail_std:
                if return_std:
                    prediction = np.concatenate([[prediction[0][:,0]],[prediction[1][:,0]]],axis=0)
                else:
                    prediction = np.array([prediction[0][:,0]])
            else:
                raise print('WARNING matrix prediction of group is not tested without standard deviation yet.')
            output.append(prediction)
    output = np.concatenate(output,axis=0)

    return output,method_output_list

#Method to do a LOO analysis on a data set. It uses the standard building method to build surrogates.
def do_LOO_on_data_set(data_set,input_columns,output_columns,step_size=10,job_ids=[]):
    from fusedwind.fused_data_set import FUSED_Data_Set

    #Extracting data arrays:
    input = data_set.get_numpy_array(input_columns)
    output = data_set.get_numpy_array(output_columns)
    #List for errors:
    LOO_error_list = []
    
    print('Conducting LOO analysis')
    
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
        surrogate_group = Create_Group_Of_Surrogates_On_Dataset(current_data_set,input_columns,output_columns)
        prediction = get_matrix_prediction_from_group(surrogate_group,[current_benchmark_input],return_std=False)[0]
        
        #Calculating the error at the current point:
        error = np.abs(np.subtract(current_benchmark_output,np.transpose(prediction)[0]))
        LOO_error_list.append(error)
    return LOO_error_list
