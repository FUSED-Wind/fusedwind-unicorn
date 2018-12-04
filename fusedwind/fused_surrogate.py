from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn import linear_model
from sklearn.externals import joblib
from fusedwind.fused_wind import FUSED_Object, Independent_Variable, get_execution_order
import numpy as np

class FUSED_Surrogate(FUSED_Object):
    def __init__(self, DOE_object_in=None, ifc={}, input_key_map={}, output_key_map={}, object_name_in='unnamed_surrogate_object',state_version_in=None):
        super(FUSED_Surrogate, self).__init__(object_name_in, state_version_in)
        self.DOE_object = DOE_object_in
        self.input_ifc = ifc
        self.input_key_map = input_key_map
        self.output_key_map = output_key_map
        self.model = None

    def _build_interface(self):
        self.add_output('prediction')

    def set_model(self,model_obj,input_names=None):
        self.model = model_obj
        if (input_names is None) and (not hasattr(self.model, 'indep_list')):
            print('No input objects were found in the model or given in set_model. This is needed for the model to act as a FUSED_model. Add them with .add_input(name,shape)')
        elif not input_names is None:
            for name in input_names:
                self.add_input(name)
            self.model.indep_list = input_names
        else:
            for name in self.model.indep_list:
                self.add_input(name)
    
    #Fused wind way of getting output:
    def compute(self,inputs,outputs):
        import pdb; pdb.set_trace()
        print(inputs)
        np_array = np.empty(len(inputs))
        for n, var_obj in enumerate(inputs):
            np_array[n] = var_obj
            print(var_obj)
        outputs['prediction'] = self.get_prediction(np_array)

    def get_prediction(self,np_input):
        #If the model is external is should just be an object with the function get_prediction which takes in a numpy array.
        if self.model is None:
            print('No Model')
        else:
            return(self.model.get_prediction(np_input))


class apply_to_numpy_inp(FUSED_Object):
    def __init__(self, indep_list, object_name_in='unnamed_dummy_object'):
        super(apply_to_numpy_inp, self).__init__(object_name_in)
        self.indep_list = indep_list

class Multi_Fidelity_Surrogate(FUSED_Surrogate):
    
    def __init__(self, data_set_object_cheap=None, data_set_object_exp=None, data_intersections=None, ifc={}, input_key_map={}, output_key_map={}, object_name_in='unnamed_surrogate_object',state_version_in=None):
        super(Multi_Fidelity_Surrogate, self).__init__(data_set_object_cheap, ifc, input_key_map, output_key_map, object_name_in, state_version_in)

        self.data_set_object_cheap = data_set_object_cheap
        self.data_set_object_exp = data_set_object_exp

        #List of index for the intersections:
        self.data_set_object_intersections = data_intersections

        #The intersection array should be an array with 0 or 1 of the length of the cheap data.
        self.data_intersections = data_intersections
        
        built = 'False'

    def build_model(self):
        self.cheap_input = self.data_set_object_cheap.get_numpy_array()['inputs']
        self.cheap_output = self.data_set_object_cheap.get_numpy_array()['outputs']

        self.exp_input = self.data_set_object_exp.get_numpy_array()['inputs']
        self.exp_output = self.data_set_object_cheap.get_numpy_array()['outputs']

        ###########
        self.exp_correction = self.exp_output
        ###########

        self.cheap_linear_model = Linear_Model(self.cheap_input,self.cheap_output)
        self.cheap_linear_model.build()
        
        linear_prediction_cheap = self.cheap_linear_model.get_prediction(self.cheap_input)
        linear_remainder = self.cheap_output-linear_prediction_cheap

        self.cheap_GP_model = Kriging_Model(self.cheap_input,linear_remainder)
        self.cheap_GP_model.build()

        self.exp_correction_linear_model = Linear_Model(self.exp_input,self.exp_correction)
        self.exp_correction_linear_model.build()

        linear_prediction_correction = self.exp_correction_linear_model.get_prediction(self.exp_input)
        linear_remainder_correction = self.exp_correction-linear_prediction_correction

        self.exp_correction_GP_model = Kriging_Model(self.exp_input,linear_remainder_correction)
        self.exp_correction_GP_model.build()

        built = 'True'
 
    #Fast way of getting a prediction on a np-array data set: 
    def get_prediction(self,input):
        prediction = self.cheap_linear_model.get_prediction(input)+self.cheap_GP_model.get_prediction(input)+self.exp_correction_linear_model.get_prediction(input)+self.exp_correction_GP_model.get_prediction(input)
        return prediction
 
class Linear_Model(linear_model.LinearRegression):
 
    def __init__(self,input=None,output=None,linear_order=3):
        super(Linear_Model, self).__init__()
        self.input = input
        self.output = output
        self.linear_order = linear_order
        self.is_build = 'False'

    def set_input(self,input=None):
        self.input = input

    def set_output(self,output=None):
        self.output = output

    def build(self):
        if self.input is None or self.output is None:
            print('#ERR# Input or output is not defined')
        else:
            self.covariates = self.create_covariates(self.input,self.linear_order)
            self.fit(self.covariates,self.output)
            self.is_build = 'True'

    def get_prediction(self,input):
        if input is self.input:
            return self.predict(self.covariates)
        else:
            covariates = self.create_covariates(input,self.linear_order)
            return self.predict(covariates)

    def create_covariates(self,input_matrix,order):
        # -------------- Linear model covariate building ------------------
        # Number of variables:
        n_var = np.size(input_matrix[0,:])
        orderGroups = {}
        orderGroups[1] = np.array(range(1,n_var+1))
        orderGroups[1] = orderGroups[1][np.newaxis,:]
        # insert constant and first order term:
        covariates = np.concatenate([np.vstack(np.ones(len(input_matrix[:,0]))),input_matrix],axis=1)
        # Going up through the orders of the covariates:
        for order_loop in range(2,order+1):
            # Going through all the existing covariates to add one factor:
            for covariate_loop in  range(0,np.size(orderGroups[order_loop-1][0,:])):
                # Inserting a variable at the end of the existing covariates. This creates dublicates, which are removed on a later stage:
                for variable_loop in range(0,n_var):
                    testGroup = np.vstack(np.sort(np.concatenate([orderGroups[order_loop-1][:,covariate_loop],[variable_loop+1]])))
                    #Adding the first number to the testgroup:
                    if not order_loop in orderGroups:
                        orderGroups[order_loop] = testGroup
                        covariate = covariates[:,testGroup[0]]
                        for k in testGroup[1:]:
                            covariate = covariate*covariates[:,k]
                        covariate = np.concatenate([covariates,covariate],axis=1)
                    else:
                        #Running through the existing covariates of same order to see if it is a duplicate:
                        for kernels in range(0,np.size(orderGroups[order_loop][0,:])):
                            if np.array_equal(testGroup, np.vstack(orderGroups[order_loop][:,kernels])):
                                break
                            #If we are at the end of the line:
                            if kernels is np.size(orderGroups[order_loop][0,:])-1:
                                orderGroups[order_loop] = np.concatenate([orderGroups[order_loop],testGroup],axis=1)
                                covariate = covariates[:,testGroup[0]]
                                for k in testGroup[1:]:
                                    covariate = covariate*covariates[:,k]
                                covariates = np.concatenate([covariates,covariate],axis=1)
        return covariates

class Kriging_Model(GaussianProcessRegressor):
    
    def __init__(self,input=None,output=None):
        super(Kriging_Model, self).__init__(input,output)
        self.input = input
        self.output = output

        self.kernel = C(1.0, (1e-2,1e2))*RBF(10,(1e-3,1e3))
        self.n_restarts_optimizer = 9
        self.optimizer = 'fmin_l_bfgs_b'
        self.alpha = 1e-6
        self.normalize_y = True
        self.copy_X_train = False
        self.random_state = None
        self.is_build = 'False'

    def build(self):
        if self.input is None or self.output is None:
            print('#ERR# Input for Kriging model not defined')
        else:
            self.fit(self.input,self.output)
        self.is_build = 'True'

    def get_prediction(self,input):
        return self.predict(input)
