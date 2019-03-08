
import copy
import numpy as np
from fusedwind.fused_wind import FUSED_Object

class Shift_Data(FUSED_Object):

    def __init__(self, shift_value, input_name_in='unnamed_input', input_meta_in={}, output_name_in='unnamed_output', output_meta_in={}, object_name_in='unnamed_shift_object', state_version_in=None):
        super(Shift_Data, self).__init__(object_name_in, state_version_in)
        self.shift_value = shift_value
        self.input_name=input_name_in
        self.input_meta=input_meta_in
        self.output_name=output_name_in
        self.output_meta=output_meta_in

    def set_input_variable(self, input_name_in, input_meta_in):
        self.input_name=input_name_in
        self.input_meta=input_meta_in

    def set_output_variable(self, output_name_in, output_meta_in):
        self.output_name=output_name_in
        self.output_meta=output_meta_in

    def set_shift_value(self, shift_value):
        self.shift_value=shift_value

    def _build_interface(self):
        self.add_input(self.input_name, **self.input_meta)
        self.add_output(self.output_name, **self.output_meta)

    def compute(self, input_values, output_values):
        # get the interface
        ifc = self.get_interface()
        # get the input name
        input_name = list(ifc['input'])
        if len(input_name)!=1:
            raise Exception('The object can only shift one variable')
        input_name=input_name[0]
        # get the output name
        output_name = list(ifc['output'])
        if len(output_name)!=1:
            raise Exception('The object can only shift one variable')
        output_name=output_name[0]
        # calculate the shifted value
        input=input_values[input_name]
        output=input+self.shift_value
        output_values[output_name]=output

class Split_Vector(FUSED_Object):

    def __init__(self, input_name_in='unnamed_input', input_meta_in={}, object_name_in='unnamed_split_vector_object', state_version_in=None):

        super(Split_Vector,self).__init__(object_name_in, state_version_in)

        self.input_name = input_name_in
        self.input_meta = input_meta_in
        self.output = {}
        self.size = 0

    def _build_interface(self):

        # Verify the that input vector is the correct size
        meta_shape=self.size
        if 'shape' in self.input_meta:
            if isinstance(self.input_meta['shape'],float):
                if self.input_meta['shape']<self.size:
                    raise Exception('The input size specified in the meta data is too small for the registered output')
                meta_shape=self.input_meta['shape']
            else:
                if len(self.input_meta['shape'])!=1:
                    raise Exception('The input shape specified in the meta data is not a vector')
                if self.input_meta['shape'][0]<self.size:
                    raise Exception('The input size specified in the meta data is too small for the registered output')
                meta_shape=self.input_meta['shape'][0]
        else:
            if 'val' in self.input_meta:
                self.input_meta['shape']=self.input_meta['val'].shape
                if len(self.input_meta['shape'])!=1:
                    raise Exception('The input shape specified in the meta data is not a vector')
                if self.input_meta['shape'][0]<self.size:
                    raise Exception('The input size specified in the meta data is too small for the registered output')
                meta_shape=self.input_meta['shape'][0]
            else:
                self.input_meta['shape']=self.size
        if 'val' in self.input_meta:
            def_val = self.input_meta['val']
            if len(def_val.shape)!=1:
                raise Exception('The input value specified in the meta data is not a vector')
            if def_val.shape[0]!=meta_shape:
                raise Exception('The input value specified in the meta data is not consistent with the value')
        else:
            self.input_meta['val']=np.zeros(meta_shape)

        # add the input
        self.add_input(self.input_name, **self.input_meta)

        # add the output
        for name, range_tuple in self.output.items():
            size = range_tuple[1]-range_tuple[0]
            meta = {'shape':(size,),'val':np.zeros(size)}
            self.add_output(name, **meta)

    def compute(self, input_values, output_values):

        var_name = self.output.keys()

        input_vector = input_values[self.input_name]
        for name in var_name:
            if not name in self.output.keys():
                raise Exception('The output does not exist')
            range_tuple = self.output[name]
            output_values[name]=input_vector[range_tuple[0]:range_tuple[1]]
    
    def add_output_split(self, name, param_1, param_2=None):

        if param_2 is None:
            param_2 = self.size+param_1
            param_1 = self.size
        self.output[name]=(param_1,param_2)
        if param_2>self.size:
            self.size=param_2

# This will build a vector based on a set of scalars
class FUSED_Build_Vector(FUSED_Object):

    # The constructor
    def __init__(self, size, input_var_name = 'scalar', output_var_name = 'vector', object_name='unnamed_build_vector_object', state_version=None):
        super(FUSED_Build_Vector,self).__init__(object_name_in=object_name, state_version_in=state_version)

        self.size = size
        self.input_var_name = input_var_name
        self.output_var_name = output_var_name

    # Build the interface
    def _build_interface(self):

        for i in range(0, self.size):
            self.add_input(self.input_var_name+'_'+str(i), val=0.0)
        self.add_output(self.output_var_name, val=np.zeros(self.size))

    # Construct the vector
    def compute(self, input_values, output_values):

        output_values[self.output_var_name] = np.zeros(self.size)
        for i in range(0, self.size):
            output_values[self.output_var_name][i]=input_values[self.input_var_name+'_'+str(i)]

#Combining two or more np_arrays:
class FUSED_Build_Vector_From_Vectors(FUSED_Object):

    def __init__(self, size_tuple=(1), input_var_name='input_vector', output_var_name='output_vector', object_name='unnamed_build_vector_from_vector_object', state_version=None):
        super(FUSED_Build_Vector_From_Vectors,self).__init__(object_name_in=object_name, state_version_in=state_version)
        
        self.input_var_name = input_var_name
        self.output_var_name = output_var_name
        self.size_tuple=size_tuple

    def _build_interface(self):
        for index,size in enumerate(self.size_tuple):
            self.add_input('%s_%i'%(self.input_var_name,index),val=np.zeros(size))

        self.add_output(self.output_var_name,val=np.zeros(sum(self.size_tuple)))

    def compute(self, input_values, output_values):
        outvec = np.array([])
        for input in input_values:
            outvec = np.append(outvec,input_values[input])

        output_values[self.output_var_name] = outvec

# This will multiply two values
class FUSED_Multiply(FUSED_Object):

    def __init__(self, lhs_name='lhs', lhs_default_value=1.0, rhs_name='rhs', rhs_default_value=1.0, output_name='solution', output_default_value=1.0, object_name='unnamed_multiply_object', state_version=None):
        super(FUSED_Multiply, self).__init__(object_name_in=object_name, state_version_in=state_version)
        self.lhs_name = lhs_name
        self.lhs_default_value = lhs_default_value
        self.rhs_name = rhs_name
        self.rhs_default_value = rhs_default_value
        self.output_name = output_name
        self.output_default_value = output_default_value
        if not self.lhs_default_value is None and not self.rhs_default_value is None:
            self.output_default_value = self.lhs_default_value*self.rhs_default_value
        
    def _build_interface(self):
        self.add_input(self.lhs_name, val=self.lhs_default_value)
        self.add_input(self.rhs_name, val=self.rhs_default_value)
        self.add_output(self.output_name, val=self.output_default_value)

    def compute(self, input_values, output_values):

        output_values[self.output_name] = input_values[self.lhs_name] * input_values[self.rhs_name]

# This function is for building workflows that must multiple simulations
def create_workflow_by_cases(case_list, builder_function, build_args={}, case_argument="case_definition", label_function=None, grouper_function=None, grouper_arguments={}, objects_argument='objects', group_case_list_argument=None, group_base_name=None, group_label_argument='group_base_name'):
    '''
    This function will generate a work flow that is based on multiple simulations defined by cases.

    Usage:

        workflow_from_cases = create_workflow_by_cases(case_list = my_case_list, builder_function = my_builder_function, build_args = my_build_args, case_argument = my_case_argument, label_function = my_label_function, grouper_function = my_grouper_function, grouper_arguments = my_grouper_arguments, objects_argument = my_objects_argument, group_base_name=my_group_base_name, group_label_argument=my_group_label_argument)

    Input:

        case_list:             This is a list of cases
        builder_function:      This is the function that will build an object based on the case definition
        build_args:            This is a dictionary for the arguments that must be passed to the builder_function
        case_argument:         This is the name of the argument that defines the case definition in the builder function
        label_function:        This is a function that takes the object and the case definition that returns a label. The label identifies the object within a dictionary.
        grouper_function:      This is a function that will perform further processing on the set of objectsi and return the workflow as a single object.
                               Generally it will add any pre/post operations and then wrap it all into a group.
                               It must return an object that contains the work-flow
        grouper_arguments:     This contains the arguments that must be passed to the grouper function.
        objects_argument:      This is the name of the argument in the grouper function that should contain the objects
        group_base_name:       This is the name for the group
        group_label_argument:  This is the name of the argument for the group label

    Output:

        workflow_from_cases:   This is the work flow that was generated from the cases.
                               When a grouper function is given, this is the output of that object
                               Otherwise, when a label function is given, it is a dictionary of the objects
                               Otherwise it is a list of the objects generated
    '''

    # Create the container for the objects
    if label_function is None:
        objects = []
        group_case_list = case_list
    else:
        objects = {}
        group_case_list = {}

    # start generating the objects based on case
    for i, case_definition in enumerate(case_list):
        # build the object
        my_args = copy.copy(build_args)
        my_args[case_argument]=case_definition
        obj = builder_function(**my_args)
        # If we cannot label, then store the object in a list
        if label_function is None:
            objects.append(obj)
        # If we can label an object, then store the object in a dictionary
        else:
            label = label_function(obj, case_definition, i)
            if label in objects:
                raise Exception('It appears there are duplicates in the labels')
            objects[label] = obj
            group_case_list[label] = case_definition
    # If we cannot group an object, then simply return our object container
    if grouper_function is None:
        return objects
    # Build my group and return the results
    else:
        my_args = copy.copy(grouper_arguments)
        my_args[objects_argument]=objects
        if not group_base_name is None:
            my_args[group_label_argument]=group_base_name
        if not group_case_list_argument is None:
            my_args[group_case_list_argument]=group_case_list
        return grouper_function(**my_args)

# This function is for building a set of work flows based on case_definition
def create_workflow_by_cases_and_case_definition(case_definition, case_definition_to_args=None, base_create_workflow_by_cases_args={}):
    '''
    This method is used inconjunction with create_workflow_by_cases to create a work flow by cases, where some parameters are defined by a case-definition. This is useful when multiple case driven work-flows need to be generated.

    Usage:

        workflow_group = create_workflow_by_cases_and_case_definition(case_definition=my_case_definition, case_definition_to_args=my_case_definition_argument, base_create_workflow_by_cases_args=my_base_create_workflow_by_cases_args)

    The input:

        case_definition:                    This contains the data that defines this version of the workflow
        case_definition_to_args:            Controls how the data in case_definition are used to populate base_create_workflow_by_cases_args
                                            There are several versions:
                                                1) It is a function that takes the case_definition and the base_create_workflow_by_cases_args to give the updated arguments
                                                2) It is a dictionary that maps the case definition key to the argument key
                                                3) It is None (default), the case definition is transferred directly to the arguments
        base_create_workflow_by_cases_args: The default arguments for create_workflow_by_cases that do not change

    The output:

        workflow_group:                     The output of 'create_workflow_by_cases' a workflow based on cases
    '''

    # Empty arguments
    my_case_args = copy.copy(base_create_workflow_by_cases_args)
    # Test if the map is callable. If so, then call it
    if callable(case_definition_to_args):
        my_case_args = case_definition_to_args(case_definition, my_case_args)
    # Otherwise lets use the dictionary to map case-definition to the args
    elif isinstance(case_definition_to_args, dict):
        for case_def_key, arg_key in case_definition_to_args.items():
            my_case_args[arg_key]=case_definition[case_def_key]
    # If there is no conversion, then just use the case definition
    elif case_definition_to_args is None:
        for key, value in case_definition.items():
            my_case_args[key]=value
    # Lets throw exception
    else:
        raise Exception('Failed to recognize case_definition_to_args')

    # Lets create the workflow
    return create_workflow_by_cases(**my_case_args)

# This is a function to mae the name unique
def make_unique_name(name, name_set):

    if not name in name_set:
        name_set.add(name)
        return name

    idx = 2
    new_name = name+'_'+str(idx)
    while new_name in name_set:
        idx+=1
        new_name = name+'_'+str(idx)
    name_set.add(new_name)
    return new_name

