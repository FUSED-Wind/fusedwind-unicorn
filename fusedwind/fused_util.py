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

