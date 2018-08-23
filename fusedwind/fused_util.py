import numpy as np
from fusedwind.fused_wind import FUSED_Object

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

    def compute(self, input_values, output_values, var_name=[]):

        if len(var_name)==0:
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

