from openmdao.api import ExplicitComponent

import numpy as np

# The following are creates an OpenMDAO 2 component from a FUSED object
#######################################################################

class FUSED_OpenMDAO2(ExplicitComponent):

    def __init__(self, model):

        super(FUSED_OpenMDAO2,self).__init__()

        self.model = model

        self._process_io(self.model.interface['input'], self.add_input)
        self._process_io(self.model.interface['output'], self.add_output)

    def _process_io(self, interface, add_method):

        for k, v in interface.items():

            # Apply the sizes of arrays
            if 'shape' in v.keys():
                for i, sz in enumerate(v['shape']):
                    if type(sz) is not int:
                        my_name = sz['name']
                        if my_name not in kwargs.keys():
                            print('The interface requires that the size '+my_name+' is specified')
                            raise Exception
                        v['shape'][i]=kwargs[my_name]
                if 'val' in v.keys():
                    v['val']=np.zeros(v['shape'], dtype=float)
            else:
                v['val'] = float(v['val'])

            add_method(k, v['val'])

    def compute(self, inputs, outputs):

        self.model.compute(inputs, outputs)


