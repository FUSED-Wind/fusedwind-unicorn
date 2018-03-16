from openmdao.api import Component

import numpy as np

# The following are creates an OpenMDAO 1 component from a FUSED object
#######################################################################

class FUSED_OpenMDAO1(Component):

    def __init__(self, model):

        super(FUSED_OpenMDAO1,self).__init__()

        self.model = model

        self._process_io(self.model.interface['input'], self.add_param)
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

    def solve_nonlinear(self, params, unknowns, resids):

        self.model.compute(params, unknowns)


