# Create OpenMDAO components, groups and problems from fused objects and inputs

import openmdao as op
import numpy as np

# Return FUSED Component based on version of OpenMDAO 1.x or 2.x
################################################################
def FUSED_Component(*args, **kwargs):

    if int(op.__version__[0]) > 1:
        from openmdao.api import ExplicitComponent
        class FUSED_OpenMDAO(ExplicitComponent):
        
            def __init__(self, model):
        
                super(FUSED_OpenMDAO,self).__init__()
        
                self.model = model
        
                process_io(self, self.model.interface['input'], 'add_input')
                process_io(self, self.model.interface['output'], 'add_output')
        
            def compute(self, inputs, outputs):
        
                self.model.compute(inputs, outputs)

        return FUSED_OpenMDAO(*args, **kwargs)
    else:
        from openmdao.api import Component
        class FUSED_OpenMDAO(Component):
        
            def __init__(self, model):
        
                super(FUSED_OpenMDAO,self).__init__()
        
                self.model = model
        
                process_io(self, self.model.interface['input'], 'add_input')
                process_io(self, self.model.interface['output'], 'add_output')
        
            def solve_nonlinear(self, params, unknowns, resids):
        
                self.model.compute(params, unknowns)

        return FUSED_OpenMDAO(*args, **kwargs)

# Add inputs and outputs to a class
def process_io(component, interface, add_method):

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

        if int(op.__version__[0]) > 1:
            if add_method == 'add_input':
                component.add_input(k, v['val'])
            elif add_method == 'add_output':
                component.add_output(k, v['val'])
        else:
            if add_method == 'add_input':
                component.add_param(k, v['val'])
            elif add_method == 'add_output':
                component.add_output(k, v['val'])

# Return FUSED Group based on version of OpenMDAO 1.x or 2.x
############################################################
def FUSED_Group(*args, **kwargs):

    if int(op.__version__[0]) > 1:
        from openmdao.api import Group
        return Group(*args, **kwargs)
    else:
        from openmdao.api import Group
        return Group(*args, **kwargs)

# Add component or subsystem to group based on version of OpenMDAO 1.x or 2.x
def FUSED_add(group, component_name, component, promoters=['']):
  
    if int(op.__version__[0]) > 1:
        return group.add_subsystem(component_name, component, promotes=promoters)
    else:
        return group.add(component_name, component, promotes=promoters)

# Add explicit connections between group components based on version of OpenMDAO 1.x or 2.x
def FUSED_connect(group, output_connection, input_connections):
  
    if int(op.__version__[0]) > 1:
        return group.connect(output_connection, input_connections)
    else:
        return group.connect(output_connection, input_connections)

# Add ability to print output for different openmdao versions
def FUSED_print(group):

    if int(op.__version__[0]) > 1:
        group.list_outputs()
    else:
        for io in group.unknowns:
            print(io + ' ' + str(group.unknowns[io]))

# Return FUSED Problem based on version of OpenMDAO 1.x or 2.x
# Redundancy kept for consistency with other FUSED functions
############################################################
def FUSED_Problem(*args, **kwargs):

    if int(op.__version__[0]) > 1:
        from openmdao.api import Problem
        return Problem(*args, **kwargs)
    else:
        from openmdao.api import Problem
        return Problem(*args, **kwargs)

# Add independent variable components to a problem
def FUSED_VarComp(*args, **kwargs):

    if int(op.__version__[0]) > 1:
        from openmdao.api import IndepVarComp
        return IndepVarComp(*args, **kwargs)
    else:
        from openmdao.api import IndepVarComp
        return IndepVarComp(*args, **kwargs)

# Add ability to print output for different openmdao versions
def FUSED_setup(problem):

    if int(op.__version__[0]) > 1:
        problem.setup()
    else:
        problem.setup()

# Run problem based on version of OpenMDAO 1.x or 2.x
def FUSED_run(problem):
  
    if int(op.__version__[0]) > 1:
        return problem.run_driver()
    else:
        return problem.run()
