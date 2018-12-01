# Create OpenMDAO components, groups and problems from fused objects and inputs

import openmdao as op
import numpy as np
import copy

from fusedwind.fused_wind import split_worflow

# This is the base class for OpenMDAO wrapped objects
#####################################################

class FUSED_OpenMDAOBase(object):

    has_been_split = False

    staged_objects = set()
    staged_wraps = set()
    staged_models = {}

    all_objects = set()
    all_wraps = set()
    all_models = {}

    def __init__(self, model):
        super(FUSED_OpenMDAOBase, self).__init__()

        # Add this as an object
        FUSED_OpenMDAOBase.staged_objects.add(model)
        FUSED_OpenMDAOBase.staged_wraps.add(self)
        FUSED_OpenMDAOBase.has_been_split = False

        self.my_hash = model._hash_value
        self.model = model

    @staticmethod
    def setup_splits(group=None):

        #print('MIMC calling FUSED_OpenMDAOBase.setup_splits(group)')

        # first solve the splits
        if not FUSED_OpenMDAOBase.has_been_split:
            (FUSED_OpenMDAOBase.staged_models, input_output_map) = split_worflow(FUSED_OpenMDAOBase.staged_objects)
            FUSED_OpenMDAOBase.has_been_split = True

        # Generate the connections in the group
        if not group is None:
            id_to_obj_map = {}
            # generate a map from ids to the object
            for obj in FUSED_OpenMDAOBase.staged_objects:
                id_to_obj_map[obj._hash_value]=obj
            # process the input_output_map to create all the connections
            for source_sub_system, dest_sub_system_map in input_output_map.items():
                source_object_name = id_to_obj_map[source_sub_system].object_name
                for dest_sub_system, source_name_map in dest_sub_system_map.items():
                    dest_object_name = id_to_obj_map[dest_sub_system].object_name
                    for source_name, dest_name_list in source_name_map.items():
                        for dest_name in dest_name_list:
                            #print('MIMC Connection:',source_object_name+'.'+source_name, dest_object_name+'.'+dest_name)
                            group.connect(source_object_name+'.'+source_name, dest_object_name+'.'+dest_name)

        # second set our model based on the results
        for obj in FUSED_OpenMDAOBase.staged_wraps:
            obj.model = FUSED_OpenMDAOBase.staged_models[obj.my_hash]

        # Call the set-up method for the wraps
        if int(op.__version__[0]) <= 1:
            for wrap in FUSED_OpenMDAOBase.staged_wraps:
                wrap.setup()

        # Reset staging
        FUSED_OpenMDAOBase.all_objects|=FUSED_OpenMDAOBase.staged_objects
        FUSED_OpenMDAOBase.all_wraps|=FUSED_OpenMDAOBase.staged_wraps
        FUSED_OpenMDAOBase.all_models.update(FUSED_OpenMDAOBase.staged_models)
        FUSED_OpenMDAOBase.staged_objects = set()
        FUSED_OpenMDAOBase.staged_wraps = set()
        FUSED_OpenMDAOBase.staged_models = {}

        #print('MIMC leaving FUSED_OpenMDAOBase.setup_splits(group)')

# Return FUSED Component based on version of OpenMDAO 1.x or 2.x
################################################################
def FUSED_Component(*args, **kwargs):

    model = args[0]
    #print('MIMC wrapping model %s'%(model.object_name))
    if model.is_independent_variable():

        # Collect the independent variable data
        val = 1.0
        name = model.get_name()
        if model.has_data():
            val = model.get_output_value()[name]
        meta = model.get_meta()
        if meta is None:
            meta={}

        # create the independent variable
        if int(op.__version__[0]) > 1:
            from openmdao.api import IndepVarComp
            class FUSED_IndepVarComp(IndepVarComp, FUSED_OpenMDAOBase):

                def __init__(self, model=None, name=None, val=1.0, **kwargs):
                    IndepVarComp.__init__(self, name, val, **kwargs)
                    FUSED_OpenMDAOBase.__init__(self, model)

                def setup(self):
                    if not FUSED_OpenMDAOBase.has_been_split:
                        FUSED_OpenMDAOBase.setup_splits()
                    super(FUSED_IndepVarComp,self).setup()

            if not 'name' in meta:
                meta['name']=name
            if not 'val' in meta:
                meta['val']=val
            return FUSED_IndepVarComp(model, **meta)
        else:
            from openmdao.api import IndepVarComp
            class FUSED_IndepVarComp(IndepVarComp, FUSED_OpenMDAOBase):

                def __init__(self, model=None, name=None, val=1.0, **kwargs):
                    IndepVarComp.__init__(self, name, val, **kwargs)
                    FUSED_OpenMDAOBase.__init__(self, model)

                def setup(self):
                    if not FUSED_OpenMDAOBase.has_been_split:
                        FUSED_OpenMDAOBase.setup_splits()

            if not 'name' in meta:
                meta['name']=name
            if not 'val' in meta:
                meta['val']=val
            return FUSED_IndepVarComp(model, **meta)

    if int(op.__version__[0]) > 1:
        from openmdao.api import ExplicitComponent
        class FUSED_OpenMDAO(ExplicitComponent, FUSED_OpenMDAOBase):
        

            def __init__(self, model):
                ExplicitComponent.__init__(self)
                FUSED_OpenMDAOBase.__init__(self, model)

            def setup(self):

                if not FUSED_OpenMDAOBase.has_been_split:
                    FUSED_OpenMDAOBase.setup_splits()

                #print('MIMC from model %s adding the IO'%(self.model.object_name))
                #print('MIMC there is a pdb directive')
                #import pdb; pdb.set_trace()
                ifc = self.model.get_interface()
                process_io(self, ifc['input'], 'add_input')
                process_io(self, ifc['output'], 'add_output')
        
            def compute(self, inputs, outputs):
        
                self.model.compute(inputs, outputs)

        return FUSED_OpenMDAO(*args, **kwargs)
    else:
        from openmdao.api import Component
        class FUSED_OpenMDAO(Component, FUSED_OpenMDAOBase):
        
            def __init__(self, model):
                Component.__init__(self)
                FUSED_OpenMDAOBase.__init__(self, model)

            def setup(self):

                if not FUSED_OpenMDAOBase.has_been_split:
                    FUSED_OpenMDAOBase.setup_splits()

                #print('MIMC from model %s adding the IO'%(self.model.object_name))
                #print('MIMC there is a pdb directive just before we retrieve the interface')
                #import pdb; pdb.set_trace()
                ifc = self.model.get_interface()
                process_io(self, ifc['input'], 'add_input')
                process_io(self, ifc['output'], 'add_output')
        
            def solve_nonlinear(self, params, unknowns, resids):
        
                self.model.compute(params, unknowns)

        return FUSED_OpenMDAO(*args, **kwargs)

# Add inputs and outputs to a class
def process_io(component, interface, add_method):

    for k, v in interface.items():

        # print the names of things
        #if add_method == 'add_input':
        #    print('MIMC adding input %s'%(k))
        #else:
        #    print('MIMC adding output %s'%(k))
        
        # Apply the sizes of arrays
        if 'shape' in v.keys():
            if not isinstance(v['shape'],int):
                for i, sz in enumerate(v['shape']):
                    if type(sz) is not int:
                        my_name = sz['name']
                        if my_name not in kwargs.keys():
                            print('The interface requires that the size '+my_name+' is specified')
                            raise Exception
                        v['shape'][i]=kwargs[my_name]
            if not 'val' in v.keys():
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
def FUSED_add(group, component_name, component, promoters=None):
  
    #print('MIMC adding to group:', component_name)
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
