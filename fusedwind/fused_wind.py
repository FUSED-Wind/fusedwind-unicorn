
#########################################################################
#
# Things that I am working on
#
#      1) Creating the capability to wrap a work-flow as a group so that it behaves like a component
#      2) Creating the capability to wrap a work-flow as a group so that is becomes a job in a case runner
#
# Basic tasks
#
#      1) The group has been created for grouping objects and having an overall interface
#      2) It has connection methods now for connecting both up-stream and down-stream
#      3) The group has methods now for triggering the upstream calculations
#      4) The system-base can now identify the up-stream and down stream
#      5) Need to get some methods for gathers ... 
#      6) Need to get some methods for triggering case runner
#

import numpy as np
import copy
import time

try:
    from mpi4py import MPI
    #print('MIMC debug stuff is here')
    #bcast_cnt = 0
except:
    print('It seems that we are not able to import MPI')
    MPI = None

# The following are helper functions to create a custom interface
#################################################################

def create_interface():

    return {'output': {}, 'input': {}}

def set_variable(inner_dict, variable):

    inner_dict[variable['name']]=copy.deepcopy(variable)

def set_input(fifc, variable):

    set_variable(fifc['input'], variable)

def set_output(fifc, variable):

    set_variable(fifc['output'], variable)

def extend_interface(base, extension):

    for k, v in extension['input'].items():
        set_input(base, v)

    for k, v in extension['output'].items():
        set_output(base, v)

    return base

def create_variable(name, val=None, desc='', shape=None):

    retval = {'name' : name, 'desc' : ''}
    if not val is None:
        if isinstance(val, np.ndarray):
            shape = val.shape
        else:
            shape = None
    if not val is None:
        retval['val']=val
    if not shape is None:
        retval['shape']=shape

    return retval

def print_interface(fifc, print_meta=False):

    out_ifc = fifc['output']
    in_ifc = fifc['input']
    print("Input:")
    for name, meta in in_ifc.items():
        if print_meta:
            print('\t%s: %s'%(name, str(meta)))
        else:
            print('\t%s'%(name))
    print("Output:")
    for name, meta in out_ifc.items():
        if print_meta:
            print('\t%s: %s'%(name, str(meta)))
        else:
            print('\t%s'%(name))

# This is a class that tracks where a value is located within an MPI environment
# ##############################################################################

# How do we differentiate between local access and distributed access?
#
#     All access is assumed to be local
#

# THIS SEEMS A LITTLE TOO CLUNKY class MPI_Value(object):
# THIS SEEMS A LITTLE TOO CLUNKY 
# THIS SEEMS A LITTLE TOO CLUNKY     def __init__(self, comm = None):
# THIS SEEMS A LITTLE TOO CLUNKY 
# THIS SEEMS A LITTLE TOO CLUNKY         self.value = None
# THIS SEEMS A LITTLE TOO CLUNKY         self.need_sync = False
# THIS SEEMS A LITTLE TOO CLUNKY         self.at_rank = -1
# THIS SEEMS A LITTLE TOO CLUNKY 
# THIS SEEMS A LITTLE TOO CLUNKY         if not MPI is None and self.comm==None:
# THIS SEEMS A LITTLE TOO CLUNKY             self.comm=MPI.COMM_WORLD
# THIS SEEMS A LITTLE TOO CLUNKY 
# THIS SEEMS A LITTLE TOO CLUNKY     def get_value(self):
# THIS SEEMS A LITTLE TOO CLUNKY 
# THIS SEEMS A LITTLE TOO CLUNKY         if self.need_sync or self.at_rank>=0:
# THIS SEEMS A LITTLE TOO CLUNKY             if MPI is None:
# THIS SEEMS A LITTLE TOO CLUNKY                 raise ValueError('It seems the value is not owned, but we are not running under MPI')
# THIS SEEMS A LITTLE TOO CLUNKY             self.comm.bcast(self.value, self.at_rank)
# THIS SEEMS A LITTLE TOO CLUNKY             self.owned = True
# THIS SEEMS A LITTLE TOO CLUNKY         return self.value
# THIS SEEMS A LITTLE TOO CLUNKY 
# THIS SEEMS A LITTLE TOO CLUNKY     def set_value(self, value_in, at_rank_in = -1):
# THIS SEEMS A LITTLE TOO CLUNKY 
# THIS SEEMS A LITTLE TOO CLUNKY         if at_rank_in<0 or self.comm.rank==at_rank_in
# THIS SEEMS A LITTLE TOO CLUNKY         self.value = value_in
# THIS SEEMS A LITTLE TOO CLUNKY         self.at_rank = at_rank_in

# This is a state version object
################################

# This tracks the state version of a work-flow
class StateVersion(object):

    def __init__(self):

        self.version = 0
        self.isSet = True

    # This indicates that the state is being modifed
    def modifying_state(self):

        if self.isSet:
            self.isSet=False
            self.version+=1

    # Retrieve the current state version
    def get_state_version(self):

        self.isSet=True
        return self.version

# The following is a helper function for parsing the arguments for a connect method
###################################################################################

def parse_connect_args(dest_object, source_object, var_name_dest=None, var_name_source=None, alias={}):

    '''
    These are the different ways of specifying a connection
    
    1) dst.connect(src)
          The assumption here is that all common variables in the interface are connected
    2) dst.connect(src, 'var1')
          This specifies that var1 in both objects should be connected
    3) dst.connect(src, ['var1','var2'])
          This specifies that var1 and var2 in both objects should be connected
    4) dst.connect(src, 'var1', 'var2')
          This specifies that 'var1' in the destination should be connected to 'var2' in the source
    5) dst.connect(src, ['var1', 'var3'], ['var2', 'var4'])
          This specifies that 'var1' and 'var3' in the destination should be connected to 'var2' and 'var4' in the source respectively
    6) dst.connect(src, {'var1':'var2', 'var3':'var4'})
          This specifies that 'var1' and 'var3' in the destination should be connected to 'var2' and 'var4' in the source respectively
    7) dst.connect(src, ..., alias={'var1':'var2', 'var3':'var4'})
          This specifies that 'var1' and 'var3' in the destination is equivalent to 'var2' and 'var4' in the source respectively.
          This does not mean they are connected though, that depends on the other arguments
          This is used in cases where the naming scheme differs for a small group of variables, and the connections are not explicit
    '''

    # First task is to create maps between the variable names
    #########################################################

    # Retrieve the interfaces
    dst_ifc = dest_object.get_interface()
    src_ifc = source_object.get_interface()
    dst_var = dst_ifc['input'].keys()
    src_var = src_ifc['output'].keys()

    # These are the variable name maps
    src_dst_map = {}
    dst_src_map = {}

    # construct reverse alias if needed
    r_alias={}
    if var_name_dest is None and not var_name_source is None:
        for k,v in alias.items():
            r_alias[v]=k

    # Is my dest name None? Then we construct based on source type
    if var_name_dest is None:
        # In this case we must find all common names
        if var_name_source is None:
            for dst_name in dst_var:
                src_name = dst_name
                if dst_name in alias:
                    src_name = alias[dst_name]
                if src_name in src_var:
                    src_dst_map[src_name]=[dst_name]
                    dst_src_map[dst_name]=src_name
        # In this case only the source name is specified
        elif isinstance(var_name_source, str):
            src_name = var_name_source
            # Use Alias and Verify
            if not src_name in src_var:
                raise Exception('That source variable name does not exist')
            dst_name = src_name
            # Apply alias
            if src_name in r_alias:
                dst_name = r_alias[src_name]
            if dst_name not in dst_var:
                raise Exception('That destination variable name does not exist')
            if dst_name in dst_src_map:
                raise Exception('That destination variable name specified twice')
            # Add to the map
            if src_name in src_dst_map:
                src_dst_map[src_name].append(dst_name)
            else:
                src_dst_map[src_name]=[dst_name]
            dst_src_map[dst_name]=src_name
        # In this case a list of source names is specified
        elif hasattr(var_name_source, '__iter__'):
            for src_name in var_name_source:
                # Use Alias and Verify
                if not src_name in src_var:
                    raise Exception('That source variable name does not exist')
                dst_name = src_name
                if src_name in r_alias:
                    dst_name = r_alias[src_name]
                if not dst_name in dst_var:
                    raise Exception('That destination variable name does not exist')
                if dst_name in dst_src_map:
                    raise Exception('That destination variable name specified twice')
                # Add to the map
                if src_name in src_dst_map:
                    src_dst_map[src_name].append(dst_name)
                else:
                    src_dst_map[src_name]=[dst_name]
                dst_src_map[dst_name]=src_name
        # In this case we have a type error
        else:
            raise Exception('Cannot use the type passed as the source variable list')
    # A single variable name is specified
    elif isinstance(var_name_dest, str):
        # In this case only the destination name matters
        if var_name_source is None:
            dst_name = var_name_dest
            if not dst_name in dst_var:
                raise Exception('That destination variable name does not exist')
            src_name = dst_name
            if dst_name in alias:
                src_name = alias[dst_name]
            if not src_name in src_var:
                raise Exception('That source variable name does not exist')
            if dst_name in dst_src_map:
                raise Exception('That destination variable name specified twice')
            # Add to the map
            if src_name in src_dst_map:
                src_dst_map[src_name].append(dst_name)
            else:
                src_dst_map[src_name]=[dst_name]
            dst_src_map[dst_name]=src_name
        # In this case only the source name is also specified
        elif isinstance(var_name_source, str):
            dst_name = var_name_dest
            src_name = var_name_source
            # Verify things are OK
            if not dst_name in dst_var:
                raise Exception('That destination variable name does not exist')
            if not src_name in src_var:
                raise Exception('That source variable name does not exist')
            if dst_name in dst_src_map:
                raise Exception('That destination variable name specified twice')
            # Add to the map
            if src_name in src_dst_map:
                src_dst_map[src_name].append(dst_name)
            else:
                src_dst_map[src_name]=[dst_name]
            dst_src_map[dst_name]=src_name
        # In this case a list of source names is specified
        elif hasattr(var_name_source, '__iter__'):
            dst_name = var_name_dest
            if len(var_name_source)!=1:
                raise Exception('Different sizes in the source and destination variables')
            src_name = var_name_source[0]
            # Verify things are OK
            if not dst_name in dst_var:
                raise Exception('That destination variable name does not exist')
            if not src_name in src_var:
                raise Exception('That source variable name does not exist')
            if dst_name in dst_src_map:
                raise Exception('That destination variable name specified twice')
            # Add to the map
            if src_name in src_dst_map:
                src_dst_map[src_name].append(dst_name)
            else:
                src_dst_map[src_name]=[dst_name]
            dst_src_map[dst_name]=src_name
        # In this case we have a type error
        else:
            raise Exception('Cannot use the type passed as the source variable list')
    # In this case the dst->src variable map is given directly
    elif isinstance(var_name_dest,dict):
        # In this case only the destination name matters
        if var_name_source is None:
            for k, v in var_name_dest.items():
                dst_name = k
                src_name = v
                # Verify things are OK
                if not dst_name in dst_var:
                    raise Exception('That destination variable name does not exist')
                if not src_name in src_var:
                    raise Exception('That source variable name does not exist')
                if dst_name in dst_src_map:
                    raise Exception('That destination variable name specified twice')
                # Add to the map
                if src_name in src_dst_map:
                    src_dst_map[src_name].append(dst_name)
                else:
                    src_dst_map[src_name]=[dst_name]
                dst_src_map[dst_name]=src_name
        # In this case we have a type error
        else:
            raise Exception('Cannot use the type passed as the source variable list')
    elif hasattr(var_name_dest, '__iter__'):
        # In this case only the destination name matters
        if var_name_source is None:
            for dst_name in var_name_dest:
                if not dst_name in dst_var:
                    raise Exception('That destination variable name does not exist')
                src_name = dst_name
                if dst_name in alias:
                    src_name = alias[dst_name]
                if not src_name in src_var:
                    raise Exception('That source variable name does not exist')
                if dst_name in dst_src_map:
                    raise Exception('That destination variable name specified twice')
                # Add to the map
                if src_name in src_dst_map:
                    src_dst_map[src_name].append(dst_name)
                else:
                    src_dst_map[src_name]=[dst_name]
                dst_src_map[dst_name]=src_name
        # In this case only the source name is also specified
        elif isinstance(var_name_source, str):
            if len(var_name_dest)!=1:
                raise Exception('Different sizes in the source and destination variables')
            dst_name = var_name_dest[0]
            src_name = var_name_source
            # Verify things are OK
            if not dst_name in dst_var:
                raise Exception('That destination variable name does not exist')
            if not src_name in src_var:
                raise Exception('That source variable name does not exist')
            if dst_name in dst_src_map:
                raise Exception('That destination variable name specified twice')
            # Add to the map
            if src_name in src_dst_map:
                src_dst_map[src_name].append(dst_name)
            else:
                src_dst_map[src_name]=[dst_name]
            dst_src_map[dst_name]=src_name
        # In this case a list of source names is specified
        elif hasattr(var_name_source, '__iter__'):
            if len(var_name_source)!=len(var_name_dest):
                raise Exception('Different sizes in the source and destination variables')
            for i in range(0,len(var_name_dest)):
                dst_name = var_name_dest[i]
                src_name = var_name_source[i]
                # Verify things are OK
                if not dst_name in dst_var:
                    raise Exception('That destination variable name does not exist')
                if not src_name in src_var:
                    raise Exception('That source variable name does not exist')
                if dst_name in dst_src_map:
                    raise Exception('That destination variable name specified twice')
                # Add to the map
                if src_name in src_dst_map:
                    src_dst_map[src_name].append(dst_name)
                else:
                    src_dst_map[src_name]=[dst_name]
                dst_src_map[dst_name]=src_name
        # In this case we have a type error
        else:
            raise Exception('Cannot use the type passed as the source variable list')
    else:
        raise Exception('Cannot use the type passed as the destination variable list')

    return src_dst_map, dst_src_map

# Unique fused object
#########################################################################

class FUSED_Unique(object):

    _object_count = 0

    def __init__(self):
        object.__init__(self)

        self._hash_value = FUSED_Unique._object_count
        FUSED_Unique._object_count += 1

    # The following are used for generating hash keys
    #################################################

    def __cmp__(self, other):
        if self._hash_value == other._hash_value:
            return 0
        if self._hash_value < other._hash_value:
            return -1
        return 1

    def __eq__(self, other):
        return bool(self.__cmp__(other)==0)

    def __ne__(self, other):
        return bool(self.__cmp__(other)!=0)

    def __lt__(self, other):
        return bool(self.__cmp__(other)==-1)

    def __le__(self, other):
        return bool(self.__cmp__(other)!=1)

    def __gt__(self, other):
        return bool(self.__cmp__(other)==1)

    def __ge__(self, other):
        return bool(self.__cmp__(other)!=-1)

    def __hash__(self):
        return self._hash_value

# The following is the FUSED Object
#########################################################################

class FUSED_Object(FUSED_Unique):

    '''
    This is the base class for any calculation in a Fused-Wind work-flow

    The user should inherit this and must implement the following methods:

        def compute(self, input_values, output_values):
        def _build_interface(self):
    '''

    default_state_version = StateVersion()
    print_level = 0
    all_objects = []

    def __init__(self, object_name_in='unnamed_object', state_version_in=None, comm = None):
        FUSED_Unique.__init__(self)

        # This is the name of the object. Useful for printing helpful messages
        self.object_name = object_name_in

        # This is the interface
        self.ifc_built = False
        self.interface = create_interface()

        # Variables for a default input vector
        self.is_default_input_built = False
        self.default_input = {}

        # This is the connection information
        ####################################

        # Registers where the input of this object is from
        # indexed by
        # conn_dict[my_input_name]=(source_object, source_output_name)
        self.conn_dict={}
        # Stores input connections
        # indexed by
        # self.connections[source_object][source_output_name]=['my_input_name_1', 'my_input_name_2', ...]
        self.connections={}
        # Stores output connections
        # indexed by
        # self.output_connections[my_output_name][dest_object]=['dest_input_name_1', 'dest_input_name_2', ...]
        self.output_connections={}

        # This is the state version
        if state_version_in is None:
            self.state_version = FUSED_Object.default_state_version
        else:
            self.state_version = state_version_in
        self.my_state_version = 0
        self.output_values = {}
        self.output_at_rank = {}
        self.my_case_runner = None

        # This is the MPI comm
        self.comm = comm
        if not MPI is None and self.comm==None:
            self.comm=MPI.COMM_WORLD

        # Ensure these objects can be indexed
        FUSED_Object.all_objects.append(self)

    # Print the interface
    def print_interface(self, print_meta=False):
        ifc = self.get_interface()
        print_interface(ifc, print_meta)

    # Lets for the fun of it, lets print the connections
    def print_object_calculations(self, object_set=set()):

        # if this is an indep var then we just print and start the chain reactions
        if self.is_independent_variable():
            if not self in object_set:
                print('Object %s of type %s provides its output'%(self.object_name, str(type(self))))
                object_set.add(self)
            return

        # lets print the execution of out sub-objects
        for src_obj, output_dict in self.connections.items():
            src_obj.print_object_calculations(object_set=object_set)

        # Now lets print the execution of my object
        if not self in object_set:
            print('Object %s of type %s provides its output'%(self.object_name, str(type(self))))
            object_set.add(self)

    # This is for solving some properties of a work-flow
    ####################################################

    @staticmethod
    def get_all_objects():

        return FUSED_Object.all_objects

    # Identifies whether it is an independent variable or Group
    ###########################################################

    def is_independent_variable(self):
        return False

    def is_group(self):
        return False

    # These are state version methods
    #################################
    
    def set_state_version(self, state_version_in):

        # This is the state version
        if state_version_in is None:
            self.state_version = default_state_version
        else:
            self.state_version = state_version_in
        self.my_state_version = 0

    def _update_needed(self):

        return self.my_state_version!=self.state_version.get_state_version()

    def _updating_data(self):

        self.my_state_version=self.state_version.get_state_version()

    # These are interface methods
    #############################

    # This method is a place-holder. It is meant to tell the class that it should build it's interface
    def _build_interface(self):
        return

    def implement_fifc(self, fifc, **kwargs):

        for k, v in fifc['input'].items():

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
                    v['val']=np.zeros(v['shape'])

            # Add our parameter
            self.add_input(**v)

        for k, v in fifc['output'].items():

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
                    v['val']=np.zeros(v['shape'])

            # add out output
            self.add_output(**v)

    def add_input(self, name, **kwargs):

        kwargs['name']=name
        set_input(self.interface, kwargs)

    def add_output(self, name, **kwargs):

        kwargs['name']=name
        set_output(self.interface, kwargs)

    def remove_input(self, name, meta=None):

        retval=self.ifc['input'][name]
        del self.ifc['input'][name]
        return retval

    def remove_output(self, name, meta=None):

        retval=self.ifc['output'][name]
        del self.ifc['output'][name]
        return retval

    def get_interface(self):

        if not self.ifc_built:
            self._build_interface()
            self.ifc_built = True
        return self.interface

    # This is the all important connect method. It tells where to get the input data from
    #####################################################################################

    def connect_input_from(self, source_object, var_name_dest=None, var_name_source=None, alias={}):
        self.connect(source_object, var_name_source, var_name_dest, alias)

    def connect_output_to(self, dest_object, var_name_source=None, var_name_dest=None, alias={}):

        if not isinstance(dest_object, FUSED_Object):
            for obj in dest_object:
                obj.connect(self, var_name_dest=var_name_dest, var_name_source=var_name_source, alias=alias)
            return

        dest_object.connect(self, var_name_dest=var_name_dest, var_name_source=var_name_source, alias=alias)

    # This will specify connections
    def connect(self, source_object, var_name_dest=None, var_name_source=None, alias={}):

        '''
        These are the different ways of specifying a connection
        
        1) dst.connect(src)
              The assumption here is that all common variables in the interface are connected
        2) dst.connect(src, 'var1')
              This specifies that var1 in both objects should be connected
        3) dst.connect(src, ['var1','var2'])
              This specifies that var1 and var2 in both objects should be connected
        4) dst.connect(src, 'var1', 'var2')
              This specifies that 'var1' in the destination should be connected to 'var2' in the source
        5) dst.connect(src, ['var1', 'var3'], ['var2', 'var4'])
              This specifies that 'var1' and 'var3' in the destination should be connected to 'var2' and 'var4' in the source respectively
        6) dst.connect(src, {'var1':'var2', 'var3':'var4'})
              This specifies that 'var1' and 'var3' in the destination should be connected to 'var2' and 'var4' in the source respectively
        7) dst.connect(src, ..., alias={'var1':'var2', 'var3':'var4'})
              This specifies that 'var1' and 'var3' in the destination is equivalent to 'var2' and 'var4' in the source respectively.
              This does not mean they are connected though, that depends on the other arguments
              This is used in cases where the naming scheme differs for a small group of variables, and the connections are not explicit
        '''

        # If there are multiple source objects
        #########################################################

        if not isinstance(source_object, FUSED_Object) and not isinstance(source_object, FUSED_Group):
            for obj in source_object:
                self.connect(obj, var_name_dest, var_name_source, alias)
            return

        # First task is to create maps between the variable names
        #########################################################

        # This will parse the arguments to get the variables that are suppose to be connected
        src_dst_map, dst_src_map = parse_connect_args(self, source_object, var_name_dest, var_name_source, alias)

        # If there are no common variables then just return
        if len(src_dst_map)==0 or len(dst_src_map)==0:
            if FUSED_Object.print_level>=2:
                print('There appears to be no valid connection between '+self.object_name+' and '+source_object.object_name)
            return

        # Now that we have parsed the arguments... add the connections that we found
        self._add_connection(source_object, src_dst_map, dst_src_map)

    # Add a connection dirctly to the data structures
    def _add_connection(self, source_object, src_dst_map, dst_src_map):

        '''
        In special cases, the variable mapping is known and can be added directly.
        This method is used to do this task
        '''

        # Lets print the data
        if FUSED_Object.print_level>=2:
            for src_name, dst_list in src_dst_map.items():
                for dst_name in dst_list:
                    print('%s %s.%s <- %s %s.%s'%(type(self), self.object_name, dst_name, type(source_object), source_object.object_name, src_name))

        # If connecting to a group, then extract the output objects and connect directly to them
        if source_object.is_group():

            # Collect all the objects and their maps
            source_object_dict = {}
            for source_name, dest_list in src_dst_map.items():
                obj, local_name = source_object.get_object_and_local_from_global_output(source_name)
                if obj in source_object_dict:
                    local_src_dst_map = source_object_dict[obj][0]
                    local_dst_src_map = source_object_dict[obj][1]
                    local_src_dst_map[source_name]=dest_list
                    for dest_name in dest_list:
                        local_dst_src_map[dest_name]=source_name
                else:
                    local_dst_src_map = {}
                    for dest_name in dest_list:
                        local_dst_src_map[dest_name]=source_name
                    source_object_dict[obj]=({local_name:dest_list},local_dst_src_map)

            # now loop through all the extracted objects and connect to them
            for source_object_local, map_pair in source_object_dict.items():
                local_src_dst_map = map_pair[0]
                local_dst_src_map = map_pair[1]
                self._add_connection(source_object_local, local_src_dst_map, local_dst_src_map)

            return

        # Now we build the connection data structure
        ############################################

        if source_object in self.connections:
            for new_source_name, new_dst_list in src_dst_map.items():
                if new_source_name in self.connections[source_object]:
                    # Merge in the new connections
                    for new_dst_name in new_dst_list:
                        if not new_dst_name in self.connections[source_object][new_source_name]:
                            self.connections[source_object][new_source_name].append(new_dst_name)
                            # add the upstream connection
                            if not new_source_name in source_object.output_connections:
                                source_object.output_connections[new_source_name]={}
                            if not self in source_object.output_connections[new_source_name]:
                                source_object.output_connections[new_source_name][self]=[]
                            source_object.output_connections[new_source_name][self].append(new_dst_name)
                        elif FUSED_Object.print_level>=2:
                            print('The connection between '+self.object_name+' and '+source_object.object_name+' connected '+new_dst_name+' and '+new_source_name+' again')
                else:
                    self.connections[source_object][new_source_name]=new_dst_list
                    # add the upstream connection
                    if not new_source_name in source_object.output_connections:
                        source_object.output_connections[new_source_name]={}
                    source_object.output_connections[new_source_name][self]=copy.copy(new_dst_list)
        else:
            self.connections[source_object]=src_dst_map
            for source_name, dest_list in src_dst_map.items():
                if not source_name in source_object.output_connections:
                    source_object.output_connections[source_name]={}
                source_object.output_connections[source_name][self]=copy.copy(dest_list)

        # this will then update the connection dictionaries
        ###################################################

        # Basically, the task is to ensure that each destination variable has a unique source
        for dst_name in dst_src_map.keys():
            # remove any variables from old connections
            if dst_name in self.conn_dict:
                if FUSED_Object.print_level>=2:
                    print('Connecting variable '+dst_name+' replaces the old connection between '+self.object_name+' and '+self.conn_dict[dst_name][0].object_name)
                tmp_source_object = self.conn_dict[dst_name][0]
                tmp_src_name = self.conn_dict[dst_name][1]
                if not source_object == tmp_source_object or not tmp_src_name == dst_src_map[dst_name]:
                    # Remove the over-ride from the connections
                    if dst_name in self.connections[tmp_source_object][tmp_src_name]:
                        self.connections[tmp_source_object][tmp_src_name].remove(dst_name)
                        # 
                        # Remove that source name, if it is no longer used
                        if len(self.connections[tmp_source_object][tmp_src_name])==0:
                            del self.connections[tmp_source_object][tmp_src_name]
                            # Remove the source object, if it is no longer used
                            if len(self.connections[tmp_source_object])==0:
                                del self.connections[tmp_source_object]
                    else:
                        raise Exception('Connection data structure corrupted')
                # Remove the upstream connection
                if dst_name in tmp_source_object.output_connections[tmp_src_name][self]:
                    tmp_source_object.output_connections[tmp_src_name][self].remove(dst_name)
                    if len(tmp_source_object.output_connections[tmp_src_name][self])==0:
                        del tmp_source_object.output_connections[tmp_src_name][self]
                        if len(tmp_source_object.output_connections[tmp_src_name])==0:
                            del tmp_source_object.output_connections[tmp_src_name]
                else:
                    raise Exception('Connection data structure corrupted')
            # Add the new variable to the conn_dict
            self.conn_dict[dst_name]=(source_object, dst_src_map[dst_name])

        # Make sure the state version is updated
        ########################################

        self.state_version.modifying_state()

    # This will verify the connections on both inputs and outputs
    def _verify_connections(self):

        # Test all data from connections
        for source_object, src_dst_map in self.connections.items():
            for source_name, dest_list in src_dst_map.items():
                for dest_name in dest_list:
                    # Verify that the data in conn_dict is correct
                    if not dest_name in self.conn_dict:
                        raise KeyError('The connections are not correct')
                    source_pair = self.conn_dict[dst_name]
                    if not source_object == source_pair[0]:
                        raise KeyError('The connections are not correct')
                    if not source_name == source_pair[1]:
                        raise KeyError('The connections are not correct')
                    # Verify that the data in output_connections is correct
                    if not source_name in source_object.output_connections:
                        raise KeyError('The connections are not correct')
                    if not self in source_object.output_connections[source_name]:
                        raise KeyError('The connections are not correct')
                    if not dest_name in source_object.output_connections[source_name][self]:
                        raise KeyError('The connections are not correct')

        # Test all data from conn_dict
        for dest_name, source_pair in self.conn_dict.items():
            source_object = source_pair[0]
            source_name = source_pair[1]
            # Verify that the data in output_connections is correct
            if not source_name in source_object.output_connections:
                raise KeyError('The connections are not correct')
            if not self in source_object.output_connections[source_name]:
                raise KeyError('The connections are not correct')
            if not dest_name in source_object.output_connections[source_name][self]:
                raise KeyError('The connections are not correct')
            # Verify all the data from connections
            if not source_object in self.connections:
                raise KeyError('The connections are not correct')
            src_dst_map = self.connections[source_object]
            if not source_name in src_dst_map:
                raise KeyError('The connections are not correct')
            dest_list = src_dst_map[source_name]
            if not dest_name in dest_list:
                raise KeyError('The connections are not correct')

        # Test all data from output_connections
        for source_name, dest_dict in self.output_connections.items():
            for dest_object, dest_list in dest_dict.items():
                for dest_name in dest_list:
                    # Verify that the data in conn_dict is correct
                    if not dest_name in dest_object.conn_dict:
                        raise KeyError('The connections are not correct')
                    source_pair = dest_object.conn_dict[dst_name]
                    if not self == source_pair[0]:
                        raise KeyError('The connections are not correct')
                    if not source_name == source_pair[1]:
                        raise KeyError('The connections are not correct')
                    # Verify all the data from connections
                    if not self in dest_object.connections:
                        raise KeyError('The connections are not correct')
                    src_dst_map = dest_object.connections[self]
                    if not source_name in src_dst_map:
                        raise KeyError('The connections are not correct')
                    if not dest_name in src_dst_map[source_name]:
                        raise KeyError('The connections are not correct')

    @staticmethod
    def _verify_all_connections():
        for obj in FUSED_Object.all_objects:
            obj._verify_connections()

    # This will list the connections associated with an object
    def get_connection_with_object(self, obj):

        if obj in self.connections:
            return self.connections[obj]
        else:
            return {}

    # This will generate a set of independent variable objects to replace a connection from a certain object
    def split_connection(self, from_object):

        #print('=================== MIMC in split_connection ===================')
        #print('MIMC from_object:', from_object)
        #print('MIMC self.connections:', self.connections)

        # First determine the hash
        hash_value = 0
        if isinstance(from_object, FUSED_Object):
            hash_value = from_object._hash_value
        else:
            hash_value = from_object

        #print('MIMC hash_value:', hash_value)

        # find the object and the destination variables
        indep_config_list = []
        for obj, src_dst_map in self.connections.items():
            if obj._hash_value == hash_value:
                for list_obj in src_dst_map.values():
                    indep_config_list.extend(list_obj)
                break

        #print('MIMC indep_config_list:', indep_config_list)

        # Loop over destination variables and generate independent variables
        indep_obj_set = set()
        input_ifc = self.get_interface()['input']
        for var in indep_config_list:
            if not var in input_ifc:
                raise Exception('The variable is not in the input interface')
            new_object_name = self.object_name+'_'+var+'_sink'
            indep_obj_set.add(Independent_Variable(var_name_in=var, var_meta_in=input_ifc[var], object_name_in=new_object_name))

        #print('MIMC indep_obj_set:', indep_obj_set)

        # Loop over the input objects and make the new connections
        for obj in indep_obj_set:
            self.connect(obj)

        # return the list of input objects
        return indep_obj_set

    def get_source_list(self):

        return self.connections.keys()

    def get_dependency_list(self):

        retval = [self]
        for dep in self.connections:
            tmp_list = dep.get_dependency_list()
            for tmp_dep in tmp_list:
                if not tmp_dep in retval:
                    retval.append(tmp_dep)
        return retval

    # These are the methods for calculating 
    ###################################################

    # This will build the default input vector
    def _build_default_input_vector(self):
        ifc = self.get_interface()
        for name, meta in ifc['input'].items():
            if 'val' in meta:
                self.default_input[name]=meta['val']
            elif 'shape' in meta:
                self.default_input[name]=np.zeros(meta['shape'])
        self.is_default_input_built = True

    # This is for collecting the input data from connections
    def _build_input_vector(self):

        # Collect the default input values
        if not self.is_default_input_built:
            self._build_default_input_vector()

        # Loop through all connections and collect the data
        self.input_values = copy.copy(self.default_input)
        for obj, src_dst_map in self.connections.items():
            #output = obj.get_output_value()
            for src_name, dst_list in src_dst_map.items():
                for dst_name in dst_list:
                    self.input_values[dst_name]=obj[src_name]
                    #self.input_values[dst_name]=output[src_name]

        # return the results
        return self.input_values

    # This is the calculation method that is called
    def compute(self, input_values, output_values):

        raise Exception('The compute method has not been implemented')

    # This instructs this class to collect the input data
    def collect_input_data(self):
        self._build_input_vector()

    # This will set the case runner for this object
    def set_case_runner(self, case_runner_in = None):

        if not self.my_case_runner is None:
            print('It appears that you are setting a case runner to this object twice. Note, that nested case-runners are not supported at this time.')
        self.my_case_runner = case_runner_in

    # This instructs this class to update it's data through calculation
    def update_output_data(self):
        if not self._update_needed():
            return
        if self.my_case_runner is None or self.my_case_runner.i_am_executing:
            self._build_input_vector()
            self.compute(self.input_values, self.output_values)
            self._updating_data()
            # labeling the object as not distributed
            self.output_at_rank = {}
        else:
            self.my_case_runner.execute()

    # This will retrieve a specific variable
    def __getitem__(self, key):
        
        # First verify this is a valid key
        ifc = self.get_interface()
        if not key in ifc['output'].keys():
            raise KeyError('That data is not in the interface')
        # Verify that the data is calculated based on the latest state
        self.update_output_data()
        # Verify that the data is synchronized properly
        if key in self.output_at_rank and self.output_at_rank[key]>=0:
            self.sync_output(key)
        # Return the result
        return self.output_values[key]

    # This will label all variables as remotely calculated
    def set_as_remotely_calculated(self, at_rank):

        ifc = self.get_interface()
        for out_name in ifc['output'].keys():
            self.output_at_rank[out_name] = at_rank
        self._updating_data()

    # This will synchronize the variables across MPI processes
    def sync_output(self, var_name = None):
        # var_name indicates the variables that need to be synchronized
        # None indicates all variables
        # '__downstream__' indicates all downstream involved in connections

        #global bcast_cnt

        # check if we are running in MPI
        if self.comm is None or self.comm.size <= 1:
            return
        if len(self.output_at_rank) == 0:
            return

        my_rank = self.comm.rank

        # If we tranfer all
        if var_name is None:
            cont = True
            at_rank = -1
            var_name = []
            for k in self.get_interface()['output'].keys():
                if not k in self.output_at_rank:
                    cont = False
                else:
                    at_rank = self.output_at_rank[k]
                    if at_rank<0:
                        cont = False
                    else:
                        var_name.append(k)
            if cont:
                # broadcast everything
                if my_rank == at_rank:
                    #print('MIMC MPI broadcast %d at rank: %d, obj name: %s, obj number: %d, dictionary in whole -> SENDING'%(bcast_cnt, my_rank, self.object_name, self._hash_value))
                    self.comm.bcast(self.output_values, at_rank)
                else:
                    self.output_values = self.comm.bcast(None, at_rank)
                    #print('MIMC MPI broadcast %d at rank: %d, obj name: %s, obj number: %d, dictionary in whole <- RECIEVING'%(bcast_cnt, my_rank, self.object_name, self._hash_value))
                #bcast_cnt+=1
                # MIMC #############
                # print('MIMC a barrier is set here after the transfer')
                # self.comm.Barrier()
                # time.sleep(0.2)
                ####################
                # reset the output at rank data structure
                self.output_at_rank = {}
                # return
                return

        # if we transfer a single variable
        if isinstance(var_name, str):
            var_name = [var_name]

        # if we transfer downstream
        if '__downstream__' in var_name:
            for output_name in self.output_connections.keys():
                if output_name in self.output_at_rank and self.output_at_rank[output_name]>=0 and not output_name in var_name:
                    var_name.append(output_name)
            while '__downstream__' in var_name:
                var_name.remove('__downstream__')

        # process a list of variables
        for name in sorted(var_name):
            # Only if the list contains a valid variable
            if name in self.output_at_rank and self.output_at_rank[name]>=0:
                at_rank = self.output_at_rank[name]
                if not name in self.output_values:
                    self.output_values[name] = None
                if my_rank == at_rank:
                    #if isinstance(self.output_values[name], np.ndarray):
                    #    print('MIMC MPI broadcast %d at rank: %d, obj name: %s, obj number: %d, var name: %s, value: %s -> SENDING'%(bcast_cnt, my_rank, self.object_name, self._hash_value, name, str(self.output_values[name])))
                    #else:
                    #    print('MIMC MPI broadcast %d at rank: %d, obj name: %s, obj number: %d, var name: %s, value: %d -> SENDING'%(bcast_cnt, my_rank, self.object_name, self._hash_value, name, self.output_values[name]))
                    self.comm.bcast(self.output_values[name], at_rank)
                else:
                    self.output_values[name] = self.comm.bcast(None, at_rank)
                    #if isinstance(self.output_values[name], np.ndarray):
                    #    print('MIMC MPI broadcast %d at rank: %d, obj name: %s, obj number: %d, var name: %s, value: %s <- RECIEVING'%(bcast_cnt, my_rank, self.object_name, self._hash_value, name, str(self.output_values[name])))
                    #else:
                    #    print('MIMC MPI broadcast %d at rank: %d, obj name: %s, obj number: %d, var name: %s, value: %d <- RECIEVING'%(bcast_cnt, my_rank, self.object_name, self._hash_value, name, self.output_values[name]))
                #bcast_cnt+=1
                # MIMC #############
                # print('MIMC a barrier is set here after the transfer')
                # self.comm.Barrier()
                # time.sleep(0.2)
                ####################
                del self.output_at_rank[name]
            else:
                raise KeyError('Tried to sync a variable that is not distributed')

    # The following is depricated
    ##############################

    # This will retrieve the output dictionary
    def get_output_value(self):

        ans = self._update_needed()
        if ans:
            self.update_output_data()
        self.sync_output()
        return self.output_values

    # This method is used by case generators to set the output values from other objects
    def _set_output_values(self, output_values):

        print('WARNING: This method may be depricated')
        if not isinstance(output_values, dict):
            raise ValueError('The output values must be a dictionary')
        self.output_values=output_values
        self._updating_data()

    # This method is used by case generators to set the output values from other objects
    def _set_output_value(self, output_key, output_value):

        print('WARNING: This method may be depricated')
        self.output_values[output_key]=output_value
        self._updating_data()

    # Retrieve the input value
    def get_input_value(self, var_name=None):

        if isinstance(var_name,str):
            var_name = [var_name]
        self._build_input_vector()
        if var_name is None:
            return self.input_values
        retval = {}
        for name in var_name:
            if not name in self.input_values:
                raise KeyError('A requested input is not in the input vector')
            retval[name] = self.input_values[name]

    # This will set the default input value
    def set_default_input_value(self, name, value):

        # Collect the default input values
        if not self.is_default_input_built:
            self._build_default_input_vector()

        # Verify it in the interface and then set the value in the input interface
        ifc = self.get_interface()
        if name not in ifc['input']:
            raise KeyError('That variable does not exist in the input interface')
        ifc['input'][name]['val'] = value

        # set the default input
        self.default_input[name] = value

        # update the state version if this default will actually be used
        if not name in self.conn_dict.keys():
            self.state_version.modifying_state()

    # This will set the default input value
    def get_default_input_value(self, name):

        # Collect the default input values
        if not self.is_default_input_built:
            self._build_default_input_vector()

        # Verify it in the interface
        if name not in self.default_input:
            raise KeyError('That variable does not exist in the input interface')

        # set the default input
        return self.default_input[name]

# independent variable class is used to represent a source object for a calculation chain
class Independent_Variable(FUSED_Object):

    ''' This represents a source to a calculation '''

    def __init__(self, data_in=None, var_name_in='unnamed_variable', var_meta_in=None, object_name_in='unnamed_object', state_version_in=None):
        super(Independent_Variable, self).__init__(object_name_in, state_version_in)

        self.name = var_name_in
        self.data = None
        self.retval = {self.name:self.data}
        if not data_in is None:
            self.set_data(data_in)
        self.meta = var_meta_in
        if self.meta is None:
            self.add_output(var_name_in, val=self.data)
        else:
            if 'name' in self.meta:
                if self.name!=self.meta['name']:
                    raise Exception('the variable name and the name in the meta data is not consistent')
            else:
                self.meta['name']=self.name
            if not data_in is None:
                self.meta['val']=self.data
            self.add_output(**self.meta)

    def is_independent_variable(self):
        return True

    def has_data(self):
        if self.data is None:
            return False
        else:
            return True
    
    def get_meta(self):
        return self.meta

    def get_name(self):
        return self.name

    def set_data(self, data_in):

        self.state_version.modifying_state()
        self.data = data_in
        self.retval = {self.name:self.data}

    def get_output_value(self):

        return self.retval

    def __getitem__(self, key):

        if not key == self.name:
            raise KeyError('That variable does not exist')
        return self.data

# This will take a list or dictionary and seperate out the objects and names
############################################################################

# this will tell if something is an active fused wind object
def is_fused_object_or_group(object):

    # simply apply the test and return
    return isinstance(object, FUSED_System_Base) or isinstance(object, FUSED_Object)

# get the dictionary and object list
def get_object_dict_and_list(object_container):

    # initialize the object list
    object_list = []
    object_dictionary = {}
    fused_object_list = []
    fused_object_dictionary = {}

    # If the objects are a dictionary then set that
    if isinstance(object_container, dict):
        # set the dictionary
        object_dictionary = object_container
        # loop through all the object dictionary
        for name, obj in object_dictionary.items():
            object_list.append(obj)
            if is_fused_object_or_group(obj):
                fused_object_list.append(obj)
                fused_object_dictionary[name] = obj
    else:
        # basically store the objects
        object_list = object_container
        # this set maintains unique names
        object_name_set = set()
        # loop through all the objects
        for obj in object_container:
            name = None
            if hasattr(obj, 'object_name'):
                name = obj.object_name
                name = make_unique_name(name, object_name_set)
                object_dictionary[name] = obj
            if is_fused_object_or_group(obj):
                fused_object_list.append(obj)
                if not name is None:
                    fused_object_dictionary[name] = obj

    return object_dictionary, object_list, fused_object_dictionary, fused_object_list

# This will reduce the object list to only the objects
def dissolve_groups(object_list, group_set = None):

    # if group_set is None, give an empty one
    if group_set is None:
        group_set = set()

    # This is the return set
    retval = set()

    # This is the map from the original object to the sub object
    orig_obj_to_sub_obj = {}

    # This is the map from the sub object to the original object
    sub_obj_to_orig_obj = {}

    # First, dissolve any groups
    for obj in object_list:
        if isinstance(obj, FUSED_System_Base):
            if not obj in group_set:
                group_set.add(obj)
                sub_set, null1, null2 = dissolve_groups(obj.system_objects, group_set=group_set)
                for sub_obj in sub_set:
                    if not obj in orig_obj_to_sub_obj:
                        orig_obj_to_sub_obj[obj]={sub_obj}
                    else:
                        orig_obj_to_sub_obj[obj].add(sub_obj)
                    if sub_obj in sub_obj_to_orig_obj:
                        raise Exception('It appears an object is contained twice')
                    sub_obj_to_orig_obj[sub_obj]=obj
                    retval.add(sub_obj)
        elif isinstance(obj, FUSED_Object):
            if not obj in orig_obj_to_sub_obj:
                orig_obj_to_sub_obj[obj]={obj}
            else:
                orig_obj_to_sub_obj[obj].add(obj)
            if obj in sub_obj_to_orig_obj:
                raise Exception('It appears an object is contained twice')
            sub_obj_to_orig_obj[obj]=obj
            retval.add(obj)

    # return the results
    return retval, orig_obj_to_sub_obj, sub_obj_to_orig_obj

# The following function will solve the split configuration of a work-flow
##########################################################################

# This is a stand-alone system object
# It contains a set of fused wind objects:
#    Some are just pure input objects
#    Some are tagged as giving output to the larger system
# Together these objects are expected to operate as a system
# The system base class packages these objects so they act as a single object

class FUSED_System_Base(FUSED_Unique):

    def __init__(self, objects_in={}, group_name='unnamed_base_group', **kwargs):
        super(FUSED_System_Base, self).__init__()

        # This is the group name
        self.object_name = group_name
        # If the objects are a dictionary then set that
        if isinstance(objects_in, dict):
            # Add the kw-arg objects
            for key, obj in kwargs.items():
                if key in objects_in:
                    raise Exception('Two different objects specified for key "%s"'%key)
                objects_in[key]=obj
        else:
            # Check if we are adding the objects as key-work args
            if len(kwargs)!=0:
                raise Exception('Objects cannot be added as key-word arguments when other objects are added as a list')

        # set the internal objects
        self.object_dictionary, null1, null2, self.system_objects = get_object_dict_and_list(objects_in)

        # whether it has been configured or not
        self.system_has_been_configured = False

        # The following data is for building the input connection interface and data structures
        #
        # Some temporary variables:
        #
        #   #This basically registers whether a given input variables has been registered in the automatic interface building methods
        #   self._input_obj_var_is_found[local_obj] = [ local_dest_name_1, local_dest_name_2 ...]
        self._input_obj_var_is_found = {}
        #
        #   #In the automatic interface construction methods.
        #   #This registers how the public name is associated with the internal objects
        #   #This can contain multiple inputs associated with a global name, however in the configuration methods, this will be renamed so there are unique inputs
        #   self._input_var_to_obj_pair[global_name] = [ ({obj_1: [local_dest_name_1, local_dest_name_2 ...] }, indep_var) , ... ]
        self._input_var_to_obj_pair = {}
        #
        # Internal input connection data structure:
        #
        #   #This shows how a local object-input is mapped to a given global name
        #   self.system_input_map[obj][lcl_name]                -> global_input_name   * Note: obj can be the input objects and the independent variable
        self.system_input_map = {}
        #
        # Internal gobal_input_name to object/local name
        #
        #   # This shows how a global input name is mapped to local objects and input names
        #   self.system_input_gbl_to_lcl_map[global_input_name] -> ( input_dict[obj] -> list( local_input_name_1, local_input_name_2, ... ) , indep_var )
        self.system_input_gbl_to_lcl_map = {}
        #   

        #
        # Some temporary variables:
        #
        #   # This is used in the automatic interface generation methods.
        #   # This tracks whether a given output has been associated with an output
        #   self._output_obj_var_is_found[local_obj] = [ local_src_name_1, local_src_name_2, ...]
        self._output_obj_var_is_found = {}
        #
        #   # This is used by the automatic interface generation methods
        #   # This shows how the global output name is associated with outputs. 
        #   # In this case, there can be multiple sources but when it is configred, the global output name is re-named so the different sources are unique
        #   self._output_var_to_obj_pair[global_name] = [ (obj_1, local_src_name_1), (obj_2, local_src_name_2) ]
        self._output_var_to_obj_pair = {}
        #
        # Internal output connection data structure:
        #
        #   # This will show how an local object and output name are mapped to global output name
        #   self.system_output_map[obj][lcl_name]                 -> global_output_name
        self.system_output_map = {}
        #
        # Internal global_output_name to object/local name
        #
        #   # This will show how a global output name is mapped to a local object and local output name
        #   self.system_output_gbl_to_lcl_map[global_output_name] -> ( obj, local_src_name )
        self.system_output_gbl_to_lcl_map = {}
        #

    # This will retrieve a data-structure that will indicate the extent a group input variable is connected externally
    #
    #     dict[dest_obj][dest_name]=(src_obj,src_name,is_external)
    #
    def get_all_input_connections(self, global_input_name , internal_object_list = None):

        # Configure the system if needed
        if not self.system_has_been_configured:
            self.configure_system()

        # dissolve the system objects
        if internal_object_list is None:
            internal_object_list, system_objects_orig_to_sub, system_objects_sub_to_orig = dissolve_groups(self.system_objects)

        if not global_input_name in self.system_input_gbl_to_lcl_map:
            raise KeyError('The input %s does not exist for this group'%(global_input_name))

        # get our return value
        retval = {}

        # Loop through our data structure and determine the connections
        input_dest_dict = self.system_input_gbl_to_lcl_map[global_input_name][0]
        for obj, dest_list in input_dest_dict.items():
            # If we have another group, then we have to proceed recursively
            if isinstance(obj, FUSED_System_Base):
                # Loop through all the names in the destination list
                for name in dest_list:
                    # Retrieve all the connections for that name
                    sub_conn_dict = obj.get_all_input_connections(name, internal_object_list)
                    # Loop through all the dest objects
                    for sub_dest_obj, sub_dest_name_dict in sub_conn_dict.items():
                        # Add connection information if the object has not been seen
                        if not sub_dest_obj in retval:
                            retval[sub_dest_obj]=sub_dest_name_dict
                        else:
                            # loop through all the dest names
                            for sub_dest_name, sub_data in sub_dest_name_dict.items():
                                # Add the connection data if that destination info has not been seen
                                if not sub_dest_name in retval[sub_dest_obj]:
                                    retval[sub_dest_obj][sub_dest_name] = sub_data
            # If we have an object then we have to access the data directly
            else:
                # Add the object if it has not been seen
                if not obj in retval:
                    retval[obj] = {}
                # Loop through the destination names
                for dest_name in dest_list:
                    # Add the data if it has not been added already
                    if not dest_name in retval[obj]:
                        # Default assumption is that there is no connection so all is None
                        src_obj = None
                        src_name = None
                        src_ext = None
                        # If it is connected, then we grab all the data
                        if dest_name in obj.conn_dict:
                            src_tuple = obj.conn_dict[dest_name]
                            src_obj = src_tuple[0]
                            src_name = src_tuple[1]
                            src_ext = not src_obj in internal_object_list
                        # add the here
                        retval[obj][dest_name]=(src_obj, src_name, src_ext)

        return retval

    # This will retrieve a data-structure that will indicate the extent a group input variable is connected externally
    #
    #     retval = (src_obj, src_name, {dst_obj:([dst_name_1, dst_name_2, ..., dst_name_N], is_external)}, is_external)
    #
    def get_all_output_connections(self, global_output_name , internal_object_list = None):

        # Configure the system if needed
        if not self.system_has_been_configured:
            self.configure_system()

        # dissolve the system objects
        if internal_object_list is None:
            internal_object_list, system_objects_orig_to_sub, system_objects_sub_to_orig = dissolve_groups(self.system_objects)

        if not global_output_name in self.system_output_gbl_to_lcl_map:
            raise KeyError('The input %s does not exist for this group'%(global_output_name))

        # get our return value
        retval = {}

        # Loop through our data structure and determine the connections
        output_tuple = self.system_output_gbl_to_lcl_map[global_output_name]
        src_obj = output_tuple[0]
        src_name = output_tuple[1]
        is_external = False
        dest_dict = {}
        # If we have another group, then we have to proceed recursively
        if isinstance(src_obj, FUSED_System_Base):
            src_obj, src_name, dest_dict, is_external = src_obj.get_all_output_connections(src_name, internal_object_list=internal_object_list)
        else:
            # If we have a connection, then we process it to build the dest dictionary
            if src_name in src_obj.output_connections:
                # loop through all the destination objects
                for sub_dst_obj, sub_dest_list in src_obj.output_connections[src_name].items():
                    # test if external
                    sub_is_external = not sub_dst_obj in internal_object_list
                    if sub_is_external:
                        is_external = True
                    # Add the data for the object
                    dest_dict[sub_dst_obj]=(sub_dest_list, sub_is_external)

        # Construct the return value
        retval = (src_obj, src_name, dest_dict, is_external)

        # Return the tuple
        return retval

    # This will retrieve the objects contained within this group
    def get_object(self, object_name):

        sub_key = None
        if '.' in object_name:
            object_name, *remainder = object_name.split('.')
            sub_key = '.'.join(remainder)
        if not object_name in self.object_dictionary:
            raise KeyError('The object %s does not exist within this group'%(object_name))
        my_obj = self.object_dictionary[object_name]
        if sub_key is None:
            return my_obj
        if not isinstance(my_obj, FUSED_System_Base):
            raise Exception('Can only access objects from a FUSED_System_Base')
        return my_obj.get_object(sub_key)

    # This will retrieve the object keys
    def get_object_keys(self):

        # simply return my object keys
        return self.object_dictionary.keys()

    # This will return all keys for my objects and my childrens objects
    def get_all_object_keys(self):

        # Populate the return value with my object keys first
        retval = list(self.object_dictionary.keys())

        # loop through my objects and get the sub-keys
        for key, obj in self.object_dictionary.items():
            # test if it ass the method we want
            if hasattr(obj, 'get_all_object_keys'):
                # get the sub-keys
                sub_keys = obj.get_all_object_keys()
                # now append them to my return value
                for sub in sub_keys:
                    retval.append(key+'.'+sub)
                    
        # Return the result
        return retval

    # This is suppose to use the independent variables to build an input interface
    def add_input_interface_from_independent_variables(self, object_list = None):
        # The objects in the object list must be objects within this object
        # When the object list is None, then all the objects in this object are considered
        # The method will search for independent variables from within the list
        # This is an automated interface generation scheme

        # cannot add variables if the system is already configured
        if self.system_has_been_configured:
            raise Exception('The system has already been configured')

        # get the object list from the system objects
        if object_list is None:
            object_list = self.system_objects

        # This stores the obj->[dest_name, ...] structure associated with indep-var
        iv_dict = {}

        # Now we loop and discover our independent variables
        for iv_obj in object_list:
            # is the object an indep ... then we must continue
            if iv_obj.is_independent_variable():
                # This is the var_name
                var_name = iv_obj.name
                # now we must loop over all the down-stream
                for src_name, dest_dict in iv_obj.output_connections.items():
                    # loop through the dest objects
                    for obj, dest_list in dest_dict.items():
                        # loop through the dest name
                        for name in dest_list:
                            # If this is a legitimate input, then lets add it as a candidate
                            if (not obj in self.system_input_map or not name in self.system_input_map[obj]) and (not obj in self._input_obj_var_is_found or not name in self._input_obj_var_is_found[obj]):
                                # Then populate the iv_dict
                                if not obj in iv_dict:
                                    iv_dict[iv_obj]=({}, iv_obj)
                                if not obj in iv_dict[iv_obj][0]:
                                    iv_dict[iv_obj][0][obj] = []
                                iv_dict[iv_obj][0][obj].append(name)
                                # register that a variable has been found
                                if not obj in self._input_obj_var_is_found:
                                    self._input_obj_var_is_found[obj]=[name]
                                else:
                                    self._input_obj_var_is_found[obj].append(name)
        # Find the default names
        for iv_obj, iv_result in iv_dict.items():
            var_name = iv_result[1].name
            if var_name in self._input_var_to_obj_pair:
                self._input_var_to_obj_pair[var_name].append(iv_result)
            else:
                self._input_var_to_obj_pair[var_name] = [iv_result]

    # This is suppose to get an input interface from connections
    def add_input_interface_from_connections(self, object_list = None, use_set_connections = True, prepend_name=None):
        # This will build an input interface from the existing connections
        # In cases where a single input goes to two variables within this group, only one input variable is declared
        # Furthermore, in this case, the default name for this variable is based on the first associated object in the internal object list
        # When 'use_set_connections' is true, then existing connections are used
        # When 'use_set_connections' is false, then each empty connection is used
        # This is an automated interface generation scheme

        # cannot add variables if the system is already configured
        if self.system_has_been_configured:
            raise Exception('The system has already been configured')

        # get the object list from the system objects
        if object_list is None:
            object_list = self.system_objects

        # Get my internal object data
        system_objects_dissolved, null1, null2 = dissolve_groups(self.system_objects)

        # search for the external connections
        if use_set_connections:
            ext_conn_dict = {}
            # loop through all the objects in consideration
            for orig_obj in object_list:
                if not orig_obj in self.system_objects:
                    raise Exception('Trying to build interface based objects outside the group')
                # If we are working with objects directly, then we just access their connection data structures
                if isinstance(orig_obj, FUSED_Object):
                    # loop through every external object
                    for ext_obj, src_dst_map in orig_obj.connections.items():
                        # test if that external object is external
                        if not ext_obj in system_objects_dissolved:
                            # loop over source names
                            for src_name, dest_list in src_dst_map.items():
                                # loop over the destination names
                                for name in dest_list:
                                    # Verify that we have a valid candidate
                                    if (not orig_obj in self.system_input_map or not name in self.system_input_map[orig_obj]) and (not orig_obj in self._input_obj_var_is_found or not name in self._input_obj_var_is_found[orig_obj]):
                                        # add ext obj to the conn dict
                                        if not ext_obj in ext_conn_dict:
                                            ext_conn_dict[ext_obj]={}
                                        # add src_name to the con dict
                                        if not src_name in ext_conn_dict[ext_obj]:
                                            ext_conn_dict[ext_obj][src_name]=({}, None)
                                        # add the object if not already
                                        if not orig_obj in ext_conn_dict[ext_obj][src_name][0]:
                                            ext_conn_dict[ext_obj][src_name][0][orig_obj]=[]
                                        ext_conn_dict[ext_obj][src_name][0][orig_obj].append(name)
                                        # Register that a variable has been found
                                        if not orig_obj in self._input_obj_var_is_found:
                                            self._input_obj_var_is_found[orig_obj] = [name]
                                        else:
                                            self._input_obj_var_is_found[orig_obj].append(name)
                # Now we are dealing with the situation that we have a group
                else:
                    # Retrieve the interface
                    obj_ifc = orig_obj.get_interface()
                    # Loop through the input names
                    for name in obj_ifc['input'].keys():
                        # collect the connection structure
                        input_conn_data = orig_obj.get_all_input_connections(name, internal_object_list=system_objects_dissolved)
                        is_external = False
                        ext_obj = None
                        # Now test if we have an external connection
                        for sub_dest_obj, sub_dest_dict in input_conn_data.items():
                            # test the sub_dest_names
                            for sub_dest_name, sub_dest_data in sub_dest_dict.items():
                                # test if we have an external connection
                                if not sub_dest_data[2] is None and sub_dest_data[2]:
                                    ext_obj = sub_dest_data[0]
                                    src_name = sub_dest_data[1]
                                    is_external = True
                                    break
                            if is_external:
                                break
                        # If we have an external connection, then proceed...
                        if is_external:
                            # Verify that we have a valid candidate
                            if (not orig_obj in self.system_input_map or not name in self.system_input_map[orig_obj]) and (not orig_obj in self._input_obj_var_is_found or not name in self._input_obj_var_is_found[orig_obj]):
                                # add ext obj to the conn dict
                                if not ext_obj in ext_conn_dict:
                                    ext_conn_dict[ext_obj]={}
                                # add src_name to the con dict
                                if not src_name in ext_conn_dict[ext_obj]:
                                    ext_conn_dict[ext_obj][src_name]=({}, None)
                                # add the object if not already
                                if not orig_obj in ext_conn_dict[ext_obj][src_name][0]:
                                    ext_conn_dict[ext_obj][src_name][0][orig_obj]=[]
                                ext_conn_dict[ext_obj][src_name][0][orig_obj].append(name)
                                # Register that a variable has been found
                                if not orig_obj in self._input_obj_var_is_found:
                                    self._input_obj_var_is_found[orig_obj] = [name]
                                else:
                                    self._input_obj_var_is_found[orig_obj].append(name)

            # now we have all our connection groups, figure out the default global name and list the group as a candidate
            for ext_obj, src_dict in ext_conn_dict.items():
                for src_name, dest_dict in src_dict.items():
                    first_obj = sorted(dest_dict[0].keys())[0]
                    name = dest_dict[0][first_obj][0]
                    # Add it to a candidate name
                    global_input_name = name
                    if not prepend_name is None:
                        global_input_name = prepend_name + name
                    if global_input_name in self._input_var_to_obj_pair:
                        self._input_var_to_obj_pair[global_input_name].append(dest_dict)
                    else:
                        self._input_var_to_obj_pair[global_input_name] = [dest_dict]

        else:
            # loop over the objects
            for orig_obj in object_list:
                obj_ifc = orig_obj.get_interface()['input']
                # Loop through all the inputs
                for name in obj_ifc.keys():
                    # Assume we do not have a connection
                    is_connected = False
                    # If we have an object, we can access the data structure directly
                    if isinstance(orig_obj, FUSED_Object) and name in obj.conn_dict:
                        is_connected = True
                    # if we have a group, then we access collect the connection information
                    elif isinstance(orig_obj, FUSED_System_Base):
                        # collect the connection structure
                        input_conn_data = orig_obj.get_all_input_connections(name, internal_object_list=system_objects_dissolved)
                        # Now we loop through all the destination objects
                        for sub_dest_obj, sub_dest_dict in input_conn_data.items():
                            # Now we loop over all the destination names
                            for sub_dest_name, sub_dest_data in sub_dest_dict.items():
                                # test if we have a connection
                                if not sub_dest_data[0] is None:
                                    is_connected = True
                                    break
                            if is_connected:
                                break
                    # Add the variable if not not connectd and not already added
                    if not is_connected and ((not orig_obj in self.system_input_map or not name in self.system_input_map[orig_obj]) and (not orig_obj in self._input_obj_var_is_found or not name in self._input_obj_var_is_found[orig_obj])):
                        # get the global input name
                        global_input_name = name
                        if not prepend_name is None:
                            global_input_name = prepend_name + name
                        # Add it to a candidate name
                        if name in self._input_var_to_obj_pair:
                            self._input_var_to_obj_pair[global_input_name].append(({orig_obj:[name]}, None))
                        else:
                            self._input_var_to_obj_pair[global_input_name] = [({orig_obj:[name]}, None)]
                        # register that a variable has been found
                        if not orig_obj in self._input_obj_var_is_found:
                            self._input_obj_var_is_found[orig_obj]=[name]
                        else:
                            self._input_obj_var_is_found[orig_obj].append(name)

    # This is suppose to assume that all inputs of all objects in the list are public input variables.
    def add_input_interface_from_objects(self, object_list = None, merge_by_input_name=False, prepend_name=None, append_name=None):
        # The object list must be a list of objects contained within this object
        # When object_list is none, Then it is assumed that all objects are used
        # This is an automated interface generation scheme

        # cannot add variables if the system is already configured
        if self.system_has_been_configured:
            raise Exception('The system has already been configured')

        # get the object list from the system objects
        if object_list is None:
            object_list = self.system_objects

        # The assumption is that object_list is a list, if not, put it in a list
        if not isinstance(object_list, list) and not isinstance(object_list,tuple):
            object_list = [object_list]

        # loop over the objects
        for obj in object_list:
            obj_ifc = obj.get_interface()['input']
            for name in obj_ifc.keys():
                # Add the variable if not already
                if (not obj in self.system_input_map or not name in self.system_input_map[obj]) and (not obj in self._input_obj_var_is_found or not name in self._input_obj_var_is_found[obj]):
                    # get the global name
                    global_name = name
                    if not prepend_name is None:
                        global_name = prepend_name+global_name
                    if not append_name is None:
                        global_name = global_name+append_name
                    # Add it to a candidate name
                    if name in self._input_var_to_obj_pair:
                        if merge_by_input_name:
                            if len(self._input_var_to_obj_pair[global_name])!=1:
                                raise Exception('Cannot merge inputs by name when there are already multiple candidates')
                            obj_dict = self._input_var_to_obj_pair[global_name][0][0]
                            if not obj in obj_dict:
                                obj_dict[obj]=[name]
                            else:
                                obj_dict[obj].append(name)
                        else:
                            self._input_var_to_obj_pair[global_name].append(({obj:[name]}, None))
                    else:
                        self._input_var_to_obj_pair[global_name] = [({obj:[name]}, None)]
                    # register that a variable has been found
                    if not obj in self._input_obj_var_is_found:
                        self._input_obj_var_is_found[obj]=[name]
                    else:
                        self._input_obj_var_is_found[obj].append(name)

    # This is suppose to set the input variable based on and internal independent variable
    # Note, that this interface item takes precedent over all automated interface constructions
    def set_input_interface_from_independent_variables(self, var_name, indep_var = None):
        # var_name is the global name for the independent variable
        # indep_var is an object in the group which is the independent variable for the input
        # when indep_var is None, then the indep_var is searched for in the object list

        # cannot add variables if the system is already configured
        if self.system_has_been_configured:
            raise Exception('The system has already been configured')

        # Cannot add a variable that is already added
        if var_name in self.system_input_gbl_to_lcl_map:
            raise Exception('That input variable already exists in the group')

        # Test if we need to search
        if indep_var is None:
            for obj in self.system_objects:
                if obj.is_independent_variable():
                    if var_name == obj.name:
                        if not indep_var is None:
                            raise('Two independent variables with the same matching names has been found')
                        indep_var = obj
        if not indep_var is None:
            if not indep_var in self.system_objects:
                raise Exception('The independent variable is not included in the system object list')
            local_name = indep_var.name
            if not indep_var in self.system_input_map:
                self.system_input_map[indep_var] = {}
            self.system_input_map[indep_var][local_name] = var_name
            # Set the variable in the global -> local
            if not var_name in self.system_input_gbl_to_lcl_map:
                self.system_input_gbl_to_lcl_map[var_name] = ({},indep_var)
            # Set the ouput objects and variables based on the output connections
            for tmp_output_name, dest_dict in indep_var.output_connections.items():
                for dest_obj, dest_list in dest_dict.items():
                    if dest_obj in self.system_objects:
                        for dest_name in dest_list:
                            # Add the variable to the input map
                            if not dest_obj in self.system_input_map:
                                self.system_input_map[dest_obj] = {}
                            self.system_input_map[dest_obj][dest_name] = var_name
                            # Add the variable to the gbl to lcl map
                            if not dest_obj in self.system_input_gbl_to_lcl_map[var_name][0]:
                                self.system_input_gbl_to_lcl_map[var_name][0][dest_obj] = [dest_name]
                            else:
                                self.system_input_gbl_to_lcl_map[var_name][0][dest_obj].append(dest_name)
        else:
            raise Exception('Cannot find an independent variable')

    # This is suppose to set an input based on a fused var or name.
    # Note, that this interface item takes precedent over all automated interface constructions
    def set_input_variable(self, var_name, obj_dict = None, dest_list = None):
        # var_name:               Can be a name for the variable, or
        #                         Actual fused variable, Note the meta data from the object will be absorbed
        # obj_dict:               A dictionary of objects to make connections to
        #                         The entries of dictionary contain the input variables of that object that should be associated with this var_name
        #                         The entries of the dictionary can be None, in which case the input with the same name is assumed
        #                         When this is None, then all objects will be searched for input variables that could match the name
        # dest_list:              All objects are searched for names in dest-list
        # Note 1: That when both obj_dict and dest_list are None, all objects are searched for candidates based on the var_name
        # Note 2: It will be assumed that multiple destination variables found will all recieve the same data

        # If the variable is already in the interface, then we need to exit
        if var_name in self.system_input_gbl_to_lcl_map:
            raise Exception('That variable already exists in the input interface')

        # cannot add variables if the system is already configured
        if self.system_has_been_configured:
            raise Exception('The system has already been configured')

        # If it is a variable then take the name
        if isinstance(var_name, dict):
            var_name = var_name['name']

        # If no guidance, then use the name
        if obj_dict is None and dest_list is None:
            dest_list = [var_name]

        # Use the object dictionary
        if not obj_dict is None:
            # loop over the objects
            for obj, var_list in obj_dict.items():
                if not obj in self.system_objects:
                    raise Exception('The object is not contained with this variable')
                if not obj in self.system_input_map:
                    self.system_input_map[obj]={}
                obj_ifc = obj.get_interface()['input']
                if var_list is None:
                    var_list = [var_name]
                for name in var_list:
                    if not name in obj_ifc.keys():
                        raise KeyError('That variable does not exist in the object')
                    # Set the variable in the local -> global
                    self.system_input_map[obj][name] = var_name
                    # Set the variable in the global -> local
                    if not var_name in self.system_input_gbl_to_lcl_map:
                        self.system_input_gbl_to_lcl_map[var_name] = ({},None)
                    if not obj in self.system_input_gbl_to_lcl_map[var_name][0]:
                        self.system_input_gbl_to_lcl_map[var_name][0][obj] = [name]
                    else:
                        self.system_input_gbl_to_lcl_map[var_name][0][obj].append(name)

        # Now lets search via the dest_list
        if not dest_list is None:
            fnd_conn = False
            for dest_name in dest_list:
                fnd_local_conn = False
                for obj in self.system_objects:
                    obj_ifc = obj.get_interface()['input']
                    if dest_name in obj_ifc:
                        fnd_local_conn = True
                        # Set the variable in the local -> global
                        if not obj in self.system_input_map:
                            self.system_input_map[obj] = {}
                        self.system_input_map[obj][dest_name] = var_name
                        # Set the variable in the global -> local
                        if not var_name in self.system_input_gbl_to_lcl_map:
                            self.system_input_gbl_to_lcl_map[var_name] = ({},None)
                        if not obj in self.system_input_gbl_to_lcl_map[var_name][0]:
                            self.system_input_gbl_to_lcl_map[var_name][0][obj] = [dest_name]
                        else:
                            self.system_input_gbl_to_lcl_map[var_name][0][obj].append(dest_name)
                if not fnd_local_conn:
                    print('Warning, could not find a connection for global variable %s with the local name %s'%(var_name, dest_name))
                else:
                    fnd_conn = True
            if not fnd_conn:
                print('Warning, could not find a connection for global variable %s'%(var_name))

    # This is suppose to assume that all outputs of all objects in the list are public output variables.
    def add_output_interface_from_objects(self, object_list = None, var_name_list=None, exclude_list=[], prepend_name=None, append_name=None):
        # The object list must be a list of objects contained within this object
        # When object_list is none, Then it is assumed that all objects are used
        # This is an automated interface generation scheme

        # cannot add variables if the system is already configured
        if self.system_has_been_configured:
            raise Exception('The system has already been configured')

        # Get the object list
        if object_list is None:
            object_list = self.system_objects

        # The assumption is that object_list is a list, if not, put it in a list
        if not isinstance(object_list, list) and not isinstance(object_list,tuple):
            object_list = [object_list]
 
        # loop through all the connections
        for obj in object_list:
            # Verify this is a valid object
            if not obj in self.system_objects:
                raise Exception('Cannot build an interface based on objects outside the group')
            # Loop through the output names
            for src_name in obj.get_interface()['output'].keys():
                # proceed only if allowed
                if (var_name_list is None and not src_name in exclude_list) or (not var_name_list is None and src_name in var_name_list):
                    # set global name
                    global_src_name = src_name
                    if not prepend_name is None:
                        global_src_name = prepend_name+global_src_name
                    if not append_name is None:
                        global_src_name = global_src_name+append_name
                    # Verify that it meets whether it is already added criteria:
                    if (not obj in self.system_output_map or not src_name in self.system_output_map[obj]) and (not obj in self._output_obj_var_is_found or not src_name in self._output_obj_var_is_found[obj]):
                        # Add it to the output structure
                        if not global_src_name in self._output_var_to_obj_pair:
                            self._output_var_to_obj_pair[global_src_name] = [(obj,src_name)]
                        else:
                            self._output_var_to_obj_pair[global_src_name].append( (obj,src_name) )
                        # add to 'is-found'
                        if not obj in self._output_obj_var_is_found:
                            self._output_obj_var_is_found[obj] = [src_name]
                        else:
                            self._output_obj_var_is_found[obj].append(src_name)

    # This is suppose to collect the interface from existing connections
    def add_output_interface_from_connections(self, object_list = None, use_set_connections = True):
        # When 'use_set_connections' is true, then existing connections are used
        # When 'use_set_connections' is false, then empty connections are used
        # This is an automated interface generation scheme

        # cannot add variables if the system is already configured
        if self.system_has_been_configured:
            raise Exception('The system has already been configured')

        # Get the object list
        if object_list is None:
            object_list = self.system_objects

        # Get the internal object list
        internal_object_list, null1, null2 = dissolve_groups(self.system_objects)
 
        # loop through all the connections
        for obj in object_list:
            # Verify this is a valid object
            if not obj in self.system_objects:
                raise Exception('Cannot build an interface based on objects outside the group')
            # loop through all the output variables
            for src_name in obj.get_interface()['output'].keys():
                # Assume it is not connected
                is_connected = False
                # process directly if we have a simple object
                if isinstance(obj, FUSED_Object) and src_name in obj.output_connections.keys():
                    # loop through all the destination objects
                    for sub_dst_obj, sub_dest_list in obj.output_connections[src_name].items():
                        # Test if external connection
                        if not sub_dst_obj in internal_object_list:
                            is_connected = True
                            break
                # Call the output function if it is a group
                else:
                    null1, null2, null3, is_connected = obj.get_all_output_connections(src_name, internal_object_list)
                # Verify that it meets the connection criteria
                if (use_set_connections and is_connected) or (not use_set_connections and not is_connected):
                    # Verify that it meets whether it is already added criteria:
                    if (not obj in self.system_output_map or not src_name in self.system_output_map[obj]) and (not obj in self._output_obj_var_is_found or not src_name in self._output_obj_var_is_found[obj]):
                        # Add it to the output structure
                        if not src_name in self._output_var_to_obj_pair:
                            self._output_var_to_obj_pair[src_name] = [(obj,src_name)]
                        else:
                            self._output_var_to_obj_pair[src_name].append( (obj,src_name) )
                        # add to 'is-found'
                        if not obj in self._output_obj_var_is_found:
                            self._output_obj_var_is_found[obj] = [src_name]
                        else:
                            self._output_obj_var_is_found[obj].append(src_name)

    # This is suppose to add an output based on a fused var.
    # Note, that this interface item takes precedent over all automated interface constructions
    def set_output_variable(self, var_name, obj = None, local_output_name = None):
        # var_name:               Can be a name for the variable
        #                         Actual fused variable, Note the meta data from the object will be absorbed
        # obj:                    The internal object where the output is from.
        #                         If None, all objects searched for the variable.
        #                         If duplicates are found an error is raised
        # local_output_name:      The variable name in the object name-space

        # This will verify that no duplicates are in the interface
        if var_name in self.system_output_gbl_to_lcl_map:
            raise Exception('That variable already exists in the output interface')

        # If the system is already configured then we cannot accept changes
        if self.system_has_been_configured:
            raise Exception('System has been configured')

        # Use the var_name for the local if not specified
        if local_output_name is None:
            local_output_name = var_name

        # If no object is given, then search for it
        if obj is None:
            # loop through all objects
            for tmp_obj in self.system_objects:
                obj_ifc = tmp_obj.get_interface()['output']
                # test if we have a match
                if local_output_name in obj_ifc:
                    if not obj is None:
                        raise Exception('Two objects with the same output discovered')
                    obj = tmp_obj

        # If no object is given, then faile
        if obj is None:
            raise Exception('Cannot find a matching object')

        # Lets verify that the object is in there
        if not obj in self.system_objects:
            raise Exception('That object does not exist in the system')

        # Lets verify that the variable is in the output
        if not local_output_name in obj.get_interface()['output']:
            raise Exception('That object does not have the local variable requested')

        # Then lets set the result
        if not obj in self.system_output_map:
            self.system_output_map[obj] = {}
        self.system_output_map[obj][local_output_name] = var_name
        self.system_output_gbl_to_lcl_map[var_name] = ( obj, local_output_name )

    # This will resolve the output names
    def resolve_interface_names(self, var_to_obj_pair, exclusion_set):

        # The new dictionary
        tmp_var_to_obj_pair = {}

        # Now loop through the old one
        for var_name, conn_list in var_to_obj_pair.items():
            # If there is not interference, then just keep it as it is
            if len(conn_list)==1 and not var_name in exclusion_set:
                tmp_var_to_obj_pair[var_name]=conn_list[0]
            else:
                # Ok try to include the object name to differentiate
                sub_dict = {}
                is_good = True
                for conn_data in conn_list:
                    first_obj = conn_data[0]
                    if isinstance(first_obj, dict):
                        first_obj = sorted(first_obj.keys())[0]
                    new_name = '%s__%s'%(first_obj.object_name, var_name)
                    if var_name in exclusion_set or var_name in sub_dict or var_name in tmp_var_to_obj_pair:
                        is_good = False
                        break
                    sub_dict[new_name]=conn_data
                if is_good:
                    tmp_var_to_obj_pair.update(sub_dict)
                else:
                    # Ok try to include the object hash and other counters to differentiate
                    sub_dict = {}
                    is_good = True
                    for conn_data in conn_list:
                        first_obj = conn_data[0]
                        if isinstance(first_obj, dict):
                            first_obj = sorted(first_obj.keys())[0]
                        new_name = '%s_%d__%s'%(first_obj.object_name, first_obj._hash_value, var_name)
                        extra = 1
                        while var_name in exclusion_set or var_name in sub_dict or var_name in tmp_var_to_obj_pair:
                            new_name = '%s_%d_%d__%s'%(first_obj.object_name, first_obj._hash_value, extra, var_name)
                            extra+=1
                        sub_dict[new_name]=conn_data
                    # Now we have a unique set
                    tmp_var_to_obj_pair.update(sub_dict)
        
        # Return the unique candidates now
        return tmp_var_to_obj_pair

    # This is called to configure the system
    def configure_system(self):

        # if configured, then just exit
        if self.system_has_been_configured:
            return

        # First make sure the automatic variables are made unique
        self._input_var_to_obj_pair = self.resolve_interface_names(self._input_var_to_obj_pair, set(self.system_input_gbl_to_lcl_map.keys()))
        self._output_var_to_obj_pair = self.resolve_interface_names(self._output_var_to_obj_pair, set(self.system_output_gbl_to_lcl_map.keys()))

        # Now lets add the automatic input variables to the permanent record
        for global_name, input_data in self._input_var_to_obj_pair.items():
            if global_name in self.system_input_gbl_to_lcl_map:
                raise Exception('The global name already exists in the interface')
            # Add the indep var if it is not already added
            if not input_data[1] is None:
                if not input_data[1] in self.system_input_map:
                    new_data = ({}, input_data[1])
                    self.system_input_map[input_data[1]] = {input_data[1].name:global_name}
            else:
                new_data = ({}, None)
            # Loop through the connect data
            new_dict = new_data[0]
            for obj, dest_list in input_data[0].items():
                for dest_name in dest_list:
                    if not obj in self.system_input_map or not dest_name in self.system_input_map[obj]:
                        # mark the variable
                        if not obj in self.system_input_map:
                            self.system_input_map[obj] = {}
                        self.system_input_map[obj][dest_name] = global_name
                        # Add to the new local data object
                        if not obj in new_dict:
                            new_dict[obj] = [dest_name]
                        else:
                            new_dict[obj].append(dest_name)
            # Verify that we have something worth adding
            if len(new_dict)>0:
                # Add the local data object to the gbl -> lcl
                self.system_input_gbl_to_lcl_map[global_name]=new_data

        # Now lets add all the automatic output variables to the permanent record
        for global_name, output_data in self._output_var_to_obj_pair.items():
            if global_name in self.system_output_gbl_to_lcl_map:
                raise Exception('The global name already exists in the interface')
            obj = output_data[0]
            name = output_data[1]
            # Add only if not already set
            if not obj in self.system_output_map or not name in self.system_output_map[obj]:
                if not obj in self.system_output_map:
                    self.system_output_map[obj] = {}
                self.system_output_map[obj][name] = global_name
                self.system_output_gbl_to_lcl_map[global_name] = output_data

        # Now lets construct the interface
        self.system_ifc = create_interface()

        # Add the input:
        for name, data in self.system_input_gbl_to_lcl_map.items():
            first_obj = sorted(data[0].keys())[0]
            lcl_name = data[0][first_obj][0]
            obj_ifc = first_obj.get_interface()
            meta = copy.copy(obj_ifc['input'][lcl_name])
            meta['name'] = name
            set_input(self.system_ifc, meta)

        # Add the output:
        for name, data in self.system_output_gbl_to_lcl_map.items():
            first_obj = data[0]
            lcl_name = data[1]
            obj_ifc = first_obj.get_interface()
            meta = copy.copy(obj_ifc['output'][lcl_name])
            meta['name'] = name
            set_output(self.system_ifc, meta)

        # Flag as configured
        self.system_has_been_configured = True

    # This will reduce the object list to only the objects
    def dissolve_groups(self, object_list = None):

        # If no object list is given, use my own
        my_objects = False
        if object_list is None:
            my_objects = True
            object_list = self.system_objects
 
        # First, dissolve any groups
        need_check = True
        while need_check:
            tmp_system_objects = []
            need_check = False
            for obj in object_list:
                if isinstance(obj, FUSED_System_Base) and not isinstance(obj, FUSED_Object):
                    need_check = True
                    for sub_obj in obj.system_objects:
                        tmp_system_objects.append(sub_obj)
                else:
                    tmp_system_objects.append(obj)
            object_list = tmp_system_objects

        # Set my objects if needed
        if my_objects:
            self.system_objects = object_list

        return object_list

    # OLD METHOD:# This is called to configure the system
    # OLD METHOD:def configure_system(self):

    # OLD METHOD:    # Lets dissolve my groups
    # OLD METHOD:    self.dissolve_groups()

    # OLD METHOD:    # find all the input objects
    # OLD METHOD:    for obj in self.system_objects:
    # OLD METHOD:        if obj.is_independent_variable():
    # OLD METHOD:            # Save the input object
    # OLD METHOD:            self.system_input_objects.add(obj)
    # OLD METHOD:            # Save the output connections
    # OLD METHOD:            self.system_input_connections[obj]={}
    # OLD METHOD:            oc0=self.system_input_connections[obj]
    # OLD METHOD:            # Now create the data structure recursively
    # OLD METHOD:            # loop over the output names
    # OLD METHOD:            for output_name, dest_obj_dict in obj.output_connections.items():
    # OLD METHOD:                if not output_name in oc0:
    # OLD METHOD:                    oc0[output_name]={}
    # OLD METHOD:                oc1=oc0[output_name]
    # OLD METHOD:                # loop over the destination objects
    # OLD METHOD:                for dest_obj, dest_list in dest_obj_dict.items():
    # OLD METHOD:                    if not dest_obj in oc1:
    # OLD METHOD:                        oc1[dest_obj]=[]
    # OLD METHOD:                    oc2=oc1[dest_obj]
    # OLD METHOD:                    # loop over the destination names
    # OLD METHOD:                    for dest_name in dest_list:
    # OLD METHOD:                        oc2.append(dest_name)

    # OLD METHOD:    # check if the lengths are acceptable
    # OLD METHOD:    if len(self.system_objects)==0 or len(self.system_output_objects)==0 or len(self.system_input_objects)==0:
    # OLD METHOD:        raise Exception('The object lists are empty')
    # OLD METHOD:    if not set(self.system_output_objects)<=set(self.system_objects):
    # OLD METHOD:        raise Exception('The output objects are not within the object set')

    # OLD METHOD:    # We will be building the interface
    # OLD METHOD:    self.system_ifc = create_interface()

    # OLD METHOD:    # This will be a map for the input and output variables
    # OLD METHOD:    #     output_map[obj][lcl_var]=sys_var
    # OLD METHOD:    self.system_output_map = {}

    # OLD METHOD:    # now lets collect the output names
    # OLD METHOD:    #
    # OLD METHOD:    # This basically creates a map from local_output_name to an object that has that output and it's meta
    # OLD METHOD:    #    output_data[local_output_name][list_index]=(object, output_meta)
    # OLD METHOD:    #
    # OLD METHOD:    output_data = {}
    # OLD METHOD:    for obj in self.system_output_objects:
    # OLD METHOD:        output_ifc = obj.get_interface()['output']
    # OLD METHOD:        for output_name, output_meta in output_ifc.items():
    # OLD METHOD:            if output_name in output_data.keys():
    # OLD METHOD:                output_data[output_name].append((obj, output_meta))
    # OLD METHOD:            else:
    # OLD METHOD:                output_data[output_name]=[(obj, output_meta)]

    # OLD METHOD:    # Loop through the output data and build the interface
    # OLD METHOD:    #
    # OLD METHOD:    # This basically creates a map from local_output_name to an object that has that output and it's meta
    # OLD METHOD:    #    output_map[output_obj]=([], {})
    # OLD METHOD:    #        # This form is used to get a list of all the outputs from a given object that are needed
    # OLD METHOD:    #        output_map[output_obj][0][list_index]=local_output_name
    # OLD METHOD:    #        # This form shows how the local output name maps to the global output name
    # OLD METHOD:    #        output_map[output_obj][1][local_output_name]=global_output_name
    # OLD METHOD:    #
    # OLD METHOD:    self.system_output_map = {}
    # OLD METHOD:    for output_name, output_list in output_data.items():
    # OLD METHOD:        if len(output_list)==1:
    # OLD METHOD:            output_pair = output_list[0]
    # OLD METHOD:            output_obj = output_pair[0]
    # OLD METHOD:            output_meta = output_pair[1]
    # OLD METHOD:            output_meta['name'] = output_name
    # OLD METHOD:            #print('MIMC adding %s to the output'%(output_name))
    # OLD METHOD:            set_output(self.system_ifc, output_meta)
    # OLD METHOD:            if output_obj in self.system_output_map:
    # OLD METHOD:                self.system_output_map[output_obj][0].append(output_name)
    # OLD METHOD:                self.system_output_map[output_obj][1][output_name]=output_name
    # OLD METHOD:            else:
    # OLD METHOD:                self.system_output_map[output_obj] = ([output_name], {output_name:output_name})
    # OLD METHOD:        else:
    # OLD METHOD:            has_duplicate = False
    # OLD METHOD:            name_dict = {}
    # OLD METHOD:            for output_obj, output_meta in output_list:
    # OLD METHOD:                name = output_obj.object_name+'__'+output_name
    # OLD METHOD:                if name in name_dict.keys():
    # OLD METHOD:                    has_duplicate = True
    # OLD METHOD:                    break
    # OLD METHOD:                name_dict[name] = (output_obj, output_meta)
    # OLD METHOD:            if has_duplicate:
    # OLD METHOD:                name_dict = {}
    # OLD METHOD:                for output_obj, output_meta in output_list:
    # OLD METHOD:                    name = output_obj.object_name+'_'+str(output_obj._hash_value)+'__'+output_name
    # OLD METHOD:                    name_dict[name] = (output_obj, output_meta)
    # OLD METHOD:            for name, output_pair in name_dict.items():
    # OLD METHOD:                output_obj = output_pair[0]
    # OLD METHOD:                output_meta = output_pair[1]
    # OLD METHOD:                output_meta['name'] = name
    # OLD METHOD:                #print('MIMC adding %s to the output'%(name))
    # OLD METHOD:                set_output(self.system_ifc, output_meta)
    # OLD METHOD:                if output_obj in self.system_output_map:
    # OLD METHOD:                    self.system_output_map[output_obj][0].append(output_name)
    # OLD METHOD:                    self.system_output_map[output_obj][1][output_name]=name
    # OLD METHOD:                else:
    # OLD METHOD:                    self.system_output_map[output_obj] = ([output_name], {output_name:name})

    # OLD METHOD:    # Now lets configure the input interface
    # OLD METHOD:    input_data = {}
    # OLD METHOD:    for obj in self.system_input_objects:
    # OLD METHOD:        input_ifc = obj.get_interface()['output']
    # OLD METHOD:        for input_name, input_meta in input_ifc.items():
    # OLD METHOD:            if input_name in input_data.keys():
    # OLD METHOD:                input_data[input_name].append((obj, input_meta))
    # OLD METHOD:            else:
    # OLD METHOD:                input_data[input_name]=[(obj, input_meta)]

    # OLD METHOD:    # Loop through the input data and build the interface
    # OLD METHOD:    self.system_input_map = {}
    # OLD METHOD:    for input_name, input_list in input_data.items():
    # OLD METHOD:        if len(input_list)==1:
    # OLD METHOD:            input_pair = input_list[0]
    # OLD METHOD:            input_obj = input_pair[0]
    # OLD METHOD:            input_meta = input_pair[1]
    # OLD METHOD:            input_meta['name'] = input_name
    # OLD METHOD:            #print('MIMC adding %s to the input'%(input_name))
    # OLD METHOD:            set_input(self.system_ifc, input_meta)
    # OLD METHOD:            if input_obj in self.system_input_map:
    # OLD METHOD:                self.system_input_map[input_obj][input_name]=input_name
    # OLD METHOD:            else:
    # OLD METHOD:                self.system_input_map[input_obj] = {input_name:input_name}
    # OLD METHOD:        else:
    # OLD METHOD:            has_duplicate = False
    # OLD METHOD:            name_dict = {}
    # OLD METHOD:            for input_obj, input_meta in input_list:
    # OLD METHOD:                name = input_obj.object_name+'__'+input_name
    # OLD METHOD:                if name in name_dict.keys():
    # OLD METHOD:                    has_duplicate = True
    # OLD METHOD:                    break
    # OLD METHOD:                name_dict[name] = (input_obj, input_meta)
    # OLD METHOD:            if has_duplicate:
    # OLD METHOD:                name_dict = {}
    # OLD METHOD:                for input_obj, input_meta in input_list:
    # OLD METHOD:                    name = input_obj.object_name+'_'+str(input_obj._hash_value)+'__'+input_name
    # OLD METHOD:                    name_dict[name] = (input_obj, input_meta)
    # OLD METHOD:            for name, input_pair in name_dict.items():
    # OLD METHOD:                input_obj = input_pair[0]
    # OLD METHOD:                input_meta = input_pair[1]
    # OLD METHOD:                input_meta['name'] = name
    # OLD METHOD:                #print('MIMC adding %s to the input'%(name))
    # OLD METHOD:                set_input(self.system_ifc, input_meta)
    # OLD METHOD:                if input_obj in self.system_input_map:
    # OLD METHOD:                    self.system_input_map[input_obj][input_name]=name
    # OLD METHOD:                else:
    # OLD METHOD:                    self.system_input_map[input_obj] = {input_name:name}

    # OLD METHOD:    # Generate the gbl to lcl map for inputs
    # OLD METHOD:    self.system_input_gbl_to_lcl_map={}
    # OLD METHOD:    for obj, name_map in self.system_input_map.items():
    # OLD METHOD:        for lcl_name, gbl_name in name_map.items():
    # OLD METHOD:            self.system_input_gbl_to_lcl_map[gbl_name]=(obj, lcl_name)

    # OLD METHOD:    # Generate the gbl to lcl map for outputs
    # OLD METHOD:    self.system_output_gbl_to_lcl_map={}
    # OLD METHOD:    for obj, output_pair in self.system_output_map.items():
    # OLD METHOD:        name_map = output_pair[1]
    # OLD METHOD:        for lcl_name, gbl_name in name_map.items():
    # OLD METHOD:            self.system_output_gbl_to_lcl_map[gbl_name]=(obj, lcl_name)

    # OLD METHOD:    self.system_has_been_configured = True

    # OLD METHOD:    #print('MIMC after configuration, the interface looks like this:', self.system_ifc)

    def system_set_state_version(self, state_version_in=None):

        # create a new state version if nothing has been specified
        if state_version_in is None:
            state_version_in = FUSED_Object.default_state_version

        # Add a new state version for this sub-system
        self.system_state_version = state_version_in
        for obj in self.system_objects:
            obj.set_state_version(self.system_state_version)

    def get_object_and_local_from_global_input(self, gbl_name):

        if not self.system_has_been_configured:
            self.configure_system()
        return self.system_input_gbl_to_lcl_map[gbl_name][0]

    def get_indep_var_from_global_input(self, gbl_name):

        if not self.system_has_been_configured:
            self.configure_system()
        return self.system_input_gbl_to_lcl_map[gbl_name][1]

    def get_object_and_local_from_global_output(self, gbl_name):

        if not self.system_has_been_configured:
            self.configure_system()
        return self.system_output_gbl_to_lcl_map[gbl_name]

    def get_global_from_object_and_local_input(self, obj, lcl_name):

        if not self.system_has_been_configured:
            self.configure_system()
        return self.system_input_map[obj][lcl_name]

    def get_global_from_object_and_local_output(self, obj, lcl_name):

        if not self.system_has_been_configured:
            self.configure_system()
        return self.system_output_map[obj][lcl_name]

    def get_system_interface(self):

        if not self.system_has_been_configured:
            self.configure_system()
        return self.system_ifc

    # This will tell this system to compute
    def system_compute(self, input_values, output_values):

        # configure if needed
        if not self.system_has_been_configured:
            self.configure_system()

        # First set the inputs on the input objects
        for global_name, input_pair in self.system_input_gbl_to_lcl_map.items():
            if input_pair[1] is None:
                raise Exception('It seems the independent variable was not set')
            input_pair[1].set_data(input_values[global_name])

        # Now collect the output
        for obj, output_map in self.system_output_map.items():
            #output = obj.get_output_value()
            for local_name, global_name in output_map.items():
                output_values[global_name] = obj[local_name]
                #output_values[global_name] = output[local_name]

    # This will find all the objects and the connection information that provides information to this group. 
    # This will find the connections from objects outside this group
    def find_source_connections(self):

        # the result, a dictionary with all the connection information {source_object: {dest_object: (src_dst_map, dst_src_map)}}
        retval = {}

        # loop through all my objects to get my source objects
        for obj in self.system_objects:
            # Look at all the source connections
            for src_obj, src_dst_map in obj.connections.items():
                # test if that object is external
                if not src_obj in self.system_objects:
                    # Build the entries for the retval
                    if not src_obj in retval:
                        retval[src_obj]={obj:({},{})}
                    if not obj in retval[src_obj]:
                        retval[src_obj][obj]=({},{})
                    # Retrieve the maps that we must update
                    system_src_dst_map, system_dst_src_map = retval[src_obj][obj]
                    # Loop through all the source variables
                    for src_name, dst_list in src_dst_map.items():
                        # loop through all the dest variables
                        for dest_name in dst_list:
                            # Add it to the system_dst_src_map
                            if src_name in system_src_dst_map:
                                system_src_dst_map[src_name].append(dest_name)
                            else:
                                system_src_dst_map[src_name]=[dest_name]
                            # Add it to the system_dst_src_map
                            if dest_name in system_dst_src_map:
                                raise Exception('It seems the connection configuration has been corrupted')
                            system_dst_src_map[dest_name]=src_name
                    retval[src_obj][obj]=(system_src_dst_map, system_dst_src_map)

        # Return the result
        return retval

    # This will find all the objects and the connection information that recieves information from this group. 
    # This will find the connections from objects outside this group
    def find_dest_connections(self):

        # the result, a dictionary with all the connection information {dest_object: {source_object: (src_dst_map, dst_src_map)}}
        retval = {}

        # loop through all my objects to get my source objects
        for obj in self.system_objects:
            # Look at all the output connections
            for src_name, dest_dict in obj.output_connections.items():
                # Look at all the dest objects
                for dest_obj, dest_list in dest_dict.items():
                    # test if that object is external
                    if not dest_obj in self.system_objects:
                        # Build the entries for the retval
                        if not dest_obj in retval:
                            retval[dest_obj]={obj:({},{})}
                        if not obj in retval[dest_obj]:
                            retval[dest_obj][obj]=({},{})
                        # Retrieve the maps that we must update
                        system_src_dst_map, system_dst_src_map = retval[dest_obj][obj]
                        # loop through all the dest variables
                        for dest_name in dest_list:
                            # Add it to the system_dst_src_map
                            if src_name in system_src_dst_map:
                                system_src_dst_map[src_name].append(dest_name)
                            else:
                                system_src_dst_map[src_name]=[dest_name]
                            # Add it to the system_dst_src_map
                            if dest_name in system_dst_src_map:
                                raise Exception('It seems the connection configuration has been corrupted')
                            system_dst_src_map[dest_name]=src_name
                        retval[dest_obj][obj]=(system_src_dst_map, system_dst_src_map)

        # Return the result
        return retval

# This is a group based on system base, here the object does not have a seperate StateVersion
class FUSED_Group(FUSED_System_Base):

    # This is the constructor
    def __init__(self, objects_in=[], group_name='unnamed_fused_group'):
        super(FUSED_Group, self).__init__(objects_in=objects_in, group_name=group_name)

    # Identifies whether it is an independent variable
    def is_independent_variable(self):
        return False

    # Identifies if group
    def is_group(self):
        return True

    # This will build the interface
    def get_interface(self):
        return self.get_system_interface()

    # This will print the interface of the group
    def print_interface(self, print_meta=False):
        ifc = self.get_interface()
        print_interface(ifc, print_meta)

    # This will connect the input from an object
    def connect_input_from(self, source_object, var_name_dest=None, var_name_source=None, alias={}):
        self.connect(source_object, var_name_source, var_name_dest, alias)

    # This will connect the output to a certain object
    def connect_output_to(self, dest_object, var_name_source=None, var_name_dest=None, alias={}):

        if not isinstance(dest_object, FUSED_Object):
            for obj in dest_object:
                obj.connect(self, var_name_dest=var_name_dest, var_name_source=var_name_source, alias=alias)
            return

        dest_object.connect(self, var_name_dest=var_name_dest, var_name_source=var_name_source, alias=alias)

    # This will specify connections
    def connect(self, source_object, var_name_dest=None, var_name_source=None, alias={}):

        # If there are multiple source objects
        #########################################################

        if not isinstance(source_object, FUSED_Object) and not isinstance(source_object, FUSED_Group):
            for obj in source_object:
                self.connect(obj, var_name_dest, var_name_source, alias)
            return

        # Now process the connections
        #########################################################
 
        # 1) Parse the arguments
        src_dst_map, dst_src_map = parse_connect_args(self, source_object, var_name_dest, var_name_source, alias)

        # obj, dest name -> src name
        dest_conn_struct = {}

        # 2) for every destination create a dest map
        for dest_name, source_name in dst_src_map.items():
            #   2a) collect the true dest object
            obj_dict = self.get_object_and_local_from_global_input(dest_name)
            # loop through the objects
            for dest_obj, local_dest_list in obj_dict.items():
                if not dest_obj in dest_conn_struct:
                    dest_conn_struct[dest_obj]={}
                for local_dest_name in local_dest_list:
                    dest_conn_struct[dest_obj][local_dest_name] = source_name

        # 3) for every dest object, run connect on that object
        for obj, conn_dict in dest_conn_struct.items():
            obj.connect(source_object, conn_dict)

    # This will call on all objects feeding objects in this group to update their output data
    def collect_input_data(self):

        # Collect the external source objects
        source_dict = self.find_source_connections()

        # Then update the output data for the input object
        for src_obj in source_dict.keys():
            src_obj.update_output_data()

    # This will call objects in this group to update their data
    def update_output_data(self):

        # Loop through each object and update output data
        for obj in self.system_objects:
            obj.update_output_data()

    # This will retrieve a specific variable
    def __getitem__(self, key):

        local_object, local_name = self.get_object_and_local_from_global_output(key)
        return local_object[local_name]

    # This will label all variables as remotely calculated
    def set_as_remotely_calculated(self, at_rank):

        # loop through all objects
        for obj in self.system_objects:
            obj.set_as_remotely_calculated(at_rank)

    # This will synchronize the variables across MPI processes
    def sync_output(self, var_name = None):

        # If it is NONE, then we just loop and sync everything
        if var_name is None:
            for obj in self.system_objects:
                obj.sync_output()
            return

        # In the event that we are transfering a single variable name
        if isinstance(var_name, str):
            var_name = [var_name]

        # This is the data structure that determines all the sync calls to objects
        sync_dict = {}

        # Check if we are suppose to transfer downstream variables
        if '__downstream__' in var_name:
            dest_conn = self.find_dest_connections()
            for dest_obj, src_dict in dest_conn.items():
                for src_obj, map_pair in src_dict.items():
                    if not src_obj in sync_dict:
                        sync_dict[src_obj] = list(map_pair[0].keys())
                    else:
                        for src_name in map_pair[0].keys():
                            if not src_name in sync_dict[src_obj]:
                                sync_dict[src_obj].append(src_name)
            while '__downstream__' in var_name:
                var_name.remove('__downstream__')

        # Now collect the data from simple variable names
        for name in var_name:
            obj, local = self.get_object_and_local_from_global_output(name)
            if obj in sync_dict:
                if not local in sync_dict[obj]:
                    sync_dict[obj].append(local)
            else:
                sync_dict[obj]=[local]

        # Perform the sync
        for obj in sorted(sync_dict.keys()):
            var_list = sync_dict[obj]
            obj.sync_output(var_list)

    # This will retrieve the output values for the group
    def get_output_value(self):

        #print('This method "get_output_value" is going to be deprecated')

        # this is the return value
        retval = {}

        # retrieve the interface
        ifc = self.get_interface()

        # loop over the outputs
        for out_name, out_meta in ifc['output'].items():
            local_object, local_name = self.get_object_and_local_from_global_output(out_name)
            retval[out_name]=local_object[local_name]
            #output_values = local_object.get_output_value()
            #retval[out_name]=output_values[local_name]

        return retval

    # This will set the case runner for all objects
    def set_case_runner(self, case_runner_in=None):

        # Loop through each object and update output data
        for obj in self.system_objects:
            obj.set_case_runner(case_runner_in)

# This is a merge of the FUSED_Object and FUSED_System_Base
class FUSED_System(FUSED_Object, FUSED_System_Base):

    # This is the constructor
    def __init__(self, objects_in=[], object_name_in='unnamed_system_object', state_version_in=None, comm=None):
        FUSED_Object.__init__(self, object_name_in=object_name_in, state_version_in=state_version_in, comm=comm)
        FUSED_System_Base.__init__(self, objects_in=objects_in, group_name=object_name_in)

    # Over-ride the configuration method to create an independent sub-system
    def configure_system(self):
        FUSED_System_Base.configure_system(self)
        self.system_set_state_version(StateVersion())

    # This is the compute method
    def compute(self, input_values, output_values):

        if not self.system_has_been_configured:
            self.configure_system()

        self.system_compute(input_values, output_values)

    # This will build the interface
    def _build_interface(self):

        if not self.system_has_been_configured:
            self.configure_system()
        self.interface = self.get_system_interface()

def obj_list_to_id_set(obj_list):
    retval = set()
    for obj in obj_list:
        retval.add(obj._hash_value)
    return retval

def id_set_to_object_set(id_set, id_obj_map):
    retval = set()
    for hash_value in id_set:
        retval.add(id_obj_map[hash_value])
    return retval

def get_execution_order(objects=None):

    # Retrieve all objects if none are specified
    if objects is None:
        objects = FUSED_Object.get_all_objects()

    # 1) Collect all the information about the objects
    id_obj_map = {}
    dep_dict = {}
    split_set = obj_list_to_id_set(objects)
    for module in objects:
        id_val = module._hash_value
        id_obj_map[id_val]=module
        dep_dict[id_val]=obj_list_to_id_set(module.get_dependency_list())

    # 2) Solve the execution order
    retval = []
    while len(split_set)>0:
        l0 = len(split_set)
        group = set()
        key = 0
        for k in split_set:
            v = dep_dict[k]
            if len(split_set & v)==1:
                group = v
                key = k
                split_set -= {key}
                del dep_dict[key]
                for k, v in dep_dict.items():
                    v -= group
                retval.append(id_obj_map[key])
                break
        if len(split_set)==l0:
            raise Exception('There is a circular dependency')

    return retval

# Solves the split configuration
# It will group the objects into sub-systems
# It will solve the objects within the sub-system that provide output for other sub-systems
# It will solve the objects that accept input from other sub-system objects
def get_split_configuration(split_points):

    # 1) Collect all the modules in this work-flow
    all_modules = set(split_points)
    dep_dict_obj = {}
    for split in split_points:
        new_mods = set(split.get_dependency_list())
        all_modules |= new_mods
        dep_dict_obj[split._hash_value]=new_mods

    # 2) Collect all the information about the objects
    id_obj_map = {}
    dep_dict = {}
    src_dict = {}
    indep_set = set()
    split_set = obj_list_to_id_set(split_points)
    for module in all_modules:
        id_val = module._hash_value
        id_obj_map[id_val]=module
        if id_val in dep_dict_obj.keys():
            dep_dict[id_val]=obj_list_to_id_set(dep_dict_obj[id_val])
        else:
            dep_dict[id_val]=obj_list_to_id_set(module.get_dependency_list())
        src_dict[id_val]=obj_list_to_id_set(module.get_source_list())
        if module.is_independent_variable():
            indep_set.add(id_val)
    
    # 3) update the split_points list
    indep_sub_sys_set = set()
    for iv in indep_set:
        for sp in split_set:
            if iv in dep_dict[sp]:
                indep_sub_sys_set.add(iv)
    all_split_set = split_set|indep_sub_sys_set

    # 4) declare some data structures
    sub_system_groups = {}
    sub_system_output = {}
    sub_system_input = {}
    dep_dict_working = copy.deepcopy(dep_dict)

    # 5) Solve how all the modules are grouped into sub-systems
    split_set_working = copy.deepcopy(all_split_set)
    while len(split_set_working)>0:
        l0 = len(split_set_working)
        group = set()
        key = 0
        for k in split_set_working:
            v = dep_dict_working[k]
            if len(split_set_working & v)==1:
                group = v
                key = k
                sub_system_groups[key] = group
                split_set_working -= {key}
                del dep_dict_working[key]
                del_list = set()
                for k, v in dep_dict_working.items():
                    v -= group
                    if len(v)==0:
                        del_list.add(k)
                for k in del_list:
                    del dep_dict_working[k]
                break
        if len(split_set_working)==l0:
            raise Exception('There is a circular dependency')

    # 6) Solve the modules that provide output from the sub-systems
    # Loop through all sub-system groups
    for ssk, ssm in sub_system_groups.items():
        # Add the top sub-system to that group
        sub_system_output[ssk] = {ssk}
        # loop through each member of sub-system
        for mem in ssm:
            # Loop over only the members that are not yet part of the output list
            if not mem in sub_system_output[ssk]:
                # Loop over the other sub-system groups
                for ssk_up, ssm_up in sub_system_groups.items():
                    keep_looping = True
                    # Loop over only the members that are not part of the upper group
                    if ssk_up != ssk:
                        # loop over every member of that group
                        for mem_up in ssm_up:
                            # check if our member is part of the src set
                            if mem in src_dict[mem_up]:
                                sub_system_output[ssk].add(mem)
                                keep_looping=False
                                break
                    if not keep_looping:
                        break

    # 7) Solve the modules that take input from other sub-systems
    # Loop through all sub-system groups
    for ssk, ssm in sub_system_groups.items():
        sub_system_input[ssk] = {}
        # loop through each member of sub-system
        for mem in ssm:
            # Loop through each source
            for mem_src in src_dict[mem]:
                # Check if not in group
                if not mem_src in ssm:
                    if mem not in sub_system_input[ssk]:
                        sub_system_input[ssk][mem]={mem_src}
                    else:
                        sub_system_input[ssk][mem].add(mem_src)

    # 8) Return the split configuration
    return (id_obj_map, sub_system_groups, sub_system_output, sub_system_input, indep_sub_sys_set)

# This will take the split configuration and actually configure the work-flow according to the result
def split_worflow(split_points):

    # 1-8) Collect the split configuration
    (id_obj_map, sub_system_groups, sub_system_output, sub_system_input, indep_sub_sys_set) = get_split_configuration(split_points)

    #print('=================== MIMC This is the split configuration ===================')
    #print('MIMC id_obj_map:',id_obj_map)
    #print('MIMC sub_system_groups:',sub_system_groups)
    #print('MIMC sub_system_output:',sub_system_output)
    #print('MIMC sub_system_input:',sub_system_input)
    #print('MIMC indep_sub_sys_set:',indep_sub_sys_set)

    # 9) Generate a map from object to sub-system
    object_id_to_sub_system_map = {}
    for sub_system_hash_value, sub_system_group in sub_system_groups.items():
        for member_hash_value in sub_system_group:
            object_id_to_sub_system_map[member_hash_value]=sub_system_hash_value

    #print('=================== MIMC After generating the object id to sub-system map ===================')
    #print('MIMC object_id_to_sub_system_map:',object_id_to_sub_system_map)

    # 10) generate a dictionary of sets for the objects
    sub_system_group_objects = {}
    for hash_value, member_set in sub_system_groups.items():
        sub_system_group_objects[hash_value]=id_set_to_object_set(member_set, id_obj_map)

    #print('=================== MIMC Transforming the system groups from ids to objects ===================')
    #print('MIMC sub_system_group_objects:',sub_system_group_objects)

    # 11) generate a dictionary of sets for the output members
    sub_system_output_objects = {}
    for hash_value, member_set in sub_system_output.items():
        sub_system_output_objects[hash_value]=id_set_to_object_set(member_set, id_obj_map)

    #print('=================== MIMC Transforming the system outputs from ids to objects ===================')
    #print('MIMC sub_system_output_objects:',sub_system_output_objects)

    # 12) build the object input map
    sub_system_object_input_map = {}
    for sub_system_hash_value, sub_system_inputs in sub_system_input.items():
        for dest_hash_value, source_hash_set in sub_system_inputs.items():
            dest_obj = id_obj_map[dest_hash_value]
            for source_hash_value in source_hash_set:
                # Add the inputs
                if not sub_system_hash_value in sub_system_object_input_map.keys():
                    sub_system_object_input_map[sub_system_hash_value] = {}
                if not source_hash_value in sub_system_object_input_map[sub_system_hash_value].keys():
                    sub_system_object_input_map[sub_system_hash_value][source_hash_value] = {}
                sub_system_object_input_map[sub_system_hash_value][source_hash_value][dest_hash_value] = copy.deepcopy(id_obj_map[dest_hash_value].get_connection_with_object(id_obj_map[source_hash_value]))

    #print('=================== MIMC After generating a new input map ===================')
    #print('MIMC sub_system_object_input_map:',sub_system_object_input_map)

    # 13) split the inputs
    old_to_new_dest_object_map = {}
    for sub_system_hash_value, sub_system_inputs in sub_system_input.items():
        for dest_hash_value, source_hash_set in sub_system_inputs.items():
            dest_obj = id_obj_map[dest_hash_value]
            for source_hash_value in source_hash_set:
                # only split if we have multiple objects in a set
                if len(sub_system_group_objects[sub_system_hash_value])>1:
                    # split the connection
                    new_indeps = dest_obj.split_connection(source_hash_value)
                    # add the indeps to the sub system group
                    sub_system_group_objects[sub_system_hash_value] |= new_indeps
                    # Generate the new IO object maps
                    if not dest_hash_value in old_to_new_dest_object_map.keys():
                        old_to_new_dest_object_map[dest_hash_value]={}
                    for indep in new_indeps:
                        old_to_new_dest_object_map[dest_hash_value][indep.get_name()]=indep
    
    #print('=================== MIMC After the input splits, new indeps added to the groups ===================')
    #print('MIMC old_to_new_dest_object_map:',old_to_new_dest_object_map)
    #print('MIMC sub_system_group_objects:',sub_system_group_objects)

    # 14) Now create the models dictionary
    sub_system_models = {}
    sub_system_models_is_system = {}
    for sub_system_hash_value, object_set in sub_system_group_objects.items():
        # if there is only 1 object, then simply pass that object
        if len(sub_system_group_objects[sub_system_hash_value])==1:
            sub_system_models[sub_system_hash_value] = object_set.pop()
            sub_system_models_is_system[sub_system_hash_value] = False
        # build a system
        else:
            sub_system_models[sub_system_hash_value] = FUSED_System(object_set, object_name_in=id_obj_map[sub_system_hash_value].object_name)
            sub_system_models[sub_system_hash_value].add_input_interface_from_independent_variables()
            sub_system_models[sub_system_hash_value].add_output_interface_from_objects(sub_system_output_objects[sub_system_hash_value])
            #print('MIMC adding pdb directive here. Looking at configure syste for system object %s'%(id_obj_map[sub_system_hash_value].object_name))
            #import pdb; pdb.set_trace()
            sub_system_models[sub_system_hash_value].configure_system()
            sub_system_models_is_system[sub_system_hash_value] = True

    #print('=================== MIMC After generating the sub-system models ===================')
    #print('MIMC sub_system_models:',sub_system_models)
    #print('MIMC sub_system_models_is_system:',sub_system_models_is_system)

    # 15) Build the IO map
    sub_system_input_map = {}
    for sub_system_hash_value, sub_system_source_map in sub_system_object_input_map.items():
        for source_hash_value, dest_id_map in sub_system_source_map.items():
            for dest_hash_value, src_dst_map in dest_id_map.items():
                source_sub_system_hash_value = object_id_to_sub_system_map[source_hash_value]
                if not source_sub_system_hash_value in sub_system_input_map.keys():
                    sub_system_input_map[source_sub_system_hash_value] = {}
                if not sub_system_hash_value in sub_system_input_map[source_sub_system_hash_value].keys():
                    sub_system_input_map[source_sub_system_hash_value][sub_system_hash_value] = {}
                for source_lcl_name, dest_lcl_name_list in src_dst_map.items():
                    source_name = source_lcl_name
                    if sub_system_models_is_system[source_sub_system_hash_value]:
                        source_name = sub_system_models[source_sub_system_hash_value].get_global_from_object_and_local_output(id_obj_map[source_hash_value], source_lcl_name)
                    for dest_lcl_name in dest_lcl_name_list:
                        dest_name = dest_lcl_name
                        if sub_system_models_is_system[sub_system_hash_value]:
                            new_obj = old_to_new_dest_object_map[dest_hash_value][dest_lcl_name]
                            dest_name = sub_system_models[sub_system_hash_value].get_global_from_object_and_local_input(new_obj, dest_lcl_name)
                        if source_name in sub_system_input_map[source_sub_system_hash_value][sub_system_hash_value].keys():
                            sub_system_input_map[source_sub_system_hash_value][sub_system_hash_value][source_name].append(dest_name)
                        else:
                            sub_system_input_map[source_sub_system_hash_value][sub_system_hash_value][source_name]=[dest_name]
 
    #print('=================== MIMC Building the input map ===================')
    #print('MIMC sub_system_input_map:', sub_system_input_map)

    # 16) Return the models
    return (sub_system_models, sub_system_input_map)
 
