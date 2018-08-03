import numpy as np
import copy

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

'''
# Consider adding to simplify including inputs into FUSED_Objects
def fusedvar(name,val,desc='',shape=None):

    return {'name' : name, 'val' : val, 'desc' : desc, 'shape' : shape}
'''

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

# The following are helper functions to help objects implement interfaces
#########################################################################

class FUSED_Object(object):

    default_state_version = StateVersion()
    print_level = 0
    _object_count = 0
    all_objects = []

    def __init__(self, object_name_in='unnamed_object', state_version_in=None):

        super(FUSED_Object,self).__init__()

        # This is the name of the object. Useful for printing helpful messages
        self.object_name = object_name_in

        # This is the interface
        self.ifc_built = False
        self.interface = create_interface()

		# Variables for a default input vector
		self.is_default_input_built = False
		self.default_input = {}

        # This is the connection information
        self.conn_dict={}
        self.connections={}

        # This is the state version
        if state_version_in is None:
            self.state_version = FUSED_Object.default_state_version
        else:
            self.state_version = state_version_in
        self.my_state_version = 0
        self.var_state_version = {}
        self.output_values = {}

        # Ensure these objects can be indeced
        self._hash_value = FUSED_Object._object_count
        FUSED_Object._object_count += 1
        FUSED_Object.all_objects.append(self)

    # This is for solving some properties of a work-flow
    ####################################################

    @staticmethod
    def get_all_objects():

        return FUSED_Object.all_objects

    # Identifies whether it is an independent variable
    ##################################################

    def is_independent_variable(self):
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

    def _update_needed(self, var_list=[]):

        if len(var_list)==0:
            return self.my_state_version!=self.state_version.get_state_version()
        else:
            # loop through the variables and find what needs to be updated
            var_list_out = []
            for var in var_list:
                if not var in self.var_state_version:
                    self.var_state_version[var]=0
                if self.var_state_version[var]!=self.state_version.get_state_version():
                    var_list_out.append(var)
            return var_list_out

    def _updating_data(self, var_list=[]):

        if len(var_list)==0:
            self.my_state_version=self.state_version.get_state_version()
        else:
            # loop through the variables and register what was updated
            for var in var_list:
                self.var_state_version[var]=self.state_version.get_state_version()

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

    # This is the all important connect method. It tells where to get the input data from
    #####################################################################################

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

        # First task is to create maps between the variable names
        #########################################################

        # Retrieve the interfaces
        dst_ifc = self.get_interface()
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

        # Now we build the connection data structure
        ############################################

        if source_object in self.connections:
            for new_source_name, new_dst_list in src_dst_map.items():
                if new_source_name in self.connections[source_object]:
                    # Merge in the new connections
                    for new_dst_name in new_dst_list:
                        if not new_dst_name in self.connections[source_object][new_source_name]:
                            self.connections[source_object][new_source_name].append(new_dst_name)
                        elif FUSED_Object.print_level>=2:
                            print('The connection between '+self.object_name+' and '+source_object.object_name+' connected '+new_dst_name+' and '+new_source_name+' again')
                else:
                    self.connections[source_object][new_source_name]=new_dst_list
        else:
            self.connections[source_object]=src_dst_map

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
                # Remove the over-ride from the connections
                if dst_name in self.connections[tmp_source_object][tmp_src_name]:
                    self.connections[tmp_source_object][tmp_src_name].remove(dst_name)
                    # Remove that source name, if it is no longer used
                    if len(self.connections[tmp_source_object][tmp_src_name])==0:
                        del self.connections[tmp_source_object][tmp_src_name]
                        # Remove the source object, if it is no longer used
                        if len(self.connections[tmp_source_object]):
                            del self.connections[tmp_source_object]
                else:
                    raise Exception('Connection data structure corrupted')
            # Add the new variable to the conn_dict
            self.conn_dict[dst_name]=(source_object, dst_src_map[dst_name])

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

    # This is for collecting the input data from connections
    def _build_input_vector(self):

		# Collect the default input values
		if not self.is_default_input_built:
			ifc = self.get_interface()
			for name, meta in ifc['input'].items():
				if 'val' in meta:
					self.default_input[name]=meta['val']
				elif 'shape' in meta:
					self.default_input[name]=np.zeros(meta['shape'])
			self.is_default_input_built = True

        # Loop through all connections and collect the data
        retval = copy.copy(self.default_input)
        for obj, src_dst_map in self.connections.items():
            output = obj.get_output_value(list(src_dst_map.keys()))
            for src_name, dst_list in src_dst_map.items():
                for dst_name in dst_list:
                    retval[dst_name]=output[src_name]

        # return the results
        return retval

    # This is the calculation method that is called
    def compute(self, input_values, output_values, var_name=[]):

        raise Exception('The _calculate_output method has not been implemented')

    def get_output_value(self, var_name=[]):

        ans = self._update_needed(var_name)
        if (len(var_name)==0 and ans) or (len(var_name)!=0 and len(ans)!=0):
            input_values = self._build_input_vector()
            if len(var_name)==0:
                self.compute(input_values, self.output_values)
                self._updating_data()
            else:
                self.compute(input_values, self.output_values, ans)
                self._updating_data(ans)
        return self.output_values

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

    def get_output_value(self, var_name=[]):

        # quick check for errors
        if len(var_name)>1 or (len(var_name)==1 and var_name[0]!=self.name):
            raise Exception('That name does not exist')

        return self.retval

# The following function will solve the split configuration of a work-flow
##########################################################################

class FUSED_System(object):

    def __init__(self, objects_in, output_objects_in):
        super(FUSED_System, self).__init__()

        # basically store the objects
        self.objects = objects_in
        self.output_objects = output_objects_in
        self.input_objects = set()

        # find all the input objects
        for obj in self.objects:
            if obj.is_independent_variable():
                self.input_objects.add(obj)

        # check if the lengths are acceptable
        if len(self.objects)==0 or len(self.output_objects)==0 or len(self.input_objects)==0:
            raise Exception('The object lists are empty')
        if not set(self.output_objects)<=set(self.objects):
            raise Exception('The output objects are not within the object set')

        # We will be building the interface
        self.ifc = create_interface()

        # This will be a map for the input and output variables
        #     *put_map[obj][lcl_var]=sys_var
        self.output_map = {}

        # now lets collect the output names
        output_data = {}
        for obj in self.output_objects:
            output_ifc = obj.get_interface()['output']
            for output_name, output_meta in output_ifc.items():
                if output_name in output_data.keys():
                    output_data[output_name].append((obj, output_meta))
                else:
                    output_data[output_name]=[(obj, output_meta)]
        # Loop through the output data and build the inteface
        self.output_map = {}
        for output_name, output_list in output_data.items():
            if len(output_list)==1:
                output_pair = output_list[0]
                output_obj = output_pair[0]
                output_meta = output_pair[1]
                output_meta['name'] = output_name
                set_output(self.ifc, output_meta)
                if output_obj in self.output_map:
                    self.output_map[output_obj][0].append(output_name)
                    self.output_map[output_obj][1][output_name]=output_name
                else:
                    self.output_map[output_obj] = ([output_name], {output_name:output_name})
            else:
                has_duplicate = False
                name_dict = {}
                for output_obj, output_meta in output_list:
                    name = output_obj.object_name+'>'+output_name
                    if name in name_dict.keys():
                        has_duplicate = True
                        break
                    name_dict[name] = (output_obj, output_meta)
                if has_duplicate:
                    name_dict = {}
                    for output_obj, output_meta in output_list:
                        name = output_obj.object_name+'_'+str(output_obj._hash_value)+'>'+output_name
                        name_dict[name] = (output_obj, output_meta)
                for name, output_pair in name_dict.items():
                    output_obj = output_pair[0]
                    output_meta = output_pair[1]
                    output_meta['name'] = name
                    set_output(self.ifc, output_meta)
                    if output_obj in self.output_map:
                        self.output_map[output_obj][0].append(output_name)
                        self.output_map[output_obj][1][output_name]=name
                    else:
                        self.output_map[output_obj] = ([output_name], {output_name:name})

        # Now lets configure the input interface
        input_data = {}
        for obj in self.input_objects:
            input_ifc = obj.get_interface()['output']
            for input_name, input_meta in input_ifc.items():
                if input_name in input_data.keys():
                    input_data[input_name].append((obj, input_meta))
                else:
                    input_data[input_name]=[(obj, input_meta)]
        # Loop through the input data and build the inteface
        self.input_map = {}
        for input_name, input_list in input_data.items():
            if len(input_list)==1:
                input_pair = input_list[0]
                input_obj = input_pair[0]
                input_meta = input_pair[1]
                input_meta['name'] = input_name
                set_input(self.ifc, input_meta)
                if input_obj in self.input_map:
                    self.input_map[input_obj][input_name]=input_name
                else:
                    self.input_map[input_obj] = {input_name:input_name}
            else:
                has_duplicate = False
                name_dict = {}
                for input_obj, input_meta in input_list:
                    name = input_obj.object_name+'>'+input_name
                    if name in name_dict.keys():
                        has_duplicate = True
                        break
                    name_dict[name] = (input_obj, input_meta)
                if has_duplicate:
                    name_dict = {}
                    for input_obj, input_meta in input_list:
                        name = input_obj.object_name+'_'+str(input_obj._hash_value)+'>'+input_name
                        name_dict[name] = (input_obj, input_meta)
                for name, input_pair in name_dict.items():
                    input_obj = input_pair[0]
                    input_meta = input_pair[1]
                    input_meta['name'] = name
                    set_input(self.ifc, input_meta)
                    if input_obj in self.input_map:
                        self.input_map[input_obj][input_name]=name
                    else:
                        self.input_map[input_obj] = {input_name:name}

        # Generate the gbl to lcl map for inputs
        self.input_gbl_to_lcl_map={}
        for obj, name_map in self.input_map.items():
            for lcl_name, gbl_name in name_map.items():
                self.input_gbl_to_lcl_map[gbl_name]=(obj, lcl_name)

        # Generate the gbl to lcl map for outputs
        self.output_gbl_to_lcl_map={}
        for obj, output_pair in self.output_map.items():
            name_map = output_pair[1]
            for lcl_name, gbl_name in name_map.items():
                self.output_gbl_to_lcl_map[gbl_name]=(obj, lcl_name)

        # Add a new state version for this sub-system
        self.state_version = StateVersion()
        for obj in self.objects:
            obj.set_state_version(self.state_version)

    def get_object_and_local_from_global_input(self, gbl_name):

        return self.input_gbl_to_lcl_map[gbl_name]

    def get_object_and_local_from_global_output(self, gbl_name):

        return self.output_gbl_to_lcl_map[gbl_name]

    def get_global_from_object_and_local_input(self, obj, lcl_name):

        return self.input_map[obj][lcl_name]

    def get_global_from_object_and_local_output(self, obj, lcl_name):

        return self.output_map[obj][1][lcl_name]

    def get_interface(self):

        return self.ifc

    def compute(self, input_values, output_values):

        # First set the inputs on the input objects
        for obj, name_map in self.input_map.items():
            if len(name_map)!=1:
                raise Exception('Currently, the expectation is that independent variable objects only contain 1 variable')
            for local_name, global_name in name_map.items():
                obj.set_data(input_values[global_name])

        # Now collect the output
        for obj, name_pair in self.output_map.items():
            output = obj.get_output_value(name_pair[0])
            output_map = name_pair[1]
            for local_name, global_name in output_map.items():
                output_values[global_name] = output[local_name]

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
# It will solve the objects that accept intput from other sub-system objects
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
            sub_system_models[sub_system_hash_value] = FUSED_System(object_set, sub_system_output_objects[sub_system_hash_value])
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
 
