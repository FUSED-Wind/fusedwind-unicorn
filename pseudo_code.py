
# This is just a pseudo code to design the smart-tip SBO code

########################################################
# First lets create some helpful class's
########################################################

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
	
	def get_state_version(self):

		self.isSet=True
		return self.version

# This is the base class for all objects, it keeps track of 
class MDOBase(object):

	default_state_version = StateVersion()
	print_level = 2
	_object_count = 0

	def __init__(self, object_name_in='unnamed_object', state_version_in=None):

		# This is the name
		self.object_name = object_name_in

		# This is the interface
		self.ifc={'input':{},'output':{}}

		# This is the connection information
		self.conn_dict={}
		self.connections=[]

		# This is the state version
		if state_version_in is None:
			self.state_version = default_state_version
		else:
			self.state_version = state_version_in

		# Ensure these objects can be indeced
		self._hash_value = _object_count
		self._object_count += 1

	def add_input(self, name, meta=None):

		self.ifc['input'][name]=meta

	def add_output(self, name, meta=None):

		self.ifc['output'][name]=meta

	def remove_input(self, name, meta=None):

		del self.ifc['input'][name]

	def remove_output(self, name, meta=None):

		del self.ifc['output'][name]

	def get_interface(self):

		return self.ifc

	def get_output_value(self, var_name=[]):

		raise Exception('The get_output method has not been implemented')

	# The following are used for generating hash keys
	#################################################

	def __cmp__(self, other):
		if self._hash_value == other._hash_value:
			return 0
		if self._hash_value < other._hash_value:
			return -1
		return 1

	def __eq__(self, other):
		return bool(self.__cmp__(self,other)==0)

	def __ne__(self, other):
		return bool(self.__cmp__(self,other)!=0)

	def __lt__(self, other):
		return bool(self.__cmp__(self,other)==-1)

	def __le__(self, other):
		return bool(self.__cmp__(self,other)!=1)

	def __gt__(self, other):
		return bool(self.__cmp__(self,other)==1)

	def __ge__(self, other):
		return bool(self.__cmp__(self,other)!=-1)

	def __hash__(self):
		return self._hash_value

	# This will specify connections
	def connect(self, source_object, var_name_dest=None, var_name_source=None, alias={}):

		# These are the different ways of specifying a connection
		#
		# 1) dst.connect(src)
		#       The assumption here is that all common variables in the interface are connected
		# 2) dst.connect(src, 'var1')
		#       This specifies that var1 in both objects should be connected
		# 3) dst.connect(src, ['var1','var2'])
		#       This specifies that var1 and var2 in both objects should be connected
		# 4) dst.connect(src, 'var1', 'var2')
		#       This specifies that 'var1' in the destination should be connected to 'var2' in the source
		# 5) dst.connect(src, ['var1', 'var3'], ['var2', 'var4'])
		#       This specifies that 'var1' and 'var3' in the destination should be connected to 'var2' and 'var4' in the source respectively
		# 6) dst.connect(src, {'var1':'var2', 'var3':'var4'})
		#       This specifies that 'var1' and 'var3' in the destination should be connected to 'var2' and 'var4' in the source respectively
		# 7) dst.connect(src, ..., alias={'var1':'var2', 'var3':'var4'})
		#       This specifies that 'var1' and 'var3' in the destination is equivalent to 'var2' and 'var4' in the source respectively.
		#       This does not mean they are connected though, that depends on the other arguments
		#       This is used in cases where the naming scheme differs for a small group of variables, and the connections are not explicit

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
			for k,v in alias.iteritems():
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
			elif isinstance(var_name_source, basestring):
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
		elif isinstance(var_name_dest, basestring):
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
			elif isinstance(var_name_source, basestring):
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
				for k, v in var_name_dest.iteritems():
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
			elif isinstance(var_name_source, basestring):
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
			if self.print_messages>=2:
				print 'There appears to be no valid connection between', self.object_name, 'and', source_object.object_name
			return

		# Now we build the connection data structure
		############################################

		if source in self.connections:
			for new_source_name, new_dst_list in src_dst_map.iteritems():
				if new_source_name in self.connections[source]:
					# Merge in the new connections
					for new_dst_name in new_dst_list:
						if not new_dst_name in self.connections[source][new_source_name]:
							self.connections[source][new_source_name].append(new_dst_name)
						elif self.print_messages>=2:
							print 'The connection between', self.object_name, 'and', source_object.object_name, 'connected', new_dst_name, 'and', new_source_name, 'again'
				else:
					self.connections[source][new_source_name]=new_dst_list
		else:
			self.connections[source]=src_dst_map

		# this will then update the connection dictionaries
		###################################################

		# Basically, the task is to ensure that each destination variable has a unique source
		for dst_name in dst_src_map.keys():
			# remove any variables from old connections
			if dst_name in self.conn_dict:
				if self.print_messages>=2:
					print 'Connecting variable', dst_name, 'replaces the old connection between', self.object_name, 'and', self.conn_dict[v].object_name
				tmp_source_object = self.conn_dict[dst_name][0]
				tmp_source_name = self.conn_dict[dst_name][1]
				tmp_src_dst_map = self.connections[tmp_source_object]
				# look for the dst_name within the values of that map
				tmp_dst_list=tmp_src_dst_map[tmp_source_name]:
				if dst_name in tmp_dst_list:
					tmp_dst_list.remove(dst_name)
				else:
					raise Exception('Connection data structure corrupted')
				# if there is nothing left in the list, the just remove it
				if len(tmp_dst_list)==0:
					tmp_src_dst_map.pop(tmp_src_name)
				# If the connection is totally over-ridden then just remove it
				if len(tmp_src_dst_map)==0:
					self.connections.pop(tmp_source_object)
			# Add the new variable to the conn_dict
			self.conn_dict[dst_name]=(source, dst_src_map[dst_name])

	def _build_input_vector(self):

		# Loop through all connections and collect the data
		retval = {}
		for obj, src_dst_map in self.connections.iteritems():
			output = obj.get_output_value(src_dst_map.keys())
			for src_name, dst_list in src_dst_map.iteritems():
				for dst_name in dst_list:
					retval[dst_name]=output[src_name]

		# return the results
		return retval


# independent variable class is used to represent a source object for a calculation chain
class Independent_Variable(MDOBase):

	''' This represents a source to a calculation '''

	def __init__(self, var_name_in='unnamed_variable', object_name_in='unnamed_object'):

		self.name = var_name_in
		self.add_input(var_name_in)
		self.data = None
		self.retval = {self.name:self.data}

	def set_data(self, data_in):

		self.state_version.modifying_state()
		self.data = data_in
		self.retval = {self.name:self.data}

	def get_output_value(self, var_name=[]):

		# quick check for errors
		if len(var_name)>1 or (len(var_name)==1 and var_name[0]!=self.name):
			raise Exception('That name does not exist')

		return self.retval

# spline modules are used to transform control points to distributed quantities
###############################################################################

# There are two class's the first is one instance of a spline solution, the second is the actual spline data

# This is the spline solution.
# It stores the grid that defines where spline data should be collected from
# It will simply go to the spline module and collect that data
class SplineSolution(MDOBase):

	# This is the constructor
	def __init__(self, spline_module_in, grid_in, var_name_in='unnamed_variable', object_name_in='unnamed_object'):
		super(SplineSolution, self).__init__(object_name_in)

		self.spline_module=spline_module_in
		self.grid=grid_in
		self.name=var_name_in

	# This will retrieve the data by accessing the spline_module and calculating with the grid
	def get_output_value(self, var_name=[]):
		
		return self.spline_module.get_spline_value(self.grid)

# This is the spline module. It takes control points and 
class SplineModuleBase(MDOBase):

	def __init__(self, var_name_in='unnamed_variable', object_name_in='unnamed_object'):
		super(SplineModule, self).__init__(object_name_in)
		self.var_name=var_name_in

	# This will return a spline object
	def get_spline_solution(self, grid_in, var_name_in=None, spline_solution_name_in=None):
		
		if var_name_in is None:
			var_name_in=self.var_name

		if spline_solution_name_in is None:
			spline_solution_name_in=self.var_name

		# create the spline
		retval = SplineSolution(self, grid_in, var_name_in, spline_solution_name_in)

		# connect the objects
		retval.connect(self)

		return retval

	# This will retrieve the value for the spline
	def get_spline_value(self, grid):
		pass

# This is an example of a spline module based on piece-wise linear functions
class SplineModule_Constant(SplineModuleBase):

	def __init__(self, , var_name_in='unnamed_variable', object_name_in='unnamed_object'):

# Then a simulation module is used to 

# The starting point is an array of design variables
x = np.ndarray(design_vector)
