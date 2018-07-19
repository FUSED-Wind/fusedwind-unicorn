
import numpy as np

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

	# Retrieve the current state version
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
		self.connections={}

		# This is the state version
		if state_version_in is None:
			self.state_version = self.default_state_version
		else:
			self.state_version = state_version_in
		self.my_state_version = 0

		# Ensure these objects can be indeced
		self._hash_value = self._object_count
		self._object_count += 1
	
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

	def add_input(self, name, meta=None):

		self.ifc['input'][name]=meta

	def add_output(self, name, meta=None):

		self.ifc['output'][name]=meta

	def remove_input(self, name, meta=None):

		retval=self.ifc['input'][name]
		del self.ifc['input'][name]
		return retval

	def remove_output(self, name, meta=None):

		retval=self.ifc['output'][name]
		del self.ifc['output'][name]
		return retval

	def get_interface(self):

		self._build_interface()
		return self.ifc

	# This is the output method
	###########################

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
			if self.print_level>=2:
				print 'There appears to be no valid connection between', self.object_name, 'and', source_object.object_name
			return

		# Now we build the connection data structure
		############################################

		if source_object in self.connections:
			for new_source_name, new_dst_list in src_dst_map.iteritems():
				if new_source_name in self.connections[source_object]:
					# Merge in the new connections
					for new_dst_name in new_dst_list:
						if not new_dst_name in self.connections[source_object][new_source_name]:
							self.connections[source_object][new_source_name].append(new_dst_name)
						elif self.print_level>=2:
							print 'The connection between', self.object_name, 'and', source_object.object_name, 'connected', new_dst_name, 'and', new_source_name, 'again'
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
				if self.print_messages>=2:
					print 'Connecting variable', dst_name, 'replaces the old connection between', self.object_name, 'and', self.conn_dict[v].object_name
				tmp_source_object = self.conn_dict[dst_name][0]
				tmp_source_name = self.conn_dict[dst_name][1]
				tmp_src_dst_map = self.connections[tmp_source_object]
				# look for the dst_name within the values of that map
				tmp_dst_list=tmp_src_dst_map[tmp_source_name]
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
			self.conn_dict[dst_name]=(source_object, dst_src_map[dst_name])

	# This is for collecting the input data from connections
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

	def __init__(self, data_in=None, var_name_in='unnamed_variable', object_name_in='unnamed_object', state_version_in=None):
		super(Independent_Variable, self).__init__(object_name_in, state_version_in)

		self.name = var_name_in
		self.add_output(var_name_in)
		self.data = None
		self.retval = {self.name:self.data}
		if not data_in is None:
			self.set_data(data_in)

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

# So the basic idea of the spline modules is that we have a given quantity that is distributed continuously over a grid
# Now computers cannot represent such data. They are ussually a set of control points and equations.
# The result of the spline is another set of discrete data
# So this is a situation where there is one set of discrete inputs, the can produce multiple sets of discrete output, where each output is a different grid
# So the solution is one instance of output, the spline module is a single spline that produces all the output
# In this framework, the spline module creates spline solution. However, the spline solution is a component that takes control points to produce results
# ... however, all solutions from the same module have the same input configuration.
# So a call to connect on one is a call to connect on all... the base class facilitates this.

# This is the spline solution.
# It stores the grid that defines where spline data should be collected from
# It will simply go to the spline module and collect that data
class SplineSolutionBase(MDOBase):

	# This is the constructor
	def __init__(self, spline_module_in, var_name_in='unnamed_spline_solution', object_name_in='unnamed_spline_solution_object'):
		super(SplineSolutionBase, self).__init__(object_name_in)

		self.spline_module=spline_module_in
		self.var_name=var_name_in

	# Now we over-ride the connect method so all solutions are connected
	def connect(self, source_object, var_name_dest=None, var_name_source=None, alias={}):

		self.spline_module.connect(source_object, var_name_dest, var_name_source, alias)
	
	# This is the connect method that is called from the spline module
	def _connect(self, source_object, var_name_dest=None, var_name_source=None, alias={}):

		MDOBase.connect(self, source_object, var_name_dest, var_name_source, alias)

# This is the spline module. It takes control points and 
class SplineModuleBase(MDOBase):

	def __init__(self, var_name_in='unnamed_spline_module_control_points', object_name_in='unnamed_spline_module_object'):
		super(SplineModuleBase, self).__init__(object_name_in)
		self.var_name=var_name_in
		self.connect_log = []
		self.solution_list = []

	# This will return a spline object
	def get_spline_solution(self, grid_in, var_name_in=None, spline_solution_name_in=None):

		raise Exception('The method has not been implemented')
		
	# this will configure a spline solution
	def _configure_solution(self, solution):

		for conn in self.connect_log:
			solution._connect(conn[0], conn[1], conn[2], conn[3])
		self.solution_list.append(solution)

	# This will call the connect method on all solutions
	def connect(self, source_object, var_name_dest=None, var_name_source=None, alias={}):

		MDOBase.connect(self, source_object, var_name_dest, var_name_source, alias)
		for solution in self.solution_list:
			solution._connect(source_object, var_name_dest, var_name_source, alias)
		self.connect_log.append( (source_object, var_name_dest, var_name_source, alias) )

# This is the spline solution for a piece-wise linear curve
class SplineSolution_PiecewiseLinear(SplineSolutionBase):

	def __init__(self, spline_module_in, grid_in, var_name_in='unnamed_piecewise_spline_values', object_name_in='unnamed_piecewise_spline_solution_object'):
		super(SplineSolution_PiecewiseLinear, self).__init__(spline_module_in, var_name_in, object_name_in)

		self.ifc_built = False
		self.set_grid(grid_in)

	def _build_interface(self):

		if not self.ifc_built:
			self.add_output(self.var_name)
			self.spline_module._build_interface()
			self.ifc_built=True

	def _build_input_interface(self):

		cp_name = self.spline_module.var_name
		self.add_input(cp_name)

	# This will set the name of the input
	def set_spline_name(self, var_name_in='unnamed_piecewise_linear_spline_control_point_variable'):

		# get the old name
		old_name = self.var_name
		self.var_name = var_name_in

		if self.ifc_built:
			old_meta=self.remove_output(old_name)
			self.add_output(self.var_name, old_meta)

	def get_output_value(self, var_name_in=[]):

		if len(var_name_in)>1 or (len(var_name_in)==1 and var_name_in[0]!=self.var_name):
			msg = "The variables requested do not match those in this spline solution"
			raise Exception(msg)

		self.spline_module._update_control_points()
		self.cps=self.spline_module.cps
		values=np.zeros(len(self.grid))
		retval={self.var_name:values}

		if self.cps.size==1:
			for I in range(0,retval.size):
				values[I]=self.cps[0]
		else:
			for I in range(0,len(self.index_list)):
				values[I]=self.weight_list[I]*self.cps[self.index_list[I]]+(1.0-self.weight_list[I])*self.cps[self.index_list[I]+1]

		return retval

	# This will update the grid
	def set_grid(self, grid_in):

		self.grid=grid_in
		self._update_cp_grid()

	# This is called by the spline module to update the blend coefficients
	def _update_cp_grid(self):
		
		self.index_list = []
		self.weight_list = []
		cp_grid=self.spline_module.cp_grid
		if cp_grid.size>1:
			for I in range(0, self.grid.size):
				grid_value = self.grid[I]
				J = 0
				while (J+1)<=cp_grid.size and cp_grid[J+1]<grid_value:
					J+=1
				self.index_list.append(J)
				w=(cp_grid[J+1]-grid_value)/(cp_grid[J+1]-cp_grid[J])
				self.weight_list.append(w)


# This is an example of a spline module based on piece-wise linear functions
class SplineModule_PiecewiseLinear(SplineModuleBase):

	def __init__(self, cp_grid_in, var_name_in='unnamed_piecewise_linear_spline_control_point_variable', object_name_in='unnamed_piecewise_linear_spline_object'):
		super(SplineModule_PiecewiseLinear, self).__init__(var_name_in, object_name_in)

		self.ifc_built = False
		self.set_cp_grid(cp_grid_in)

	# This will generate a new spline solution
	def get_spline_solution(self, grid_in, var_name_in='unnamed_piecewise_spline_values', spline_solution_name_in='unnamed_piecewise_spline_solution_object'):

		retval = SplineSolution_PiecewiseLinear(self, grid_in, var_name_in, spline_solution_name_in)
		self._configure_solution(retval)
		return retval

	# This will set the new spline control points
	def set_cp_grid(self, cp_grid_in):

		self.cp_grid = cp_grid_in
		for solution in self.solution_list:
			solution._update_cp_grid()

	# This will set the name of the input
	def set_control_point_name(self, var_name_in='unnamed_piecewise_linear_spline_control_point_variable'):

		# get the old name
		old_name = self.var_name
		self.var_name = var_name_in

		if self.ifc_built:
			old_meta=self.remove_input(old_name)
			self.add_input(self.var_name, old_meta)

	# This is called by solutions to inform the module to collect the control points
	def _update_control_points(self):

		# collect the new control points if needed
		if self._update_needed():
			# Build the new input
			input_val = self._build_input_vector()
			# store the values at a fixed variable
			self.cps = input_val[self.var_name]
			# Save state version, to avoid duplicat calculations
			self._updating_data()

	# This is a method that will trigger the construction of the input interface
	def _build_interface(self):

		if not self.ifc_built:
			self.add_input(self.var_name)
			for solution in self.solution_list:
				solution._build_input_interface()
			self.ifc_built = True

# Then lets, have a dummy simulation module
class DummySimulation(MDOBase):

	def __init__(self, object_name_in='unnamed_dummy_object'):
		super(DummySimulation,self).__init__(object_name_in)
		self.add_input('input_value')
		self.add_output('sum_value')

	def get_output_value(self, var_name=[]):

		input_val = self._build_input_vector()
		retval={}
		retval['sum_value']=np.sum(input_val['input_value'])
		return retval

if __name__ == "__main__":

	print 'Starting the example'

	# This is a simple example of a work-flow
	# Basically it is a simple spline that takes control point values
	# Then there are 3 simulations, that basically sum the spline values.
	# The difference is that the 3 simulations have 3 different sample locations from the spline
	# The example builds the work-flow and accesses all the data from the 3 spline solutions and 3 simulations

	# Prepare some data to work and example around
	##############################################

	# The names in this problem
	cp_name='cp_values'
	spline_name='spline_values'
	sim_input_name='input_value'
	sim_output_name='sum_value'

	# create an alias between spline and dummy simulation
	ds_alias={sim_input_name:spline_name}

	# The grids and control point values
	cp_grid_list=[0.0, 1.0, 2.0]
	grid1_list=[0.25, 0.75]
	grid2_list=[0.5 , 1.5 ]
	grid3_list=[1.25, 1.75]
	cp_values_list=[1.0,2.0,3.0]

	# construct the arrays
	cp_grid = np.array(cp_grid_list)
	grid1 = np.array(grid1_list)
	grid2 = np.array(grid2_list)
	grid3 = np.array(grid3_list)
	cp_values = np.array(cp_values_list)

	# Now lets build our work-flow objects
	######################################

	# now lets create an independent variable
	cp = Independent_Variable(cp_values, cp_name, 'cp_source_object')

	# now lets create a spline module and some solutions
	sm = SplineModule_PiecewiseLinear(cp_grid, cp_name, 'spline_module')
	ss1 = sm.get_spline_solution(grid1, spline_name, 'spline_solution_1')
	ss2 = sm.get_spline_solution(grid2, spline_name, 'spline_solution_2')
	ss3 = sm.get_spline_solution(grid3, spline_name, 'spline_solution_3')

	# Then this is the dummy simulation
	ds1=DummySimulation('dummy_simulation_1')
	ds2=DummySimulation('dummy_simulation_2')
	ds3=DummySimulation('dummy_simulation_3')

	# connect things in our work-flow
	#################################

	# Auto-connect
	sm.connect(cp)
	# Auto-connect with alias
	ds1.connect(ss1, alias=ds_alias)
	# Explicit connection
	ds2.connect(ss2, sim_input_name, spline_name)
	# Explicit connection with alias
	ds3.connect(ss3, sim_input_name, alias=ds_alias)

	# Now lets start to access the results
	######################################

	# access the spline solution
	output1 = ss1.get_output_value([spline_name])
	output2 = ss2.get_output_value([spline_name])
	output3 = ss3.get_output_value([spline_name])

	# now lets print the answer
	print 'About to print the spline output'
	print output1
	print output2
	print output3

	# access the dummy simulation
	output1 = ds1.get_output_value([sim_output_name])
	output2 = ds2.get_output_value([sim_output_name])
	output3 = ds3.get_output_value([sim_output_name])

	# now lets print the answer
	print 'About to print the simulation output, should be 3, 4, 5'
	print output1
	print output2
	print output3

