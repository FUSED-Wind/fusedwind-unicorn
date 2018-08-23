import numpy as np
from fusedwind.fused_wind import FUSED_Object

# spline modules are used to transform control points to distributed quantities
###############################################################################

# There are two class's the first is one instance of a spline solution, the second is the actual spline data

# So the basic idea of the spline modules is that we have a given quantity that is distributed continuously over a grid
# Now computers cannot represent such data. They are ussually a set of control points and equations.
# The result of the spline is another set of discrete data
# So this is a situation where there is one set of discrete inputs, the can produce multiple sets of discrete output, where each output is a different grid
# So the 'solution' is one instance of output, the 'spline module' is a single spline that produces all the output
# In this framework, the spline module creates spline solutions. However, the spline solution is a component that takes control points to produce results
# ... however, all solutions from the same module have the same input configuration.
#
# The connection diagram is below:
#
#                                     /-> spline-solution-1
#       cp-source -> spline-module --+--> spline-solution-2
#                                     \-> spline-solution-3
#
#

# This is the spline solution.
# It stores the grid that defines where spline data should be collected from
# It will simply go to the spline module and collect that data
class SplineSolutionBase(FUSED_Object):

    # This is the constructor
    def __init__(self, spline_module_in, var_name_in='unnamed_spline_solution', object_name_in='unnamed_spline_solution_object', state_version_in=None):
        super(SplineSolutionBase, self).__init__(object_name_in, state_version_in)

        self.spline_module=spline_module_in
        self.var_name=var_name_in

    def _build_interface(self):

        # collect the input
        control_point_name=self.spline_module.var_name
        self.add_input(control_point_name)
        src_dst_map={control_point_name:[control_point_name]}
        dst_src_map={control_point_name:control_point_name}
        self._add_connection(self.spline_module,src_dst_map,dst_src_map)

# This is the spline module. It takes control points and 
class SplineModuleBase(FUSED_Object):

    def __init__(self, var_name_in='unnamed_spline_module_control_points', object_name_in='unnamed_spline_module_object', state_version_in=None):
        super(SplineModuleBase, self).__init__(object_name_in, state_version_in)
        self.var_name=var_name_in
        self.solution_list = []

    # This will return a spline object
    def get_spline_solution(self, grid_in, var_name_in=None, spline_solution_name_in=None):

        raise Exception('The method has not been implemented')
        
    # this will configure a spline solution
    def _configure_solution(self, solution):

        self.solution_list.append(solution)
        solution.set_state_version(self.state_version)

# This is the spline solution for a piece-wise linear curve
class SplineSolution_PiecewiseLinear(SplineSolutionBase):

    def __init__(self, spline_module_in, grid_in, var_name_in='unnamed_piecewise_spline_values', object_name_in='unnamed_piecewise_spline_solution_object', state_version_in=None):
        super(SplineSolution_PiecewiseLinear, self).__init__(spline_module_in, var_name_in, object_name_in)

        self.set_grid(grid_in)

    def _build_interface(self):
        super(SplineSolution_PiecewiseLinear, self)._build_interface()
        self.add_output(self.var_name)

    # This will set the name of the input
    def set_spline_name(self, var_name_in='unnamed_piecewise_linear_spline_control_point_variable'):

        # get the old name
        old_name = self.var_name
        self.var_name = var_name_in

        if self.ifc_built:
            old_meta=self.remove_output(old_name)
            self.add_output(self.var_name, old_meta)

    def compute(self, input_values, output_values, var_name_in=[]):

        if len(var_name_in)>1 or (len(var_name_in)==1 and var_name_in[0]!=self.var_name):
            msg = "The variables requested do not match those in this spline solution"
            raise Exception(msg)

        control_point_name=self.spline_module.var_name
        values=np.zeros(len(self.grid))
        output_values[self.var_name]=values
        cps=input_values[control_point_name]

        if cps.size==1:
            for I in range(0,retval.size):
                values[I]=cps[0]
        else:
            for I in range(0,len(self.index_list)):
                values[I]=self.weight_list[I]*cps[self.index_list[I]]+(1.0-self.weight_list[I])*cps[self.index_list[I]+1]

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
                while (J+2)<cp_grid.size and cp_grid[J+1]<grid_value:
                    J+=1
                self.index_list.append(J)
                w=(cp_grid[J+1]-grid_value)/(cp_grid[J+1]-cp_grid[J])
                self.weight_list.append(w)


# This is an example of a spline module based on piece-wise linear functions
class SplineModule_PiecewiseLinear(SplineModuleBase):

    def __init__(self, cp_grid_in, var_name_in='unnamed_piecewise_linear_spline_control_point_variable', object_name_in='unnamed_piecewise_linear_spline_object', state_version_in=None):
        super(SplineModule_PiecewiseLinear, self).__init__(var_name_in, object_name_in)

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
            old_meta=self.remove_output(old_name)
            self.add_output(self.var_name, old_meta)

    # This is a method that will trigger the construction of the input interface
    def _build_interface(self):

        self.add_input(self.var_name)
        self.add_output(self.var_name)

    def compute(self, input_values, output_values, var_name=[]):

        output_values[self.var_name] = input_values[self.var_name]

