# This is a work-flow the demonstrates importing a work flow and connecting it to a dakota case-runner

# import the workflow
from fusedwind.examples.fused_dummy_dak_pure_workflow import indep_list, dakota_output_list
# import the dakota objects
from fusedwind.fused_dakota import Dakota_Work_Flow, Dakota_Job
# import the case runner
from fusedwind.fused_mpi_cases import FUSED_MPI_Cases

# Create a list of parameter files
param_list = ['dakota_param_files/params.in.1', \
                'dakota_param_files/params.in.2', \
                'dakota_param_files/params.in.3', \
                'dakota_param_files/params.in.4', \
                'dakota_param_files/params.in.5', \
                'dakota_param_files/params.in.6', \
                'dakota_param_files/params.in.7', \
                'dakota_param_files/params.in.8']

# create a dakota workflow
work_flow = Dakota_Work_Flow(indep_list, dakota_output_list)

# Create a list of dakota cases
dakota_cases = []
for param_file in param_list:
    result_file = param_file+'.out'
    dakota_cases.append(Dakota_Job(work_flow, param_file, result_file))

# create the case runner
case_runner = FUSED_MPI_Cases(dakota_cases)

# execute the cases
case_runner.execute()
