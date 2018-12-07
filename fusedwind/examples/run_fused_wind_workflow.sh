#! /bin/bash

LST="\
    fused_dummy_dak_pure_workflow.py\
	fused_dummy_dak_case_runner.py\
    fused_dummy_example.py\
    fused_dummy_group_example.py\
    fused_spline_dakota_example.py\
	fused_dummy_parallel_group.py\
	fused_group_interfaces.py\
    fused_nrel_csm.py\
    "

for F in ${LST}
do
    echo -e "\n=========================== $F =========================\n"
    python $F
	read -p "Continue ... "
done

