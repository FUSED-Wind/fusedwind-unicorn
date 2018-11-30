#! /bin/bash

LST="\
    fused_dummy_dak_pure_workflow.py\
    fused_dummy_example.py\
    fused_dummy_group_example.py\
    fused_nrel_csm.py\
    fused_spline_dakota_example.py\
    "

for F in ${LST}
do
    echo -e "\n=========================== $F =========================\n"
    python $F
done

