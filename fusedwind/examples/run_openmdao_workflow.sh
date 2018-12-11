#! /bin/bash

LST="\
        fused_dummy_om_partial.py\
        fused_dummy_om_total.py\
        fused_om_csm_examples.py\
    "

for F in $LST
do
    echo -e "\n=========================== $F =========================\n"
    python $F
done

