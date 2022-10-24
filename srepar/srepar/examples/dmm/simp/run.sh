#!/bin/bash

PYFILE="dmm_simp.py"
ARGS="-n 50 -lr 1e-4"

for ((i=0;i<3;i++)); do
    LOG=res_$(date "+%Y%m%d-%H%M%S").log
    python3 $PYFILE $ARGS -l $LOG
done
