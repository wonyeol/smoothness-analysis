#!/bin/bash

PYFILE="experiment_simp.py"
ARGS="-t 10 -g continuous -i continuous"

for ((i=0;i<1;i++)); do
    # LOG=res_$(date "+%Y%m%d-%H%M%S").log
    time python3 $PYFILE $ARGS # > $LOG
done
