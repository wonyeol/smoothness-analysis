#!/bin/bash

PYFILE="br.py"
ARGS="-n 5000 -ef 100 -lr 1e-2"

for ((i=0;i<3;i++)); do
    LOG=res_$(date "+%Y%m%d-%H%M%S").log
    python3 $PYFILE $ARGS > $LOG
done
