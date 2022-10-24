#!/bin/bash

PYFILE="csis.py"
ARGS="-n 2000 -ef 10 -lr 1e-3"

for ((i=0;i<3;i++)); do
    LOG=res_$(date "+%Y%m%d-%H%M%S").log
    python3 $PYFILE $ARGS > $LOG
done
