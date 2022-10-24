#!/bin/bash

PYFILE="main_simp.py"
ARGS="-n 50000 -lr 1e-4 -blr 0.1 --z-pres-prior 0.01\
      --progress-every 10"

for ((i=0;i<3;i++)); do
    LOG=res_$(date "+%Y%m%d-%H%M%S").log
    python3 $PYFILE $ARGS > $LOG
done
