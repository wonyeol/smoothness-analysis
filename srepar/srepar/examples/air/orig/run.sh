#!/bin/bash

PYFILE="main.py"
ARGS="-n 50000 -blr 0.1 --z-pres-prior 0.01 \
      --scale-prior-sd 0.2 --predict-net 200 --bl-predict-net 200 \
      --decoder-output-use-sigmoid --decoder-output-bias -2 \
      --progress-every 10"

for ((i=0;i<3;i++)); do
    LOG=res_$(date "+%Y%m%d-%H%M%S").log
    python3 $PYFILE $ARGS > $LOG
done
