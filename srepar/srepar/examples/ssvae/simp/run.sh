#!/bin/bash

PYFILE="ssvae_simp.py"
ARGS="-n 50 -lr 1e-4 -b1 0.95"

for ((i=0;i<3;i++)); do
    LOG=res_$(date "+%Y%m%d-%H%M%S").log
    python3 $PYFILE $ARGS -log $LOG
done
