#!/bin/bash

time python3 scanvi.py -n 0 #> results-n0.log #<--- 20 sec
time python3 scanvi.py -n 3 #> results-n3.log #<--- 8 min
