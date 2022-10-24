#!/bin/bash

# python3 main.py             # num_epochs=101
# python3 main.py -nq 2 -n 1  # num_quadrant_inputs=[2,], num_epochs=1 <--- 2 min
# python3 main.py -nq 1 -n 10 # num_quadrant_inputs=[1,], num_epochs=10 <--- 16 min
python3 main.py -nq 1 -n 5  # num_quadrant_inputs=[1,], num_epochs=5 <--- 8 min
