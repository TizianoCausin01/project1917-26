import sys, os, yaml
import warnings
import numpy as np
import random
from scipy.io import savemat
ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["useful_stuff_path"])
sys.path.append("..")

'''
run2part
Converts the current run into the correct movie part. 1, 2, 3 stay the same, 4, 5, 6 become 1, 2, 3 respectively.
INPUT:
    - run: int -> the current run
OUTPUT:
    - movie_part -> the corresponding movie part
'''
def run2part(run: int) -> int:
    if run > 3:
        movie_part = run - 3
    else: 
        movie_part = run
    # end if run > 3:
    return movie_part 
# EOF


