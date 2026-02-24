import sys, os
import h5py

sys.path.append("..")
from project_specific_utils.dataloader import load_eyetracking_data
from general_utils.utils import print_wise, TimeSeries

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



