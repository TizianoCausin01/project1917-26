import sys, os
import h5py
import random
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


"""
get_spaced_pseudotrials
Randomly samples starting indices for pseudotrials from a time series, ensuring that selected trials do not overlap by enforcing a minimum separation.

INPUT:
    - signal: array-like -> time series from which pseudotrials are extracted
    - pseudotrials_len_tps: int -> length of each pseudotrial in timepoints
    - pseudotrials_n: int -> number of pseudotrials to sample
    - min_separation: int -> minimum required separation (in timepoints) between pseudotrials (in addition to their length)

OUTPUT:
    - pseudotrials_idx: list of ints -> starting indices of non-overlapping pseudotrials

RAISES:
    - ValueError: if the signal is too short to fit the required number of spaced pseudotrials
"""
def get_spaced_pseudotrials(signal, pseudotrials_len_tps, pseudotrials_n, min_separation):
    candidates = list(range(len(signal) - pseudotrials_len_tps)) # all the possible points that can be chosen as pseudotrial start [0, 1, 2, ..., n-1]
    pseudotrials_idx = []
    tot_separation = pseudotrials_len_tps+min_separation # the minimum separation between two pseudotrials starts (so that they don't overlap)
    if tot_separation*pseudotrials_n > len(signal):
        raise ValueError(f"The length of the TimeSeries {len(signal)} is not enough for the number and length of pseudotrials required")
    # while still there are candidates available and while there are and while we haven't reached the quota of pseudotrials
    while candidates and len(pseudotrials_idx) < pseudotrials_n: 
        idx = random.choice(candidates) # randomly chooses among the available timepoints
        pseudotrials_idx.append(idx) # adds it to the list of pseudotrials starts
        # excludes the candidates that would overlap with the just chosen index
        candidates = [c for c in candidates if abs(c - idx) >= tot_separation] 
    if len(pseudotrials_idx) < pseudotrials_n: 
        raise ValueError("Not enough non-overlapping pseudotrials available for the given max_lag.")
    return pseudotrials_idx
# EOF
