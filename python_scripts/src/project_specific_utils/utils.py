import sys, os
import warnings
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
    - discontinuities: list -> the points we don't want to cross with our pseudotrials (e.g. because it's crossing some run boundary)
    - fail_safety: bool -> whether or not having a deterministic failure if there is very little room for wiggle the pseudotrials
OUTPUT:
    - pseudotrials_idx: list of ints -> starting indices of non-overlapping pseudotrials

RAISES:
    - ValueError: if the signal is too short to fit the required number of spaced pseudotrials
"""
def get_spaced_pseudotrials(signal: TimeSeries, pseudotrials_len_tps: int, pseudotrials_n: int, min_separation: int, discontinuities: list=[], fail_safety: bool=True):
    candidates = list(range(len(signal) - pseudotrials_len_tps)) # all the possible points that can be chosen as pseudotrial start [0, 1, 2, ..., n-1]
    pseudotrials_idx = []
    tot_separation = pseudotrials_len_tps+min_separation # the minimum separation between two pseudotrials starts (so that they don't overlap)
    if discontinuities:
        to_remove = []
        for el in discontinuities:
            to_remove.extend(list(range(max(el-pseudotrials_len_tps,0), el)))
        to_remove = set(to_remove) # for O(1) lookup
        candidates = [x for x in candidates if x not in to_remove]
    # tot_separation is multiplied by the amount of pseudotrials_n and then we check if the projected pseudotrials duration surpasses the whole signal duration
    if tot_separation*pseudotrials_n - 2*min_separation > len(signal) - len(discontinuities):  #-2*min_separation bc the first and last trials don't have it, -len(discontinuities)*pseudotrials_len_tps not to count the pts we don't want to cross 
        raise ValueError(f"The length of the TimeSeries {len(signal)} is not enough for the number and length of pseudotrials required")
    rho = len(candidates)/(pseudotrials_n*tot_separation) # rho=1 means that you have barely enough space, rho=1.2 means that you have just 20% of room, etc...
    if rho < 1.2:
        if fail_safety:
            raise IndexError(f"For {pseudotrials_n=} and pseudotrial_len={pseudotrials_len_tps} it's likely you'll encounter a failure")
        else:
            warnings.warn(f"For {pseudotrials_n=} and pseudotrial_len={pseudotrials_len_tps} it's likely you'll encounter a failure")
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
