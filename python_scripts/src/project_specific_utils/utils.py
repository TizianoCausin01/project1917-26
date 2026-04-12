import sys, os, yaml
import warnings
import numpy as np
import random
ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["useful_stuff_path"])
sys.path.append("..")
from useful_stuff.general_utils.RSA import dRSA
from useful_stuff.general_utils.II import dynInformationImbalance
from useful_stuff.general_utils.regression import dyn_linear_encoding
from useful_stuff.general_utils.utils import print_wise, TimeSeries, compatible_TimeSeries_check
from project_specific_utils.dataloader import load_eyetracking_data

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

'''
subsampling_lagged_comparisons
Performs subsampled lagged comparisons between signal and model across multiple pseudotrials and iterations, averaging the resulting similarity matrices.

INPUT:
    - signal: TimeSeries
    - model: TimeSeries
    - pseudotrials_len_tps: int -> length of each pseudotrial in timepoints
    - iteration_n: int -> number of subsampling iterations
    - pseudotrials_n: int -> number of pseudotrials per iteration
    - discontinuities: list[int] -> timepoints to avoid when sampling pseudotrials
    - func: function -> function computing similarity (e.g. RSA, II)
    - rank: int -> rank for distributed logging
    - *args: additional positional arguments for func
    - min_separation: int -> minimum separation between pseudotrials
    - **kwargs: additional keyword arguments for func

OUTPUT:
    - tot_similarity_A2B: ndarray -> averaged similarity matrix (A2B)
    - tot_similarity_B2A: ndarray -> averaged similarity matrix (B2A)
'''
def subsampling_lagged_comparisons(signal, model, pseudotrials_len_tps, iteration_n, pseudotrials_n, discontinuities, func, rank, *args, min_separation=5, **kwargs):
    compatible_TimeSeries_check(signal, model)    
    tot_similarity_A2B = np.zeros((pseudotrials_len_tps, pseudotrials_len_tps))
    tot_similarity_B2A = np.zeros((pseudotrials_len_tps, pseudotrials_len_tps))
    # end if measure=='RSA':
    for i_iter in range(iteration_n):
        if i_iter %10 ==0:
            print_wise(f"starting iteration {i_iter} of {iteration_n-1}", rank=rank)
        signal_trials = []
        model_trials = []
        pseudotrials_idx = get_spaced_pseudotrials(signal, pseudotrials_len_tps, pseudotrials_n, min_separation, discontinuities=discontinuities)
        for n in pseudotrials_idx:
            signal_trials.append(signal.get_array()[:,n:n+pseudotrials_len_tps])
            model_trials.append(model.get_array()[:,n:n+pseudotrials_len_tps])
        # end for n in pseudotrials_idx:
        signal_trials = TimeSeries(np.stack(signal_trials, axis=2), signal.get_fs())
        model_trials = TimeSeries(np.stack(model_trials, axis=2), model.get_fs())
        # if func is subsampling_RSA the second output is just a 0 for compatibility with the double output of II
        similarity_corr_A2B, similarity_corr_B2A = func(signal_trials, model_trials, *args, **kwargs) 
        tot_similarity_A2B += similarity_corr_A2B
        tot_similarity_B2A += similarity_corr_B2A
    # end for i_iter in range(iteration_n):
    tot_similarity_A2B = tot_similarity_A2B/iteration_n
    tot_similarity_B2A = tot_similarity_B2A/iteration_n
    return tot_similarity_A2B, tot_similarity_B2A
# EOF

'''
subsampling_II
Computes dynamic information imbalance (II) between neural and model pseudotrials.

INPUT:
    - neu_trials: TimeSeries -> neural data pseudotrials
    - mod_trials: TimeSeries -> model data pseudotrials
    - neu_metric: callable -> metric for neural data
    - mod_metric: callable -> metric for model data
    - k: int -> number of neighbors for II computation

OUTPUT:
    - II_A2B: ndarray -> information imbalance from neural to model
    - II_B2A: ndarray -> information imbalance from model to neural
'''
def subsampling_II(neu_trials, mod_trials, neu_metric, mod_metric, k=1):
    II_obj = dynInformationImbalance(neu_metric, mod_metric, k=k)
    II_obj.compute_both_RDM_timeseries(neu_trials, mod_trials)
    II_obj.compute_both_distance_ranks_timeseries()
    II_A2B, II_B2A = II_obj.compute_both_dynII()
    return II_A2B, II_B2A
# EOF


'''
subsampling_RSA
Computes representational similarity analysis (RSA) between neural and model pseudotrials.

INPUT:
    - neu_trials: TimeSeries -> neural data pseudotrials
    - mod_trials: TimeSeries -> model data pseudotrials
    - neu_metric: callable -> metric for neural data
    - mod_metric: callable -> metric for model data
    - RSA_metric: str -> similarity metric used for RSA

OUTPUT:
    - rsa_corr: ndarray -> RSA similarity matrix (A2B)
    - _: int -> placeholder (compatibility with II output format)
'''
def subsampling_RSA(neu_trials, mod_trials, neu_metric, mod_metric, RSA_metric="correlation"):
    drsa_obj_corr = dRSA(neu_metric, model_RDM_metric=mod_metric, RSA_metric=RSA_metric)
    drsa_obj_corr.compute_both_RDM_timeseries(neu_trials, mod_trials)
    rsa_corr = drsa_obj_corr.compute_dRSA()
    _ = 0 # just for compatibility with II
    return rsa_corr, _
# EOF


'''
subsampling_encoding
Computes dynamic encoding performance between model and neural pseudotrials using cross-validated regression.

INPUT:
    - neu_trials: TimeSeries -> neural data pseudotrials (targets)
    - mod_trials: TimeSeries -> model data pseudotrials (features)
    - regression_type: str -> type of regression model
    - cv_type: str -> cross-validation strategy
    - max_lag: int -> maximum temporal lag considered
    - score_type: str -> scoring metric (default 'r2')
    - n_splits: int -> number of CV splits

OUTPUT:
    - final_score: ndarray -> encoding performance scores
'''
def subsampling_encoding(neu_trials, mod_trials, regression_type, cv_type, max_lag, score_type='r2', n_splits=5):
    regression_obj = dyn_linear_encoding(regression_type, cv_type, max_lag, score_type=score_type, n_splits=n_splits)
    final_score = regression_obj.crossvalidate_general_dyn(mod_trials, neu_trials)
    print(final_score.shape())
    return final_score.get_array()
# EOF 

"""
save_lagged_comparisons
Generates the filename for saving lagged comparison results (e.g., encoding, RSA, II).

INPUT:
    - paths: dict[str, str] -> project path dictionary (not used here but kept for consistency)
    - analysis: str -> type of analysis ("encoding", "RSA", "II", etc.)
    - sub_num: int -> subject number
    - sensors_group: str -> MEG sensors group
    - full_model_name: str -> model/layer name
    - iterations_n: int -> number of subsampling iterations
    - len_or_lag: int -> pseudotrial length or max lag (in timepoints)
    - neu_fs: float -> sampling frequency of neural data (used to convert to seconds)
    - signal_metric: str | None -> metric for neural data (e.g., distance metric)
    - model_metric: str | None -> metric for model data
    - regression_type: str | None -> regression type (for encoding analysis)
    - PCs_used: int | None -> number of PCs used (for encoding analysis)
    - score_metric: str | None -> scoring metric (e.g., r2, correlation)
    - pseudotrials_n: int | None -> number of pseudotrials (for RSA/II)
    - sq_side: int | None -> patch size (if applicable)
    - regress_out_gaze: str | int (default="0") -> whether/how gaze was regressed out

OUTPUT:
    - general_filename: str -> filename encoding all analysis parameters
"""
def save_lagged_comparisons(paths, analysis, sub_num, sensors_group, full_model_name, iterations_n, len_or_lag, neu_fs, signal_metric=None, model_metric=None, regression_type=None, PCs_used=None, score_metric=None, pseudotrials_n=None, sq_side=None, regress_out_gaze="0"):
    if analysis == "encoding":
        general_filename = f"{paths['data_path']}/results/{analysis}_sub{sub_num:03d}_{sensors_group}_{full_model_name}_{PCs_used}PCs_{regression_type}_{score_metric}_{iterations_n}iter_lag_{round(len_or_lag/neu_fs)}s"
    else:
        general_filename = f"{paths['data_path']}/results/{analysis}_sub{sub_num:03d}_{sensors_group}_{full_model_name}_{signal_metric}-{model_metric}_{iterations_n}iter_{pseudotrials_n}pst_len_{round(len_or_lag/neu_fs)}s"
    if sq_side:
        general_filename = general_filename + f"_{sq_side}x{sq_side}patch_regr_out_gaze_{regress_out_gaze}"
    general_filename = general_filename +".mat"
    return general_filename
# EOF
