
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
from useful_stuff.general_utils.RSA import dRSA
from useful_stuff.general_utils.II import dynInformationImbalance
from useful_stuff.general_utils.regression import dyn_linear_encoding
from useful_stuff.general_utils.utils import print_wise, TimeSeries, compatible_TimeSeries_check
from project_specific_utils.dataloader import load_eyetracking_data, load_concat_regressout_meg, load_concat_regressout_mod, load_concat_gaze
from image_processing.gaze_dep_models import save_ANN_features


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
def save_lagged_comparisons(paths, analysis, sub_num, sensors_group, repetition, full_model_name, iterations_n, len_or_lag, neu_fs, signal_metric=None, model_metric=None, regression_type=None, PCs_used=None, score_metric=None, pseudotrials_n=None, sq_side=None, regress_out_gaze="0", k=1):
    if analysis == "encoding":
        general_filename = f"{paths['data_path']}/results/{analysis}_sub{sub_num:03d}_{sensors_group}_rep{repetition}_{full_model_name}_{PCs_used}PCs_{regression_type}_{score_metric}_lag_{round(len_or_lag/neu_fs)}s"
    else:
        general_filename = f"{paths['data_path']}/results/{analysis}_sub{sub_num:03d}_{sensors_group}_rep{repetition}_{full_model_name}_{signal_metric}-{model_metric}_{iterations_n}iter_{pseudotrials_n}pst_len_{round(len_or_lag/neu_fs)}s"
    if sq_side:
        general_filename = general_filename + f"_{sq_side}x{sq_side}patch_regr_out_gaze_{regress_out_gaze}"
    general_filename = general_filename +".mat"
    return general_filename
# EOF


"""
multivariate_lagged_comparisons
Performs multivariate lagged comparisons (RSA or II) between neural and model data, and saves the results to disk.

INPUT:
    - paths: dict[str, str] -> project path dictionary
    - n: TimeSeries -> neural signal (features x timepoints)
    - analysis_type: str -> type of analysis ("RSA" or "II")
    - sub_num: int -> subject number
    - sensors_group: str -> MEG sensors group
    - repetition: int -> repetition index (selects runs)
    - full_model_name: str -> model/layer name
    - iterations_n: int -> number of subsampling iterations
    - pseudotrial_len: int -> length of pseudotrials (in timepoints)
    - neu_fs: float -> sampling frequency of neural data
    - signal_metric: str -> distance/similarity metric for neural data
    - model_metric: str -> distance/similarity metric for model data
    - pseudotrials_n: int -> number of pseudotrials per iteration
    - sq_side: int -> spatial patch size used for model features
    - regress_out_gaze: bool | str -> whether/how gaze regression is applied (used for naming)
    - n_model_components: int -> number of PCA components for model features
    - pooling: str (default="all") -> pooling type for model features
    - rank: int (default=0) -> process rank for logging

OUTPUT:
    - None -> results are saved to disk as .mat files
        - RSA: saves a single matrix (time x time)
        - II: saves two matrices (A2B and B2A directions)
"""
def multivariate_lagged_comparisons(paths, rank, full_model_name, n, analysis_type, sub_num, sensors_group, repetition, iterations_n, pseudotrial_len, neu_fs, mod_fs, model_len, signal_metric, model_metric, pseudotrials_n, sq_side, regress_out_gaze, n_model_components, pooling="all",):
    regress_out_type = regress_out_gaze if regress_out_gaze else "0"
    if analysis_type == "RSA":
        p = [save_lagged_comparisons(paths, analysis_type, sub_num, sensors_group, repetition, full_model_name, iterations_n, pseudotrial_len, neu_fs, signal_metric=signal_metric, model_metric=model_metric, pseudotrials_n=pseudotrials_n, sq_side=sq_side, regress_out_gaze=regress_out_type),]
    elif analysis_type == "II":
        p = []
        pA2B = save_lagged_comparisons(paths, analysis_type+"A2B", sub_num, sensors_group, repetition, full_model_name, iterations_n, pseudotrial_len, neu_fs, signal_metric=signal_metric, model_metric=model_metric, pseudotrials_n=pseudotrials_n, sq_side=sq_side, regress_out_gaze=regress_out_type)
        p.append(pA2B)
        pB2A = save_lagged_comparisons(paths, analysis_type+"B2A", sub_num, sensors_group, repetition, full_model_name, iterations_n, pseudotrial_len, neu_fs, signal_metric=signal_metric, model_metric=model_metric, pseudotrials_n=pseudotrials_n, sq_side=sq_side, regress_out_gaze=regress_out_gaze)
        p.append(pB2A)
    if all(os.path.exists(path) for path in p):
        print(f"{p[0]} already exists")
        return None
    mod_fs = config["movie_fs"]
    model_len = [round(i*neu_fs/config["movie_fs"]) for i in config["model_len"]]
    m = load_concat_regressout_mod(paths, sub_num, save_ANN_features, full_model_name, repetition, mod_fs, neu_fs, *(sq_side, n_model_components, pooling), regress_out_gaze=False, gaze_dep=True, gaze_fs=50, rank=rank,)
    func = subsampling_RSA if analysis_type == "RSA" else subsampling_II
    tot_A2B, tot_B2A = subsampling_lagged_comparisons(n, m, pseudotrial_len, iterations_n, pseudotrials_n, model_len, func, rank, *(signal_metric, model_metric))
    if analysis_type == "RSA":
        savemat(p[0], {"RSA": tot_A2B})
        print_wise(f"{full_model_name} saved at {p[0]}")
    elif analysis_type == "II":
        savemat(p[0], {"II": tot_A2B})
        savemat(p[1], {"II": tot_B2A})
        print_wise(f"{full_model_name} saved at {p[0]}")
    return None
# EOF


def lagged_encoding_comparisons(paths, rank, full_model_name, n, analysis_type, sub_num, sensors_group, repetition, max_lag, neu_fs, mod_fs, regression_type, score_type, sq_side, regress_out_gaze, n_model_components, model_PCs_to_keep, pooling="all",):
    regress_out_type = regress_out_gaze if regress_out_gaze else "0"
    p = save_lagged_comparisons(paths, analysis_type, sub_num, sensors_group, repetition, full_model_name, None, max_lag, neu_fs, regression_type=regression_type, score_metric=score_type, PCs_used=model_PCs_to_keep, sq_side=sq_side, regress_out_gaze=regress_out_type)
    if os.path.exists(p):
        print(f"{p} already exists")
        return None
    mod_fs = config["movie_fs"]
    m = load_concat_regressout_mod(paths, sub_num, save_ANN_features, full_model_name, repetition, mod_fs, neu_fs, *(sq_side, n_model_components, pooling), regress_out_gaze=False, gaze_dep=True, gaze_fs=50, rank=rank,)
    m = TimeSeries(m.get_array()[:model_PCs_to_keep, :, np.newaxis], neu_fs)
    regression_obj = dyn_linear_encoding(regression_type, 'kf', max_lag, score_type=score_type, n_splits=5)
    s = regression_obj.crossvalidate_general_dyn(m, n)
    savemat(p, {"encoding": s.get_array()})
    print_wise(f"{full_model_name} saved at {p}")
    return None
# EOF
