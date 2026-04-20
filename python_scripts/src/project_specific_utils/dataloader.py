
import os, yaml, sys
import numpy as np
import h5py
from scipy.io import loadmat
ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["useful_stuff_path"])
sys.path.append("..")
from useful_stuff.general_utils.utils import TimeSeries, print_wise
from useful_stuff.general_utils.regression import dyn_linear_encoding

'''
load_eyetracking_data
Loads eyetracking data from a .mat file and returns as a TimeSeries object.

INPUT:
    - paths: dict[str, str] -> project path dictionary
    - sub_num: int -> subject number
    - run: int -> run number
    - fs: float -> sampling frequency
    - xy: bool -> if True, returns only x and y coordinates

OUTPUT:
    - gaze: TimeSeries -> gaze coordinates (2 x T if xy=True, else full)
    - timestamps: np.ndarray -> corresponding time points
'''
def load_eyetracking_data(paths: dict[str: str], sub_num: int, run: int, fs: float, xy=True):
    eyetracking_dir = f"{paths['data_path']}/data/eyetracker_data"
    eyetracking_file = f"{eyetracking_dir}/gaze_sub{sub_num:03d}_run{run:02d}_{fs}Hz.mat"
    with h5py.File(eyetracking_file, "r") as f:
        gaze = f['gaze'][:].T
        timestamps = f['tNew'][:].T
    # end with h5py.File(eyetracking_file, "r") as f:
    if xy:
        gaze = gaze[:2, :]
    # end if xy:
    gaze = TimeSeries(gaze, fs)
    return gaze, timestamps
# EOF


'''
load_meg_data
Loads preprocessed MEG data from a .mat file and returns as a TimeSeries object.

INPUT:
    - paths: dict[str, str] -> project path dictionary
    - sub_num: int -> subject number
    - run: int -> run number
    - sensors_group: str -> group of sensors to load (e.g., 'mag', 'grad')
    - fs: float -> sampling frequency

OUTPUT:
    - data_neural: TimeSeries -> MEG data for the specified run
    - labels: list[str] -> sensor labels
'''
def load_meg_data(paths: dict[str: str], sub_num: int, run: int, sensors_group: str, fs: float):
    file_path = f"{paths['data_path']}/data/sub{sub_num:03d}/preprocessed/sub{sub_num:03d}_{sensors_group}_{fs}Hz_MNN0_badmuscle0_badlowfreq1_badsegint1_badcomp2.mat"    
    data = loadmat(file_path)
    labels = [str(x[0]) for x in data['label'].flatten()]
    data_neural = data['data_final'][run-1][0]
    data_neural = TimeSeries(data_neural, fs)
    return data_neural, labels
# EOF


"""
load_concat_regressout_meg
Loads MEG data across 3 runs, optionally regresses out gaze, and concatenates into a single TimeSeries.

INPUT:
    - paths: dict[str, str] -> project path dictionary
    - sub_num: int -> subject number
    - repetition: int -> repetition index (0-based, selects runs [1–3] or [4–6])
    - sensors_group: str -> MEG sensors group to load
    - neu_fs: float -> target sampling frequency for MEG data
    - gaze_fs: float -> sampling frequency of gaze data
    - regress_out_gaze: bool | str -> whether/how to regress out gaze
        - False -> no regression
        - "lag0" -> regress gaze at same timepoints
        - "PCR" -> delay-embedded PCR regression
    - PCs_to_regress_out: int -> number of PCs for PCR regression
    - timepts_to_regress_out: tuple[int, int] (default=(-100,100)) -> temporal window for delay embedding
    - rank: int -> process rank for logging

OUTPUT:
    - neu: TimeSeries -> concatenated MEG signal (features x timepoints)
"""
def load_concat_regressout_meg_(paths, sub_num, repetition, sensors_group, neu_fs, gaze_fs, regress_out_gaze,  PCs_to_regress_out, timepts_to_regress_out=(-100,100), rank=0):
    neu = []
    print_wise(f"Loading MEG signal: {regress_out_gaze=}", rank)
    runs = np.arange(1,4)+3*repetition 
    nans_rows = []
    for idx, i_run in enumerate(runs):
        run_neu, labels = load_meg_data(paths, sub_num, i_run, sensors_group, neu_fs)
        if np.any(np.isnan(run_neu)):
            print_wise(i_run, "contains nan")
        run_neu.z_score_feats()
        model_len = round(config["model_len"][idx]*neu_fs/config["movie_fs"])
        run_neu = TimeSeries(run_neu[:model_len], run_neu.get_fs())
        # checks if there is any sensor that has been discarded
        nan_positions = np.argwhere(np.isnan(run_neu.array))
        if nan_positions.size > 0:
            print(f"Total NaNs: {nan_positions.shape[0]}")
        rows_with_nan = np.unique(nan_positions[:, 0])
        if rows_with_nan.size > 0:
            for bad_row in rows_with_nan:
                if np.all(np.isnan(run_neu.array[bad_row, :])):  
                    nans_rows.append(bad_row)
                else:
                    raise ValueError(f"Not all elements in {bad_row} are NaNs")
                # end
        if idx == 1: # i.e. if it is the 2nd part of the movie (run 2 or run 5)
            run_neu = run_neu[3*neu_fs:]
        neu.append(run_neu[:])
    # end for i_run in range(1,4):
    for idx, run in enumerate(neu):
        neu[idx] = np.delete(run, nans_rows, axis=0)

    for idx, run_neu in enumerate(neu):
        if regress_out_gaze:
            run_gaze, _ = load_eyetracking_data(paths, sub_num, idx +1+ repetition*3, gaze_fs, xy=True)
            run_gaze.z_score_feats()
            run_gaze.resample(run_gaze.get_fs())
            run_gaze = TimeSeries(run_gaze[:model_len], run_gaze.get_fs())
            run_neu = TimeSeries(run_neu, neu_fs)
            dyn_regr_obj = dyn_linear_encoding('lr', 'same', None)
            if regress_out_gaze == "PCR":
                run_neu, _ = dyn_regr_obj.delay_embed_PCR_regress_out(run_gaze, run_neu, timepts_to_regress_out, PCs_to_keep=PCs_to_regress_out, pad_mode='edge', crop_end=True)
            elif regress_out_gaze == "lag0":
                run_neu = dyn_regr_obj.pointwise_regress_out(run_gaze, run_neu, regression_type=None) 
            # end if cfg.regr_out_eyes == "PCR":
        neu[idx] = run_neu[:]
    # end if cfg.regr_out_eyes:
    len_runs = [i.shape for i in neu]
    print_wise(f"Shape runs {runs}: {len_runs}", rank=rank)
    neu = np.concatenate(neu, axis=1)
    neu = TimeSeries(neu, neu_fs)
    return neu
# EOF


"""
load_concat_gaze
Loads gaze data across 3 runs, preprocesses, and concatenates into a single TimeSeries.

INPUT:
    - paths: dict[str, str] -> project path dictionary
    - sub_num: int -> subject number
    - repetition: int -> repetition index (0-based, selects runs [1–3] or [4–6])
    - gaze_fs: float -> original gaze sampling frequency
    - new_fs: float -> target sampling frequency
    - rank: int -> process rank for logging

OUTPUT:
    - gaze: TimeSeries -> concatenated gaze signal (2 x timepoints)
"""
def load_concat_gaze(paths, sub_num, repetition, gaze_fs, new_fs, rank=0):
    gaze = []
    runs = np.arange(1,4)+3*repetition 
    for idx, i_run in enumerate(runs):
        run_gaze, _ = load_eyetracking_data(paths, sub_num, i_run, gaze_fs, xy=True)
        run_gaze.z_score_feats()
        run_gaze.resample(new_fs)
        run_gaze = TimeSeries(run_gaze[:len_mod[idx]], run_gaze.get_fs())
        model_len = round(config["model_len"][idx]*new_fs/config["movie_fs"])
        run_gaze = run_gaze[:, :model_len]
        if idx == 1: # i.e. if it is the 2nd part of the movie (run 2 or run 5)
            print(i_run)
            run_gaze = run_gaze[3*new_fs:]
        gaze.append(run_gaze)
        # end if cfg.regr_out_eyes:
    # end for i_run in range(1,4):
    len_runs = [i.shape for i in gaze]
    print_wise(f"Shape runs {runs}: {len_runs}", rank=rank)
    gaze = TimeSeries(np.concatenate(gaze, axis=1), new_fs)
    return gaze
# EOF


"""
load_concat_regressout_mod
Loads model features across runs, optionally regresses out gaze, and concatenates into a single TimeSeries.

INPUT:
    - paths: dict[str, str] -> project path dictionary
    - sub_num: int -> subject number
    - save_func: callable -> function returning the model file path
    - model_name: str -> name of the model/layer to load
    - repetition: int -> repetition index (0-based, selects runs [1–3] or [4–6] if gaze_dep=True)
    - mod_fs: float -> original model sampling frequency
    - new_fs: float -> target sampling frequency
    - *args -> additional arguments passed to save_func
    - regress_out_gaze: bool (default=True) -> whether to regress out gaze (lag-0)
    - gaze_dep: bool (default=True) -> whether model is gaze-dependent (affects run indexing)
    - gaze_fs: float (default=50) -> sampling frequency of gaze data
    - rank: int -> process rank for logging
    - **kwargs -> additional keyword arguments passed to save_func

OUTPUT:
    - mod: TimeSeries -> concatenated model features (features x timepoints)
"""
def load_concat_regressout_mod(paths, sub_num, save_func, model_name, repetition, mod_fs, new_fs, *args, regress_out_gaze=True, gaze_dep=True, gaze_fs=50, rank=0, **kwargs):
    print_wise(f"Loading model {model_name}: {regress_out_gaze=}", rank=0)
    runs = np.arange(1,4)+3*repetition if gaze_dep else np.arange(1,4)
    mod = []
    for idx, i_run in enumerate(runs):
        model_filename = save_func(paths, model_name, sub_num, i_run, mod_fs, *args, **kwargs) 
        with h5py.File(model_filename, "r") as f:
            run_mod = f['vecrep'][:]
        # end with h5py.File(model_filename, "r") as f:
        run_mod = TimeSeries(run_mod, mod_fs)
        run_mod.resample(new_fs)
        new_len = round(config["model_len"][idx]*new_fs/config["movie_fs"])
        assert new_len == len(run_mod), f"The length of the model ({len(run_mod)}) doesn't match its expected length ({new_len})"
        if regress_out_gaze:
            run_gaze, _ = load_eyetracking_data(paths, sub_num, i_run, gaze_fs, xy=True)
            run_gaze.z_score_feats()
            run_gaze.resample(new_fs)
            run_gaze = TimeSeries(run_gaze[:new_len], run_gaze.get_fs())
            dyn_regr_obj = dyn_linear_encoding('lr', 'same', None)
            run_mod = dyn_regr_obj.pointwise_regress_out(run_gaze, run_mod, regression_type=None)
        if idx == 1: # i.e. if it is the 2nd part of the movie (run 2 or run 5)
            run_mod = run_mod[3*new_fs:]
        mod.append(run_mod[:])
    len_runs = [i.shape for i in mod]
    print_wise(f"{model_name}: shape runs {runs}: {len_runs}", rank=rank)
    # end for i_run in range(1,4):
    mod = TimeSeries(np.concatenate(mod, axis=1), new_fs)
    return mod
# EOF
