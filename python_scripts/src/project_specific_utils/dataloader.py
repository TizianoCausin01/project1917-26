
import os, yaml, sys
import numpy as np
import h5py

sys.path.append("..")
from general_utils.utils import TimeSeries

def load_eyetracking_data(paths: dict[str: str], sub_num: int, run: int, fs: float, xy=True):
    eyetracking_dir = f"{paths['data_dir']}/data/eyetracker_data"
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
