
import os, yaml, sys
import numpy as np
import h5py
from scipy.io import loadmat
sys.path.append("..")
from general_utils.utils import TimeSeries

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
    file_path = f"{paths['data_dir']}/data/sub{sub_num:03d}/preprocessed/sub{sub_num:03d}_{sensors_group}_{fs}Hz_MNN0_badmuscle0_badlowfreq1_badsegint1_badcomp2.mat"    
    data = loadmat(file_path)
    labels = [str(x[0]) for x in data['label'].flatten()]
    data_neural = data['data_final'][run-1][0]
    data_neural = TimeSeries(data_neural, fs)
    return data_neural, labels
# EOF
