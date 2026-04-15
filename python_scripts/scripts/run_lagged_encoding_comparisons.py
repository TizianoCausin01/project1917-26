import os, yaml, sys
import argparse
import numpy as np
import joblib
import torch
ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
subjects = config['subjects']
sys.path.append(paths["src_path"])
sys.path.append(paths["useful_stuff_path"])
from analyses.subsampling_lagged_comparisons import lagged_encoding_comparisons
from project_specific_utils.dataloader import load_concat_regressout_meg
from useful_stuff.image_processing.computational_models import get_relevant_output_layers
from useful_stuff.parallel.parallel_funcs import parallel_setup, master_workers_queue
from useful_stuff.general_utils.utils import TimeSeries

# e.g. to call it:
# mpiexec -np 5 python3 run_lagged_encoding_comparisons.py --analysis_type "encoding" --sub_num 3 --sensors_group occ --repetition 0 --max_lag 600 --regression_type ridge --score_type r2 --neu_fs 100 --gaze_fs 50 --regress_out_gaze PCR --timepts_to_regress_out 100 --PCs_to_regress_out 50 --model_name vit_l_16 --n_model_components 1000 --sq_side 384 --model_PCs_to_keep 50 --pooling all --pkg timm

parser = argparse.ArgumentParser()
parser.add_argument("--analysis_type", type=str)
parser.add_argument("--sub_num", type=int)
parser.add_argument("--sensors_group", type=str)
parser.add_argument("--repetition", type=int)
parser.add_argument("--max_lag", type=int)
parser.add_argument("--regression_type", type=str)
parser.add_argument("--score_type", type=str)
parser.add_argument("--neu_fs", type=int)
parser.add_argument("--gaze_fs", type=int)
parser.add_argument("--regress_out_gaze", type=str, default=None)
parser.add_argument("--PCs_to_regress_out", type=int)
parser.add_argument("--timepts_to_regress_out", type=int)
parser.add_argument("--model_name", type=str)
parser.add_argument("--n_model_components", type=int)
parser.add_argument("--sq_side", type=int)
parser.add_argument("--model_PCs_to_keep", type=int)
parser.add_argument("--pkg", type=str)
parser.add_argument("--pooling", type=str, default='all')


cfg = parser.parse_args()
layer_names = get_relevant_output_layers(cfg.model_name, pkg=cfg.pkg) 
task_list = [f"{cfg.model_name}_{l}" for l in layer_names]
mod_fs = config["movie_fs"]
model_len = [round(i*cfg.neu_fs/config["movie_fs"]) for i in config["model_len"]]
cfg.timepts_to_regress_out = (-cfg.timepts_to_regress_out, cfg.timepts_to_regress_out)
_, rank, _ = parallel_setup()
if rank != 0:
    n = load_concat_regressout_meg(paths, cfg.sub_num, cfg.repetition, cfg.sensors_group, cfg.neu_fs, cfg.gaze_fs, cfg.regress_out_gaze, cfg.PCs_to_regress_out, timepts_to_regress_out=cfg.timepts_to_regress_out, rank=rank)
    n.z_score_feats()
    n = TimeSeries(n.get_array()[:,:,np.newaxis], cfg.neu_fs)
else:
    n = None
# end if rank != 0:

master_workers_queue(task_list, paths, lagged_encoding_comparisons, *(n, cfg.analysis_type, cfg.sub_num, cfg.sensors_group, cfg.repetition, cfg.max_lag, cfg.neu_fs, mod_fs, cfg.regression_type, cfg.score_type, cfg.sq_side, cfg.regress_out_gaze, cfg.n_model_components, cfg.model_PCs_to_keep,), **{"pooling": cfg.pooling}) 
