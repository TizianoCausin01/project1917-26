import os, yaml, sys
import argparse
import joblib
import torch
ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
subjects = config['subjects']
sys.path.append(paths["src_path"])
sys.path.append(paths["useful_stuff_path"])
from useful_stuff.general_utils.utils import print_wise
from analyses.subsampling_lagged_comparisons import multivariate_lagged_comparisons
from project_specific_utils.dataloader import load_concat_regressout_meg
from useful_stuff.image_processing.computational_models import get_relevant_output_layers
from useful_stuff.parallel.parallel_funcs import parallel_setup, master_workers_queue
_, rank, _ = parallel_setup()
print(rank)
