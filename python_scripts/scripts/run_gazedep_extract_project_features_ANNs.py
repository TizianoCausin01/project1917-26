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
from image_processing.gaze_dep_models import ANN_extraction_projection_1917_wrapper, save_ipca_patch
from useful_stuff.general_utils.utils import print_wise
from useful_stuff.parallel.parallel_funcs import parallel_setup, master_workers_queue
from useful_stuff.image_processing.computational_models import imgANN

# e.g. to call it:
# mpiexec -np 5 python3 run_gazedep_extract_project_features_ANNs.py --sub_to_proc 1 --eye_fs 50 --model_name vit_l_16 --layer_idx 10 --n_components 1000 --sq_size 384 --input_size 384 --pooling all --pkg timm

parser = argparse.ArgumentParser()
parser.add_argument("--eye_fs", type=int)
parser.add_argument("--model_name", type=str)
parser.add_argument("--n_components", type=int)
parser.add_argument("--sq_size", type=int)
parser.add_argument("--input_size", type=int)
parser.add_argument("--pooling", type=str)
parser.add_argument("--pkg", type=str)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--model_url", type=str, default="facebook/dinov3-vitl16-pretrain-lvd1689m")

cfg = parser.parse_args()
task_list = config["subjects"]
_, rank, _ = parallel_setup()
if rank != 0:
    m = imgANN(cfg.model_name, cfg.pkg, cfg.input_size, dtype=torch.float16, attn_implementation='sdpa', repo_url=cfg.model_url)
    print_wise(m, rank=rank)
    m.model.eval()
    PCs_dict = {}
    m.set_relevant_layers(m.get_relevant_layers())
    for l in m.relevant_layers:
        ipca_path = save_ipca_patch(paths, m.model_name, l, cfg.n_components, cfg.sq_size, cfg.pooling,) 
        ipca_obj = joblib.load(ipca_path)
        PCs = ipca_obj.components_.T
        PCs_dict[l] = PCs
        print_wise(f"{l} loaded", rank=rank)
    # end for l in ANN.relevant_layers:
else:
    m = None
    PCs_dict = None
# end if rank != 0:

master_workers_queue(task_list, paths, ANN_extraction_projection_1917_wrapper, *(m, cfg.sq_size, cfg.n_components, PCs_dict, cfg.eye_fs, cfg.batch_size)) 
