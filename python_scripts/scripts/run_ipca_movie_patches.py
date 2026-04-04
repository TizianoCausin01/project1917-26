import os, yaml, sys
import argparse
import torch
ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
subjects = config['subjects']
sys.path.append(paths["src_path"])
sys.path.append(paths["useful_stuff_path"])
from image_processing.gaze_dep_models import ipca_movie_patches
from useful_stuff.parallel.parallel_funcs import parallel_setup, master_workers_queue
from useful_stuff.general_utils.utils import get_device
from useful_stuff.image_processing.computational_models import imgANN, get_relevant_output_layers
# ipca_movie_patches(paths, rank, layer_name, model_name, model, n_components, batch_size, patches_per_frame, frames_step, patches_overhead_sampling, sq_size, input_size, pooling, secs_to_skip=5)
# e.g. to call it:
# mpiexec -np 5 python3 run_ipca_movie_patches.py --model_name vit_l_16 --n_components 1000 --batch_size 1024 --patches_per_frame 3 --frames_step 3 --patches_overhead_sampling 2 --sq_size 384 --input_size 384 --pooling all --pkg timm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--n_components", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--patches_per_frame", type=int)
    parser.add_argument("--frames_step", type=int)
    parser.add_argument("--patches_overhead_sampling", type=int)
    parser.add_argument("--sq_size", type=int)
    parser.add_argument("--input_size", type=int)
    parser.add_argument("--pooling", type=str)
    parser.add_argument("--pkg", type=str)
    parser.add_argument("--model_url", type=str, default="facebook/dinov3-vitl16-pretrain-lvd1689m")

    cfg = parser.parse_args()
    _, rank, _ = parallel_setup()

    if rank != 0:
        m = imgANN(cfg.model_name, cfg.pkg, cfg.input_size, dtype=torch.float16, attn_implementation='sdpa', repo_url=cfg.model_url)
        print(m)
        task_list = m.relevant_layers
    else:
        m = None
        task_list = get_relevant_output_layers(cfg.model_name)

    master_workers_queue(task_list, paths, ipca_movie_patches, *(m, cfg.n_components, cfg.batch_size, cfg.patches_per_frame, cfg.frames_step, cfg.patches_overhead_sampling, cfg.sq_size,)) 
