import os, yaml, sys
import argparse
from torchvision.datasets import ImageFolder
ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
subjects = config['subjects']
sys.path.append(paths["src_path"])
from image_processing.gaze_dep_models import wrapper_run_sequential_gaze_dep_mod, pixelwise_luminance, save_pixelwise_luminance
from parallel.parallel_funcs import master_workers_queue

# e.g. to call it:
# mpiexec -np 5 python3 run_gaze_dep_pixelwise_luminance.py --sq_side=250 --downsampled_sq_size=50

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sq_side", type=int)
    parser.add_argument("--downsampled_sq_size", type=int)

    cfg = parser.parse_args()
    gaze_fs = 50
    task_list = subjects
    model_name = "pixelwise_luminance"
    master_workers_queue(task_list, paths, wrapper_run_sequential_gaze_dep_mod, *(pixelwise_luminance, save_pixelwise_luminance, cfg.sq_side, model_name, gaze_fs, *(cfg.downsampled_sq_size,))) 
