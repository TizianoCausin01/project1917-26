import os, yaml, sys
import argparse
from torchvision.datasets import ImageFolder
ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
subjects = config['subjects']
sys.path.append(paths["src_path"])
from image_processing.gaze_dep_models import OF_wrapper
from parallel.parallel_funcs import master_workers_queue


#OF_wrapper(paths, 0, 4, cfg.eye_fs, cfg.mod_fs, cfg.sq_side, cfg.downsampled_sq_size,)
# e.g. to call it:
# mpiexec -np 5 python3 run_gaze_dep_pixelwise_luminance.py --sq_side=250 --downsampled_sq_size=50

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sq_side", type=int)
    parser.add_argument("--downsampled_sq_size", type=int)
    cfg = parser.parse_args()
    mod_fs = 24
    eye_fs = 50
    task_list = subjects
    model_name = "pixelwise_luminance"
    master_workers_queue(task_list, paths, OF_wrapper, *(eye_fs, mod_fs, cfg.sq_side, cfg.downsampled_sq_size)) 
