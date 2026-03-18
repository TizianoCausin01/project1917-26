import os, yaml, sys
import argparse
from torchvision.datasets import ImageFolder
ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
subjects = config['subjects']
sys.path.append(paths["src_path"])
from image_processing.gaze_dep_models import eyeOF_wrapper
from parallel.parallel_funcs import master_workers_queue

#eyeOF_wrapper(paths: dict[str:str], rank: int, sub_num: int, fs: float, sq_side: int, sq_size_resized: int,):


# e.g. to call it:
# mpiexec -np 5 python3 run_gaze_dep_pixelwise_luminance.py --sq_side=250 --downsampled_sq_size=50

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sq_side", type=int)
    parser.add_argument("--downsampled_sq_size", type=int)

    cfg = parser.parse_args()
    mod_fs = 24
    task_list = subjects
    model_name = "pixelwise_luminance"
    master_workers_queue(task_list, paths, eyeOF_wrapper, *(mod_fs, cfg.sq_side, cfg.downsampled_sq_size)) 
