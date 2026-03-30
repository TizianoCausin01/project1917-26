import os, yaml, sys
import argparse
from torchvision.datasets import ImageFolder
ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
subjects = config['subjects']
sys.path.append(paths["src_path"])
sys.path.append(paths["useful_stuff_path"])
from image_processing.gaze_dep_models import ANN_extraction_projection_1917_wrapper
from parallel.parallel_funcs import master_workers_queue
from useful_stuff.general_utils.utils import get_device
from useful_stuff.image_processing.utils import get_relevant_output_layers, load_torchvision_model, load_timm_model

# e.g. to call it:
# mpiexec -np 5 python3 run_extract_project_features_ANNs.py --sub_to_proc 1 --eye_fs 50 --model_name vit_l_16 --layer_idx 10 --n_components 1000 --sq_size 384 --input_size 384 --pooling all --pkg timm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub_to_proc", type=int) # eventually take it out
    parser.add_argument("--eye_fs", type=int)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--layer_idx", type=int)
    parser.add_argument("--n_components", type=int)
    parser.add_argument("--sq_size", type=int)
    parser.add_argument("--input_size", type=int)
    parser.add_argument("--pooling", type=str)
    parser.add_argument("--pkg", type=str)

    cfg = parser.parse_args()
    device = get_device(verbose=True)
    task_list = config["subjects"][:cfg.sub_to_proc]
    if cfg.pkg == 'torchvision':
        model = load_torchvision_model(cfg.model_name, device)
    elif cfg.pkg == 'timm':
        model = load_timm_model(cfg.model_name, device)
    # end if cfg.pkg == 'torchvision':
    layers = get_relevant_output_layers(cfg.model_name, cfg.pkg)
    layer_name = layers[cfg.layer_idx]
    master_workers_queue(task_list, paths, ANN_extraction_projection_1917_wrapper, *(model, cfg.sq_size, cfg.input_size, cfg.model_name, layer_name, cfg.n_components, cfg.pooling, cfg.eye_fs, device)) 
