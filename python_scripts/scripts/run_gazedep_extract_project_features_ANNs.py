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
from useful_stuff.parallel.parallel_funcs import master_workers_queue
from useful_stuff.general_utils.utils import get_device
from useful_stuff.image_processing.computational_models import get_relevant_output_layers, load_torchvision_model, load_timm_model

# e.g. to call it:
# mpiexec -np 5 python3 run_gazedep_extract_project_features_ANNs.py --sub_to_proc 1 --eye_fs 50 --model_name vit_l_16 --layer_idx 10 --n_components 1000 --sq_size 384 --input_size 384 --pooling all --pkg timm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub_to_proc", type=int) # eventually take it out
    parser.add_argument("--eye_fs", type=int)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--n_components", type=int)
    parser.add_argument("--sq_size", type=int)
    parser.add_argument("--input_size", type=int)
    parser.add_argument("--pooling", type=str)
    parser.add_argument("--pkg", type=str)
    parser.add_argument("--model_url", type=str, default="facebook/dinov3-vitl16-pretrain-lvd1689m")

    cfg = parser.parse_args()
    task_list = config["subjects"][:cfg.sub_to_proc]
    if rank != 0:
        m = imgANN(cfg.model_name, cfg.pkg, cfg.input_size, dtype=torch.float16, attn_implementation='sdpa', repo_url=cfg.model_url)
        PCs_dict = {}
        for l in ANN.relevant_layers:
            ipca_path = save_ipca_patch(paths, m.model_name, l, cfg.n_components, cfg.sq_side, cfg.pooling,) 
            ipca_obj = joblib.load(ipca_path)
            PCs = ipca_obj.components_.T
        # end for l in ANN.relevant_layers:
    else:
        m = None
        PCs_dict = None
    # end if rank != 0:

    master_workers_queue(task_list, paths, ANN_extraction_projection_1917_wrapper, *(m, cfg.sq_size, cfg.n_components, cfg.eye_fs)) 
