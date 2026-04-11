import os, yaml, sys
import joblib
from sklearn.decomposition import IncrementalPCA
from einops import rearrange
import numpy as np
import argparse
ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["useful_stuff_path"])
sys.path.append(paths["src_path"])
from image_processing.gaze_dep_models import save_ipca_patch
from useful_stuff.image_processing.computational_models import imgANN
from useful_stuff.general_utils.utils import print_wise
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str)
parser.add_argument("--pkg", type=str)
parser.add_argument("--n_components", type=int)
parser.add_argument("--sq_size", type=int)
parser.add_argument("--input_size", type=int)
parser.add_argument("--pooling", type=str)

cfg = parser.parse_args()
m = imgANN(cfg.model_name, cfg.pkg, cfg.input_size)
layers = m.get_relevant_layers()
for l in layers:
    fn = save_ipca_patch(paths, cfg.model_name, l, cfg.n_components, cfg.sq_size, cfg.pooling)
    ipca_obj = joblib.load(fn)
    if not hasattr(ipca_obj, "order"): # to avoid reshaping twice
        print_wise(f"Reshaping {l}")
        components = ipca_obj.components_
        in_size = m.get_layer_output_shape(l)
        intermed_size = (cfg.n_components,*in_size) 
        components = components.reshape(intermed_size, order="F")
        components = rearrange(components, 'n_comp ... -> n_comp (...)')
        ipca_obj.components_ = components
        ipca_obj.order = "C" 
        joblib.dump(ipca_obj, fn)
