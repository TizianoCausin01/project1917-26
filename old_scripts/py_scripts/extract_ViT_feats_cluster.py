from PIL import Image
from transformers import pipeline
from transformers import (
    AutoImageProcessor,
    ViTConfig,
    ViTModel,
    ViTForImageClassification,
)
import torch
import sys
import numpy as np
import cv2
import h5py

pipeline = pipeline(
    task="image-classification",
    model="/leonardo/home/userexternal/tcausin0/virtual_envs/1917_py_env/vit_model",
    torch_dtype=torch.float16,
    device=-1,
)

# Initializing a ViT vit-base-patch16-224 style configuration
configuration = ViTConfig()

# Initializing a model (with random weights) from the vit-base-patch16-224 style configuration
model = ViTModel(configuration)

# Accessing the model configuration
configuration = model.config

image_processor = AutoImageProcessor.from_pretrained(
    "google/vit-base-patch16-224"
)  # resizes and crops the img just picking up the center
image_processor.image_mean = [
    0.3806,
    0.4242,
    0.3794,
]  # custom parameters for normalization
image_processor.image_std = [0.2447, 0.2732, 0.2561]
model = ViTForImageClassification.from_pretrained("/leonardo/home/userexternal/tcausin0/virtual_envs/1917_py_env/vit_model")
num_patches = 14**2
layer_dim = 768
out_dim = (num_patches + 1) * layer_dim
encoder_blocks = 12
rand_idx = []
for block in range(encoder_blocks):
    rand_idx.append(
        np.random.choice(np.arange(out_dim), size=out_dim // 10, replace=False)
    )


def wrapper_hook(layer, rand_idx):
    def hook_func(module, input, output):
        out = output.detach().half().reshape(-1)
        out = out[rand_idx]
        feats[str(layer)].append(
            out
        )  # half makes it become float16, reshape(-1) vectorizes it

    return hook_func


hook_handle = []
# here we are hooking the output of of the 12 transformers blocks
# they all end with an MLP 768>3072>768 -> importantly, it processes each of the patches indepently and identically
# so the output will be (batch_size, num_tokens, hidden_dim).
# Leaving aside the batch_size, the tokens (embeddings for the patches) are on the rows, they are (224^2 / 16^2) +1 = 196+1 (the +1 is given by the classification token, a summary of the img)
#for block_idx in range(encoder_blocks):
#    hook_handle.append(
#        model.vit.encoder.layer[block_idx].output.register_forward_hook(
#            wrapper_hook(block_idx, rand_idx[block_idx])
#        )
#    )


path2mod = "/leonardo_work/Sis25_piasini/tcausin/Project1917/models"
#path2mod = "/Volumes/TIZIANO/models"
runs = [1, 2, 3]
for irun in runs:
    feats = {str(i): [] for i in range(encoder_blocks + 1)}  # adds the logits
    feats = {str(12): []}
    print("irun:", irun)
    path2vid = f"/leonardo_scratch/fast/Sis25_piasini/tcausin/Project1917/stimuli/Project1917_movie_part{irun}_24Hz.mp4"
    #path2vid = f"/Volumes/TIZIANO/stimuli/Project1917_movie_part{irun}_24Hz.mp4"
    print("path2vid", path2vid)
    sys.stdout.flush()
    reader = cv2.VideoCapture(path2vid)
    count=0
    while True:
    #for i in range(3):
        ret, frame = reader.read()
        count+=1
        if ret == False:
            break
        # end if ret==False:
        frame_rgb = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )  # converts to bgr to rgb color codes
        img_frame = Image.fromarray(frame_rgb)
        inputs = image_processor(frame_rgb, return_tensors="pt")
        with torch.no_grad():
            feats["12"].append(model(**inputs).logits)
        
        print(f"\n FRAME{count}")
        sys.stdout.flush()
    # end while True:
    with h5py.File(f"{path2mod}/Project1917_ViT_logits_run0{irun}.h5", "w") as f:
    # Iterate over dictionary items and save them in the HDF5 file
        for key, value in feats.items():
            f.create_dataset(
                key, data=value
            )  # Create a dataset for each key-value pair
