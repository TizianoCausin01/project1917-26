# %%
# /Users/tizianocausin/Desktop/1917_py_env/bin/activate
# %%
import torch
import cv2
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from transformers import ViTImageProcessor, ViTForImageClassification
from torchvision import models
import numpy as np
import h5py

# %%
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
ViT = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")


# %%

path2vid = "/Volumes/TIZIANO/stimuli/Project1917_movie_part3_24Hz.mp4"
reader = cv2.VideoCapture(path2vid)
# %%
conv_layers = [
    "real_conv_layer1",
    "real_conv_layer4",
    "real_conv_layer7",
    "real_conv_layer9",
    "real_conv_layer11",
]
conv_layers_idx = [0, 3, 6, 8, 10]

fc_layers = ["real_fc_layer2", "real_fc_layer5"]
fc_layers_idx = [1, 4]
# %%
output_len_conv = [193600, 139968, 64896, 43264, 43264]
rand_idx_conv = []
for len_repr in output_len_conv:
    rand_idx_conv.append(
        np.random.choice(np.arange(len_repr - 1), size=len_repr // 50, replace=False)
    )

output_len_fc = [46656, 32448]
rand_idx_fc = []
for len_repr in output_len_fc:
    rand_idx_fc.append(
        np.random.choice(np.arange(len_repr - 1), size=len_repr // 50, replace=False)
    )


# %%
def wrapper_hook(layer, rand_idx):
    def hook_func(module, input, output):
        out = output.detach().half().reshape(-1)
        out = out[rand_idx]
        feats[layer].append(
            out
        )  # half makes it become float16, reshape(-1) vectorizes it

    return hook_func


hook_handle = []
for conv_idx in range(len(conv_layers_idx)):
    hook_handle.append(
        alexnet.features[conv_layers_idx[conv_idx]].register_forward_hook(
            wrapper_hook(conv_layers[conv_idx], rand_idx_conv[conv_idx])
        )
    )


for fc_idx in range(len(fc_layers_idx)):
    hook_handle.append(
        alexnet.features[fc_layers_idx[fc_idx]].register_forward_hook(
            wrapper_hook(fc_layers[fc_idx], rand_idx_fc[fc_idx])
        )
    )


feats = {
    "real_conv_layer1": [],
    "real_conv_layer4": [],
    "real_conv_layer7": [],
    "real_conv_layer9": [],
    "real_conv_layer11": [],
    "real_fc_layer2": [],
    "real_fc_layer5": [],
}
# while True:
# %%
for i in range(10):
    print("NEW FRAME")
    ret, frame = reader.read()
    frame_tr = processor(images=frame, return_tensors="pt")
    with torch.no_grad():
        out = ViT(**frame_tr)
    logits = out.logits
    top5 = torch.topk(logits, k=5)
    top5_scores = top5.values
    top5_indices = top5.indices
    for i in range(len(top5_scores[0])):
        print(i)
        print(f"{ViT.config.id2label[top5_indices[0][i].item()]}")

    # %%
    if ret == False:
        break
    # end if ret==False:

    frame_rgb = cv2.cvtColor(
        frame, cv2.COLOR_BGR2RGB
    )  # converts to bgr to rgb color codes
    input_tensor = preprocess(frame_rgb).unsqueeze(
        0
    )  # unsqueeze adds the batch size in front of the img
    with torch.no_grad():
        alexnet(input_tensor)
# end for i in range(1):
# %%
irun = 5
path2mod = "/Volumes/TIZIANO/models"
with h5py.File(f"{path2mod}/Project1917_real_alexnet_run0{irun}.h5", "w") as f:
    # Iterate over dictionary items and save them in the HDF5 file
    for key, value in feats.items():
        f.create_dataset(key, data=value)  # Create a dataset for each key-value pair
# %%
