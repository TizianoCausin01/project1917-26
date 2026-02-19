# %%
# /Users/tizianocausin/Desktop/1917_py_env/bin/activate
# %%
import torch
import cv2
import torchvision.transforms as transforms
from torchvision import models
#import imageio
import numpy as np
import h5py
print("inside")
#alexnet = models.alexnet(pretrained=True).eval()
alexnet = models.alexnet(weights=True).eval()
preprocess = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.3806, 0.4242, 0.3794], std=[0.2447, 0.2732, 0.2561]
        ),  # Normalization for pretrained model
    ]
)

#reader = imageio.get_reader(path2vid)
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

for irun in range(3):
    print("irun:", irun)
    path2vid = "/leonardo_scratch/fast/Sis25_piasini/tcausin/Project1917/stimuli/Project1917_movie_part{irun+1}_24Hz.mp4"
    reader = cv2.VideoCapture(path2vid)
    feats = {
        "conv_layer1": [],
        "conv_layer4": [],
        "conv_layer7": [],
        "conv_layer9": [],
        "conv_layer11": [],
        "fc_layer2": [],
        "fc_layer5": [],
    }
    count = 0
    while True:
        count += 1
        ret, frame = reader.read()
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

    path2mod = "/leonardo_scratch/fast/Sis25_piasini/tcausin/Project1917/models"
    with h5py.File(f"{path2mod}/Project1917_alexnet_run0{irun+1}.h5", "w") as f:
    # Iterate over dictionary items and save them in the HDF5 file
        for key, value in feats.items():
            f.create_dataset(key, data=value)  # Create a dataset for each key-value pair

