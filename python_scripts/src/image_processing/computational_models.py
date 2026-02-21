import os, yaml, sys
import numpy as np
import torch
import re
from pathlib import Path
import joblib
import h5py
import cv2
from scipy.io import savemat
from scipy.ndimage import zoom
from scipy.special import logsumexp
from torchvision.models.feature_extraction import create_feature_extractor
from sklearn.decomposition import PCA
sys.path.append("../DeepGaze")
import deepgaze_pytorch

sys.path.append("..")
from image_processing.utils import read_video, resize_video_array, pool_features
from general_utils.utils import print_wise, decode_matlab_strings, create_RDM


"""
make_corners
For plotting the coords on a picture
"""
def make_corners(x1, y1, x2, y2):
    points = np.array([
        [x1, y1],  # top-left
        [x2, y1],  # top-right
        [x2, y2],  # bottom-right
        [x1, y2],  # bottom-left
        [x1, y1]   # close
    ])
    return points


"""
get_height_width
To get the height and width of a box
"""
def get_height_width(x1, y1, x2, y2):
    w = x2 - x1
    h = y2 - y1
    return h, w


"""
compute_center
To compute the center of a box
"""
def compute_center(x1, y1, x2, y2):
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return cx, cy


"""
make_square_box
From a rectangular box, it makes it square. It's used in the context of face boxes before
scaling them.
"""
def make_square_box(x1, y1, x2, y2):
    h, w = get_height_width(x1, y1, x2, y2)
    cx, cy = compute_center(x1, y1, x2, y2)
    side = max(w, h) # make it square by expanding the smaller dimension
    # New coordinates (keep center fixed)
    new_x1 = cx - side / 2
    new_x2 = cx + side / 2
    new_y1 = cy - side / 2
    new_y2 = cy + side / 2
    return new_x1, new_y1, new_x2, new_y2


"""
scale_box
Increases/decreases proportionally the height and width of a box. scale is the factor with which to scale. scale > 1 , the box increases; scale < 1 the box decreases.
"""
def scale_box(x1, y1, x2, y2, scale):
    h, w = get_height_width(x1, y1, x2, y2)
    cx, cy = compute_center(x1, y1, x2, y2)
    new_h = h * scale
    new_w = w * scale
    new_x1 = cx - new_w / 2
    new_x2 = cx + new_w / 2
    new_y1 = cy - new_h / 2
    new_y2 = cy + new_h / 2
    return new_x1, new_y1, new_x2, new_y2


"""
estimate_head_box
From a person box, it estimates the head position. The guesses are that it's gonna be at the top n% (top_h) along the y-axis and in the center (cnt_width) of the x-axis.
"""
def estimate_head_box(x1, y1, x2, y2, top_h=0.4, cnt_width=0.5):
    h, w = get_height_width(x1, y1, x2, y2)
    new_y1 = y1
    new_y2 = y1 + top_h * h
    x_center = (x1 + x2) / 2
    new_width = cnt_width * w
    new_x1 = x_center - new_width / 2
    new_x2 = x_center + new_width / 2
    return new_x1, new_y1, new_x2, new_y2


"""
compute_face_box
Computes the face box given a face model and a scaling factor (see scale_box). 
In case of no detections found or confidence < .75 , it will raise an IndexError that will be caught by detect_faces.
face_model is usually imported like this:

from huggingface_hub import hf_hub_download
face_model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
face_model = YOLO(face_model_path)
"""
def compute_face_box(frame, face_model, scale):
    results = face_model(frame, classes=[0], verbose=False)  # Class 0 = person
    if len(results[0].boxes) == 0:
        raise IndexError("No detections found.")
    confidence = round(results[0].boxes[0].conf[0].item(), 3)
    if confidence < 0.75:
        raise IndexError
    # end if confidence < 0.75:
    box = results[0].boxes.xyxy[0] 
    x1, y1, x2, y2 = box
    new_x1, new_y1, new_x2, new_y2 = make_square_box(x1, y1, x2, y2)
    new_x1, new_y1, new_x2, new_y2 = scale_box(new_x1, new_y1, new_x2, new_y2, scale)
    new_x1, new_y1, new_x2, new_y2 = map(int, [new_x1, new_y1, new_x2, new_y2])
    return new_x1, new_y1, new_x2, new_y2, confidence


"""
compute_occluded_face_box
Computes a face box in the case of no face detection by the face_model. It estimates the approximate position of a face based on the person box 
"""
def compute_occluded_face_box(frame, person_model):
    results = person_model(frame, classes=[0], verbose=False)
    if len(results[0].boxes) == 0:
        raise IndexError("No detections found.")
    confidence = round(results[0].boxes[0].conf[0].item(), 3)
    if confidence < 0.7:
        raise IndexError
    # end if confidence < 0.7:
    x1, y1, x2, y2 = results[0].boxes.xyxy[0]
    # Original width and height
    new_x1, new_y1, new_x2, new_y2 = estimate_head_box(x1, y1, x2, y2, top_h=0.4, cnt_width=0.5)
    new_x1, new_y1, new_x2, new_y2 = map(int, [new_x1, new_y1, new_x2, new_y2])
    return new_x1, new_y1, new_x2, new_y2, confidence


"""
smooth_face_detection
Takes care of the situations in which there is a too abrupt change in the face detection
INPUT:
    - coords: np.ndarray (6, time) -> its rows are these [face_presence, confidence, x1, y1, x2, y2] , face presence being 0 if no face was detected, 1 if a face was detected by a person model, 2 if a face was detected by the person model
"""
def smooth_face_detection(coords):
    for i in range(1, coords.shape[1] - 1): # loops over all the timepoints except from the 0th and the last, because they don't have either a previous or a successive point
        if coords[0, i-1] == coords[0, i+1] and coords[0, i] != coords[0, i-1]: # if the previous and successive points are the same, but still they are different from the point to be considered, then we gotta change the current point
            if coords[0, i] == 0: # if the model didn't detect any face but still before and after it did, we plug the previous coords
                coords[1:, i] = coords[1:, i-1]
            # end if a[0, i] == 0:
            coords[0, i] = coords[0, i-1]
        # end if a[0, i-1] == a[0, i+1] and a[0, i] != a[0, i-1]:
    # end for i in range(1, a.shape[1] - 1):
    return coords


"""
detect_faces
Loops through all the frames of a video and detects one face for each of them. If the face model doesn't reliably detect any face, it falls back to the person model, and if no person is detected with high confidence, it says that there is no face. face_model detections are normal faces, person_model detections are occluded faces. 
INPUT:
    - video: np.ndarray (t, h, w, c) -> the video, read with read_video in image_processing.utils
    - face_model -> usually imported like this:
          from huggingface_hub import hf_hub_download
          face_model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
          face_model = YOLO(face_model_path)
    - person_model -> usually imported like this:
          person_model_path = f'{path_to_weights}/yolov8n.pt'
          person_model = YOLO(person_model_path)
    - scale: float -> the degree to which we have to increase the face box detected by the face model

OUTPUT:
    - coords: np.ndarray (6, t) -> its rows are these [face_presence, confidence, x1, y1, x2, y2] , face presence being 0 if no face was detected, 1 if a face was detected by a person model, 2 if a face was detected by the person model

"""
def detect_faces(video, face_model, person_model, scale):
    coords = []
    for i in range(video.shape[0]):
        frame = video[i,:,:,:]
        try: # tries to detect a face with the face_model
            x1, y1, x2, y2, confidence = compute_face_box(frame, face_model, scale=scale)
            face_presence = 1 # face present
            confidence = round(confidence, 3)
        except IndexError: # if face_model detects no face
            try: # tries  to detect a face with the person_model
                x1, y1, x2, y2, confidence = compute_occluded_face_box(frame, person_model)
                confidence = round(confidence, 3)
                face_presence = 2 # occluded
            except IndexError: # if person_model detects no person
                x1, y1, x2, y2, confidence = np.nan, np.nan, np.nan, np.nan, np.nan 
                face_presence = 0 # face absent
            # end except IndexError:
        # end except IndexError:
        coords.append(np.array([face_presence, confidence, x1, y1, x2, y2])); # updates coords
    # end for i in range(video.shape[0]):
    coords = np.stack(coords, axis=1) # converts it into an array
    coords = smooth_face_detection(coords) # smooths it so that we don't have isolated points
    return coords
# EOF


"""
par_detect_faces
Another version for the function above adapted for parallel processing
"""
def par_detect_faces(paths, rank, fn, face_model, person_model, scale):
    outfn = f"{paths['livingstone_lab']}/tiziano/models/human_face_detection_{fn[:-4]}.mat" # [:-4] slice to take off the mp4 extension
    if os.path.exists(outfn):
        print_wise(f"model already exists at {outfn}", rank=rank)
        return None
    else:
        video = read_video(paths, rank, fn, vid_duration=0)
        coords = detect_faces(video, face_model, person_model, scale)
        print_wise(f"model saved at {outfn}", rank=rank)
        savemat(outfn, {"coords": coords})


"""
compute_centerbias
Creates centerbias for deepgaze by loading it from the model_weights folder
"""
def compute_centerbias(paths, h, w):
    centerbias_template = np.load(f'{paths['livingstone_lab']}/tiziano/model_weights/deep_gaze/centerbias_mit1003.npy') # downloaded at https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/centerbias_mit1003.npy
    # rescale to match image size
    centerbias = zoom(centerbias_template, (h/centerbias_template.shape[0], w/centerbias_template.shape[1]), order=0, mode='nearest')
    # renormalize log density
    centerbias -= logsumexp(centerbias)
    centerbias = torch.from_numpy(centerbias) #.float().to(device)
    centerbias = centerbias.unsqueeze(0)
    return centerbias
#EOF


"""
prepare_dg_input
prepares the input for deep gaze such that it's [B C H W] and torch. B=1
"""
def prepare_dg_input(frame):
    input = frame.transpose(2,0,1)
    input = torch.from_numpy(input) #.float().to(device)
    input = input.unsqueeze(0)
    return input
#EOF


"""
preprocess_log_density
First it brings the log density into probability mass, and then it reduces the size of the PMF, 
renormalizes and flattens according to matlab's order
"""
def preprocess_log_density(log_density_prediction, new_dims):
    if type(log_density_prediction) !=  np.ndarray:
        log_density_prediction = log_density_prediction.numpy()
    pmf = np.exp(log_density_prediction)/np.sum(np.exp(log_density_prediction))
    pmf = cv2.resize(pmf.squeeze(), new_dims)
    pmf_flat = pmf.flatten(order="F")
    pmf_flat = pmf_flat.astype(np.float16)
    return pmf_flat


"""
dg_pass
Computes the deep gaze output and translates it back to probability from log probability.
"""
def dg_pass(input, model, centerbias, new_dims):
    with torch.no_grad():
        log_density_prediction = model(input, centerbias)
    d_flat = preprocess_log_density(log_density_prediction, new_dims) 
    return d_flat
#EOF


"""
compute_dg_saliency
Wrapper to compute the deep-gaze visual saliency in parallel
"""
def compute_dg_saliency(paths, rank, fn, model, resize_factor):
    outfn = f"{paths['livingstone_lab']}/tiziano/models/dgIIE_{fn[:-4]}.npz" # [:-4] slice to take off the mp4 extension
    if os.path.exists(outfn):
        print_wise(f"model already exists at {outfn}", rank=rank)
        return None
    else: 
        video = read_video(paths, rank, fn)
        h, w = video.shape[1:3] 
        new_dims = (round(w*resize_factor), round(h*resize_factor))
        centerbias = compute_centerbias(paths, h, w)
        video_saliency = []
        for i_frame in range(video.shape[0]):
            current_frame = video[i_frame, :, :, :]
            input = prepare_dg_input(current_frame)
            dg_saliency = dg_pass(input, model, centerbias, new_dims)
            video_saliency.append(dg_saliency)
            if i_frame%10 == 0: # such that every tenth frame it prints out the progression
                print_wise(f"frame {i_frame} computed", rank=rank)
        video_saliency = np.stack(video_saliency, axis=1)
        #savemat(outfn, {"features" : video_saliency}) 
        np.savez_compressed(outfn, data=video_saliency)
        print_wise(f"model saved at {outfn}", rank=rank)
#EOF


def ICF_setup(paths):
    tf.reset_default_graph()
    tf.disable_v2_behavior()
    check_point = f"{paths['livingstone_lab']}/tiziano/model_weights/deep_gaze/ICF.ckpt"  # DeepGaze II
    new_saver = tf.train.import_meta_graph('{}.meta'.format(check_point))
    input_tensor = tf.get_collection('input_tensor')[0]
    # in case we wanted the center bias
    # centerbias_tensor = tf.get_collection('centerbias_tensor')[0]
    # log_density = tf.get_collection('log_density')[0]
    log_density_wo_centerbias = tf.get_collection('log_density_wo_centerbias')[0]
    return check_point, new_saver, input_tensor, log_density_wo_centerbias


def compute_ICF_saliency(paths, rank, fn, resize_factor, check_point, new_saver, input_tensor, log_density_wo_centerbias):
    outfn = f"{paths['livingstone_lab']}/tiziano/models/ICF_{fn[:-4]}.npz" # [:-4] slice to take off the mp4 extension
    if os.path.exists(outfn):
        print_wise(f"model already exists at {outfn}", rank=rank)
        return None
    else:
        check_point, new_saver, input_tensor, log_density_wo_centerbias = ICF_setup(paths)
        video = read_video(paths, rank, fn, vid_duration=0)
        features = ICF_loop(video, resize_factor, check_point, new_saver, input_tensor, log_density_wo_centerbias)
        np.savez_compressed(outfn, data=features)
        print_wise(f"model saved at {outfn}", rank=rank)


def ICF_loop(video, resize_factor, check_point, new_saver, input_tensor, log_density_wo_centerbias    ):
     video_saliency = []
     h, w = video.shape[1:3] 
     for i_frame in range(video.shape[0]):
         input = video[i_frame, :,:,:]
         with tf.Session() as sess:
             new_saver.restore(sess, check_point)
             new_dims = (round(w*resize_factor), round(h*resize_factor))
             log_density_prediction = sess.run(log_density_wo_centerbias, {
             input_tensor: input[np.newaxis, :,:,:],
             })  
         # end with tf.Session() as sess:
         d_flat = preprocess_log_density(log_density_prediction, new_dims)
         video_saliency.append(d_flat)
     # end for i_frame in range(video.shape[0]):
     video_saliency = np.stack(video_saliency, axis=1)
     return video_saliency


"""
pass_video
What this function does:
1. the target video
2. makes it suitable for the model (i.e. reduces the size to (224, 224, 3) and normalizes)
3. does the model forward pass
INPUTS:
    - paths: dict -> paths to directories
    - rank: int -> worker id for logging
    - feature_extractor: torch model -> feature extractor object
    - layer_name: str -> layer from which to extract features
    - fn: str -> video filename
    - device: torch.device -> device to run the model on
    - max_len: int, optional -> maximum duration of video (in seconds)
    - new_height: int, optional -> height to resize frames
    - new_width: int, optional -> width to resize frames

OUTPUT:
    - feats: np.ndarray -> (frames, features) array of extracted features
"""
def pass_video(paths, rank, feature_extractor, layer_name, fn, device, max_len=20, new_height=224, new_width=224):
    video = read_video(paths, rank, fn, vid_duration=max_len)
    inputs = resize_video_array(video, new_height, new_width, normalize=True)
    inputs = torch.from_numpy(inputs).float().to(device)
    inputs = inputs.permute(0, 3, 1, 2)
    with torch.no_grad():
        feats = feature_extractor(inputs)[layer_name]
        feats = feats.reshape(feats.size(0), -1).cpu().numpy()
    return feats
# EOF


"""
compute_torchvision_model
Processes all videos in a directory through a given model and layer, optionally projects features onto PCA components, and saves the result.

INPUTS:
    - paths: dict -> paths to directories
    - rank: int -> worker id for logging
    - layer_name: str -> layer from which to extract features
    - model_name: str -> name of the torchvision model
    - model: torch model -> pretrained model
    - max_len: int, optional -> maximum duration of video (in seconds)
    - pca_opt: bool, optional -> whether to project features using precomputed PCA components

SIDE EFFECTS:
    - Saves feature arrays for each video as compressed .npz files
    - Prints status messages with worker rank
"""
def compute_torchvision_model(paths, rank, layer_name, model_name, model, device, max_len=20, pca_opt=True):
    feature_extractor = create_feature_extractor(model, return_nodes=[layer_name]).to(device)
    videos_dir = f"{paths['livingstone_lab']}/Stimuli/Movies/all_videos"
    models_path = f"{paths['livingstone_lab']}/tiziano/models"
    all_files = os.listdir(videos_dir) 
    if pca_opt:
        ipca_ydx = joblib.load(f"{models_path}/YDX_{model_name}_{layer_name}_ipca_1000_PCs.pkl")
        ipca_img = joblib.load(f"{models_path}/IMG_{model_name}_{layer_name}_ipca_1000_PCs.pkl")
        ipca_faceswap = joblib.load(f"{models_path}/faceswap_{model_name}_{layer_name}_ipca_1000_PCs.pkl")
    # end if pca_opt:
    for fn in all_files:
        outfn = f"{models_path}/{model_name}_{layer_name}_{fn[:-4]}.npz"
        if os.path.exists(outfn):
            print_wise(f"model already exists at {outfn}", rank=rank)
        else:
            if "YDX" in fn:
                curr_ipca = ipca_ydx     
            elif "IMG" in fn:
                curr_ipca = ipca_img
            else:
                curr_ipca = ipca_faceswap
            # end if "YDX" in fn:
            feats = pass_video(paths, rank, feature_extractor, layer_name, fn, device, max_len=max_len)
            feats_proj = feats @ curr_ipca.components_.T if pca_opt else feats
            feats_proj = feats_proj.T
            np.savez_compressed(outfn, data=feats_proj)
            print_wise(f"model of size {feats_proj.shape} saved at {outfn}", rank=rank)
        # end if os.path.exists(outfn):
    # end for fn in all_files:
# EOF


"""
img_dataloader_feature_extraction_loop
Iterates over an image DataLoader and extracts features from a given model layer.

What this function does:
1) Iterates over batches of images from a PyTorch DataLoader
2) Performs a forward pass through a feature extractor
3) Flattens and stores the features batch by batch
4) Concatenates all features into a single array

INPUT:
- rank: int -> process rank (used for controlled printing)
- feature_extractor: torch.nn.Module -> model wrapped with create_feature_extractor
- dataloader: torch.utils.data.DataLoader -> image dataloader

OUTPUT:
- all_feats: np.ndarray (n_images, n_features) -> extracted features
"""
def img_dataloader_feature_extraction_loop(rank, feature_extractor, layer_name, dataloader, device, pooling='all'):
    all_feats = []
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            feats =feature_extractor(images.to(device))[layer_name]
            feats = pool_features(feats, pooling)
            feats = feats.reshape(feats.size(0), -1).cpu().numpy()
            all_feats.append(feats)
            print_wise(f"loaded batch {batch_idx} of shape {feats.shape}", rank=rank)
    all_feats = np.concatenate(all_feats, axis=0)
    return all_feats
# EOF

"""
img_feats_extraction
Computes and stores image features, PCA, and RDM for a given model layer.

What this function does:
1) Checks whether PCA, features, and RDM files already exist
2) Extracts image features from the specified model layer
3) Reorders features to match monkey presentation order
4) Computes the RDM from full features
5) Fits PCA and saves explained variance ratios
6) Projects features onto PCA space and saves them

INPUT:
- paths: dict -> dictionary with base paths
- rank: int -> process rank
- layer_name: str -> model layer used for feature extraction
- model_name: str -> name of the model
- model: torch.nn.Module -> pretrained ANN
- dataloader: DataLoader -> image dataloader
- mapping_idx: list[int] -> index mapping from ANN order to monkey order
- monkey_name: str -> monkey identifier
- date: str -> experiment date
- n_components: int -> number of PCA components
- device: torch.device -> cpu / cuda / mps

OUTPUT:
- None
(Saves PCA explained variance, projected features, and RDM to disk)
"""
def img_feats_extraction(paths, rank, layer_name, model_name, model, dataloader, mapping_idx, monkey_name, date, img_size, n_components, device):
    pca_save_name = f"{paths['livingstone_lab']}/tiziano/models/{monkey_name}_{date}_{model_name}_{img_size}_{layer_name}_{n_components}PCs.pkl"
    feats_save_name = f"{paths['livingstone_lab']}/tiziano/models/{monkey_name}_{date}_{model_name}_{img_size}_{layer_name}_features.npz"
    RDM_save_name = f"{paths['livingstone_lab']}/tiziano/models/{monkey_name}_{date}_{model_name}_{img_size}_{layer_name}_RDM.npz"
    paths_exist = all([
        os.path.exists(pca_save_name),
        os.path.exists(feats_save_name),
        os.path.exists(RDM_save_name),
    ])
    if paths_exist:
        print_wise(f"feature already computed at {feats_save_name}")
    else:
        feature_extractor = create_feature_extractor(model, return_nodes=[layer_name]).to(device)
        all_feats = img_dataloader_feature_extraction_loop(rank, feature_extractor, layer_name, dataloader, device)
        all_feats = all_feats[mapping_idx, :]
        RDM_vec = create_RDM(all_feats.T)
        np.savez_compressed(RDM_save_name, RDM_vec)
        print_wise(f"saved RDM at {RDM_save_name}", rank=rank)
        pca = PCA(n_components=n_components)
        pca.fit(all_feats)
        #np.savez_compressed(pca_save_name, pca.explained_variance_ratio_)
        joblib.dump(pca, pca_save_name)
        print_wise(f"saved pca explained ratio at {pca_save_name}", rank=rank)
        #joblib.dump(pca, pca_save_name)
        all_feats_redu = pca.transform(all_feats)
        np.savez_compressed(feats_save_name, all_feats_redu.T)
        print_wise(f"saved features at {feats_save_name}", rank=rank)
# EOF


def img_feats_extraction_pooling(paths, rank, layer_name, model_name, model, dataloader, dataset_name, img_size, pooling, device):
    feats_save_name = f"{paths['livingstone_lab']}/tiziano/models/{dataset_name}_{model_name}_{img_size}_{layer_name}_features_{pooling}pool.npz"
    if os.path.exists(feats_save_name):
        print_wise(f"feature already computed at {feats_save_name}")
    else:
        feature_extractor = create_feature_extractor(model, return_nodes=[layer_name]).to(device)
        all_feats = img_dataloader_feature_extraction_loop(rank, feature_extractor, layer_name, dataloader, device, pooling=pooling)
        np.savez_compressed(feats_save_name, all_feats.T)
        print_wise(f"saved features at {feats_save_name}", rank=rank)
# EOF


"""
rename_talia_dataset
just renaming the names the same way I did in the folder also in the uniqueImages file, 
otherwise I wouldn't be able to do the correct mapping. 
We add an underscore between the image name and the number and we take off the spaces.
"""
def rename_talia_dataset(monkey_presentation_order):
    monkey_presentation_order_renamed = []
    for f in monkey_presentation_order:
        # Step 1: insert underscore before first number following a letter
        newname = re.sub(r'([a-zA-Z])([0-9])', r'\1_\2', f)
        # Step 2: remove spaces
        newname = newname.replace(' ', '')
        # Rename if changed
        monkey_presentation_order_renamed.append(newname)
    # end for f in monkey_presentation_order:
    return monkey_presentation_order_renamed
# EOF


"""
map_image_order_from_ann_to_monkey
Creates an index mapping to align ANN image order with monkey presentation order.

What this function does:
1) Loads the list of images presented to the monkey from a MATLAB file
2) Decodes MATLAB string references into Python strings
3) Removes duplicate image names while preserving order
4) Extracts the ANN image presentation order from the dataset
5) Computes the index mapping from monkey order to ANN order

INPUT:
- paths: dict -> dictionary with base paths
- monkey_name: str -> monkey identifier
- date: str -> experiment date
- dataset: torchvision.datasets.ImageFolder -> ANN image dataset

OUTPUT:
- mapping_idx: list[int] -> indices to reorder ANN features to monkey order
"""
def map_image_order_from_ann_to_monkey(paths, monkey_name, date, dataset):
    allimgs_path = f"{paths['livingstone_lab']}/tiziano/data/{monkey_name}_allimages{date}.mat"
    with h5py.File(allimgs_path, "r") as f:
        try:
            refs = f["allimages"][:]      # shape (N, 1) of object refs
        except KeyError:
            refs = f["uniqueImage"][:]
        # end try:
        monkey_presentation_order = decode_matlab_strings(f, refs)
        monkey_presentation_order = sorted(set(monkey_presentation_order))
    ann_presentation_order = [os.path.basename(path) for path, _ in dataset.samples] # creates the order with which images are presented to the ANN
    if os.path.basename(Path(dataset.root))=="talia_20each_tizi": # little detour because I have changed the filenames for talia_20each_tizi
        monkey_presentation_order = rename_talia_dataset(monkey_presentation_order)
    # end if dataset=="talia_20each_tizi":
    mapping_idx = [ann_presentation_order.index(x) for x in monkey_presentation_order] # Creates a mapping from the monkey to the ann presentation order
    newly_ordered_ann = [ann_presentation_order[i] for i in mapping_idx]
    assert newly_ordered_ann == monkey_presentation_order
    return mapping_idx # by applying this to the ann features we'll get the same order as the monkeys'
# EOF
