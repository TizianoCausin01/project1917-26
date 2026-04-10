import yaml, sys, os
import numpy as np
import h5py
import joblib
from sklearn.decomposition import IncrementalPCA
from torchvision.models.feature_extraction import create_feature_extractor
import torch.nn.functional as F
import torch
import cv2
ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["useful_stuff_path"])
sys.path.append("..")
from useful_stuff.general_utils import print_wise, TimeSeries, get_device
from useful_stuff.image_processing.utils import get_video_dimensions, read_video 
from useful_stuff.image_processing.computational_models import pool_features, get_layer_output_shape
from project_specific_utils.dataloader import load_eyetracking_data
from project_specific_utils.utils import run2part
"""
pad_frame
Pad a movie frame into a larger screen canvas at a given offset.
INPUT:
    - frame: np.ndarray[h, w, 3] -> RGB movie frame to embed.
    - movie_dims: tuple[int, int] -> (h, w) dimensions of the movie frame.
    - offset_dims: tuple[int, int] -> (h_offset, w_offset) top-left position of the frame
        inside the canvas.
    - canvas: np.ndarray[H, W, 3] | None (default=None) -> Existing canvas to write into.
        If None, a gray canvas is created.
    - screen_res: tuple[int, int] (default=(1080, 1920)) -> (H, W) resolution of the canvas if created internally.

OUTPUT:
    - canvas: np.ndarray[H, W, 3] -> frame embedded inside the screen-sized canvas.
"""
def pad_frame(frame: np.ndarray[3], movie_dims: tuple[int, int], offset_dims: tuple[int, int], canvas=None, screen_res=(1080, 1920)):
    if canvas is None:
        canvas = np.ones((screen_res[0], screen_res[1], 3), dtype=np.uint8) * 128
    # if canvas is None:
    canvas[offset_dims[0]:offset_dims[0]+movie_dims[0], offset_dims[1]:offset_dims[1]+movie_dims[1], :] = frame
    return canvas
# EOF


"""
extract_square_patch
Extract a square patch centered at (cx, cy).
Pads with fill_value if square goes outside canvas.
INPUT:
    - canvas: np.ndarray[H, W, C] -> the screen
    - cx, cy: int -> center coordinates (usually the gaze position) of the square (pixels)
    - side: int -> the square side length
    - fill_value: uint8 -> padding value
OUTPUT:
    - patch: np.ndarray[side, side, 3] -> the extracted patch in pixels
"""
def extract_square_patch(canvas: np.ndarray[3], cx: int, cy: int, side: int, fill_value=128):

    H, W, C = canvas.shape
    half_size = side//2
    # initialize patch
    patch = np.ones((side, side, C), dtype=canvas.dtype) * fill_value
    # canvas boundaries
    if np.any(np.isnan([cx, cy])):
        raise ValueError(f"Gaze coords are NaNs")
    # end if np.any(np.isnan([cx, cy])):
    if cx<0:
        cx = 0
    if cy < 0:
        cy = 0
    # end if cy < 0
    x_start_canvas = max(cx - half_size, 0)
    x_end_canvas   = min(cx + half_size, W)
    y_start_canvas = max(cy - half_size, 0)
    y_end_canvas   = min(cy + half_size, H)
    # corresponding patch boundaries
    x_start_patch = max(0, half_size - cx)
    x_end_patch   = x_start_patch + (x_end_canvas - x_start_canvas)
    y_start_patch = max(0, half_size - cy)
    y_end_patch   = y_start_patch + (y_end_canvas - y_start_canvas)
    patch[y_start_patch:y_end_patch, x_start_patch:x_end_patch, :] = canvas[y_start_canvas:y_end_canvas, x_start_canvas:x_end_canvas, :]
    return patch
# EOF


'''
sequential_gaze_dep_loop
Processes a single video frame in a sequential gaze-dependent model.
Reads the current frame, centers it on the screen canvas, extracts the gaze-centered
square patch, and applies the feature function.

INPUT:
    - cap: cv2.VideoCapture -> opened video capture object
    - xy_gaze: np.ndarray[2, T] -> gaze coordinates (x, y) per frame
    - frame_idx: int -> current frame index
    - sq_side: int -> side length of square patch
    - movie_dims: tuple[int, int] -> (vertical_offset, horizontal_offset) of the movie display from the whole screen (the first is smaller than the second)
    - offset_dims: tuple[int, int] -> (vertical_offset, horizontal_offset) of the movie display from the whole screen (the first is smaller than the second)
    - canvas: np.ndarray[H, W, 3] or None -> screen canvas
    - features: list -> list collecting frame-wise feature vectors
    - func: callable -> feature extraction function
    - rank: int -> process rank for logging
    - *args: additional positional arguments for func
    - **kwargs: additional keyword arguments for func

OUTPUT:
    - canvas: np.ndarray[H, W, 3] -> updated screen canvas so that we don't allocate new memory

SIDE EFFECTS:
    - features: list -> gets in-place modification
'''
def sequential_gaze_dep_loop(cap: cv2.VideoCapture, xy_gaze: TimeSeries, frame_idx: int, sq_side: int, movie_dims: Tuple[int, int], offset_dims: Tuple[int, int], canvas: np.ndarray, features: list, func, rank: int, sub, run, *args, **kwargs):
    xy = xy_gaze[frame_idx]
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {movie_path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    canvas = pad_frame(frame, movie_dims, offset_dims,)
    frame_patch = extract_square_patch(canvas, round(xy[0]), round(xy[1]), sq_side)
    func(frame_patch, features, *args, **kwargs)
    if frame_idx %1000 == 0: # to print out
        print_wise(f"sub {sub} run {run}, processed first {frame_idx}th frames", rank=rank)
    # end if frame_idx %1000 == 0:
    return canvas
# EOF


'''
sequential_gaze_dep_mod
Runs a full sequential gaze-dependent model over a movie.
Loads gaze data, iterates through frames, extracts gaze-centered patches,
computes features, and saves the resulting representation to disk.

INPUT:
    - paths: dict -> project path dictionary
    - rank: int -> process rank for logging
    - sub_num: int -> subject number
    - func: callable -> frame-wise feature extraction function
    - save_func: callable -> function generating output file name
    - sq_side: int -> side length of square patch
    - model_name: str -> name of the model
    - run: int -> run number
    - fs: int -> gaze sampling frequency
    - *args: additional positional arguments passed to func and save_func
    - screen_res: tuple[int, int] -> screen resolution (H, W)
    - secs_to_skip: int -> seconds to skip at movie start
    - **kwargs: additional keyword arguments passed to func and save_func

OUTPUT:
    - None
    (Saves dataset "vecrep" to an HDF5 file)
'''
def sequential_gaze_dep_mod(paths: dict[str: str], rank: int, sub_num: int, func, save_func, sq_side: int, model_name: str, run: int, fs, *args, screen_res=(1080, 1920), secs_to_skip=5, **kwargs): 
    movie_part = run2part(run)
    movie_fn = f"{paths['data_path']}/stimuli/Project1917_movie_part{movie_part}_24Hz.mp4"
    cap = cv2.VideoCapture(movie_fn)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, round(5*fps))
    h, w, frames_n = get_video_dimensions(cap)
    save_name = save_func(paths, model_name, sub_num, run, fps, sq_side, *args, **kwargs)
    if os.path.exists(save_name):
        print_wise(f"model already exists at {save_name}", rank=rank)
        return None
    # end if os.path.exists(save_name):
    xy_gaze, _ = load_eyetracking_data(paths, sub_num, run, fs, xy=True)
    xy_gaze.resample(fps)
    frames_n -= round(secs_to_skip*fps) +2 # to be on the safe side, because when downsampled, the number of gaze-datapoints exceeds the number of frames
    if frames_n > len(xy_gaze):
        raise IndexError(f"The number of frames ({frames_n}) is larger than the number of gaze datapoints ({len(xy_gaze)}) in sub {sub_num} run {run}")
    # end if frames_n > len(xy_gaze):

    offset_dims = ((screen_res[0] -h)//2 , ( screen_res[1] - w)//2)
    canvas = None
    features = []
    for frame_idx in range(frames_n):
        canvas = sequential_gaze_dep_loop(cap, xy_gaze, frame_idx, sq_side, (h,w), offset_dims, canvas, features, func, rank, sub_num, run, *args, **kwargs)
    # end for frame_idx in range(frames_n):
    
    features = np.stack(features, axis=1)
    with h5py.File(save_name, "w") as f:
        f.create_dataset("vecrep", data=features)
    # end with h5py.File(save_name, "w") as f:
    print_wise(f"model {model_name} saved at {save_name}", rank=rank)
# EOF

"""
wrapper_run_sequential_gaze_dep_mod
Just a wrapper of the above function to loop through the runs
"""
def wrapper_run_sequential_gaze_dep_mod(paths: dict[str: str], rank: int, sub_num: int, func, save_func, sq_side: int, model_name: str, fs, *args, screen_res=(1080, 1920), secs_to_skip=5, **kwargs): 
    print_wise(f"Start running {model_name} for sub {sub_num}", rank=rank)
    for irun in range(1, 7):
        sequential_gaze_dep_mod(paths, rank, sub_num, func, save_func, sq_side, model_name, irun, fs, *args, screen_res=(1080, 1920), secs_to_skip=5, )
    # end for irun in range(1, 7):
# EOF


'''
pixelwise_lum
Computes pixel-wise luminance features from a gaze-centered patch.
Resizes the patch, flattens it in MATLAB-style (column-major), and appends
the result to the feature list.

INPUT:
    - frame_patch: np.ndarray[H, W, 3] -> gaze-centered image patch
    - features: list -> list collecting flattened frame-wise features
    - sq_side_resized: int -> side length after resizing

OUTPUT:
    - None
    (Appends flattened feature vector to features in-place)
'''
def pixelwise_luminance(frame_patch, features, sq_side_resized):
    resized = cv2.resize(frame_patch, (sq_side_resized, sq_side_resized), interpolation=cv2.INTER_LINEAR)
    resized = resized.ravel(order='F')
    features.append(resized)
    # no need of return feats as its modified in-place
# EOF


'''
save_pixelwise_lum
Generates the output file name for the pixel-wise luminance model.
The end result is something like this:
    sub004_run01_pixelwise_luminance_gazedep_224x224rect_to_50x50_24Hz.h5
INPUT:
    - paths: dict -> project path dictionary
    - model_name: str -> name of the model
    - sub_num: int -> subject number
    - run: int -> run number
    - fs: float or int -> movie frame rate
    - sq_side: int -> original square patch side length
    - *args: positional arguments (expects resized side length at args[0])
    - **kwargs: additional keyword arguments (unused)

OUTPUT:
    - save_name: str -> full path to output .h5 file
'''
def save_pixelwise_luminance(paths, model_name, sub_num, run, fs, sq_side, *args, **kwargs):
    save_name = f"{paths['data_path']}/models/sub{sub_num:03d}_run{run:02d}_{model_name}_gazedep_{sq_side}x{sq_side}rect_to_{args[0]}x{args[0]}_{round(fs)}Hz.h5"
    return save_name
# EOF


"""
save_OF
Generate the filename for saving optical flow–based models.

INPUT:
    - paths: dict[str, str] -> dictionary containing base paths (expects 'data_path')
    - model_name: str -> name of the model (e.g., 'eyeOF', 'eyeOFdir', 'eyeOFmag')
    - sub_num: int -> subject number
    - run: int -> run index
    - fs: float -> sampling frequency (Hz)
    - sq_side: int -> original square side size
    - *args: tuple -> resized square side (args[0])

OUTPUT:
    - save_name: str -> full path to .h5 file where the model will be saved
"""
def save_OF(paths, model_name, sub_num, run, fs, sq_side, *args):
    save_name = f"{paths['data_path']}/models/sub{sub_num:03d}_run{run:02d}_{model_name}_gazedep_{sq_side}x{sq_side}rect_to_{args[0]}x{args[0]}_{round(fs)}Hz.h5"
    return save_name
# EOF 

"""
eyeOF_loop
Compute optical flow, magnitude, and normalized direction from a pixelwise video representation.
Wrapper to compute and save optical flow–based features for all runs of a subject.
It's "eye" because it computes also the OF generated by the eye movements.

INPUT:
    - pixelwise_mod: np.ndarray -> (pixels, n_frames), flattened RGB frames (Fortran order)
    - downsampled_sq_size: int -> spatial size (H = W) of reshaped frames
    - eps: float -> small constant to avoid division by zero (default: 1e-8)

OUTPUT:
    - OF_tot: np.ndarray -> (pixels*2, n_frames), flattened optical flow vectors (x,y)
    - OFmag_tot: np.ndarray -> (pixels, n_frames), optical flow magnitude
    - OFdir_tot: np.ndarray -> (pixels*2, n_frames), normalized flow direction (unit vectors)

NOTES:
    - Optical flow is computed between consecutive frames using Farneback method
    - First frame is duplicated to preserve alignment with input frames
    - All outputs are flattened in Fortran order
"""
def eyeOF_loop(pixelwise_mod, downsampled_sq_size, eps=1e-8):
    OF_tot = []
    OFmag_tot = []
    OFdir_tot = []
    flow = None
    old_frame = cv2.cvtColor(pixelwise_mod[:, 0].reshape((downsampled_sq_size, downsampled_sq_size,3), order='F'), cv2.COLOR_RGB2GRAY) # initialize old_frame with the first
    for i_frame in np.arange(pixelwise_mod.shape[1]-1): # because we go one frame ahead
        new_frame = cv2.cvtColor(pixelwise_mod[:, i_frame+1].reshape((downsampled_sq_size, downsampled_sq_size,3), order='F'), cv2.COLOR_RGB2GRAY) # one frame ahead because the flow at t is the function of t-1 and t (and then we shift all frames by 1 prepending the 0th OF twice)
        # https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga5d10ebbd59fe09c5f650289ec0ece5af 
        # flow.shape=(height, width, 2)
        flow = cv2.calcOpticalFlowFarneback(
            old_frame, new_frame,
            flow,       # previous flow
            pyr_scale=0.5,
            levels=3,
            winsize=15, # the bigger win_size, the blurrier the vector field (smooths over more pixels)
            iterations=3,
            poly_n=7, # the bigger poly_n, the blurrier the vector field
            poly_sigma=1.5, # adjusted as per documentation
            flags=0
        )
        # flow has shape (H, W, 2) — x, y displacement vectors
        OFmag = np.linalg.norm(flow, axis=-1) # (H, W, xy_components) -> (H, W)
        if np.any(OFmag==0): 
            ValueError("Mag=0")
        OFdir = flow / (OFmag[...,np.newaxis]+ eps)
        # vectorizes the results
        flow = flow.ravel(order='F')
        OFmag = OFmag.ravel(order='F')
        OFdir = OFdir.ravel(order='F')
        OF_tot.append(flow)
        OFmag_tot.append(OFmag)
        OFdir_tot.append(OFdir)
        old_frame = new_frame # updates old_frame
    # end for i_frame in np.arange(mod.shape[1]):
    # duplicates the first frame at the beginning (it's just one datapoint)  
    OF_tot.insert(0, OF_tot[0])
    OFmag_tot.insert(0, OFmag_tot[0])
    OFdir_tot.insert(0, OFdir_tot[0])
    OF_tot = np.stack(OF_tot, axis=1)
    OFmag_tot = np.stack(OFmag_tot, axis=1)
    OFdir_tot = np.stack(OFdir_tot, axis=1)
    return [OF_tot, OFmag_tot, OFdir_tot]
# EOF
    
"""
eyeOF_wrapper
Wrapper to compute and save optical flow–based features for all runs of a subject.
It's "eye" because it computes also the OF generated by the eye movements.

INPUT:
    - paths: dict[str, str] -> dictionary containing base paths
    - rank: int -> process rank (for logging)
    - sub_num: int -> subject number
    - fs: float -> sampling frequency (Hz)
    - sq_side: int -> original square side size
    - sq_size_resized: int -> resized square side for processing

OUTPUT:
    - None (saves .h5 files to disk)

NOTES:
    - Computes three models: optical flow (eyeOF), direction (eyeOFdir), magnitude (eyeOFmag)
    - Skips computation if all output files already exist
    - Loads pixelwise luminance representation as input
    - Saves outputs under dataset name 'vecrep'
"""
def eyeOF_wrapper(paths: dict[str:str], rank: int, sub_num: int, fs: float, sq_side: int, sq_size_resized: int,):
    model_names=['eyeOF', 'eyeOFdir', 'eyeOFmag']
    for irun in range(1, 7):
        OF_savenames = [save_OF(paths, mn, sub_num, irun, fs, sq_side, *(sq_size_resized,)) for mn in model_names]
        if all([os.path.exists(p) for p in OF_savenames]):
            print_wise(f"all models already exists at {OF_savenames[0]}", rank=rank)
        else:
            pixelwise_savename = save_pixelwise_luminance(paths, "pixelwise_luminance", sub_num, irun, fs, sq_side, *(sq_size_resized,))
            with h5py.File(pixelwise_savename, "r") as f:
                pixelwise_mod = f['vecrep'][:]
            # end with h5py.File(model_filename, "r") as f:
            features_list = eyeOF_loop(pixelwise_mod, sq_size_resized)
            for sn, feats in zip(OF_savenames, features_list):
                with h5py.File(sn, "w") as f:
                    f.create_dataset("vecrep", data=feats)
                # end with h5py.File(save_name, "w") as f:
            # end for sn, f in zip(OF_savenames, features_list):
            print_wise(f"model saved at {OF_savenames[0]}", rank=rank)
        # end if all([os.exists(p) for p in OF_savenames]):
    # end for irun in range(1, 7):
# EOF



"""
append_OF_list
Append current optical flow features to cumulative lists (per feature type).

INPUT:
    - tot_features_list: list[list] -> list of lists storing accumulated features (one per feature type)
    - curr_features: list[np.ndarray] -> current frame features (e.g., flow, magnitude, direction)

OUTPUT:
    - None (modifies tot_features_list in place)

NOTES:
    - Each feature is flattened in Fortran order before appending
    - Assumes consistent ordering between tot_features_list and curr_features
"""
def append_OF_list(tot_features_list, curr_features):
    for tot, curr in zip(tot_features_list, curr_features):
        # TODO decrease resolution
        tot.append(curr.ravel(order='F'))
    # end for idx, OF_type in enumerate(tot_features_list):
# EOF

"""
OF_0th_frame_pad
Duplicate the first frame of each feature list at the beginning.

INPUT:
    - tot_features_list: list[list] -> list of feature lists (one per feature type)

OUTPUT:
    - None (modifies lists in place)

NOTES:
    - Ensures alignment with original frame indexing (e.g., flow defined from t-1 → t)
"""
def OF_0th_frame_pad(tot_features_list: list[list]):
    for OF_type in tot_features_list:
        OF_type.insert(0, OF_type[0])
    # end for OF_type in tot_features_list:
# EOF

"""
stack_OF_list
Stack lists of optical flow features into arrays.

INPUT:
    - tot_features_list: list[list[np.ndarray]] -> list of feature lists (one per feature type)

OUTPUT:
    - tot_features_list: list[np.ndarray] -> each element has shape (features, n_frames)

NOTES:
    - Stacking is performed along axis=1 (time dimension)
"""
def stack_OF_list(tot_features_list):
    tot_features_list = [np.stack(f, axis=1) for f in tot_features_list]
    return tot_features_list
# EOF


"""
OF_inside
Compute gaze-centered optical flow features from a video stream.

INPUT:
    - frames_n: int -> number of frames to process
    - xy_gaze: np.ndarray -> (n_frames, 2), gaze coordinates (x, y) per frame
    - cap: cv2.VideoCapture -> video capture object
    - video_dims: tuple[int, int] -> original video dimensions (H, W)
    - offset_dims: tuple[int, int] -> padding offsets to center video on screen
    - sq_side: int -> side length of extracted square patch
    - sq_size_resized: int -> resized patch side length
    - rank: int -> process rank (for logging)

OUTPUT:
    - features_list: list[list[np.ndarray]] -> lists of features (one per type: flow, magnitude, direction)

NOTES:
    - Optical flow is computed between consecutive grayscale frames (Farneback method)
    - Flow is cropped around gaze location and resized to fixed resolution
    - Outputs:
        * OF_patch: (H, W, 2) → flow vectors
        * OFmag: (H, W) → magnitude
        * OFdir: (H, W, 2) → normalized direction (NaNs set to 0)
    - Features are flattened (Fortran order) and appended via append_OF_list
"""
def OF_inside(frames_n, xy_gaze, cap, video_dims, offset_dims, sq_side, sq_size_resized, rank):
    features_list = [[], [], []]
    ret, frame = cap.read()
    old_canvas = pad_frame(frame, video_dims, offset_dims)
    old_canvas = cv2.cvtColor(old_canvas, cv2.COLOR_BGR2GRAY)
    flow = None
    for frame_idx in range(1, frames_n):    
        xy = xy_gaze[frame_idx]
        ret, frame = cap.read()
        new_canvas = pad_frame(frame, video_dims, offset_dims)
        new_canvas = cv2.cvtColor(new_canvas, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            old_canvas, new_canvas,
            flow,       # initial flow
            pyr_scale=0.5,
            levels=3,
            winsize=15, # the bigger win_size, the blurrier the vector field (smooths over more pixels)
            iterations=3,
            poly_n=7, # the bigger poly_n, the blurrier the vector field
            poly_sigma=1.5, # adjusted as per documentation
            flags=0
        )
        OF_patch = extract_square_patch(flow, round(xy[0]), round(xy[1]), sq_side, fill_value=0)
        OF_patch = cv2.resize(OF_patch, (sq_size_resized, sq_size_resized), interpolation=cv2.INTER_LINEAR)
        OFmag = np.linalg.norm(OF_patch, axis=-1) # (H, W, xy_components) -> (H, W)
        OFdir = np.divide(
            OF_patch,
            OFmag[..., np.newaxis],
            out=np.zeros_like(OF_patch),
            where=OFmag[..., np.newaxis] != 0
        )
        if not np.isfinite(OFdir).all():
            raise ValueError("Infinite or nan values encountered in the computation of optical flow direction")
        # end if not np.isfinite(OFdir).all():
        append_OF_list(features_list, [OF_patch, OFmag, OFdir])
        # magnitude, normalization, NaNs == 0
        if frame_idx % 1000 ==0:
            print_wise(f"Processed frame {frame_idx}", rank=rank)
        # end if frame_idx % 1000 ==0:
    # end for frame_idx in range(1, frames_n):
    return features_list
# EOF

"""
sequential_OF_gaze_dep
Compute gaze-dependent optical flow features sequentially from video frames.

INPUT:
    - paths: dict[str, str] -> dictionary containing base paths
    - rank: int -> process rank (for logging)
    - sub_num: int -> subject number
    - model_names: list[str] -> names of output models (e.g., ['OF', 'OFmag', 'OFdir'])
    - run: int -> run index
    - eye_fs: float -> eye-tracking sampling frequency
    - mod_fs: float -> model sampling frequency
    - sq_side: int -> side length of extracted gaze-centered patch
    - sq_size_resized: int -> resized patch side length
    - screen_res: tuple[int, int] -> screen resolution (H, W)
    - secs_to_skip: int -> seconds to skip at start of video

OUTPUT:
    - features_list: list[np.ndarray] -> list of feature arrays (one per model), shape (features, n_frames)

NOTES:
    - Video frames are aligned with gaze data (resampled to video FPS)
    - Optical flow is computed on gaze-centered patches
    - First frame is duplicated to maintain temporal alignment
    - Returns None if all output files already exist
"""
def sequential_OF_gaze_dep(paths: dict[str: str], rank: int, sub_num: int, model_names: str, run: int, eye_fs: float, mod_fs: float, sq_side: int, sq_size_resized: int, screen_res=(1080, 1920), secs_to_skip=5,): 
    movie_part = run2part(run)
    movie_fn = f"{paths['data_path']}/stimuli/Project1917_movie_part{movie_part}_24Hz.mp4"
    cap = cv2.VideoCapture(movie_fn)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, round(5*fps))
    h, w, frames_n = get_video_dimensions(cap)
    OF_savenames = [save_OF(paths, mn, sub_num, run, mod_fs, sq_side, *(sq_size_resized,)) for mn in model_names]
    if all([os.path.exists(p) for p in OF_savenames]):
        print_wise(f"all models already exists at {OF_savenames[0]}", rank=rank)
        return None
    # end if all([os.path.exists(p) for p in OF_savenames]):
    xy_gaze, _ = load_eyetracking_data(paths, sub_num, run, eye_fs, xy=True)
    xy_gaze.resample(fps)
    frames_n -= round(secs_to_skip*fps) +2 # to be on the safe side, because when downsampled, the number of gaze-datapoints exceeds the number of frames
    if frames_n > len(xy_gaze):
        raise IndexError(f"The number of frames ({frames_n}) is larger than the number of gaze datapoints ({len(xy_gaze)}) in sub {sub_num} run {run}")
    # end if frames_n > len(xy_gaze):
    offset_dims = ((screen_res[0] -h)//2 , ( screen_res[1] - w)//2)
    features_list = OF_inside(frames_n, xy_gaze, cap, (h,w), offset_dims, sq_side, sq_size_resized, rank)
    OF_0th_frame_pad(features_list)
    features_list = stack_OF_list(features_list)
    return features_list
# EOF

"""
OF_wrapper
Wrapper to compute and save optical flow–based features for all runs of a subject.

INPUT:
    - paths: dict[str, str] -> dictionary containing base paths
    - rank: int -> process rank (for logging)
    - sub_num: int -> subject number
    - eye_fs: float -> eye-tracking sampling frequency
    - mod_fs: float -> model sampling frequency
    - sq_side: int -> side length of extracted gaze-centered patch
    - sq_size_resized: int -> resized patch side length

OUTPUT:
    - None (saves .h5 files to disk)

NOTES:
    - Computes three models: OF, OFmag, OFdir
    - Skips runs where all output files already exist
    - Saves each feature under dataset name 'vecrep'
"""
def OF_wrapper(paths: dict[str:str], rank: int, sub_num: int, eye_fs: float, mod_fs: float, sq_side: int, sq_size_resized: int,):
    model_names=['OF', 'OFmag', 'OFdir']
    for irun in range(1, 7):
        OF_savenames = [save_OF(paths, mn, sub_num, irun, mod_fs, sq_side, *(sq_size_resized,)) for mn in model_names]
        if all([os.path.exists(p) for p in OF_savenames]):
            print_wise(f"all models already exists at {OF_savenames[0]}", rank=rank)
        else:
            features_list = sequential_OF_gaze_dep(paths, rank, sub_num,  model_names, irun, eye_fs, mod_fs, sq_side, sq_size_resized)
            for sn, feats in zip(OF_savenames, features_list):
                with h5py.File(sn, "w") as f:
                    f.create_dataset("vecrep", data=feats)
                # end with h5py.File(save_name, "w") as f:
            # end for sn, f in zip(OF_savenames, features_list):
            print_wise(f"model saved at {OF_savenames[0]}", rank=rank)
        # end if all([os.exists(p) for p in OF_savenames]):
    # end for irun in range(1, 7):
# EOF


"""
extract_center_patches
Extract three square patches per frame, centered vertically and spaced horizontally.

INPUT:
    - v: np.ndarray -> (n_frames, H, W, C)
    - sq_size: int -> size of square patches

OUTPUT:
    - patches: list[np.ndarray] -> list of patches of shape (sq_size, sq_size, C)
"""
def extract_center_patches(v, sq_size):
    patches = []
    h, w, c = v[0].shape
    d = (w - 3 * sq_size) // 4
    cy = h // 2
    cx1 = 2*d + sq_size//2
    cx2 = cx1 + d + sq_size
    cx3 = cx2 + d + sq_size
    cxs = [cx1, cx2, cx3]
    for frame in v:
        for cx in cxs:
            patches.append(extract_square_patch(frame, cx, cy, sq_size))
    return patches
# EOF 

"""
sample_random_patches
Randomly sample a batch of frames (without replacement), convert to torch tensor, and remove them from the source list.


INPUT:
    - tot_frames: list[np.ndarray] -> list of frames (H, W, C)
    - batch_size: int -> number of frames to sample

OUTPUT:
    - chunk: torch.Tensor -> (batch_size, H, W, C)
    
SIDE EFFECT:
    - removes sampled frames from tot_frames
"""
def sample_random_patches(tot_frames, batch_size):
    frames_indices = np.random.choice(len(tot_frames), size=batch_size, replace=False) 
    chunk = [torch.from_numpy(tot_frames[i]) for i in frames_indices]            
    chunk = torch.stack(chunk) # (N,H,W,...)
    for i in sorted(frames_indices, reverse=True):
        del tot_frames[i] # deletes frames that are being used
    return chunk
# EOF


"""
capture_1917_movie_runs
Load all video files from the stimuli directory as OpenCV VideoCapture objects.

INPUT:
    - paths: dict -> must contain key 'data_path'

OUTPUT:
    - caps_list: list[cv2.VideoCapture] -> list of opened video capture objects
"""
def capture_1917_movie_runs(paths):
    stim_dir = f"{paths['data_path']}/stimuli"
    filenames = os.listdir(stim_dir)
    caps_list = []
    for fn in filenames:
        fn_path = f"{stim_dir}/{fn}"
        cap = cv2.VideoCapture(fn_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {fn_path}")
        # end if not cap.isOpened():
        caps_list.append(cap)
    # end for fn in filenames:
    return caps_list
# EOF

import torch

"""
extract_features_1917_movie
Preprocess a batch and extract features from a specified model layer.

INPUT:
    - batch: torch.Tensor -> (B, H, W, C)
    - feature_extractor: callable -> returns dict of layer activations
    - layer_name: str -> key for desired layer
    - input_size: int -> input resolution for preprocessing
    - pooling: str -> 'all' (flatten) or pooling method

OUTPUT:
    - features: np.ndarray -> (B, n_features)
"""
def extract_features_1917_movie(batch, feature_extractor, layer_name, input_size, pooling="all", device='cpu'):
    batch = preprocess_batch(batch, input_size, device=device)
    with torch.no_grad():
        features = feature_extractor(batch)[layer_name]
    features = features.cpu().detach().numpy()
    if pooling == 'all':
        features = features.reshape(features.shape[0], -1, order='F')
    else:
        features = pool_features(features, pooling)
    # end if pooling == 'all':
    return features
# EOF


"""
save_ipca_patch
Generate a filename for saving an Incremental PCA model based on movie patches.

INPUT:
    - paths: dict -> must contain key 'data_path'
    - model_name: str -> name of the model
    - layer_name: str -> CNN/ANN layer used for features
    - n_components: int -> number of PCA components
    - sq_size: int -> square patch size
    - pooling: str -> pooling method ('all' or specific)

OUTPUT:
    - savename: str -> full path to save the IPCA model
"""
def save_ipca_patch(paths, model_name, layer_name, n_components, sq_size, pooling):
    models_path = f"{paths['data_path']}/models"
    savename = f"{models_path}/{model_name}_{layer_name}_{n_components}components_{sq_size}x{sq_size}patch_{pooling}pool.pkl"
    return savename

"""
ipca_movie_patches
Perform incremental PCA on patches extracted from the 1917 movie across multiple runs.

INPUT:
    - paths: dict -> must contain 'data_path' for stimuli
    - rank: int -> process rank for printing/logging
    - layer_name: str -> CNN/ANN layer to extract features from
    - ANN: imgANN -> the imgANN object I created
    - n_components: int -> number of PCA components
    - batch_size: int -> number of patches per batch
    - patches_per_frame: int -> patches to sample per frame
    - frames_step: int -> frame subsampling step
    - patches_overhead_sampling: float -> extra patches fraction to ensure coverage
    - sq_size: int -> size of square patches
    - secs_to_skip: int (default=5) -> seconds to skip at movie start

OUTPUT:
    - None (saves the Incremental PCA model to disk)

PROCESS:
    - Opens movie captures and computes sampling strategy
    - Extracts center patches from sampled frames
    - Preprocesses batches and extracts features
    - Fits Incremental PCA incrementally over all batches
    - Saves the trained PCA model using `joblib`
"""
def ipca_movie_patches(paths, rank, layer_name, ANN, n_components, batch_size, patches_per_frame, frames_step, patches_overhead_sampling, sq_size, secs_to_skip=5):
    device = get_device()
    PCs_savename = save_ipca_patch(paths, ANN.get_model_name(), layer_name, n_components, sq_size, ANN.get_pooling())
    sub_batch_size = batch_size//3
    if os.path.exists(PCs_savename):
        print_wise(f"PCs already exist at {PCs_savename}", rank=rank)
        return None
    # end if os.path.exists(PCs_savename):
    layer_dim = ANN.get_layer_output_shape(layer_name)
    ANN.create_forward_hook(layer_names=[layer_name,])
    ipca_obj = IncrementalPCA(n_components=min(n_components, np.prod(layer_dim)), batch_size=batch_size)
    caps_list = capture_1917_movie_runs(paths)
    n_movies = len(caps_list)
    fps = caps_list[0].get(cv2.CAP_PROP_FPS)
    frames_to_skip = round(secs_to_skip*fps)
    frames_per_run = []
    for cap in caps_list:
        _, _, n_frames = get_video_dimensions(cap)
        frames_per_run.append(n_frames)
    # end for cap in caps_list:
    patches_to_read = batch_size + round(batch_size*patches_overhead_sampling) # the number of patches we have to read at every step
    patches_per_frame_per_all_movies = n_movies*patches_per_frame/frames_step # number of patches we'll sample by counting the three movies
    frames_to_read_per_movie = round(patches_to_read/patches_per_frame_per_all_movies) # the number of patches we have to read at every step
    max_frames = max(frames_per_run)
    min_frames = min(frames_per_run)
    batch_starts = np.arange(frames_to_skip, max_frames, frames_to_read_per_movie)
    tot_batch_n = len(batch_starts) + round(len(batch_starts)*patches_overhead_sampling)
    while True: # little check that we don't surpass the three videos in the first 10 steps
        np.random.shuffle(batch_starts)
        mask = batch_starts[:10] < min_frames - batch_size
        if np.all(mask):
            break
        # end if np.all(mask):
    # end while True:
    tot_frames = None
    for idx, start_f in enumerate(batch_starts): # sample from the whole movies
        start_s = start_f/fps
        for cap, tot_f in zip(caps_list, frames_per_run): # for each of the movies in the caps_list, it reads the frames at that point
            if start_f < tot_f: # enter here only if the current start is in the movie
                end_f = min(start_f+frames_to_read_per_movie, tot_f) # because we might surpass the end of the video
                end_s = end_f/fps
                v = read_video(paths, None, cap=cap, start=start_s, end=end_s, release=False, verbose=False)
                v = v[::frames_step]
                v = extract_center_patches(v, sq_size)
                if tot_frames is None:
                    tot_frames = v
                else:
                    tot_frames.extend(v)# = torch.concatenate((tot_frames, v), dim=0)
                # end if tot_frames is None:
        batch = sample_random_patches(tot_frames, batch_size)
        features_list = []
        for i in range(0, batch_size, sub_batch_size):
            sub_batch = batch[i:i+sub_batch_size]
            sub_batch = preprocess_batch(sub_batch, ANN.img_size, device=device)
            with torch.no_grad():
                ANN.model(sub_batch)
            f = ANN.features[layer_name].detach().cpu().numpy()
            features_list.append(f)
            ANN.features[layer_name] = None
            torch.cuda.empty_cache() 
        f_full = np.concatenate(features_list, axis=0)
        ipca_obj.fit(f_full)
        print_wise(f"processed batch {idx} of {tot_batch_n} features shape = {f_full.shape}", rank=rank)
    # end for start_f in batch_starts:
        
    for idx, b in enumerate(np.arange(0, len(tot_frames) - batch_size, batch_size)): # process the remaining overhead (skip the last one to maintain the batch size constant)
        batch = sample_random_patches(tot_frames, batch_size)
        features_list = []
        for i in range(0, batch_size, sub_batch_size):
            sub_batch = batch[i:i+sub_batch_size]
            sub_batch = preprocess_batch(sub_batch, ANN.img_size, device=device)
            with torch.no_grad():
                ANN.model(sub_batch)
            f = ANN.features[layer_name].detach().cpu().numpy()
            features_list.append(f)
            ANN.features[layer_name] = None
            torch.cuda.empty_cache() 
        f_full = np.concatenate(features_list, axis=0)
        ipca_obj.fit(f_full)
        print_wise(f"processed batch {idx + len(batch_starts)} of {tot_batch_n}, features shape = {f_full.shape}", rank=rank)

    joblib.dump(ipca_obj, PCs_savename)
    print_wise(f"Model successfully saved at {PCs_savename}", rank=rank)
# EOF


"""
save_ANN_features
Creates the savename for the ANN features
INPUT:
    - paths: {str: str} -> requires data_path
    - full_model_name: str -> the model name comprising also the layer name, like this: "{model_name}_{layer_name}"
    - fs: int -> the sampling frequency of the model
    - sub_num: int
    - run: int
    - n_components: int -> the number of components extracted from the ANN model
    - sq_side: int -> how big in pixels was the size of the square extracted
    - pooling: str -> how we pooled the ANN features ('mean', 'max', 'all')
OUTPUT:
    - save_name: str -> the full path to the file, like this: '/Users/tizianocausin/1917_local/models/sub003_run01_alexnet_classifier.2_1000components_allpooling_gazedep_384x384_24Hz.h5' 
"""
def save_ANN_features(paths, full_model_name, sub_num, run, fs, sq_side, n_components, pooling,):
    save_name = f"{paths['data_path']}/models/sub{sub_num:03d}_run{run:02d}_{full_model_name}_{n_components}components_{pooling}pooling_gazedep_{sq_side}x{sq_side}_{round(fs)}Hz.h5"
    return save_name
# EOF


"""
gaze_dep_ANN_extraction
Extract gaze-dependent ANN features from a movie by sampling square patches
centered at gaze positions and projecting them onto PCA components.

INPUT:
    - paths: dict[str, str] -> Dictionary containing data and output paths.
    - rank: int -> Process rank (for logging).
    - sub_num: int -> Subject identifier.
    - sq_side: int -> Side length of the square patch (in pixels).
    - model: torch.nn.Module -> Pretrained model used for feature extraction.
    - model_name: str -> Name of the model (used for saving).
    - layer_name: str -> Model layer from which features are extracted.
    - n_components: int -> Number of PCA components.
    - pooling: str -> Pooling strategy applied to features.
    - PCs: np.ndarray -> PCA projection matrix (features projected via @ PCs).
    - input_size: int -> Input size expected by the model.
    - run: int -> Run index (1–6).
    - eye_fs: int -> Eyetracking sampling frequency.
    - device: str -> Device for computation ("cpu", "cuda", "mps").
    - screen_res: tuple[int, int] (default=(1080, 1920)) -> Screen resolution (H, W).
    - secs_to_skip: int (default=5) -> Seconds skipped at the beginning of the movie.

OUTPUT:
    - None -> Saves extracted feature matrix to disk as an HDF5 file:
        dataset name: "vecrep", shape (n_components, n_frames).

NOTES:
    - Each frame is padded to screen resolution and sampled at gaze coordinates.
    - Features are extracted per frame and projected onto PCA space.
    - If output already exists, computation is skipped.
"""
def gaze_dep_ANN_extraction(paths: dict[str: str], rank: int, sub_num: int, sq_side: int, ANN, n_components, PCs,  run: int, eye_fs, screen_res=(1080, 1920), secs_to_skip=5,): 
    device = ANN.get_device()
    movie_part = run2part(run)
    movie_fn = f"{paths['data_path']}/stimuli/Project1917_movie_part{movie_part}_24Hz.mp4"
    cap = cv2.VideoCapture(movie_fn)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, round(5*fps))
    h, w, frames_n = get_video_dimensions(cap)
    save_names = {}
    for l in ANN.relevant_layers:
        save_fn = save_ANN_features(paths, f"{ANN.model_name}_{l}", sub_num, run, round(fps), sq_side, n_components, ANN.pooling,)
        if os.path.exists(save_fn):
            print_wise(f"model already exists at {save_fn}", rank=rank)
        else:
            save_names[l] = save_fn
        # end if os.path.exists(save_fn):
    # end for l in ANN.relevant_layers:
    layers_to_compute = list(save_names.keys())
    if not layers_to_compute:
        return None
    ANN.create_forward_hook(layer_names=layers_to_compute)
    xy_gaze, _ = load_eyetracking_data(paths, sub_num, run, eye_fs, xy=True)
    xy_gaze.resample(fps)
    frames_n -= round(secs_to_skip*fps) +2 # to be on the safe side, because when downsampled, the number of gaze-datapoints exceeds the number of frames
    if frames_n > len(xy_gaze):
        raise IndexError(f"The number of frames ({frames_n}) is larger than the number of gaze datapoints ({len(xy_gaze)}) in sub {sub_num} run {run}")
    # end if frames_n > len(xy_gaze):

    offset_dims = ((screen_res[0] -h)//2 , ( screen_res[1] - w)//2)
    canvas = None
    features = {l: [] for l in layers_to_compute}
    for frame_idx in range(frames_n):
        xy = xy_gaze[frame_idx]
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {frame_idx} from {movie_fn}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        canvas = pad_frame(frame, (h,w), offset_dims,)
        frame_patch = extract_square_patch(canvas, round(xy[0]), round(xy[1]), sq_side)
        frame_patch = torch.from_numpy(frame_patch)
        frame_patch = preprocess_batch(frame_patch[None,:, :, :], ANN.img_size, device=device)        
        with torch.no_grad():
            ANN.model(frame_patch)
        for l, f in ANN.get_features().items():
            f = f.cpu().detach().numpy()
            f_proj = np.squeeze(f @ PCs[l]) 
            features[l].append(f_proj)
        # end for l, f in curr_features:
        if frame_idx%10 == 0:
            print_wise(f"processed frame {frame_idx} of {frames_n} in run {run} of {ANN.model_name}")
    # end for frame_idx in range(frames_n):

    for l, final_f in features.items():
        final_f = np.stack(final_f, axis=1)
        with h5py.File(save_names[l], "w") as f:
            f.create_dataset("vecrep", data=final_f)
        # end with h5py.File(save_name, "w") as f:
        print_wise(f"model {ANN.model_name} saved at {save_names[l]}", rank=rank)
# EOF



"""
ANN_extraction_projection_1917_wrapper
Wrapper function to run gaze-dependent ANN feature extraction across all runs
using precomputed PCA components.

INPUT:
    - paths: dict[str, str] -> Dictionary containing data and output paths.
    - rank: int -> Process rank (for logging).
    - sub_num: int -> Subject identifier.
    - model: torch.nn.Module -> Pretrained model used for feature extraction.
    - sq_side: int -> Side length of the square patch (in pixels).
    - input_size: int -> Input size expected by the model.
    - model_name: str -> Name of the model.
    - layer_name: str -> Model layer from which features are extracted.
    - n_components: int -> Number of PCA components.
    - pooling: str -> Pooling strategy applied to features.
    - eye_fs: int -> Eyetracking sampling frequency.
    - device: str -> Device for computation ("cpu", "cuda", "mps").
    - screen_res: tuple[int, int] (default=(1080, 1920)) -> Screen resolution (H, W).
    - secs_to_skip: int (default=5) -> Seconds skipped at the beginning of each run.

OUTPUT:
    - None -> Calls `gaze_dep_ANN_extraction` for each run (1–6) and saves results to disk.

NOTES:
    - PCA components are loaded once and reused across runs.
    - Projection matrix is transposed before use.
"""
def ANN_extraction_projection_1917_wrapper(paths: dict[str: str], rank: int, sub_num: int, ANN, sq_side: int, n_components, PCs_dict, eye_fs, screen_res=(1080, 1920), secs_to_skip=5,): 
    print_wise(f"Start running {ANN.model_name} for sub {sub_num}", rank=rank)
    for irun in range(1, 7):
        gaze_dep_ANN_extraction(paths, rank, sub_num, sq_side, ANN, n_components, PCs_dict, irun, eye_fs, screen_res=(1080, 1920), secs_to_skip=5, )
    # end for irun in range(1, 7):
# EOF



"""
preprocess_batch
Convert a batch of images to model-ready format: channel-first, resized, and normalized.

INPUT:
    - batch: torch.Tensor -> (B, H, W, C)
    - input_size: int -> target spatial size (e.g. 224, 384)
    - m: list[float] -> mean for normalization (default: ImageNet)
    - std: list[float] -> std for normalization (default: ImageNet)

OUTPUT:
    - batch: torch.Tensor -> (B, 3, input_size, input_size)
"""
def preprocess_batch(batch, input_size, m=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], device='cpu'):
    # 1. Convert to float and scale to [0,1] if needed
    batch = batch.to(device)
    batch = batch.permute(0,3,1,2)
    if batch.dtype != torch.float32:
        batch = batch.float()
    if batch.max() > 1.0:
        batch = batch / 255.0
    # 2. Resize (keeps it simple: direct resize)
    batch = F.interpolate(
        batch,
        size=(input_size, input_size),
        mode='bilinear',
        align_corners=False
    )
    # 3. Normalize with ImageNet stats
    mean = torch.tensor(m, device=batch.device)[None, :, None, None]
    std  = torch.tensor(std, device=batch.device)[None, :, None, None]
    batch = (batch - mean) / std
    return batch
# EOF

