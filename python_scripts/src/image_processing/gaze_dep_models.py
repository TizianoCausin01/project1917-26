import sys, os
import numpy as np
import h5py
import cv2
sys.path.append("..")
from general_utils import print_wise, TimeSeries
from project_specific_utils.dataloader import load_eyetracking_data
from project_specific_utils.utils import run2part
from image_processing.utils import get_video_dimensions
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
    cap.set(cv2.CAP_PROP_POS_FRAMES, round(5*fps)-1)
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
    cap.set(cv2.CAP_PROP_POS_FRAMES, round(5*fps)-1)
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
