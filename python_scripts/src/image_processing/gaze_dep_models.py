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
    x_start_canvas = max(cx - half_size, 0)
    x_end_canvas   = min(cx + half_size, W)
    y_start_canvas = max(cy - half_size, 0)
    y_end_canvas   = min(cy + half_size, H)
    # corresponding patch boundaries
    x_start_patch = max(0, half_size - cx)
    x_end_patch   = x_start_patch + (x_end_canvas - x_start_canvas)
    y_start_patch = max(0, half_size - cy)
    y_end_patch   = y_start_patch + (y_end_canvas - y_start_canvas)
    # copy region
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
def sequential_gaze_dep_loop(cap: cv2.VideoCapture, xy_gaze: TimeSeries, frame_idx: int, sq_side: int, movie_dims: Tuple[int, int], offset_dims: Tuple[int, int], canvas: np.ndarray, features: list, func, rank: int, *args, **kwargs):
    xy = xy_gaze[frame_idx]
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {movie_path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    canvas = pad_frame(frame, movie_dims, offset_dims,)
    frame_patch = extract_square_patch(canvas, round(xy[0]), round(xy[1]), sq_side)
    func(frame_patch, features, *args, **kwargs)
    if frame_idx %1000 == 0: # to print out
        print_wise(f"processed first {frame_idx}th frames", rank=rank)
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
    movie_fn = f"{paths['data_dir']}/stimuli/Project1917_movie_part{movie_part}_24Hz.mp4"
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
    frames_n -= round(secs_to_skip*fps)
    offset_dims = ((screen_res[0] -h)//2 , ( screen_res[1] - w)//2)
    canvas = None
    features = []
    for frame_idx in range(frames_n):
        canvas = sequential_gaze_dep_loop(cap, xy_gaze, frame_idx, sq_side, (h,w), offset_dims, canvas, features, func, rank, *args, **kwargs)
    # end for frame_idx in range(frames_n):
    features = np.stack(features, axis=1)
    with h5py.File(save_name, "w") as f:
        f.create_dataset("vecrep", data=features)
    # end with h5py.File(save_name, "w") as f:
    print_wise(f"model {model_name} saved at {save_name}")
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
def pixelwise_lum(frame_patch, features, sq_side_resized):
    resized = cv2.resize(frame_patch, (sq_side_resized, sq_side_resized), interpolation=cv2.INTER_LINEAR)
    resized.ravel(order='F')
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
def save_pixelwise_lum(paths, model_name, sub_num, run, fs, sq_side, *args, **kwargs):
    save_name = f"{paths['data_dir']}/models/sub{sub_num:03d}_run{run:02d}_{model_name}_gazedep_{sq_side}x{sq_side}rect_to_{args[0]}x{args[0]}_{round(fs)}Hz.h5"
    return save_name
# EOF
