import numpy as np


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
