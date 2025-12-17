# memsplit_toolkit/io_mrc.py
import numpy as np
import mrcfile


def load_score_volume(path: str) -> np.ndarray:
    """
    Load the MRC score volume, similar to _run_watershed_segmentation.
    """
    with mrcfile.open(path, permissive=True) as mrc:
        # explicit float32 to be safe for filters.sobel, etc.
        return mrc.data.astype(np.float32, copy=True)


def save_segmentation(path: str, data: np.ndarray, voxel_size: float):
    """
    Save segmentation to MRC, with the same dtype fallback logic as _save_segmentation.
    """
    data = np.asarray(data)
    dtypes = (np.uint8, np.uint16, np.uint32)

    last_error = None
    for dtype in dtypes:
        try:
            with mrcfile.new(path, overwrite=True) as mrc:
                mrc.set_data(data.astype(dtype))
                mrc.voxel_size = (voxel_size, voxel_size, voxel_size)
            return
        except Exception as e:
            last_error = e
            continue

    # If we got here, all attempts failed
    raise RuntimeError(f"Failed to save MRC with dtypes {dtypes}: {last_error}")
