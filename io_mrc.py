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


def load_mrc_voxel_size(path: str) -> float | None:
    """
    Read voxel size from an MRC header and return a single isotropic value.
    Returns None if the header does not provide a usable positive voxel size.
    """
    with mrcfile.open(path, permissive=True) as mrc:
        voxel_size = mrc.voxel_size

        values = []
        for axis in ("x", "y", "z"):
            try:
                value = float(getattr(voxel_size, axis))
            except (AttributeError, TypeError, ValueError):
                value = None
            if value is not None and np.isfinite(value) and value > 0:
                values.append(value)

        if not values:
            return None

        return float(np.mean(values))


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
