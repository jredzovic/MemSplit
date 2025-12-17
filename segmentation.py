
# memsplit_toolkit/segmentation.py
import numpy as np
from skimage import filters, segmentation, measure
from skimage.measure import label as connected_components


def watershed_from_scores(scores: np.ndarray, seed_threshold: float) -> np.ndarray:
    """
    Run full watershed pipeline on a 3D score volume and return a relabelled label image.
    This matches your current _run_watershed_segmentation logic.
    """
    if scores.ndim != 3:
        raise ValueError("Watershed requires a 3D score volume.")

    # your pipeline:
    binary_mask = scores > 0
    gradient = filters.sobel(scores)
    binary_seeds = scores > seed_threshold
    seeds = measure.label(binary_seeds)
    labels = segmentation.watershed(gradient, seeds, mask=binary_mask)
    connected = labels  # changed to reflect that watershed output is already labeled
    # The output of watershed is already connected components kind of output with different labels so do not need to do another connected components step
    #connected = measure.label(labels)

    # downcast to save memory, similar to what you do elsewhere
    max_label = int(connected.max())
    if max_label <= np.iinfo(np.uint16).max:
        return connected.astype(np.uint16, copy=False)
    return connected.astype(np.uint32, copy=False)


def relabel_connected_components(data: np.ndarray, connectivity: int = 1) -> np.ndarray:
    """
    Relabel the whole volume (or 2D image) using connected components.
    Mirrors _run_connected_components.
    """
    result = connected_components(data, connectivity=connectivity)
    max_label = int(result.max())
    if max_label <= np.iinfo(np.uint16).max:
        return result.astype(np.uint16, copy=False)
    return result.astype(np.uint32, copy=False)


def clean_z_range(data: np.ndarray, z_start: int, z_stop: int, background: int = 0) -> np.ndarray:
    """
    Zero-out all voxels outside [z_start, z_stop) along axis 0.
    Mirrors _clean_z_range's core array logic.
    """
    if data.ndim != 3:
        raise ValueError("Z cleaning requires a 3D label layer.")

    z_dim = data.shape[0]
    if not (0 <= z_start < z_stop <= z_dim):
        raise ValueError(f"Invalid z range: {z_start}–{z_stop} for volume with {z_dim} slices.")

    cleaned = data.copy()
    cleaned[:z_start, :, :] = background
    cleaned[z_stop:, :, :] = background
    return cleaned
