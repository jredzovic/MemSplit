
# memsplit_toolkit/segmentation.py
import numpy as np
from skimage import filters, segmentation, measure
from skimage.measure import label as connected_components


def _compact_label_dtype(data: np.ndarray) -> np.ndarray:
    """Store labels in a compact unsigned integer dtype."""
    max_label = int(data.max())
    if max_label <= np.iinfo(np.uint16).max:
        return data.astype(np.uint16, copy=False)
    return data.astype(np.uint32, copy=False)


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
    return _compact_label_dtype(connected)


def relabel_connected_components(data: np.ndarray, connectivity: int = 1) -> np.ndarray:
    """
    Relabel the whole volume (or 2D image) using connected components.
    Mirrors _run_connected_components.
    """
    result = connected_components(data, connectivity=connectivity)
    return _compact_label_dtype(result)


def split_single_label_watershed(
    data: np.ndarray,
    scores: np.ndarray,
    selected_label: int,
    seed_threshold: float = 1.5,
) -> tuple[np.ndarray, int]:
    """
    Refine one existing label with an ROI-limited watershed and assign fresh IDs.

    This mirrors the notebook workflow:
    - isolate one label as the ROI
    - use scores > 0 inside that ROI as the watershed mask
    - create seeds from scores > seed_threshold
    - run watershed on the Sobel gradient of the score volume
    """
    if data.shape != scores.shape:
        raise ValueError(
            "Label data and score volume must have the same shape for selected-label watershed."
        )

    if selected_label == 0:
        raise ValueError("Background label 0 cannot be refined.")

    label_mask = data == selected_label
    if not np.any(label_mask):
        raise ValueError(f"No voxels found for label {selected_label}.")

    binary_mask_roi = (scores > 0) & label_mask
    if not np.any(binary_mask_roi):
        raise ValueError(
            f"Label {selected_label} has no positive-score voxels inside the selected ROI."
        )

    binary_seeds_roi = (scores > float(seed_threshold)) & label_mask
    seeds = measure.label(binary_seeds_roi)
    num_components = int(seeds.max())

    if num_components == 0:
        raise ValueError(
            f"No watershed seeds found for label {selected_label}. "
            "Lower the absolute seed threshold or check the selected score volume."
        )

    gradient = filters.sobel(scores)
    labels_refined = segmentation.watershed(gradient, seeds, mask=binary_mask_roi)

    new_data = data.copy()
    new_data[label_mask] = 0

    offset = int(data.max()) + 1
    refined_mask = labels_refined > 0
    new_data[refined_mask] = labels_refined[refined_mask] + offset - 1
    return _compact_label_dtype(new_data), num_components


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
