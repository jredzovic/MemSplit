# memsplit_toolkit/label_ops.py
import numpy as np
from skimage.measure import label as connected_components


def split_selected_labels_cc(
    data: np.ndarray,
    selected_labels: list[int],
    connectivity: int = 1,
) -> tuple[np.ndarray, int]:
    """
    Split the union of selected labels into connected components, assigning new IDs.

    This is your _split_selected_label logic, but pure and testable.
    Returns (new_data, num_components).
    """
    if not selected_labels:
        raise ValueError("No labels selected to split.")

    mask = np.isin(data, selected_labels)
    if not np.any(mask):
        raise ValueError("No voxels found for selected labels.")

    components = connected_components(mask, connectivity=connectivity)
    num_components = int(components.max())
    if num_components == 0:
        return data.copy(), 0

    new_data = data.copy()
    start_id = int(data.max()) + 1

    for i in range(1, num_components + 1):
        new_data[(components == i) & mask] = start_id
        start_id += 1

    # remove old labels inside the mask where no component was found
    new_data[(components == 0) & mask] = 0
    return new_data, num_components


def merge_labels(
    data: np.ndarray,
    selected_labels: list[int],
    target_label: int,
) -> np.ndarray:
    """
    Merge selected_labels into target_label, with the same safe-ID logic you had.
    Returns a new array.
    """
    if not selected_labels:
        return data

    data = data.copy()
    existing_labels = np.unique(data)

    # find a safe temporary ID
    safe_id = int(existing_labels.max() + 1)

    if target_label != 0 and target_label in existing_labels and target_label not in selected_labels:
        # move target_label away to avoid collision
        data[data == target_label] = safe_id

    for lbl in selected_labels:
        if lbl != target_label:
            data[data == lbl] = target_label

    return data
