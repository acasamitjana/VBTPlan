import warnings

import torch
import numpy as np

from scipy.ndimage.morphology import distance_transform_edt, distance_transform_cdt, binary_erosion

from type_utils import *
import fn_utils

def ignore_background(y_pred, y):
    """
    This function is used to remove background (the first channel) for `y_pred` and `y`.

    Args:
        y_pred: predictions. As for classification tasks,
            `y_pred` should has the shape [BN] where N is larger than 1. As for segmentation tasks,
            the shape should be [BNHW] or [BNHWD].
        y: ground truth, the first dim is batch.

    """

    y = y[:, 1:] if y.shape[1] > 1 else y  # type: ignore[assignment]
    y_pred = y_pred[:, 1:] if y_pred.shape[1] > 1 else y_pred  # type: ignore[assignment]
    return y_pred, y


def get_mask_edges(
    seg_pred,
    seg_gt,
    label_idx=1,
    crop=True,
):
    '''
    Args:
        seg_pred: predictions. Should have a 3D shape
        seg_gt: ground-truth. Should have a 3D shape
        label_idx: if multi-class, then specify the label
        crop: whether to crop to speed up the process
    '''

    # move in the funciton to avoid using all the GPUs
    if seg_pred.shape != seg_gt.shape:
        raise ValueError(f"seg_pred and seg_gt should have same shapes, got {seg_pred.shape} and {seg_gt.shape}.")


    # If not binary images, convert them
    if seg_pred.dtype != bool:
        seg_pred = seg_pred == label_idx

    if seg_gt.dtype != bool:
        seg_gt = seg_gt == label_idx

    if crop:
        or_vol = seg_pred | seg_gt
        if not or_vol.any():
            return np.zeros(seg_pred.shape, dtype=bool), np.zeros(seg_gt.shape, dtype=bool)

        _, crop_coord = fn_utils.crop_label(or_vol, margin=1)
        seg_pred = fn_utils.apply_crop(seg_pred, crop_coord)
        seg_gt = fn_utils.apply_crop(seg_gt, crop_coord)

    edges_pred = binary_erosion(seg_pred) ^ seg_pred
    edges_gt = binary_erosion(seg_gt) ^ seg_gt

    return edges_pred.astype(bool), edges_gt.astype(bool)



def get_surface_distance(
    seg_pred,
    seg_gt,
    distance_metric,
    spacing,
):
    """
    This function is used to compute the surface distances from `seg_pred` to `seg_gt`.

    Args:
        seg_pred: the edge of the predictions.
        seg_gt: the edge of the ground truth.
        distance_metric: : [``"euclidean"``, ``"chessboard"``, ``"taxicab"``]
            the metric used to compute surface distance. Defaults to ``"euclidean"``.

            - ``"euclidean"``, uses Exact Euclidean distance transform.
            - ``"chessboard"``, uses `chessboard` metric in chamfer type of transform.
            - ``"taxicab"``, uses `taxicab` metric in chamfer type of transform.
        spacing: spacing of pixel (or voxel). This parameter is relevant only if ``distance_metric`` is set to ``"euclidean"``.
            Several input options are allowed:
            (1) If a single number, isotropic spacing with that value is used.
            (2) If a sequence of numbers, the length of the sequence must be equal to the image dimensions.
            (3) If ``None``, spacing of unity is used. Defaults to ``None``.

    Note:
        If seg_pred or seg_gt is all 0, may result in nan/inf distance.

    """
    if not seg_gt.any():
        dis = np.inf * np.ones_like(seg_gt, dtype=np.float32)

    else:
        if not np.any(seg_pred):
            dis = np.inf * np.ones_like(seg_gt, dtype=np.float32)
            dis = dis[seg_gt]
            return dis

        if distance_metric == "euclidean":
            dis = distance_transform_edt((~seg_gt), sampling=spacing)  # type: ignore

        elif distance_metric in {"chessboard", "taxicab"}:
            dis = distance_transform_cdt(convert_to_numpy(~seg_gt), metric=distance_metric)

        else:
            raise ValueError(f"distance_metric {distance_metric} is not implemented.")

    return dis[seg_pred]

def get_edge_surface_distance(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    distance_metric="euclidean",
    spacing=None,
    symmetric=False,
    class_index=-1,
):
    """
    This function is used to compute the surface distance from `y_pred` to `y` using the edges of the masks.

    Args:
        y_pred: the predicted binary or labelfield image. Expected to be in format (H, W[, D]).
        y: the actual binary or labelfield image. Expected to be in format (H, W[, D]).
        distance_metric: : [``"euclidean"``, ``"chessboard"``, ``"taxicab"``]
            See :py:func:`monai.metrics.utils.get_surface_distance`.
        spacing: spacing of pixel (or voxel). This parameter is relevant only if ``distance_metric`` is set to ``"euclidean"``.
            See :py:func:`monai.metrics.utils.get_surface_distance`.
        symmetric: whether to compute the surface distance from `y_pred` to `y` and from `y` to `y_pred`.
        class_index: The class-index used for context when warning about empty ground truth or prediction.

    Returns:
        (edges_pred, edges_gt), distances_pred_to_gt

    """
    edges_pred, edges_gt = get_mask_edges(y_pred, y, crop=True)
    if not edges_gt.any():
        warnings.warn(
            f"the ground truth of class {class_index if class_index != -1 else 'Unknown'} is all 0,"
            " this may result in nan/inf distance."
        )
    if not edges_pred.any():
        warnings.warn(
            f"the prediction of class {class_index if class_index != -1 else 'Unknown'} is all 0,"
            " this may result in nan/inf distance."
        )
    if symmetric:
        distances = (
            get_surface_distance(edges_pred, edges_gt, distance_metric, spacing),
            get_surface_distance(edges_gt, edges_pred, distance_metric, spacing),
        )  # type: ignore
    else:
        distances = (get_surface_distance(edges_pred, edges_gt, distance_metric, spacing),)  # type: ignore

    return (edges_pred, edges_gt), distances



def _dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.0001
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)