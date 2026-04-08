import pdb
import re
import math

from munkres import Munkres
import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.interpolate import RegularGridInterpolator as rgi
import torch
from torch.nn import functional as F


def get_type_dict(data):
    t_dict = {}
    t_dict['type'] = type(data)
    t_dict['dtype'] = data.dtype
    if t_dict['type'] == torch.Tensor:
        t_dict['device'] = data.device

    return t_dict

def convert_to_type(data, type, dtype=None, device=None):
    if type == np.ndarray:
        return convert_to_numpy(data, dtype=dtype)
    elif type == torch.Tensor:
        return convert_to_tensor(data, dtype=dtype, device=device)

    return data

def convert_to_tensor(data, dtype=None, device=None):
    if isinstance(data, np.ndarray):
        # skip array of string classes and object, refer to:
        # https://github.com/pytorch/pytorch/blob/v1.9.0/torch/utils/data/_utils/collate.py#L13
        if re.search(r"[SaUO]", data.dtype.str) is None:
            # numpy array with 0 dims is also sequence iterable,
            # `ascontiguousarray` will add 1 dim if img has no dim, so we only apply on data with dims
            if data.ndim > 0:
                data = np.ascontiguousarray(data)

            return torch.as_tensor(data, dtype=dtype, device=device)

    elif isinstance(data, list):
        return [convert_to_tensor(i, dtype=dtype, device=device) for i in data]
    elif isinstance(data, tuple):
        return tuple(convert_to_tensor(i, dtype=dtype, device=device) for i in data)
    elif isinstance(data, dict):
        return {k: convert_to_tensor(v, dtype=dtype, device=device) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        if data.device != device:
            data = data.to(device)
        return data

    return data

def convert_to_numpy(data, dtype=None):
    """
    Utility to convert the input data to a numpy array. If passing a dictionary, list or tuple,
    recursively check every item and convert it to numpy array.

    Args:
        data: input data can be PyTorch Tensor, numpy array, list, dictionary, int, float, bool, str, etc.
            will convert Tensor, Numpy array, float, int, bool to numpy arrays, strings and objects keep the original.
            for dictionary, list or tuple, convert every item to a numpy array if applicable.
        dtype: target data type when converting to numpy array.
        wrap_sequence: if `False`, then lists will recursively call this function.
            E.g., `[1, 2]` -> `[array(1), array(2)]`. If `True`, then `[1, 2]` -> `array([1, 2])`.
        safe: if `True`, then do safe dtype convert when intensity overflow. default to `False`.
            E.g., `[256, -12]` -> `[array(0), array(244)]`. If `True`, then `[256, -12]` -> `[array(255), array(0)]`.
    """
    if isinstance(data, torch.Tensor):
        data = np.asarray(data.detach().to(device="cpu").numpy())

    elif isinstance(data, (np.ndarray, float, int, bool)):
        # Convert into a contiguous array first if the current dtype's size is smaller than the target dtype's size.
        # This help improve the performance because (convert to contiguous array) -> (convert dtype) is faster
        # than (convert dtype) -> (convert to contiguous array) when src dtype (e.g., uint8) is smaller than
        # target dtype(e.g., float32) and we are going to convert it to contiguous array anyway later in this
        # method.
        if isinstance(data, np.ndarray) and data.ndim > 0 and data.dtype.itemsize < np.dtype(dtype).itemsize:
            data = np.ascontiguousarray(data)
        data = np.asarray(data, dtype=dtype)
    elif isinstance(data, list):
        return [convert_to_numpy(i, dtype=dtype) for i in data]
    elif isinstance(data, tuple):
        return tuple(convert_to_numpy(i, dtype=dtype) for i in data)
    elif isinstance(data, dict):
        return {k: convert_to_numpy(v, dtype=dtype) for k, v in data.items()}

    if isinstance(data, np.ndarray) and data.ndim > 0:
        data = np.ascontiguousarray(data)

    return data

def align_with_identity_vox2ras0(V, vox2ras0):

    COST = np.zeros((3,3))
    for i in range(3):
        for j in range(3):

            # worker is the vector
            b = vox2ras0[:3,i]

            # task is j:th axis
            a = np.zeros((3,1))
            a[j] = 1

            COST[i, j] = - np.abs(np.dot(a.T, b))/np.linalg.norm(a, 2)/np.linalg.norm(b, 2)

    m = Munkres()
    indexes = m.compute(COST)

    v2r = np.zeros_like(vox2ras0)
    for idx in indexes:
        v2r[:, idx[1]] = vox2ras0[:, idx[0]]
    v2r[:, 3] = vox2ras0[:, 3]
    V = np.transpose(V, axes=[idx[1] for idx in indexes])

    for d in range(3):
        if v2r[d,d] < 0:
            v2r[:3, d] = -v2r[:3, d]
            v2r[:3, 3] = v2r[:3, 3] - v2r[:3, d] * (V.shape[d] -1)
            V = np.flip(V, axis=d)

    return V, v2r

def rescale_volume(volume, new_min=0, new_max=255, min_percentile=2, max_percentile=98, use_positive_only=True):
    """This function linearly rescales a volume between new_min and new_max.
    :param volume: a numpy array
    :param new_min: (optional) minimum value for the rescaled image.
    :param new_max: (optional) maximum value for the rescaled image.
    :param min_percentile: (optional) percentile for estimating robust minimum of volume (float in [0,...100]),
    where 0 = np.min
    :param max_percentile: (optional) percentile for estimating robust maximum of volume (float in [0,...100]),
    where 100 = np.max
    :param use_positive_only: (optional) whether to use only positive values when estimating the min and max percentile
    :return: rescaled volume
    """

    # select only positive intensities
    new_volume = volume.copy()
    intensities = new_volume[new_volume > 0] if use_positive_only else new_volume.flatten()

    # define min and max intensities in original image for normalisation
    robust_min = np.min(intensities) if min_percentile == 0 else np.percentile(intensities, min_percentile)
    robust_max = np.max(intensities) if max_percentile == 0 else np.percentile(intensities, max_percentile)

    # trim values outside range
    new_volume = np.clip(new_volume, robust_min, robust_max)

    # rescale image
    if robust_min != robust_max:
        return new_min + (new_volume - robust_min) / (robust_max - robust_min) * new_max
    else:  # avoid dividing by zero
        return np.zeros_like(new_volume)

def rescale_voxel_size(volume, aff, new_vox_size, not_aliasing=False, method='linear'):
    """This function resizes the voxels of a volume to a new provided size, while adjusting the header to keep the RAS
    :param volume: a numpy array
    :param aff: affine matrix of the volume
    :param new_vox_size: new voxel size (3 - element numpy vector) in mm
    :return: new volume and affine matrix
    """

    pixdim = np.sqrt(np.sum(aff * aff, axis=0))[:-1]
    new_vox_size = np.array(new_vox_size)
    if all([a==b for a, b in zip(pixdim, new_vox_size)]):
        return volume, aff

    factor = pixdim / new_vox_size
    sigmas = 0.25 / factor
    sigmas[factor > 1] = 0  # don't blur if upsampling

    if len(volume.shape) > 3:
        sigmas = np.concatenate((sigmas, [0]))

    if all(sigmas == 0) or not_aliasing or method=='nearest':
        volume_filt = volume
    else:
        volume_filt = gaussian_filter(volume, sigmas)

    # volume2 = zoom(volume_filt, factor, order=1, mode='reflect', prefilter=False)
    x = np.arange(0, volume_filt.shape[0])
    y = np.arange(0, volume_filt.shape[1])
    z = np.arange(0, volume_filt.shape[2])

    my_interpolating_function = rgi((x, y, z), volume_filt, method)

    start = - (factor - 1) / (2 * factor)
    step = 1.0 / factor
    stop = start + step * np.ceil(volume_filt.shape[:3] * factor)

    xi = np.arange(start=start[0], stop=stop[0], step=step[0])
    yi = np.arange(start=start[1], stop=stop[1], step=step[1])
    zi = np.arange(start=start[2], stop=stop[2], step=step[2])
    xi[xi < 0] = 0
    yi[yi < 0] = 0
    zi[zi < 0] = 0
    xi[xi > (volume_filt.shape[0] - 1)] = volume_filt.shape[0] - 1
    yi[yi > (volume_filt.shape[1] - 1)] = volume_filt.shape[1] - 1
    zi[zi > (volume_filt.shape[2] - 1)] = volume_filt.shape[2] - 1

    xig, yig, zig = np.meshgrid(xi, yi, zi, indexing='ij', sparse=True)
    volume2 = my_interpolating_function((xig, yig, zig))

    aff2 = aff.copy()
    for c in range(3):
        aff2[:-1, c] = aff2[:-1, c] / factor[c]
    aff2[:-1, -1] = aff2[:-1, -1] - np.matmul(aff2[:-1, :-1], 0.5 * (factor - 1))

    return volume2, aff2

def draw_value_from_distribution(hyperparameter,
                                 size=1,
                                 distribution='uniform',
                                 centre=0.,
                                 default_range=10.0,
                                 positive_only=False):
    """Sample values from a uniform, or normal distribution of given hyper-parameters.
    These hyper-parameters are to the number of 2 in both uniform and normal cases.
    :param hyperparameter: values of the hyper-parameters. Can either be:
    1) None, in each case the two hyper-parameters are given by [center-default_range, center+default_range],
    2) a number, where the two hyper-parameters are given by [centre-hyperparameter, centre+hyperparameter],
    3) a sequence of length 2, directly defining the two hyper-parameters: [min, max] if the distribution is uniform,
    [mean, std] if the distribution is normal.
    4) a numpy array, with size (2, m). In this case, the function returns a 1d array of size m, where each value has
    been sampled independently with the specified hyper-parameters. If the distribution is uniform, rows correspond to
    its lower and upper bounds, and if the distribution is normal, rows correspond to its mean and std deviation.
    5) a numpy array of size (2*n, m). Same as 4) but we first randomly select a block of two rows among the
    n possibilities.
    6) the path to a numpy array corresponding to case 4 or 5.
    7) False, in which case this function returns None.
    :param size: (optional) number of values to sample. All values are sampled independently.
    Used only if hyperparameter is not a numpy array.
    :param distribution: (optional) the distribution type. Can be 'uniform' or 'normal'. Default is 'uniform'.
    :param centre: (optional) default centre to use if hyperparameter is None or a number.
    :param default_range: (optional) default range to use if hyperparameter is None.
    :param positive_only: (optional) wheter to reset all negative values to zero.
    :return: a float, or a numpy 1d array if size > 1, or hyperparameter is itself a numpy array.
    Returns None if hyperparmeter is False.
    """

    # return False is hyperparameter is False
    if hyperparameter is False:
        return None

    # reformat parameter_range
    if not isinstance(hyperparameter, np.ndarray):
        if hyperparameter is None:
            hyperparameter = np.array([[centre - default_range] * size, [centre + default_range] * size])
        elif isinstance(hyperparameter, (int, float)):
            hyperparameter = np.array([[centre - hyperparameter] * size, [centre + hyperparameter] * size])
        elif isinstance(hyperparameter, (list, tuple)):
            assert len(hyperparameter) == 2, 'if list, parameter_range should be of length 2.'
            hyperparameter = np.transpose(np.tile(np.array(hyperparameter), (size, 1)))
        else:
            raise ValueError('parameter_range should either be None, a nummber, a sequence, or a numpy array.')
    elif isinstance(hyperparameter, np.ndarray):
        assert hyperparameter.shape[0] % 2 == 0, 'number of rows of parameter_range should be divisible by 2'
        n_modalities = int(hyperparameter.shape[0] / 2)
        modality_idx = 2 * np.random.randint(n_modalities)
        hyperparameter = hyperparameter[modality_idx: modality_idx + 2, :]

    if distribution == 'uniform':
        parameter_value = np.random.uniform(low=hyperparameter[0, :], high=hyperparameter[1, :])
    elif distribution == 'normal':
        parameter_value = np.random.normal(loc=hyperparameter[0, :], scale=hyperparameter[1, :])
    else:
        raise ValueError("Distribution not supported, should be 'uniform' or 'normal'.")

    if positive_only:
        parameter_value[parameter_value < 0] = 0

    return parameter_value

def get_GMM_parameters(sampling_scheme,
                       prior_distributions='uniform',
                       prior_means=None,
                       prior_stds=None,
                       use_specific_stats_for_channel=False,
                       mix_prior_and_random=False):

    # retrieve channel specific stats if necessary
    # mean
    if isinstance(prior_means, np.ndarray):
        if (prior_means.shape[0] > 2) & use_specific_stats_for_channel:
            tmp_prior_means = prior_means[:2, :]
        else:
            tmp_prior_means = prior_means
    else:
        tmp_prior_means = prior_means
    if (prior_means is not None) & mix_prior_and_random & (np.random.uniform() > 0.5):
        tmp_prior_means = None

    # variance
    if isinstance(prior_stds, np.ndarray):
        if (prior_stds.shape[0] > 2) & use_specific_stats_for_channel:
            tmp_prior_stds = prior_stds[:2, :]
        else:
            tmp_prior_stds = prior_stds
    else:
        tmp_prior_stds = prior_stds
    if (prior_stds is not None) & mix_prior_and_random & (np.random.uniform() > 0.5):
        tmp_prior_stds = None

    tmp_sampling_means = draw_value_from_distribution(tmp_prior_means,
                                                      size=sampling_scheme.n_sampling_labels,
                                                      distribution=prior_distributions,
                                                      centre=125.,
                                                      default_range=100.,
                                                      positive_only=True)

    tmp_sampling_stds = draw_value_from_distribution(tmp_prior_stds,
                                                     size=sampling_scheme.n_sampling_labels,
                                                     distribution=prior_distributions,
                                                     centre=15.,
                                                     default_range=10.,
                                                     positive_only=True)

    # if np.random.uniform() > 0.95:  # reset the background to 0 in 10% of cases
    #     tmp_sampling_means[0] = 0
    #     tmp_sampling_stds[0] = 0

    means, stds = sampling_scheme.compute_stats(tmp_sampling_means, tmp_sampling_stds)

    return means, stds

def get_image_from_labelmap(labelmap, mus, sigmas):
    MU_IMAGE = mus[labelmap]
    STD_IMAGE = sigmas[labelmap]
    image = MU_IMAGE + STD_IMAGE * np.random.randn(*STD_IMAGE.shape)
    image[image < 0] = 0

    return image

def convert_labels(lut):
    labels = np.zeros((np.max(list(lut.keys())) + 1,))
    for in_lab, out_lab in lut.items(): labels[in_lab] = out_lab

    return labels

def convert_labelmap(labelmap, lut):
    labels = convert_labels(lut)
    # labels = np.zeros((np.max(list(lut.keys())) + 1,))
    # for in_lab, out_lab in lut.items(): labels[in_lab] = out_lab

    L = labels[labelmap]
    return L

def one_hot_encoding(target, num_classes=None, categories=None):
    '''

    Parameters
    ----------
    target (np.array): target vector of dimension (d1, d2, ..., dN).
    num_classes (int): number of classes
    categories (None or dict): existing categories as a LUT. If set to None, we will consider only categories 0,...,num_classes

    Returns
    -------
    labels (np.array): one-hot target vector of dimension (num_classes, d1, d2, ..., dN)

    '''

    if categories is None and num_classes is None:
        categories = np.sort(np.unique(target))
        num_classes = len(categories)

    elif categories is not None:
        if isinstance(categories, list) or isinstance(categories, np.ndarray):
            categories = {cls: it_cls for it_cls, cls in enumerate(categories)}

        num_classes = len(np.unique(np.array(list(categories.values()))))

    else:
        categories = {cls: cls for cls in np.arange(num_classes)}

    labels = np.zeros((num_classes,) + target.shape, dtype='int')
    for cls, it_cls in categories.items():
        idx_class = np.where(target == cls)
        idx = (it_cls,) + idx_class
        labels[idx] = 1

    return labels

def get_gaussian_filter_3d(sigma, channels=1, truncate=4, normalize_area=True):
    # Set these to whatever you want for your gaussian filter
    kernel_size = tuple([2*round(s*truncate)+1 for s in sigma])

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    II, JJ, KK = torch.meshgrid([torch.arange(0, kernel_size[0]), torch.arange(0, kernel_size[1]), torch.arange(0, kernel_size[2])], indexing='ij')
    grid = torch.stack([II, JJ, KK], dim=-1)

    mean = torch.from_numpy(np.array([(k - 1) / 2. for k in kernel_size]).reshape((1, 1, 1, 3)))
    variance = torch.from_numpy(((sigma ** 2.).reshape((1, 1, 1, 3))))

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / ((2. * torch.pi * torch.prod(variance)))**(1/3)) * torch.exp(-torch.sum((grid - mean) ** 2. / (2 * variance), dim=-1))
    gaussian_kernel /= torch.max(gaussian_kernel)
    # Make sure sum of values in gaussian kernel equals 1.
    if normalize_area:
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 3d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, *kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1, 1)

    gaussian_filter = torch.nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                      padding='same', groups=channels, bias=False)

    gaussian_filter.weight.data = gaussian_kernel.float()
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter

def crop_label(mask, margin=10, threshold=0):

    ndim = len(mask.shape)
    if isinstance(margin, int):
        margin=[margin]*ndim

    crop_coord = []
    idx = np.where(mask>threshold)
    for it_index, index in enumerate(idx):
        clow = max(0, np.min(idx[it_index]) - margin[it_index])
        chigh = min(mask.shape[it_index], np.max(idx[it_index]) + margin[it_index])
        crop_coord.append([clow, chigh])

    mask_cropped = mask[
                   crop_coord[0][0]: crop_coord[0][1],
                   crop_coord[1][0]: crop_coord[1][1],
                   crop_coord[2][0]: crop_coord[2][1]
                   ]

    return mask_cropped, crop_coord

def apply_crop(image, crop_coord):
    return image[crop_coord[0][0]: crop_coord[0][1],
                 crop_coord[1][0]: crop_coord[1][1],
                 crop_coord[2][0]: crop_coord[2][1]
           ]



def _get_scan_interval(
    image_size, roi_size, num_spatial_dims, overlap
):
    """
    Compute scan interval according to the image size, roi size and overlap.
    Scan interval will be `int((1 - overlap) * roi_size)`, if interval is 0,
    use 1 instead to make sure sliding window works.

    """
    if len(image_size) != num_spatial_dims:
        raise ValueError(f"len(image_size) {len(image_size)} different from spatial dims {num_spatial_dims}.")
    if len(roi_size) != num_spatial_dims:
        raise ValueError(f"len(roi_size) {len(roi_size)} different from spatial dims {num_spatial_dims}.")

    scan_interval = []
    for i, o in zip(range(num_spatial_dims), overlap):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - o))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)

def _flatten_struct(seg_out):
    dict_keys = None
    seg_probs: tuple[torch.Tensor, ...]
    seg_probs = (seg_out,)
    return dict_keys, seg_probs


def _pack_struct(seg_out, dict_keys=None):
    if dict_keys is not None:
        return dict(zip(dict_keys, seg_out))
    if isinstance(seg_out, (list, tuple)) and len(seg_out) == 1:
        return seg_out[0]
    return seg_out


def compute_importance_map(
    patch_size,
    mode,
    sigma_scale=0.125,
    device="cpu",
    dtype=torch.float32,
):
    """Get importance map for different weight modes.

    Args:
        patch_size: Size of the required importance map. This should be either H, W [,D].
        mode: {``"constant"``, ``"gaussian"``}
            How to blend output of overlapping windows. Defaults to ``"constant"``.

            - ``"constant``": gives equal weight to all predictions.
            - ``"gaussian``": gives less weight to predictions on edges of windows.

        sigma_scale: Sigma_scale to calculate sigma for each dimension
            (sigma = sigma_scale * dim_size). Used for gaussian mode only.
        device: Device to put importance map on.
        dtype: Data type of the output importance map.

    Raises:
        ValueError: When ``mode`` is not one of ["constant", "gaussian"].

    Returns:
        Tensor of size patch_size.

    """
    device = torch.device(device)
    if mode == "constant":
        importance_map = torch.ones(patch_size, device=device, dtype=torch.float)
    elif mode == 'gaussian':
        if isinstance(sigma_scale, int, float):
            sigma_scale = (sigma_scale, ) * len(patch_size)

        sigmas = [i * sigma_s for i, sigma_s in zip(patch_size, sigma_scale)]

        for i in range(len(patch_size)):
            x = torch.arange(
                start=-(patch_size[i] - 1) / 2.0, end=(patch_size[i] - 1) / 2.0 + 1, dtype=torch.float, device=device
            )
            x = torch.exp(x**2 / (-2 * sigmas[i] ** 2))  # 1D gaussian
            importance_map = importance_map.unsqueeze(-1) * x[(None,) * i] if i > 0 else x
    else:
        raise ValueError(
            f"Unsupported mode: {mode}, available options are [constant, gaussian]."
        )
    # handle non-positive weights
    min_non_zero = max(torch.min(importance_map).item(), 1e-3)
    importance_map = torch.clamp_(importance_map.to(torch.float), min=min_non_zero).to(dtype)
    return importance_map

def issequenceiterable(obj) -> bool:
    """
    Determine if the object is an iterable sequence and is not a string.
    """
    from collections.abc import Iterable
    try:
        if hasattr(obj, "ndim") and obj.ndim == 0:
            return False  # a 0-d tensor is not iterable
    except Exception:
        return False
    return isinstance(obj, Iterable) and not isinstance(obj, (str, bytes))

def ensure_tuple_size(vals, dim, pad_val=0, pad_from_start: bool = False) -> tuple:
    """
    Returns a copy of `tup` with `dim` values by either shortened or padded with `pad_val` as necessary.
    """
    tup = ensure_tuple(vals)
    pad_dim = dim - len(tup)
    if pad_dim <= 0:
        return tup[:dim]
    if pad_from_start:
        return (pad_val,) * pad_dim + tup
    return tup + (pad_val,) * pad_dim
def ensure_tuple(vals, wrap_array: bool = False) -> tuple:
    """
    Returns a tuple of `vals`.

    Args:
        vals: input data to convert to a tuple.
        wrap_array: if `True`, treat the input numerical array (ndarray/tensor) as one item of the tuple.
            if `False`, try to convert the array with `tuple(vals)`, default to `False`.

    """
    if wrap_array and isinstance(vals, (np.ndarray, torch.Tensor)):
        return (vals,)
    return tuple(vals) if issequenceiterable(vals) else (vals,)

from collections.abc import Sequence
def get_valid_patch_size(image_size: Sequence[int], patch_size: Sequence[int] | int | np.ndarray) -> tuple[int, ...]:
    """
    Given an image of dimensions `image_size`, return a patch size tuple taking the dimension from `patch_size` if this is
    not 0/None. Otherwise, or if `patch_size` is shorter than `image_size`, the dimension from `image_size` is taken. This ensures
    the returned patch size is within the bounds of `image_size`. If `patch_size` is a single number this is interpreted as a
    patch of the same dimensionality of `image_size` with that size in each dimension.
    """
    ndim = len(image_size)
    patch_size_ = ensure_tuple_size(patch_size, ndim)

    # ensure patch size dimensions are not larger than image dimension, if a dimension is None or 0 use whole dimension
    return tuple(min(ms, ps or ms) for ms, ps in zip(image_size, patch_size_))

def first(iterable, default=None):
    """
    Returns the first item in the given iterable or `default` if empty, meaningful mostly with 'for' expressions.
    """
    for i in iterable:
        return i
    return default

def dense_patch_slices(
    image_size: Sequence[int], patch_size: Sequence[int], scan_interval: Sequence[int], return_slice: bool = True
) -> list[tuple[slice, ...]]:
    """
    Enumerate all slices defining ND patches of size `patch_size` from an `image_size` input image.

    Args:
        image_size: dimensions of image to iterate over
        patch_size: size of patches to generate slices
        scan_interval: dense patch sampling interval
        return_slice: whether to return a list of slices (or tuples of indices), defaults to True

    Returns:
        a list of slice objects defining each patch

    """
    num_spatial_dims = len(image_size)
    patch_size = get_valid_patch_size(image_size, patch_size)
    scan_interval = ensure_tuple_size(scan_interval, num_spatial_dims)

    scan_num = []
    for i in range(num_spatial_dims):
        if scan_interval[i] == 0:
            scan_num.append(1)
        else:
            num = int(math.ceil(float(image_size[i]) / scan_interval[i]))
            scan_dim = first(d for d in range(num) if d * scan_interval[i] + patch_size[i] >= image_size[i])
            scan_num.append(scan_dim + 1 if scan_dim is not None else 1)

    starts = []
    for dim in range(num_spatial_dims):
        dim_starts = []
        for idx in range(scan_num[dim]):
            start_idx = idx * scan_interval[dim]
            start_idx -= max(start_idx + patch_size[dim] - image_size[dim], 0)
            dim_starts.append(start_idx)
        starts.append(dim_starts)
    out = np.asarray([x.flatten() for x in np.meshgrid(*starts, indexing="ij")]).T
    if return_slice:
        return [tuple(slice(s, s + patch_size[d]) for d, s in enumerate(x)) for x in out]
    return [tuple((s, s + patch_size[d]) for d, s in enumerate(x)) for x in out]  # type: ignore

def _compute_coords(coords, z_scale, out, patch):
    """sliding window batch spatial scaling indexing for multi-resolution outputs."""
    for original_idx, p in zip(coords, patch):
        idx_zm = list(original_idx)  # 4D for 2D image, 5D for 3D image
        if z_scale:
            for axis in range(2, len(idx_zm)):
                idx_zm[axis] = slice(
                    int(original_idx[axis].start * z_scale[axis - 2]), int(original_idx[axis].stop * z_scale[axis - 2])
                )
        out[idx_zm] += p

def sliding_window_inference(
    inputs,
    roi_size,
    sw_batch_size,
    predictor,
    overlap=0.25,
    mode='constant',
    sigma_scale=0.25,
    sw_device='cpu',
    device='cpu',
    *args, **kwargs
):
    """
    Sliding window inference on `inputs` with `predictor`.

    The outputs of `predictor` could be a tensor, a tuple, or a dictionary of tensors.
    Each output in the tuple or dict value is allowed to have different resolutions with respect to the input.
    e.g., the input patch spatial size is [128,128,128], the output (a tuple of two patches) patch sizes
    could be ([128,64,256], [64,32,128]).
    In this case, the parameter `overlap` and `roi_size` need to be carefully chosen to ensure the output ROI is still
    an integer. If the predictor's input and output spatial sizes are not equal, we recommend choosing the parameters
    so that `overlap*roi_size*output_size/input_size` is an integer (for each spatial dimension).

    When roi_size is larger than the inputs' spatial size, the input image are padded during inference.
    To maintain the same spatial sizes, the output image will be cropped to the original input size.

    Args:
        inputs: input image to be processed (assuming NCHW[D])
        roi_size: the spatial window size for inferences.
            When its components have None or non-positives, the corresponding inputs dimension will be used.
            if the components of the `roi_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `roi_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        sw_batch_size: the batch size to run window slices.
        predictor: given input tensor ``patch_data`` in shape NCHW[D],
            The outputs of the function call ``predictor(patch_data)`` should be a tensor, a tuple, or a dictionary
            with Tensor values. Each output in the tuple or dict value should have the same batch_size, i.e. NM'H'W'[D'];
            where H'W'[D'] represents the output patch's spatial size, M is the number of output channels,
            N is `sw_batch_size`, e.g., the input shape is (7, 1, 128,128,128),
            the output could be a tuple of two tensors, with shapes: ((7, 5, 128, 64, 256), (7, 4, 64, 32, 128)).
            In this case, the parameter `overlap` and `roi_size` need to be carefully chosen
            to ensure the scaled output ROI sizes are still integers.
            If the `predictor`'s input and output spatial sizes are different,
            we recommend choosing the parameters so that ``overlap*roi_size*zoom_scale`` is an integer for each dimension.
        overlap: Amount of overlap between scans along each spatial dimension, defaults to ``0.25``.
        mode: {``"constant"``, ``"gaussian"``}
            How to blend output of overlapping windows. Defaults to ``"constant"``.

            - ``"constant``": gives equal weight to all predictions.
            - ``"gaussian``": gives less weight to predictions on edges of windows.

        sigma_scale: the standard deviation coefficient of the Gaussian window when `mode` is ``"gaussian"``.
            Default: 0.125. Actual window sigma is ``sigma_scale`` * ``dim_size``.
            When sigma_scale is a sequence of floats, the values denote sigma_scale at the corresponding
            spatial dimensions.
        sw_device: device for the window data.
            By default the device (and accordingly the memory) of the `inputs` is used.
            Normally `sw_device` should be consistent with the device where `predictor` is defined.
        device: device for the stitched output prediction.
            By default the device (and accordingly the memory) of the `inputs` is used. If for example
            set to device=torch.device('cpu') the gpu memory consumption is less and independent of the
            `inputs` and `roi_size`. Output is on the `device`.
    Note:
        - input must be channel-first and have a batch dim, supports N-D sliding window.

    """
    num_spatial_dims = len(inputs.shape) - 2
    overlap = (overlap, ) * num_spatial_dims
    for o in overlap:
        if o < 0 or o >= 1:
            raise ValueError(f"overlap must be >= 0 and < 1, got {overlap}.")
    compute_dtype = inputs.dtype

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    batch_size, _, *image_size_ = inputs.shape
    device = device or inputs.device
    sw_device = sw_device or inputs.device
    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))

    # Store all slices
    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)
    slices = dense_patch_slices(image_size, roi_size, scan_interval, return_slice=True)

    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows
    windows_range = range(0, total_slices, sw_batch_size)

    # Create window-level importance map
    valid_patch_size = get_valid_patch_size(image_size, roi_size)
    valid_p_size = valid_patch_size
    try:
        importance_map_ = compute_importance_map(
            valid_p_size, mode=mode, sigma_scale=sigma_scale, device=sw_device, dtype=compute_dtype
        )
        if len(importance_map_.shape) == num_spatial_dims:
            importance_map_ = importance_map_[None, None]  # adds batch, channel dimensions

    except Exception as e:
        raise RuntimeError(
            f"patch size {valid_p_size}, mode={mode}, sigma_scale={sigma_scale}, device={device}\n"
            "Seems to be OOM. Please try smaller patch size or mode='constant' instead of mode='gaussian'."
        ) from e

    # stores output and count map
    output_image_list, count_map_list, sw_device_buffer, b_s, b_i = [], [], [], 0, 0  # type: ignore
    # for each patch
    for slice_g in windows_range:
        slice_range = range(slice_g, total_slices)
        unravel_slice = [
            [slice(idx // num_win, idx // num_win + 1), slice(None)] + list(slices[idx % num_win])
            for idx in slice_range
        ]
        if sw_batch_size > 1:
            win_data = torch.cat([inputs[win_slice] for win_slice in unravel_slice]).to(sw_device)
        else:
            win_data = inputs[unravel_slice[0]].to(sw_device)

        seg_prob_out = predictor(win_data, *args, **kwargs)  # batched patch

        # convert seg_prob_out to tuple seg_tuple, this does not allocate new memory.
        dict_keys, seg_tuple = _flatten_struct(seg_prob_out)
        w_t = importance_map_
        if len(w_t.shape) == num_spatial_dims:
            w_t = w_t[None, None]
        w_t = w_t.to(dtype=compute_dtype, device=sw_device)
        sw_device_buffer = list(seg_tuple)

        for ss in range(len(sw_device_buffer)):
            b_shape = sw_device_buffer[ss].shape
            seg_chns, seg_shape = b_shape[1], b_shape[2:]
            z_scale = None
            if seg_shape != roi_size:
                z_scale = [out_w_i / float(in_w_i) for out_w_i, in_w_i in zip(seg_shape, roi_size)]
                w_t = F.interpolate(w_t, seg_shape, mode="nearest")

            if len(output_image_list) <= ss:
                output_shape = [batch_size, seg_chns]
                output_shape += [int(_i * _z) for _i, _z in zip(image_size, z_scale)] if z_scale else list(image_size)
                # allocate memory to store the full output and the count for overlapping parts
                new_tensor = torch.zeros  # type: ignore
                output_image_list.append(new_tensor(output_shape, dtype=compute_dtype, device=device))
                count_map_list.append(torch.zeros([1, 1] + output_shape[2:], dtype=compute_dtype, device=device))
                w_t_ = w_t.to(device)
                for __s in slices:
                    if z_scale is not None:
                        __s = tuple(slice(int(_si.start * z_s), int(_si.stop * z_s)) for _si, z_s in zip(__s, z_scale))
                    count_map_list[-1][(slice(None), slice(None), *__s)] += w_t_

            sw_device_buffer[ss] *= w_t
            sw_device_buffer[ss] = sw_device_buffer[ss].to(device)
            _compute_coords(unravel_slice, z_scale, output_image_list[ss], sw_device_buffer[ss])

        sw_device_buffer = []

    # account for any overlapping sections
    for ss in range(len(output_image_list)):
        output_image_list[ss] /= count_map_list.pop(0)

    final_output = _pack_struct(output_image_list, dict_keys)

    return final_output  # type: ignore