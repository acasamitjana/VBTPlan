import torch
import numpy as np

UNSUPPORTED_TYPES = {np.dtype("uint16"): np.int32, np.dtype("uint32"): np.int64, np.dtype("uint64"): np.int64}


def convert_to_tensor(
        data,
        dtype=None,
        device=None,
        wrap_sequence=False,
):
    """
    Utility to convert the input data to a PyTorch Tensor, if `track_meta` is True, the output will be a `MetaTensor`,
    otherwise, the output will be a regular torch Tensor.
    If passing a dictionary, list or tuple, recursively check every item and convert it to PyTorch Tensor.

    Args:
        data: input data can be PyTorch Tensor, numpy array, list, dictionary, int, float, bool, str, etc.
            will convert Tensor, Numpy array, float, int, bool to Tensor, strings and objects keep the original.
            for dictionary, list or tuple, convert every item to a Tensor if applicable.
        dtype: target data type to when converting to Tensor.
        device: target device to put the converted Tensor data.
        wrap_sequence: if `False`, then lists will recursively call this function.
            E.g., `[1, 2]` -> `[tensor(1), tensor(2)]`. If `True`, then `[1, 2]` -> `tensor([1, 2])`.
        track_meta: whether to track the meta information, if `True`, will convert to `MetaTensor`.
            default to `False`.
        safe: if `True`, then do safe dtype convert when intensity overflow. default to `False`.
            E.g., `[256, -12]` -> `[tensor(0), tensor(244)]`.
            If `True`, then `[256, -12]` -> `[tensor(255), tensor(0)]`.

    """

    def _convert_tensor(tensor, **kwargs):
        if not isinstance(tensor, torch.Tensor):
            # certain numpy types are not supported as being directly convertible to Pytorch tensors
            if isinstance(tensor, np.ndarray) and tensor.dtype in UNSUPPORTED_TYPES:
                tensor = tensor.astype(UNSUPPORTED_TYPES[tensor.dtype])

            # if input data is not Tensor, convert it to Tensor first
            tensor = torch.as_tensor(tensor, **kwargs)

        return tensor

    dtype = get_equivalent_dtype(dtype, torch.Tensor)
    if isinstance(data, torch.Tensor):
        return _convert_tensor(data).to(dtype=dtype, device=device, memory_format=torch.contiguous_format)

    if isinstance(data, np.ndarray):
        if data.ndim > 0:
            data = np.ascontiguousarray(data)
        return _convert_tensor(data, dtype=dtype, device=device)

    elif isinstance(data, list):
        list_ret = [convert_to_tensor(i, dtype=dtype, device=device) for i in data]
        return _convert_tensor(list_ret, dtype=dtype, device=device) if wrap_sequence else list_ret
    elif isinstance(data, tuple):
        tuple_ret = tuple(convert_to_tensor(i, dtype=dtype, device=device) for i in data)
        return _convert_tensor(tuple_ret, dtype=dtype, device=device) if wrap_sequence else tuple_ret
    elif isinstance(data, dict):
        return {k: convert_to_tensor(v, dtype=dtype, device=device) for k, v in data.items()}

    return data


def convert_to_numpy(data, dtype=None, wrap_sequence=False):
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
        data = np.asarray(data.detach().to(device="cpu").numpy(), dtype=get_equivalent_dtype(dtype, np.ndarray))
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
        list_ret = [convert_to_numpy(i, dtype=dtype) for i in data]
        return np.asarray(list_ret) if wrap_sequence else list_ret
    elif isinstance(data, tuple):
        tuple_ret = tuple(convert_to_numpy(i, dtype=dtype) for i in data)
        return np.asarray(tuple_ret) if wrap_sequence else tuple_ret
    elif isinstance(data, dict):
        return {k: convert_to_numpy(v, dtype=dtype) for k, v in data.items()}

    if isinstance(data, np.ndarray) and data.ndim > 0:
        data = np.ascontiguousarray(data)

    return data


def convert_to_list(data):
    """
    Convert to list from `torch.Tensor`/`np.ndarray`/`list`/`tuple` etc.
    Args:
        data: data to be converted
    Returns:
        a list

    """
    return data.tolist() if isinstance(data, (torch.Tensor, np.ndarray)) else list(data)


def get_equivalent_dtype(dtype, data_type):
    """Convert to the `dtype` that corresponds to `data_type`.

    The input dtype can also be a string. e.g., `"float32"` becomes `torch.float32` or
    `np.float32` as necessary.

    Example::

        im = torch.tensor(1)
        dtype = get_equivalent_dtype(np.float32, type(im))

    """
    if dtype is None:
        return None
    if data_type is torch.Tensor or data_type.__name__ == "MetaTensor":
        if isinstance(dtype, torch.dtype):
            # already a torch dtype and target `data_type` is torch.Tensor
            return dtype
        return dtype_numpy_to_torch(dtype)
    if not isinstance(dtype, torch.dtype):
        # assuming the dtype is ok if it is not a torch dtype and target `data_type` is not torch.Tensor
        return dtype
    return dtype_torch_to_numpy(dtype)


def dtype_numpy_to_torch(dtype: np.dtype) -> torch.dtype:
    """Convert a numpy dtype to its torch equivalent."""
    return torch.from_numpy(np.empty([], dtype=dtype)).dtype


def dtype_torch_to_numpy(dtype: torch.dtype) -> np.dtype:
    """Convert a torch dtype to its numpy equivalent."""
    return torch.empty([], dtype=dtype).numpy().dtype  # type: ignore


def convert_to_dst_type(
        src,
        dst,
        dtype=None,
        wrap_sequence=False,
        device=None
):
    """
    Convert source data to the same data type and device as the destination data.
    If `dst` is an instance of `torch.Tensor` or its subclass, convert `src` to `torch.Tensor` with the same data type as `dst`,
    if `dst` is an instance of `numpy.ndarray` or its subclass, convert to `numpy.ndarray` with the same data type as `dst`,
    otherwise, convert to the type of `dst` directly.

    Args:
        src: source data to convert type.
        dst: destination data that convert to the same data type as it.
        dtype: an optional argument if the target `dtype` is different from the original `dst`'s data type.
        wrap_sequence: if `False`, then lists will recursively call this function. E.g., `[1, 2]` -> `[array(1), array(2)]`.
            If `True`, then `[1, 2]` -> `array([1, 2])`.
        device: target device to put the converted Tensor data. If unspecified, `dst.device` will be used if possible.

    See Also:
        :func:`convert_data_type`
    """

    device = dst.device if device is None and isinstance(dst, torch.Tensor) else device
    if dtype is None:
        dtype = getattr(dst, "dtype", None)  # sequence has no dtype

    if isinstance(dst, torch.Tensor):
        return convert_to_tensor(data=src, dtype=dtype, device=device, wrap_sequence=wrap_sequence)
    elif isinstance(dst, np.ndarray):
        return convert_to_numpy(data=src, dtype=dtype, wrap_sequence=wrap_sequence)
    else:
        raise ValueError("Destination type not understood")

