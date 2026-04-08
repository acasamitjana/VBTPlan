#original python
import os
import pdb
import re
from types import GeneratorType as generator
from typing import List
import time
import subprocess
import csv
import functools

#third party imports
import numpy as np
import nibabel as nib
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from skimage import measure

# Functions

def fast_3D_interp_torch(X, II, JJ, KK, mode, constant=0):
    if mode == 'nearest':
        IIr = torch.round(II).long()
        JJr = torch.round(JJ).long()
        KKr = torch.round(KK).long()
        IIr[IIr < 0] = 0
        JJr[JJr < 0] = 0
        KKr[KKr < 0] = 0
        IIr[IIr > (X.shape[0] - 1)] = (X.shape[0] - 1)
        JJr[JJr > (X.shape[1] - 1)] = (X.shape[1] - 1)
        KKr[KKr > (X.shape[2] - 1)] = (X.shape[2] - 1)
        Y = X[IIr, JJr, KKr]

    elif mode == 'linear':
        ok = (II>=0) & (JJ>=0) & (KK>=0) & (II<=X.shape[0]-1) & (JJ<=X.shape[1]-1) & (KK<=X.shape[2]-1)
        IIv = II[ok]
        JJv = JJ[ok]
        KKv = KK[ok]
        #
        fx = torch.floor(IIv).long()
        cx = fx + 1
        cx[cx > (X.shape[0] - 1)] = (X.shape[0] - 1)
        wcx = IIv - fx
        wfx = 1 - wcx
        #
        fy = torch.floor(JJv).long()
        cy = fy + 1
        cy[cy > (X.shape[1] - 1)] = (X.shape[1] - 1)
        wcy = JJv - fy
        wfy = 1 - wcy
        #
        fz = torch.floor(KKv).long()
        cz = fz + 1
        cz[cz > (X.shape[2] - 1)] = (X.shape[2] - 1)
        wcz = KKv - fz
        wfz = 1 - wcz
        #
        c000 = X[fx, fy, fz]
        c100 = X[cx, fy, fz]
        c010 = X[fx, cy, fz]
        c110 = X[cx, cy, fz]
        c001 = X[fx, fy, cz]
        c101 = X[cx, fy, cz]
        c011 = X[fx, cy, cz]
        c111 = X[cx, cy, cz]
        #
        c00 = c000 * wfx + c100 * wcx
        c01 = c001 * wfx + c101 * wcx
        c10 = c010 * wfx + c110 * wcx
        c11 = c011 * wfx + c111 * wcx
        #
        c0 = c00 * wfy + c10 * wcy
        c1 = c01 * wfy + c11 * wcy
        #
        c = c0 * wfz + c1 * wcz
        #
        Y = constant * torch.ones(II.shape, device=X.device)
        Y[ok] = c.float()

    else:
        raise Exception('mode must be linear or nearest')

    return Y

def fast_3D_interp_field_torch(X, II, JJ, KK, mode='linear', constant=0):
    num_channels = X.shape[-1]
    if mode == 'nearest':
        IIr = torch.round(II).long()
        JJr = torch.round(JJ).long()
        KKr = torch.round(KK).long()
        IIr[IIr < 0] = 0
        JJr[JJr < 0] = 0
        KKr[KKr < 0] = 0
        IIr[IIr > (X.shape[0] - 1)] = (X.shape[0] - 1)
        JJr[JJr > (X.shape[1] - 1)] = (X.shape[1] - 1)
        KKr[KKr > (X.shape[2] - 1)] = (X.shape[2] - 1)
        Y = constant * torch.ones([*II.shape, num_channels], device=X.device)
        for channel in range(num_channels):
            #
            Xc = X[..., channel]
            Yc = Xc[IIr, JJr, KKr]
            Y[..., channel] = Yc

    elif mode == 'linear':
        #
        ok = (II > 0) & (JJ > 0) & (KK > 0) & (II <= X.shape[0] - 1) & (JJ <= X.shape[1] - 1) & (KK <= X.shape[2] - 1)
        IIv = II[ok]
        JJv = JJ[ok]
        KKv = KK[ok]
        #
        del JJ, KK
        #
        fx = torch.floor(IIv).long()
        cx = fx + 1
        cx[cx > (X.shape[0] - 1)] = (X.shape[0] - 1)
        wcx = IIv - fx
        wfx = 1 - wcx
        #
        fy = torch.floor(JJv).long()
        cy = fy + 1
        cy[cy > (X.shape[1] - 1)] = (X.shape[1] - 1)
        wcy = JJv - fy
        wfy = 1 - wcy
        #
        fz = torch.floor(KKv).long()
        cz = fz + 1
        cz[cz > (X.shape[2] - 1)] = (X.shape[2] - 1)
        wcz = KKv - fz
        wfz = 1 - wcz
        #
        Y = constant * torch.ones([*II.shape, num_channels], device=X.device)
        for channel in range(num_channels):
            #
            Xc = X[..., channel]
            #
            c000 = Xc[fx, fy, fz]
            c100 = Xc[cx, fy, fz]
            c010 = Xc[fx, cy, fz]
            c110 = Xc[cx, cy, fz]
            c001 = Xc[fx, fy, cz]
            c101 = Xc[cx, fy, cz]
            c011 = Xc[fx, cy, cz]
            c111 = Xc[cx, cy, cz]
            #
            c00 = c000 * wfx + c100 * wcx
            c01 = c001 * wfx + c101 * wcx
            c10 = c010 * wfx + c110 * wcx
            c11 = c011 * wfx + c111 * wcx
            #
            c0 = c00 * wfy + c10 * wcy
            c1 = c01 * wfy + c11 * wcy
            #
            c = c0 * wfz + c1 * wcz
            #
            Yc = torch.zeros(II.shape, device=X.device)
            Yc[ok] = c.float()
            #
            Y[..., channel] = Yc
        #
    return Y

def vol_resample_fast(ref_proxy, flo_proxy, proxyflow=None, mode='linear', return_np=False, constant=0):

    ref_v2r = (ref_proxy.affine).astype('float32')
    target_v2r = (flo_proxy.affine).astype('float32')

    ii = np.arange(0, ref_proxy.shape[0], dtype='int32')
    jj = np.arange(0, ref_proxy.shape[1], dtype='int32')
    kk = np.arange(0, ref_proxy.shape[2], dtype='int32')

    II, JJ, KK = np.meshgrid(ii, jj, kk, indexing='ij')

    del ii, jj, kk

    II = torch.tensor(II, device='cpu')
    JJ = torch.tensor(JJ, device='cpu')
    KK = torch.tensor(KK, device='cpu')

    if proxyflow is not None:
        flow_v2r = proxyflow.affine
        flow_v2r = flow_v2r.astype('float32')

        affine = torch.tensor(np.linalg.inv(flow_v2r) @ ref_v2r)
        II2 = affine[0, 0] * II + affine[0, 1] * JJ + affine[0, 2] * KK + affine[0, 3]
        JJ2 = affine[1, 0] * II + affine[1, 1] * JJ + affine[1, 2] * KK + affine[1, 3]
        KK2 = affine[2, 0] * II + affine[2, 1] * JJ + affine[2, 2] * KK + affine[2, 3]

        flow = np.array(proxyflow.dataobj)
        if flow.shape[-1] == 3: flow = np.transpose(flow, axes=(3, 0, 1, 2))
        flow = torch.tensor(flow)

        FIELD = fast_3D_interp_field_torch(flow, II2, JJ2, KK2)
        II3 = II2 + FIELD[:, :, :, 0]
        JJ3 = JJ2 + FIELD[:, :, :, 1]
        KK3 = KK2 + FIELD[:, :, :, 2]


        affine = torch.tensor(np.linalg.inv(target_v2r) @ flow_v2r)
        II4 = affine[0, 0] * II3 + affine[0, 1] * JJ3 + affine[0, 2] * KK3 + affine[0, 3]
        JJ4 = affine[1, 0] * II3 + affine[1, 1] * JJ3 + affine[1, 2] * KK3 + affine[1, 3]
        KK4 = affine[2, 0] * II3 + affine[2, 1] * JJ3 + affine[2, 2] * KK3 + affine[2, 3]


    else:
        affine = torch.tensor(np.linalg.inv(target_v2r) @ ref_v2r)
        II4 = affine[0, 0] * II + affine[0, 1] * JJ + affine[0, 2] * KK + affine[0, 3]
        JJ4 = affine[1, 0] * II + affine[1, 1] * JJ + affine[1, 2] * KK + affine[1, 3]
        KK4 = affine[2, 0] * II + affine[2, 1] * JJ + affine[2, 2] * KK + affine[2, 3]


    image = np.array(flo_proxy.dataobj)
    if len(flo_proxy.shape) == 3:
        reg_image = fast_3D_interp_torch(torch.tensor(image), II4, JJ4, KK4, mode, constant=constant)
    else:
        reg_image = fast_3D_interp_field_torch(torch.tensor(image), II4, JJ4, KK4, constant=constant)

    reg_image = reg_image.numpy()

    if return_np:
        return reg_image
    else:
        return nib.Nifti1Image(reg_image, ref_proxy.affine)

def create_COG_space(proxy_list):
    def run_fn(proxy):
        data = np.array(proxy.dataobj)
        mask = (data > 0).astype('float32')
        v2r_init = proxy.affine.astype('float32')

        # Compte RAS
        idx = np.where(mask > 0)
        mx, my, mz = np.median(idx[0]), np.median(idx[1]), np.median(idx[2])
        ref_cog = v2r_init @ np.array([mx, my, mz, 1])

        T_ref_cog = np.eye(4)
        T_ref_cog[0, -1] = -ref_cog[0]
        T_ref_cog[1, -1] = -ref_cog[1]
        T_ref_cog[2, -1] = -ref_cog[2]
        v2r = T_ref_cog @ v2r_init

        return nib.Nifti1Image(data, v2r), T_ref_cog

    if isinstance(proxy_list, list):
        proxy_out = []
        T = []
        for proxy in proxy_list:
            if isinstance(proxy, str):
                proxy = nib.load(proxy)

            p, t = run_fn(proxy)

            proxy_out += [p]
            T += [t]
        return proxy_out, T

    else:
        if isinstance(proxy_list, str):
            proxy_list = nib.load(proxy_list)

        return run_fn(proxy_list)

def create_template_space(proxy_list, resolution=None, mode='linear'):

    reduce_channels = False
    if isinstance(proxy_list, nib.Nifti1Image):
        proxy_list = [proxy_list]
        reduce_channels = True

    if resolution is None:
        ref_v2r = proxy_list[0].affine.astype('float32')
        resolution = np.sqrt(np.sum(ref_v2r * ref_v2r, axis=0))[:-1]

    elif isinstance(resolution, (int, float)):
        resolution = [resolution]*3

    if not isinstance(mode, list):
        mode = [mode]*len(proxy_list)

    boundaries_min = np.zeros((len(proxy_list), 3))
    boundaries_max = np.zeros((len(proxy_list), 3))
    for it_p, proxy in enumerate(proxy_list):
        if isinstance(proxy, str):
            proxy = nib.load(proxy)

        header = proxy.affine
        vox_min = [0, 0, 0, 1]
        vox_max = list(proxy.shape[:3]) + [1]

        minR, minA, minS = np.inf, np.inf, np.inf
        maxR, maxA, maxS = -np.inf, -np.inf, -np.inf

        for i in [vox_min[0], vox_max[0] + 1]:
            for j in [vox_min[1], vox_max[1] + 1]:
                for k in [vox_min[2], vox_max[2] + 1]:
                    aux = np.dot(header, np.asarray([i, j, k, 1]).T)

                    minR, maxR = min(minR, aux[0]), max(maxR, aux[0])
                    minA, maxA = min(minA, aux[1]), max(maxA, aux[1])
                    minS, maxS = min(minS, aux[2]), max(maxS, aux[2])

        boundaries_min[it_p] = [minR, minA, minS]
        boundaries_max[it_p] = [maxR, maxA, maxS]

    # Get the corners of cuboid in RAS space
    minR = np.min(boundaries_min[..., 0])
    minA = np.min(boundaries_min[..., 1])
    minS = np.min(boundaries_min[..., 2])
    maxR = np.max(boundaries_max[..., 0])
    maxA = np.max(boundaries_max[..., 1])
    maxS = np.max(boundaries_max[..., 2])

    # Define header and size
    temp_v2r = np.asarray([[resolution[0], 0, 0, minR],#/(scale[0]*ref_res[0])],#
                          [0, resolution[1], 0, minA],#/(scale[1]*ref_res[1])],#
                          [0, 0, resolution[2], minS],#/(scale[2]*ref_res[2])],#
                          [0, 0, 0, 1]]).astype('float32')

    template_size = np.asarray([int(np.ceil(maxR - minR) / (resolution[0])),
                                int(np.ceil(maxA - minA) / (resolution[1])),
                                int(np.ceil(maxS - minS) / (resolution[2]))])

    II, JJ, KK = np.meshgrid(np.arange(template_size[0]), np.arange(template_size[1]), np.arange(template_size[2]), indexing='ij')
    II, JJ, KK = torch.from_numpy(II), torch.from_numpy(JJ), torch.from_numpy(KK)

    images_out = []
    for proxy, m in zip(proxy_list, mode):
        if isinstance(proxy, str):
            proxy = nib.load(proxy)

        v2r = proxy.affine
        im_shape = proxy.shape
        data = np.array(proxy.dataobj)
        if len(im_shape) > 3:
            data = data.reshape(im_shape[:3] + (-1, ))
        data = torch.from_numpy(data)
        #
        affine = torch.tensor(np.linalg.inv(v2r) @ temp_v2r, device='cpu')
        di = affine[0, 0] * II + affine[0, 1] * JJ + affine[0, 2] * KK + affine[0, 3]
        dj = affine[1, 0] * II + affine[1, 1] * JJ + affine[1, 2] * KK + affine[1, 3]
        dk = affine[2, 0] * II + affine[2, 1] * JJ + affine[2, 2] * KK + affine[2, 3]
        #
        if len(im_shape) > 3:
            im_out = np.transpose(fast_3D_interp_field_torch(data, di, dj, dk, mode=m).numpy(), axes=(3, 0, 1, 2))
            images_out += [im_out.reshape(im_shape[3:] + tuple(template_size))]

        else:
            im_out = fast_3D_interp_torch(data, di, dj, dk, mode=m).numpy()
            images_out += [im_out]

    if len(proxy_list) == 1 and reduce_channels:
        images_out = images_out[0]

    return images_out, temp_v2r

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

def unfold(inp, kernel_size, stride=None, collapse=False):
    """Extract patches from a tensor.
    Parameters
    ----------
    inp : (..., *spatial) tensor
        Input tensor.
    kernel_size : [sequence of] int
        Patch shape.
    stride : [sequence of] int, default=`kernel_size`
        Stride.
    collapse : bool or 'view', default=False
        Collapse the original spatial dimensions.
        If 'view', forces collapsing to use the view mechanism, which ensures
        that no data copy is triggered. This can fail if the tensor's
        strides do not allow these dimensions to be collapsed.
    Returns
    -------
    out : (..., *spatial_out, *kernel_size) tensor
        Output tensor of patches.
        If `collapse`, the output spatial dimensions (`spatial_out`)
        are flattened.
    """
    inp = torch.as_tensor(inp)
    kernel_size = make_list(kernel_size)
    dim = len(kernel_size)
    batch_dim = inp.dim() - dim
    stride = make_list(stride, dim)
    stride = [st or sz for st, sz in zip(stride, kernel_size)]
    for d, (sz, st) in enumerate(zip(kernel_size, stride)):
        inp = inp.unfold(dimension=batch_dim+d, size=sz, step=st)
    if collapse:
        batch_shape = inp.shape[:-dim*2]
        if collapse == 'view':
            inp = inp.view([*batch_shape, -1, *kernel_size])
        else:
            inp = inp.reshape([*batch_shape, -1, *kernel_size])
    return inp

def unsqueeze(input, dim=0, ndim=1):
    """Adds singleton dimensions to a tensor.
    This function expands `torch.unsqueeze` with additional options.
    Parameters
    ----------
    input : tensor_like
        Input tensor.
    dim : int, default=0
        Position at which to insert singleton dimensions.
    ndim : int, default=1
        Number of singleton dimensions to insert.
    Returns
    -------
    output : tensor
        Tensor with additional singleton dimensions.
    """
    for _ in range(ndim):
        input = torch.unsqueeze(input, dim)
    return input

def create_dir(results_dir, subdirs=None):
    if subdirs is None: subdirs = []

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        for sd in subdirs:
            os.makedirs(os.path.join(results_dir, sd))
    else:
        for sd in subdirs:
            if not os.path.exists(os.path.join(results_dir, sd)):
                os.makedirs(os.path.join(results_dir, sd))
def make_sequence(input, n=None, crop=True, *args, **kwargs):
    """Ensure that the input is a sequence and pad/crop if necessary.
    Parameters
    ----------
    input : scalar or sequence or generator
        Input argument(s).
    n : int, optional
        Target length.
    crop : bool, default=True
        Crop input sequence if longer than `n`.
    default : optional
        Default value to pad with.
        If not provided, replicate the last value.
    Returns
    -------
    output : list or tuple or generator
        Output arguments.
    """
    default = None
    has_default = False
    if len(args) > 0:
        default = args[0]
        has_default = True
    elif 'default' in kwargs.keys():
        default = kwargs['default']
        has_default = True

    if isinstance(input, generator):
        # special case for generators
        def make_gen():
            last = None
            i = None
            for i, elem in input:
                if crop and (i == n):
                    return
                last = elem
                yield elem
            if i is None:
                if n is None:
                    return
                if not has_default:
                    raise ValueError('Empty sequence')
                last = default
            for j in range(i + 1, n):
                yield last

        return make_gen()
    else:
        # generic case -> induces a copy
        if not isinstance(input, (list, tuple, range)):
            input = [input]
        return_type = type(input) if isinstance(input, (list, tuple)) else list
        input = list(input)
        if len(input) == 0 and n is not None and not has_default:
            raise ValueError('Empty sequence')
        if n is not None:
            if crop:
                input = input[:min(n, len(input))]
            if not has_default:
                default = input[-1]
            input += [default] * max(0, n - len(input))
        return return_type(input)

def make_list(*args, **kwargs):
    """Ensure that the input is a list and pad/crop if necessary.
    Parameters
    ----------
    input : scalar or sequence generator
        Input argument(s).
    n : int, optional
        Target length.
    crop : bool, default=True
        Crop input sequence if longer than `n`.
    default : optional
        Default value to pad with.
        If not provided, replicate the last value.
    Returns
    -------
    output : list
        Output arguments.
    """
    return [elem for elem in make_sequence(*args, **kwargs)]

def init_weights(net, init_type='normal', init_gain=0.02, init_bias=0.0):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, init_bias)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='normal', init_gain=0.02, device='cpu', gpu_ids=[], init_bias=0):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    else:
        net = net.to(device)

    init_weights(net, init_type, init_gain=init_gain, init_bias=init_bias)
    return net

def align_LR(proxy, w_reg=0.001, device='cpu', verbose=False):
    # Read input image(s)
    v2r = proxy.affine.astype('float32')

    # RAS
    image_shape = proxy.shape[:3]
    if len(proxy.shape) == 3:
        mask = (np.array(proxy.dataobj) > 0).astype('float32')
        # idx = np.where(mask > 0)
        mask_tensor = convert_to_tensor(mask[np.newaxis, np.newaxis], dtype=torch.float16).to(device)

    elif len(proxy.shape) == 4:  # maybe more than one channel.
        mask = np.transpose((np.array(proxy.dataobj) > 0).astype('float32'),
                            axes=(3, 0, 1, 2))  # because it comes from a proxy.
        # idx = np.where(mask[0] > 0)
        mask_tensor = convert_to_tensor(mask[np.newaxis], dtype=torch.float16).to(device)

    else:  # maybe more than one channel and/or batch (i.e. different matrices). still not possible
        raise ValueError("Mask shape has len=5. Not implemented.")

    model = {'reg': InstanceAlignModelClassic(image_shape, device=device, v2r=v2r, tx_factor=50,
                                              align_lr=True, align_ap=True, align_is=False).to(device)}

    # Optimizer
    reg_lr = 1
    max_iter = 10
    optimizer = {'reg': torch.optim.LBFGS(params=model['reg'].parameters(), lr=reg_lr,
                                          max_iter=max_iter, line_search_fn='strong_wolfe')}

    # Training
    temp_dir = os.path.join('/tmp/align_lr_' + str(np.random.randint(0, 1e6)))
    create_dir(temp_dir, subdirs=['results', 'checkpoints'])

    pdict = {
        'save_model_frequency': 10,
        'starting_epoch': 0,
        'num_epochs': 10,
        'results_dir': temp_dir,
        'patience': 1
    }
    loss_dict = {
        'symmetry': {'loss': DiceLoss(name='symmetry', device=device), 'weight': 1},
        'regularizer': {'loss': L2Loss(name='regularizer'), 'weight': w_reg},
    }

    training_session = JointInstanceAlign(loss_dict, [], pdict, device=device, trainable_keys={'reg': 'reg'},
                                          verbose=verbose)
    _ = training_session.register({'mask': mask_tensor}, model, optimizer)

    subprocess.call(['rm', '-rf', temp_dir])

    return model['reg'].get_ras_matrix()[0].detach().cpu().numpy()

# Transforms
class Transform(object):
    def __call__(self, data_dict, *args, **kwargs):
        return

class Pad(Transform):
    def __init__(self, keys, multp, v2r_key=None):
        self.keys = keys
        self.multp = multp
        self.v2r_key = v2r_key

    def __call__(self, data_dict, *args, **kwargs):
        input_shape = data_dict[self.keys[0]].shape
        image_shape = input_shape[-3:]

        pad = [int(np.ceil(image_shape[0]/self.multp)*self.multp - image_shape[0]),
               int(np.ceil(image_shape[1]/self.multp)*self.multp - image_shape[1]),
               int(np.ceil(image_shape[2]/self.multp)*self.multp - image_shape[2])]

        pad_width = [[pad[0]//2, pad[0] - pad[0]//2], [pad[1]//2, pad[1] - pad[1]//2], [pad[2]//2, pad[2] - pad[2]//2]]
        # if len(input_shape) - 3 > 0:
        #     pad_width = [[0, 0] * (len(input_shape) - 3)] + pad_width
        if self.v2r_key is not None:
            T = np.eye(4)
            T[0, 3] = -pad_width[0][0]
            T[1, 3] = -pad_width[1][0]
            T[2, 3] = -pad_width[2][0]
            data_dict[self.v2r_key] = data_dict[self.v2r_key] @ T

        data_dict['pad_width'] = pad_width
        for k in self.keys:
            mask_k = 'mask_' + k + '_pad'
            if isinstance(data_dict[k], torch.Tensor):
                data_dict[mask_k] = torch.ones_like(data_dict[k])
                data_dict[mask_k] = F.pad(data_dict[mask_k], tuple(np.concatenate(pad_width)[::-1].tolist()))
                data_dict[k] = F.pad(data_dict[k], tuple(np.concatenate(pad_width)[::-1].tolist()))
            else:
                if len(data_dict[k].shape) - 3 > 0:
                    padw = [[0, 0]*(len(data_dict[k].shape) - 3)] + pad_width
                else:
                    padw = pad_width
                data_dict[mask_k] = np.ones_like(data_dict[k])
                data_dict[mask_k] = np.pad(data_dict[mask_k], np.array(padw).astype('int'))
                data_dict[k] = np.pad(data_dict[k], np.array(padw).astype('int'))

        return data_dict

class Resample(Transform):
    def __init__(self, keys, v2r_key, ref_key, ref_v2r_key, filter_type='gaussian', *args, **kwargs):
        self.keys = keys
        self.v2r_key = v2r_key
        self.ref_key = ref_key
        self.ref_v2r_key = ref_v2r_key
        self.filter_type = filter_type
        self.normalize_area = kwargs['normalize_area'] if 'normalize_area' in kwargs.keys() else False

    def _gaussian_filter_3d(self, sigma, channels=1, truncate=4):
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
        if self.normalize_area:
            gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 3d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, *kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1, 1)

        gaussian_filter = torch.nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                          padding='same', groups=channels, bias=False)

        gaussian_filter.weight.data = gaussian_kernel.float()
        gaussian_filter.weight.requires_grad = False

        return gaussian_filter

    def _average_filter_3d(self, sigma, channels=1, truncate=4):
        # Set these to whatever you want for your gaussian filter
        kernel_size = tuple([round(s * truncate / 2) * 2 + 1 for s in sigma])

        av_kernel = torch.ones(kernel_size)
        # Make sure sum of values in gaussian kernel equals 1.
        av_kernel = av_kernel / torch.sum(av_kernel)

        # Reshape to 3d depthwise convolutional weight
        av_kernel = av_kernel.view(1, 1, *kernel_size)
        av_kernel = av_kernel.repeat(channels, 1, 1, 1, 1).float()

        av_filter = torch.nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                    padding='same', groups=channels, bias=False)
        av_filter.weight.data = av_kernel
        av_filter.weight.requires_grad = False

        return av_filter

    def _blur(self, image, image_res, sampling_res):
        sigma = 0.25 * sampling_res / image_res
        sigma[sampling_res == image_res] = 0.5
        if all([s <= 0.5 for s in sigma]): return image

        if self.filter_type == 'gaussian':
            filt = self._gaussian_filter_3d(sigma)
        elif self.filter_type == 'average':
            filt = self._average_filter_3d(sigma)
        else:
            filt = lambda x: x

        filt = filt.to(image.device)
        return filt(image)

    def _interp(self, X, II, JJ, KK, mode):

        II = II.to(X.device)
        JJ = JJ.to(X.device)
        KK = KK.to(X.device)
        num_channels = X.shape[1]
        Y = torch.zeros([1, num_channels, *II.shape], device=X.device)
        if mode == 'nearest':
            IIr = torch.round(II).long()
            JJr = torch.round(JJ).long()
            KKr = torch.round(KK).long()
            IIr[IIr < 0] = 0
            JJr[JJr < 0] = 0
            KKr[KKr < 0] = 0
            IIr[IIr > (X.shape[2] - 1)] = (X.shape[2] - 1)
            JJr[JJr > (X.shape[3] - 1)] = (X.shape[3] - 1)
            KKr[KKr > (X.shape[4] - 1)] = (X.shape[4] - 1)
            for channel in range(num_channels):
                #
                Xc = X[0, channel]
                Y[0, channel] = Xc[IIr, JJr, KKr]

        elif mode == 'linear':
            #
            ok = (II >= 0) & (JJ >= 0) & (KK >= 0) & (II <= X.shape[2] - 1) & (JJ <= X.shape[3] - 1) & (KK <= X.shape[4] - 1)
            IIv = II[ok]
            JJv = JJ[ok]
            KKv = KK[ok]
            #
            del JJ, KK
            #
            fx = torch.floor(IIv).long()
            cx = fx + 1
            cx[cx > (X.shape[2] - 1)] = (X.shape[2] - 1)
            wcx = IIv - fx
            wfx = 1 - wcx
            #
            fy = torch.floor(JJv).long()
            cy = fy + 1
            cy[cy > (X.shape[3] - 1)] = (X.shape[3] - 1)
            wcy = JJv - fy
            wfy = 1 - wcy
            #
            fz = torch.floor(KKv).long()
            cz = fz + 1
            cz[cz > (X.shape[4] - 1)] = (X.shape[4] - 1)
            wcz = KKv - fz
            wfz = 1 - wcz
            #
            for channel in range(num_channels):
                #
                Xc = X[0, channel]
                #
                c000 = Xc[fx, fy, fz]
                c100 = Xc[cx, fy, fz]
                c010 = Xc[fx, cy, fz]
                c110 = Xc[cx, cy, fz]
                c001 = Xc[fx, fy, cz]
                c101 = Xc[cx, fy, cz]
                c011 = Xc[fx, cy, cz]
                c111 = Xc[cx, cy, cz]
                #
                c00 = c000 * wfx + c100 * wcx
                c01 = c001 * wfx + c101 * wcx
                c10 = c010 * wfx + c110 * wcx
                c11 = c011 * wfx + c111 * wcx
                #
                c0 = c00 * wfy + c10 * wcy
                c1 = c01 * wfy + c11 * wcy
                #
                c = c0 * wfz + c1 * wcz
                #
                Yc = torch.zeros(II.shape, device=X.device)
                Yc[ok] = c.float().to(X.device)
                #
                Y[0, channel] = Yc
            #
        return Y

    def __call__(self, data_dict, *args, **kwargs):
        # parameters
        ref_image = data_dict[self.ref_key]
        ref_v2r = data_dict[self.ref_v2r_key]

        v2r = data_dict[self.v2r_key]
        affine = np.linalg.inv(v2r) @ ref_v2r

        pixdim = np.sqrt(np.sum(v2r * v2r, axis=0))[:-1]
        new_vox_size = np.sqrt(np.sum(ref_v2r * ref_v2r, axis=0))[:-1]
        factor = pixdim / new_vox_size
        sigmas = 0.25 / factor
        sigmas[factor > 1] = 0  # don't blur if upsampling

        II, JJ, KK = np.meshgrid(np.arange(ref_image.shape[2]), np.arange(ref_image.shape[3]), np.arange(ref_image.shape[4]), indexing='ij')
        II, JJ, KK = torch.from_numpy(II), torch.from_numpy(JJ), torch.from_numpy(KK)
        di = affine[0, 0] * II + affine[0, 1] * JJ + affine[0, 2] * KK + affine[0, 3]
        dj = affine[1, 0] * II + affine[1, 1] * JJ + affine[1, 2] * KK + affine[1, 3]
        dk = affine[2, 0] * II + affine[2, 1] * JJ + affine[2, 2] * KK + affine[2, 3]

        type_dict = {}
        shape_dict = {}
        for k in self.keys:
            type_dict[k] = get_type_dict(data_dict[k])
            shape_dict[k] = data_dict[k].shape

        image_list = []
        for k in self.keys:
            image = convert_to_tensor(data_dict[k])

            if len(shape_dict[k]) == 3:
                image = [image]
            elif len(shape_dict[k]) == 4:
                image = list(torch.unbind(image, dim=0))
            elif len(shape_dict[k]) == 5:
                image = list(torch.unbind(image.view((-1, ) + shape_dict[k][2:]), dim=0))

            image_list += image

        image = torch.stack([torch.unsqueeze(f) for f in image_list], axis=0)
        if any([s > 0 for s in sigmas]):
            image = self._blur(image, pixdim, new_vox_size)
        image = torch.permute(image, (1, 0, 2, 3, 4))
        image = self._interp(image, di, dj, dk, mode='linear')

        it_image = 0
        for k in self.keys:
            data_dict[self.v2r_key] = ref_v2r
            if len(shape_dict[k]) == 3:
                data_dict[k] = convert_to_type(image[it_image], **type_dict)[0]
                it_image += 1

            elif len(shape_dict[k]) == 4:
                data_dict[k] = torch.stack([image[it_image + it][0] for it in range(shape_dict[k][0])])
                data_dict[k] = convert_to_type(data_dict[k], ** type_dict)
                it_image += shape_dict[k][0]

            elif len(shape_dict[k]) == 5:
                data_dict[k] = torch.stack([image[it_image + it, 0] for it in range(np.prod(shape_dict[k][:2]))])
                data_dict[k] = convert_to_type(data_dict[k].reshape(shape_dict[k]), **type_dict)
                it_image += np.prod(shape_dict[k][:2])

        return data_dict

class RescaleVoxelSize(Resample):
    def __init__(self, keys, v2r_key, filter_type='gaussian', target_res=1, output_shape=None, *args, **kwargs):
        super(RescaleVoxelSize, self).__init__(keys=keys, v2r_key=v2r_key, ref_key=None, ref_v2r_key=None,
                                               filter_type=filter_type, *args, **kwargs)

        self.target_res = np.array([target_res]*3) if isinstance(target_res, (float, int)) else np.array(target_res)
        self.output_shape = output_shape
        self.normalize_area = kwargs['normalize_area'] if 'normalize_area' in kwargs.keys() else False

    def _resample(self, image, v2r, sampling_res, mode='linear'):

        if len(v2r.shape) == 3:
            v2r = v2r[0]

        image_res = np.sqrt(np.sum(v2r * v2r, axis=0))[:-1]
        imshape = image.shape[-3:]
        outshape = np.int32(np.asarray(imshape) * image_res / sampling_res)

        factor = np.asarray(outshape) / np.asarray(imshape)
        start = - (factor - 1) / (2 * factor)
        step = 1.0 / factor
        stop = start + step * outshape

        xd = torch.arange(start=start[0], end=stop[0], step=step[0])
        yd = torch.arange(start=start[1], end=stop[1], step=step[1])
        zd = torch.arange(start=start[2], end=stop[2], step=step[2])
        IId, JJd, KKd = torch.meshgrid([xd, yd, zd], indexing='ij')

        if len(image.shape) == 5:
            image = torch.permute(image, (1, 0, 2, 3, 4))
            image = self._blur(image, image_res, sampling_res)
            image = torch.permute(image, (1, 0, 2, 3, 4))
        else:
            image = self._blur(image, image_res, sampling_res)

        new_aff = v2r.copy()
        for c in range(3):
            new_aff[:-1, c] = new_aff[:-1, c] / factor[c]
        new_aff[:-1, -1] = new_aff[:-1, -1] - np.matmul(new_aff[:-1, :-1], 0.5 * (factor - 1))

        return self._interp(image, IId, JJd, KKd, mode=mode), new_aff

    def __call__(self, data_dict, *args, **kwargs):
        # parameters
        v2r = data_dict[self.v2r_key]

        # get resolution
        if self.output_shape is not None:
            if isinstance(self.output_shape, str):
                output_shape = data_dict[self.output_shape].shape[2:]
            else:
                output_shape = self.output_shape
            input_shape = data_dict[self.keys[0]].shape[2:]
            image_res = np.sqrt(np.sum(v2r * v2r, axis=0))[:-1]
            sampling_res = np.asarray(input_shape) * image_res / output_shape

        else:
            sampling_res = self.target_res

        data_dict['sampling_res'] = self.target_res

        for k in self.keys:
            type_dict = get_type_dict(data_dict[k])
            image = convert_to_tensor(data_dict[k], dtype=torch.float32)
            image_shape = image.shape
            if len(image_shape) == 3:
                image = torch.unsqueeze(torch.unsqueeze(image, 0), 0)
            elif len(image_shape) == 4:
                image = torch.unsqueeze(image, 0)

            image, new_aff = self._resample(image, v2r, sampling_res)

            data_dict[k] = convert_to_type(image, **type_dict)
            if len(image_shape) == 3:
                data_dict[k] = data_dict[k][0, 0]
            elif len(image_shape) == 4:
                data_dict[k] = data_dict[k][0]

            data_dict[self.v2r_key] = new_aff

        return data_dict

class GaussianBlur(Transform):
    def __init__(self, keys, sigma, prob=1, *args, **kwargs):
        self.keys = keys
        self.sigma = sigma
        self.prob = prob
        self.normalize_area = kwargs['normalize_area'] if 'normalize_area' in kwargs.keys() else False

    def _gaussian_filter_3d(self, sigma, channels=1, truncate=4):
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
        if self.normalize_area:
            gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 3d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, *kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1, 1)

        gaussian_filter = torch.nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                          padding='same', groups=channels, bias=False)

        gaussian_filter.weight.data = gaussian_kernel.float()
        gaussian_filter.weight.requires_grad = False

        return gaussian_filter

    def _blur(self, image):
        if callable(self.sigma):
            sigma = self.sigma()
        else:
            sigma = self.sigma
        filt = self._gaussian_filter_3d(sigma)
        filt = filt.to(image.device)

        return filt(image)

    def __call__(self, data_dict, *args, **kwargs):
        if np.random.rand() > self.prob:
            return data_dict

        for k in self.keys:
            type_dict = get_type_dict(data_dict[k])
            image = convert_to_tensor(data_dict[k])
            image_shape = image.shape
            if len(image_shape) == 3:
                image = torch.unsqueeze(torch.unsqueeze(image, 0), 0)

            data_dict[k] = convert_to_type(self._blur(image), **type_dict)
            if len(image_shape) == 3:
                data_dict[k] = data_dict[k][0, 0]

        return data_dict

class MinMaxNormalization(Transform):
    def __init__(self, keys, new_min=0, new_max=1, percentile_min=0, percentile_max=100, min_val=None,
                 max_val=None):
        self.keys = keys

        self.new_min = new_min
        self.new_max = new_max
        self.percentile_min = percentile_min
        self.percentile_max = percentile_max
        self.min_val = min_val
        self.max_val = max_val

    def _get_min_max(self, data):
        if self.min_val is not None:
            if callable(self.min_val):
                m = self.min_val(data.numpy())
            else:
                m = self.min_val
        else:
            m = np.percentile(data, self.percentile_min)

        if self.max_val is not None:
            if callable(self.min_val):
                M = self.max_val(data.numpy())
            else:
                M = self.max_val
        else:
            M = np.percentile(data, self.percentile_max)

        return m, M

    def _get_min_max_torch(self, data):
        if self.min_val is not None:
            if callable(self.min_val):
                m = self.min_val(data)
            else:
                m = self.min_val
        else:
            m = torch.quantile(data, self.percentile_min/100)

        if self.max_val is not None:
            if callable(self.max_val):
                M = self.max_val(data)
            else:
                M = self.max_val
        else:
            M = torch.quantile(data, self.percentile_max/100)

        return m, M

    def __call__(self, data_dict, *args, **kwargs):
        for k in self.keys:
            if isinstance(data_dict[k], torch.Tensor):
                m, M = self._get_min_max_torch(data_dict[k])
                data_dict[k] = (data_dict[k] - m) / (M - m) * (self.new_max - self.new_min) + self.new_min
                data_dict[k] = torch.clamp(data_dict[k], self.new_min, self.new_max)

            else:
                m, M = self._get_min_max(data_dict[k])
                data_dict[k] = (data_dict[k] - m) / (M - m) * (self.new_max - self.new_min) + self.new_min
                data_dict[k] = np.clip(data_dict[k], self.new_min, self.new_max)

        return data_dict

class AddBatchAxis(Transform):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data_dict, *args, **kwargs):
        for k in self.keys:
            if len(data_dict[k].shape) == 3:
                data_dict[k] = data_dict[k][np.newaxis, np.newaxis]
            elif len(data_dict[k].shape) == 4:
                data_dict[k] = data_dict[k][np.newaxis]
            else:
                pass

        return data_dict

class ToTensor(Transform):
    def __init__(self, keys, device='cpu'):
        self.keys = keys
        self.device = device

    def __call__(self, data_dict, *args, **kwargs):
        for k in self.keys:
            if len(data_dict[k].shape) == 3:
                data_dict[k] = torch.from_numpy(data_dict[k][np.newaxis].astype('float32')).to(self.device)
            else:
                data_dict[k] = torch.from_numpy(data_dict[k].astype('float32')).to(self.device)
        return data_dict

class ToNumpy(Transform):
    def __init__(self, keys, to_nibabel=False):
        self.keys = keys
        self.to_nibabel = to_nibabel

    def __call__(self, data_dict, *args, **kwargs):
        for k in self.keys:
            if isinstance(data_dict[k], torch.Tensor):
                data_dict[k] = np.squeeze(data_dict[k].detach().cpu().numpy())
                if len(data_dict[k].shape) == 4 and self.to_nibabel:
                    data_dict[k] = np.transpose(data_dict[k], axes=(1, 2, 3, 0))

        return data_dict

class OneHot(Transform):
    def __init__(self, keys, lut=None, num_classes=None):
        self.keys = keys
        self.lut = lut
        self.num_classes = num_classes

    def _one_hot(self, target, num_classes=None, categories=None):
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

    def __call__(self, data_dict, *args, **kwargs):
        for k in self.keys:
            type_dict = get_type_dict(data_dict[k])
            image = convert_to_numpy(data_dict[k])
            image_shape = image.shape
            if len(image_shape) == 5:
                image = image[0, 0]

            if self.lut is not None:
                image = convert_labelmap(image.astype('int'), self.lut)
                num_classes = len(np.unique(np.array(list(self.lut.values()))))
            else:
                num_classes = self.num_classes

            image = self._one_hot(image, num_classes)
            data_dict[k] = convert_to_type(image, **type_dict)
            if len(image_shape) == 5:
                data_dict[k] = torch.unsqueeze(data_dict[k], 0)

        return data_dict

class Argmax(Transform):
    def __init__(self, keys, axis=None, keepdims=False, *args, **kwargs):
        self.keys = keys
        self.axis = axis
        self.keepdims = keepdims

    def _torch_argmax(self, seg):
        if self.axis is not None:
            seg = torch.argmax(seg, self.axis, keepdims=self.keepdims).float()
        elif len(seg.shape) == 5:
            seg = torch.argmax(seg, 1, keepdims=self.keepdims).float()
        elif len(seg.shape) == 4:
            seg = torch.argmax(seg, 0, keepdims=self.keepdims).float()

        return seg

    def _np_argmax(self, seg):
        if self.axis is not None:
            seg = np.argmax(seg, self.axis, keepdims=self.keepdims).astype('float32')
        elif len(seg.shape) == 5:
            seg = np.argmax(seg, 1, keepdims=self.keepdims).astype('float32')
        elif len(seg.shape) == 4:
            seg = np.argmax(seg, 0, keepdims=self.keepdims).astype('float32')

        return seg

    def __call__(self, data_dict, *args, **kwargs):
        for k in self.keys:
            if isinstance(data_dict[k], np.ndarray):
                data_dict[k] = self._np_argmax(data_dict[k])
            else:
                data_dict[k] = self._torch_argmax(data_dict[k])

        return data_dict

class RestrictVagina(Transform):
    def __init__(self, keys, num_blobs=1):
        self.keys = keys
        self.num_blobs = num_blobs


class ConnectedComponents(Transform):
    def __init__(self, keys, num_blobs=1):
        self.keys = keys
        self.num_blobs = num_blobs

    def __call__(self, data_dict, *args, **kwargs):
        for k in self.keys:
            tdict = get_type_dict(data_dict[k])
            if not isinstance(data_dict[k], np.ndarray):
                data_dict[k] = convert_to_numpy(data_dict[k])

            pred = np.squeeze(data_dict[k])
            all_blobs, num_blobs = measure.label(pred, connectivity=2, return_num=True)
            all_blobs_count = np.bincount(all_blobs[pred > 0])

            pred_refined = np.zeros_like(pred)
            idx_regions = np.argsort(all_blobs_count)[::-1]
            for num_blob in range(self.num_blobs):
                pred_refined[all_blobs == idx_regions[num_blob]] = 1

            if not isinstance(data_dict[k], np.ndarray):
                data_dict[k] = convert_to_type(pred_refined, type=tdict['type'], dtype=tdict['dtype'])
            else:
                data_dict[k] = pred_refined

        return data_dict

# Losses
class _Loss(nn.Module):
    def __init__(self, name = None):
        super().__init__()
        self.name = name

class SSIM(_Loss):
    def __init__(self, name=None, reduction='mean', *args, **kwargs):
        if name is None:
            name='SSIM'
        super().__init__(name=name)
        self.reduction = reduction

    @NotImplementedError
    def _ssim_loss(self, prediction, target, reduction='mean'):
        pass

    def forward(self, prediction, target, mask=None, weight=None, *args, **kwargs):
        ndims = len(prediction.shape)
        if mask is None and weight is None:
            return self._ssim_loss(prediction, target, reduction=self.reduction)
        else:
            res = self._ssim_loss(prediction, target, reduction='none')
            if mask is not None:
                res = res * mask
            if weight is not None:
                res = res * weight

        if self.reduction == 'mean':
            return 1/torch.sum(mask)*torch.sum(res)
            # norm_factor = torch.sum(mask, dim=[it for it in range(1,ndims)], keepdim=True)
            # wk = 1 / norm_factor * torch.sum(mask, dim=[it for it in range(2,ndims)])
            # res = 1 / norm_factor * torch.sum(res, dim=[it for it in range(2,ndims)])
            # res = torch.sum(wk*res, dim=1)
            # res = torch.mean(res)

        elif self.reduction == 'sum':
            return torch.sum(res)
        else:
            return res

        return res

class L2Loss(SSIM):
    def _ssim_loss(self, prediction, target, reduction='mean'):
        return F.mse_loss(prediction, target, reduction=reduction)

class DiceLoss(_Loss):
    def __init__(self, name=None, *args, **kwargs):

        if name is None:
            name='dice'
        super().__init__(name=name)

    def forward(self, target, prediction, mask=None, classes_compute=None, eps = 0.0000001):
        """Dice loss.
        Compute the dice similarity loss (approximation of the DSC). The foreground

        Parameters
        ----------
        prediction : torch variable of size (batch_size, num_classes, d1, d2, ..., dN) representing the post-softmax
            values

        target : torch variable of ssize (batch_size, num_classes, d1, d2, ..., dN) representing a 1-hot encoding of the
            target values

        Returns
        -------
        dice_total :

        """
        # smooth = eps #1.
        # pflat = prediction.view(-1)
        # tflat = target.view(-1)
        # intersection = (pflat * tflat).sum()
        #
        # return 1 - ((2. * intersection + smooth) / (pflat.sum() + tflat.sum() + smooth))



        # prediction = torch.clip(prediction / torch.sum(prediction, dim=1, keepdims=True), 0, 1)
        # target = torch.clip(target / torch.sum(target, dim=1, keepdims=True), 0, 1)
        if classes_compute is not None:
            prediction = prediction[:, classes_compute]
            target = target[:, classes_compute]

        if mask is not None:
            prediction = prediction * mask
            target = target * mask

        top = torch.sum(2 * prediction * target, dim=list(range(1, len(prediction.shape))))
        bottom = prediction**2 + target**2 + eps
        bottom = torch.sum(bottom, dim=list(range(1, len(prediction.shape))))

        last_tensor = top / bottom

        return torch.mean(1 - last_tensor)


# Segmentation classes
class SegmentationInference(object):
    def __init__(self, in_tf, out_tf, device, **kwargs):
        self.in_tf = in_tf
        self.out_tf = out_tf
        self.device = device

        self.num_blobs_bowel = kwargs['num_blobs_bowel'] if 'num_blobs_bowel' in kwargs else 1
        self.num_blobs_rectum =  kwargs['num_blobs_rectum'] if 'num_blobs_rectum' in kwargs else 1
        self.num_blobs_sigma =  kwargs['num_blobs_sigma'] if 'num_blobs_rectum' in kwargs else 2


    def forward(self, tensor_dict, model_dict, *kwargs):
        with torch.no_grad():
            for tf in self.in_tf:
                tensor_dict = tf(tensor_dict)

            tensor_dict['input_image'] = tensor_dict['image'].to(self.device)
            tensor_dict['output_seg'] = model_dict['seg'](tensor_dict['input_image'])

        return tensor_dict

    def forward_sliding(self, tensor_dict, model_dict, spatial_size=(256, 256, 64), **kwargs):
        from monai.inferers import sliding_window_inference
        with torch.no_grad():
            for tf in self.in_tf:
                tensor_dict = tf(tensor_dict)

            tensor_dict['input_image'] = tensor_dict['image'].to(self.device)
            tensor_dict['output_seg'] = sliding_window_inference(tensor_dict['input_image'], spatial_size, 4, model_dict['seg'], overlap=0.8)

        return tensor_dict
    def morphological_postprocessing(self, tensor_dict):

        pred = np.squeeze(tensor_dict['output_seg'])[2]
        all_blobs, num_blobs = measure.label(pred, connectivity=2, return_num=True)
        all_blobs_count = np.bincount(all_blobs[pred > 0])

        pred_refined = np.zeros_like(pred)
        idx_regions = np.argsort(all_blobs_count)[::-1]
        for num_blob in range(self.num_blobs_bowel):
            if np.sum(all_blobs == idx_regions[num_blob]) < 100000: continue
            pred_refined[all_blobs == idx_regions[num_blob]] = 1
        tensor_dict['output_seg'][2] = pred_refined

        pred = np.squeeze(tensor_dict['output_seg'])[3]
        all_blobs, num_blobs = measure.label(pred, connectivity=2, return_num=True)
        all_blobs_count = np.bincount(all_blobs[pred > 0])

        pred_refined = np.zeros_like(pred)
        idx_regions = np.argsort(all_blobs_count)[::-1]
        for num_blob in range(self.num_blobs_rectum):
            if np.sum(all_blobs == idx_regions[num_blob]) < 65000: continue
            pred_refined[all_blobs == idx_regions[num_blob]] = 1
        tensor_dict['output_seg'][3] = pred_refined

        pred = np.squeeze(tensor_dict['output_seg'])[3]
        all_blobs, num_blobs = measure.label(pred, connectivity=2, return_num=True)
        all_blobs_count = np.bincount(all_blobs[pred > 0])

        pred_refined = np.zeros_like(pred)
        idx_regions = np.argsort(all_blobs_count)[::-1]
        for num_blob in range(self.num_blobs_rectum):
            if np.sum(all_blobs == idx_regions[num_blob]) < 65000: continue
            pred_refined[all_blobs == idx_regions[num_blob]] = 1
        tensor_dict['output_seg'][3] = pred_refined

        pred = np.squeeze(tensor_dict['output_seg'])[4]
        all_blobs, num_blobs = measure.label(pred, connectivity=2, return_num=True)
        all_blobs_count = np.bincount(all_blobs[pred > 0])

        pred_refined = np.zeros_like(pred)
        idx_regions = np.argsort(all_blobs_count)[::-1]
        for num_blob in range(min(self.num_blobs_sigma, len(idx_regions))):
            if np.sum(all_blobs == idx_regions[num_blob]) < 2500: continue
            pred_refined[all_blobs == idx_regions[num_blob]] = 1
        tensor_dict['output_seg'][4] = pred_refined

        pred = np.squeeze(tensor_dict['output_seg'])[5]
        all_blobs, num_blobs = measure.label(pred, connectivity=2, return_num=True)
        all_blobs_count = np.bincount(all_blobs[pred > 0])

        pred_refined = np.zeros_like(pred)
        idx_regions = np.argsort(all_blobs_count)[::-1]
        for num_blob in range(1):
            pred_refined[all_blobs == idx_regions[num_blob]] = 1

        tensor_dict['output_seg'][5] = pred_refined

        return tensor_dict

    def step(self, data_dict, model_dict, *args, **kwargs):
        tensor_dict = self.forward(data_dict, model_dict)

        for out_tf in self.out_tf:
            tensor_dict = out_tf(tensor_dict)

        return tensor_dict

    # def step_unique(self, data_dict, model_dict):
    #     tensor_dict = self.forward(data_dict, model_dict)
    #
    #     # Register to original image
    #     v2r = np.linalg.inv(data_dict['affine']) @ data_dict['image_v2r']
    #     pred = np.transpose(tensor_dict['output_seg'][0].cpu().numpy(), axes=(1, 2, 3, 0))
    #     pred_proxy = nib.Nifti1Image(pred, v2r)
    #     orig_proxy = nib.Nifti1Image(data_dict['orig_image'], data_dict['orig_v2r'])
    #     pred_proxy = vol_resample_fast(orig_proxy, pred_proxy)
    #
    #     tensor_dict['output_seg'] = np.transpose(np.array(pred_proxy.dataobj), axes=(3, 0, 1, 2))
    #
    #     # Post-processing
    #     tensor_dict = self.morphological_postprocessing(tensor_dict)
    #
    #     return tensor_dict



    def step_full(self, data_dict, model_dict, sliding_window=False, *args, **kwargs):

        if sliding_window:
            tensor_dict = self.forward_sliding(data_dict, model_dict, **kwargs)
        else:
            tensor_dict = self.forward(data_dict, model_dict)

        # Register to original image
        if 'affine' in tensor_dict.keys():
            v2r = np.linalg.inv(tensor_dict['affine']) @ tensor_dict['image_v2r']
        else:
            v2r = tensor_dict['image_v2r']

        pred = np.transpose(tensor_dict['output_seg'][0].cpu().numpy(), axes=(1, 2, 3, 0))
        pred_proxy = nib.Nifti1Image(pred, v2r)
        orig_proxy = nib.Nifti1Image(data_dict['orig_image'], data_dict['orig_v2r'])
        pred_proxy = vol_resample_fast(orig_proxy, pred_proxy)

        tensor_dict['output_seg'] = np.transpose(np.array(pred_proxy.dataobj), axes=(3, 0, 1, 2))

        for out_tf in self.out_tf:
            tensor_dict = out_tf(tensor_dict)

        tensor_dict = self.morphological_postprocessing(tensor_dict)

        # pred = np.squeeze(tensor_dict['output_seg'][2])
        # all_blobs, num_blobs = measure.label(pred, connectivity=2, return_num=True)
        # all_blobs_count = np.bincount(all_blobs[pred > 0])
        #
        # pred_refined = np.zeros_like(pred)
        # idx_regions = np.argsort(all_blobs_count)[::-1]
        # for num_blob in range(self.num_blobs_bowel):
        #     if np.sum(all_blobs == idx_regions[num_blob]) < 100000: continue
        #     pred_refined[all_blobs == idx_regions[num_blob]] = 1
        # tensor_dict['output_seg'][2] = pred_refined
        #
        # pred = np.squeeze(tensor_dict['output_seg'][3])
        # all_blobs, num_blobs = measure.label(pred, connectivity=2, return_num=True)
        # all_blobs_count = np.bincount(all_blobs[pred > 0])
        #
        # pred_refined = np.zeros_like(pred)
        # idx_regions = np.argsort(all_blobs_count)[::-1]
        # for num_blob in range(self.num_blobs_rectum):
        #     if np.sum(all_blobs == idx_regions[num_blob]) < 65000: continue
        #     pred_refined[all_blobs == idx_regions[num_blob]] = 1
        # tensor_dict['output_seg'][3] = pred_refined
        #
        # pred = np.squeeze(tensor_dict['output_seg'][3])
        # all_blobs, num_blobs = measure.label(pred, connectivity=2, return_num=True)
        # all_blobs_count = np.bincount(all_blobs[pred > 0])
        #
        # pred_refined = np.zeros_like(pred)
        # idx_regions = np.argsort(all_blobs_count)[::-1]
        # for num_blob in range(self.num_blobs_rectum):
        #     if np.sum(all_blobs == idx_regions[num_blob]) < 65000: continue
        #     pred_refined[all_blobs == idx_regions[num_blob]] = 1
        # tensor_dict['output_seg'][3] = pred_refined
        #
        # pred = np.squeeze(tensor_dict['output_seg'][4])
        # all_blobs, num_blobs = measure.label(pred, connectivity=2, return_num=True)
        # all_blobs_count = np.bincount(all_blobs[pred > 0])
        #
        # pred_refined = np.zeros_like(pred)
        # idx_regions = np.argsort(all_blobs_count)[::-1]
        # for num_blob in range(self.num_blobs_sigma):
        #     if np.sum(all_blobs == idx_regions[num_blob]) < 2500: continue
        #     pred_refined[all_blobs == idx_regions[num_blob]] = 1
        # tensor_dict['output_seg'][4] = pred_refined
        #
        # pred = np.squeeze(tensor_dict['output_seg'][5])
        # all_blobs, num_blobs = measure.label(pred, connectivity=2, return_num=True)
        # all_blobs_count = np.bincount(all_blobs[pred > 0])
        #
        # pred_refined = np.zeros_like(pred)
        # idx_regions = np.argsort(all_blobs_count)[::-1]
        # for num_blob in range(1):
        #     pred_refined[all_blobs == idx_regions[num_blob]] = 1
        #
        # tensor_dict['output_seg'][5] = pred_refined

        return tensor_dict

class JointInstanceReg(object):

    def __init__(self, loss_dict, callbacks, main_dict, device='cpu', trainable_keys=None, verbose=True, **kwargs):
        self.loss_dict = loss_dict
        self.log_keys = ['loss_' + loss['loss'].name for loss in loss_dict.values()] + \
                        ['w_loss_' + loss['loss'].name for loss in loss_dict.values()] + \
                        ['val_loss_' + loss['loss'].name for loss in loss_dict.values()] + \
                        ['val_loss', 'loss', 'time_duration (s)']

        attach = True if main_dict['starting_epoch'] > 0 else False
        mcheck = ModelCheckpoint(os.path.join(main_dict['results_dir'], 'checkpoints'), main_dict['save_model_frequency'])
        training_printer = PrinterCallback()
        training_tocsv = ToCSVCallback(filepath=os.path.join(main_dict['results_dir'], 'results', 'training_results.csv'),
                                       keys=self.log_keys, attach=attach)
        callback_list = [mcheck, training_tocsv]
        if verbose:
            callback_list += [training_printer]

        self.callbacks = callback_list + callbacks

        self.trainable_keys = trainable_keys if trainable_keys is not None else {}
        if not isinstance(self.trainable_keys, dict): self.trainable_keys = {self.trainable_keys: self.trainable_keys}

        if 'min_epoch_val' not in main_dict.keys():
            main_dict['min_epoch_val'] = 0

        self.main_dict = main_dict
        self.device = device
        self.kwargs = kwargs

    def train(self, generator_train, model_dict, optimizer_dict, **kwargs):
        for cb in self.callbacks:
            cb.on_train_init(model_dict, starting_epoch=self.main_dict['starting_epoch'])

        logs_dict = {}
        best_loss = 100000000
        persistent_epoch = 0
        for epoch in range(self.main_dict['starting_epoch'], self.main_dict['num_epochs']):

            epoch_start_time = time.time()
            for cb in self.callbacks:
                cb.on_epoch_init(model_dict, epoch)

            for m in self.trainable_keys.values():
                model_dict[m].train()

            logs_dict = self.iterate(generator_train, model_dict, optimizer_dict, epoch, **kwargs)

            if epoch >= self.main_dict['min_epoch_val'] and 'val' in kwargs.keys():
                for m in self.trainable_keys.values():
                    model_dict[m].eval()
                with torch.no_grad():
                    val_log_dict = self.iterate_val(kwargs['val'], model_dict)
                    logs_dict = {**logs_dict, **{'val_' + k: v for k, v in val_log_dict.items()}}
            else:
                logs_dict = {**logs_dict, **{'val_' + k: v for k, v in logs_dict.items() if k[0] != 'w'}}


            epoch_end_time = time.time()
            logs_dict['time_duration (s)'] = epoch_end_time - epoch_start_time

            for cb in self.callbacks:
                cb.on_epoch_fi(logs_dict, model_dict, epoch, optimizer=optimizer_dict)

            if ((best_loss - logs_dict['val_loss']) / np.abs(best_loss)) * 100 < 0.1:  # if loss improves less than 0.1%
                persistent_epoch += 1

            else:
                checkpoint = {
                    'epoch': epoch + 1,
                }
                for model_name, model_instance in model_dict.items():
                    try:
                        checkpoint['state_dict_' + model_name] = model_instance.state_dict()
                    except:
                        pass

                for optimizer_name, optimizer_instance in optimizer_dict.items():
                    checkpoint['optimizer_' + optimizer_name] = optimizer_instance.state_dict()

                filepath = os.path.join(self.main_dict['results_dir'], 'checkpoints', 'model_checkpoint.BEST.pth')
                torch.save(checkpoint, filepath)

                persistent_epoch = 0
                best_loss = logs_dict['val_loss']

            if persistent_epoch >= self.main_dict['patience']:
                break

        for cb in self.callbacks:
            cb.on_train_fi(model_dict)

        return logs_dict

    def register(self, data_dict, model_dict, optimizer_dict, **kwargs):
        logs_dict = self.train(data_dict, model_dict, optimizer_dict, **kwargs)
        model_dict['reg'].eval()
        data_dict['parameters'] = model_dict['reg'].get_params()
        with torch.no_grad():
            data_dict['loss'] = logs_dict['loss']
            data_dict['affine'] = model_dict['reg'].get_matrix()
            data_dict['affine_ras'] = model_dict['reg'].get_ras_matrix()
            data_dict = self.forward(data_dict, model_dict)

        return data_dict

    def forward(self, data_dict, model_dict):
        if 'flo_mask' in data_dict.keys():
            im = model_dict['reg'](torch.cat((data_dict['flo_image'], data_dict['flo_mask']), axis=1))
            data_dict['reg_image'] = im[:, 0:1]
            data_dict['reg_mask'] = im[:, 1:]
        else:
            data_dict['reg_image'] = model_dict['reg'](data_dict['flo_image'])

        return data_dict

    def iterate(self, data_dict, model_dict, optimizer_dict, epoch, **kwargs):
        def closure():
            if torch.is_grad_enabled():
                for k in self.trainable_keys.values():
                    optimizer_dict[k].zero_grad()

            data_dict_closure = self.forward(data_dict, model_dict)
            loss, _ = self.compute_loss(data_dict_closure, model_dict)
            loss.backward()

            return loss

        for k in self.trainable_keys.values():
            optimizer_dict[k].step(closure=closure)

        with torch.no_grad():
            data_dict = self.forward(data_dict, model_dict)
            loss, log_dict = self.compute_loss(data_dict, model_dict)

        log_dict['loss'] = loss.item()
        return log_dict

    def compute_loss(self, data_dict, model_dict):
        log_dict = {}

        # Registration
        sim_loss = 0.
        if self.loss_dict['reg']['weight'] > 0:
            sim_loss = self.loss_dict['reg']['loss'](data_dict['reg_image'], data_dict['ref_image'],
                                                     mask=data_dict['ref_mask'])
            log_dict['loss_' + self.loss_dict['reg']['loss'].name] = sim_loss.item()
            sim_loss = self.loss_dict['reg']['weight'] * sim_loss
            log_dict['w_loss_' + self.loss_dict['reg']['loss'].name] = sim_loss.item()

        # Registration
        sim_label_loss = 0.
        if self.loss_dict['reg_label']['weight'] > 0:
            sim_label_loss = self.loss_dict['reg_label']['loss'](data_dict['reg_mask'], data_dict['ref_mask'])
            log_dict['loss_' + self.loss_dict['reg_label']['loss'].name] = sim_label_loss.item()
            sim_label_loss = self.loss_dict['reg_label']['weight'] * sim_label_loss
            log_dict['w_loss_' + self.loss_dict['reg_label']['loss'].name] = sim_label_loss.item()

        # Registration
        sym_loss = 0.
        if self.loss_dict['reg_lr']['weight'] > 0:
            sym_loss = self.loss_dict['reg_lr']['loss'](data_dict['reg_image'])
            log_dict['loss_' + self.loss_dict['reg_lr']['loss'].name] = sym_loss.item()
            sym_loss = self.loss_dict['reg_lr']['weight'] * sym_loss
            log_dict['w_loss_' + self.loss_dict['reg_lr']['loss'].name] = sym_loss.item()

        reg_loss = 0.
        if self.loss_dict['regularizer']['weight'] > 0:
            params = model_dict['reg'].get_params_scaled()
            for p in params:
                reg_loss += self.loss_dict['regularizer']['loss'](p, torch.zeros_like(p))
            log_dict['loss_' + self.loss_dict['regularizer']['loss'].name] = reg_loss.item()
            reg_loss = self.loss_dict['regularizer']['weight'] * reg_loss
            log_dict['w_loss_' + self.loss_dict['regularizer']['loss'].name] = reg_loss.item()

        r_loss = reg_loss + sim_loss + sim_label_loss + sym_loss
        log_dict['loss'] = r_loss.item()

        return r_loss, log_dict


class JointInstanceAlign(JointInstanceReg):

    def forward(self, data_dict, model_dict):
        if 'image' in data_dict.keys():
            im, im_flip = model_dict['reg'](torch.cat((data_dict['image'], data_dict['mask']), axis=1))
            data_dict['reg_image'], data_dict['reg_image_flip'] = im[:, 0:1], im_flip[:, 0:1]
            data_dict['reg_mask'], data_dict['reg_mask_flip'] = im[:, 1:], im_flip[:, 1:]
        else:
            data_dict['reg_mask'], data_dict['reg_mask_flip'] = model_dict['reg'](data_dict['mask'])

        return data_dict

    def iterate(self, data_dict, model_dict, optimizer_dict, epoch, **kwargs):
        if isinstance(optimizer_dict['reg'], torch.optim.LBFGS):
            def closure():
                optimizer_dict['reg'].zero_grad()

                data_dict_closure = self.forward(data_dict, model_dict)

                loss, _ = self.compute_loss(data_dict_closure, model_dict)
                loss.backward()

                return loss

            optimizer_dict['reg'].step(closure=closure)

        else:
            data_dict = self.forward(data_dict, model_dict)
            loss, _ = self.compute_loss(data_dict, model_dict)
            loss.backward()

            optimizer_dict['reg'].step()

        with torch.no_grad():
            data_dict = self.forward(data_dict, model_dict)
            loss, log_dict = self.compute_loss(data_dict, model_dict)
            log_dict['loss'] = loss.item()

        return log_dict

    def compute_loss(self, data_dict, model_dict):
        log_dict = {}

        # Registration
        sym_loss = 0.
        if self.loss_dict['symmetry']['weight'] > 0:
            sym_loss = self.loss_dict['symmetry']['loss'](data_dict['reg_mask'], data_dict['reg_mask_flip'])
            log_dict['loss_' + self.loss_dict['symmetry']['loss'].name] = sym_loss.item()
            sym_loss = self.loss_dict['symmetry']['weight'] * sym_loss
            log_dict['w_loss_' + self.loss_dict['symmetry']['loss'].name] = sym_loss.item()

        sym_im_loss = 0.
        if 'image' in data_dict.keys() and self.loss_dict['symmetry_im']['weight'] > 0:
            sym_im_fn = self.loss_dict['symmetry_im']['loss']
            mask = data_dict['reg_mask'] * data_dict['reg_mask_flip']

            sym_im_loss =sym_im_fn(data_dict['reg_image'], data_dict['reg_image_flip'], mask=mask)
            log_dict['loss_' + sym_im_fn.name] = sym_im_loss.item()
            sym_im_loss = self.loss_dict['symmetry_im']['weight'] * sym_im_loss
            log_dict['w_loss_' + sym_im_fn.name] = sym_im_loss.item()

        reg_loss = 0.
        if self.loss_dict['regularizer']['weight'] > 0:
            params = model_dict['reg'].get_params()
            for p in params:
                reg_loss += self.loss_dict['regularizer']['loss'](p, torch.zeros_like(p))
            log_dict['loss_' + self.loss_dict['regularizer']['loss'].name] = reg_loss.item()
            reg_loss = self.loss_dict['regularizer']['weight'] * reg_loss
            log_dict['w_loss_' + self.loss_dict['regularizer']['loss'].name] = reg_loss.item()

        r_loss = sym_loss + sym_im_loss + reg_loss
        log_dict['loss'] = r_loss.item()
        return r_loss, log_dict


# Models
class BaseConvBlock3D(nn.Module):

    def initialize_padding(self, pad_type, padding):
        # initialize padding
        if pad_type == 'replicate':
            self.pad = nn.ReplicationPad3d(padding)
        elif pad_type == 'zeros':
            self.pad = nn.ConstantPad3d(padding, 0.0)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

    def initialize_normalization(self, norm_layer, norm_dim):
        if norm_layer == 'bn_partial':
            self.norm_layer = functools.partial(nn.BatchNorm3d(norm_dim), affine=True, track_running_stats=True)
        elif norm_layer == 'bn':
            self.norm_layer = nn.BatchNorm3d(norm_dim)
        elif norm_layer == 'in_partial':
            self.norm_layer = functools.partial(nn.InstanceNorm3d(norm_dim, affine=False, track_running_stats=False))
        elif norm_layer == 'in':
            self.norm_layer = nn.InstanceNorm3d(norm_dim, affine=False, track_running_stats=False)
        elif norm_layer == 'none' or norm_layer == 'sn':
            self.norm_layer = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm_layer)

    def initialize_activation(self, activation):
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

class ConvBlock3D(BaseConvBlock3D):
    '''
    2D ConvlutionBlock performing the following operations:
        Conv2D --> BatchNormalization -> Activation function
    :param Conv2D input parameters: see nn.Conv2D
    :param norm_layer (None, PyTorch normalization layer): it can be either None if no normalization is applied or a
    Pytorch normalization layer (nn.BatchNorm2d, nn.InstanceNorm2d)
    :param activation (None or PyTorch activation): it can be either None for linear activation or any other activation
    in PyTorch (nn.ReLU, nn.LeakyReLu(alpha), nn.Sigmoid, ...)
    '''

    def __init__(self, input_filters, output_filters, kernel_size=3, padding=0, stride=1, bias=True,
                 norm_layer='bn', activation='relu', pad_type='zeros'):

        super().__init__()
        # initialize padding
        self.initialize_padding(pad_type, padding)
        self.initialize_normalization(norm_layer, norm_dim=output_filters)
        self.initialize_activation(activation)
        self.conv_layer = nn.Conv3d(input_filters, output_filters, kernel_size=kernel_size,  stride=stride, bias=bias)


    def forward(self, inputs):
        outputs = self.conv_layer(self.pad(inputs))
        if self.norm_layer is not None:
            outputs = self.norm_layer(outputs)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

class Upsample(nn.Module):

    def __init__(self, offset=0, stride=2, output_padding=0,
                 output_shape=None, fill=True):
        """
        Only one of `output_padding` or `output_shape` should be provided.
        Parameters
        ----------
        offset : [sequence of] int, default=0
        stride : [sequence of] int, default=2
        output_padding : [sequence of] int, default=0
        output_shape : [sequence of] int, optional
        fill : bool, default=True
        """
        super().__init__()
        self.offset = offset
        self.stride = stride
        self.output_padding = output_padding
        self.output_shape = output_shape
        self.fill = fill

    def forward(self, x, output_padding=None, output_shape=None):
        """
        Parameters
        ----------
        x : (batch, channel, *in_spatial) tensor
        output_padding : [sequence of] int, default=self.output_padding
        output_shape : [sequence of] int, default=self.output_shape
        Returns
        -------
        x : (batch, channel, *out_spatial) tensor
        """
        dim = x.dim() - 2
        offset = make_list(self.offset, dim)
        stride = make_list(self.stride, dim)

        new_shape = self.shape(x, output_padding=output_padding,
                               output_shape=output_shape)
        y = x.new_zeros(new_shape)
        if self.fill:
            z = unfold(y, stride)
            x = unsqueeze(x, -1, dim)
            slicer = [slice(o, o+sz*st) for sz, st, o in
                      zip(x.shape[2:], stride, offset)]
            slicer = [slice(None)]*2 + slicer
            subz = z[tuple(slicer)]
            slicer = [slice(mx) for mx in subz.shape[2:]]
            slicer = [slice(None)]*2 + slicer
            subz.copy_(x[tuple(slicer)])
        else:
            slicer = [slice(o, None, s) for o, s in zip(offset, stride)]
            slicer = [slice(None)]*2 + slicer
            suby = y[tuple(slicer)]
            slicer = [slice(mx) for mx in suby.shape[2:]]
            slicer = [slice(None)]*2 + slicer
            suby.copy_(x[tuple(slicer)])

        return y

    def shape(self, x, output_padding=None, output_shape=None):
        """
        Parameters
        ----------
        x : (batch, channel, *in_spatial) tensor or sequence[int]
        output_padding : [sequence of] int, default=self.output_padding
        output_shape : [sequence of] int, default=self.output_shape
        Returns
        -------
        shape : tuple[int]
        """
        if torch.is_tensor(x):
            x = x.shape
        x = list(x)

        if output_padding is not None and output_shape is not None:
            raise ValueError('Only one of `output_padding` or `output_shape` '
                             'should be provided.')
        elif output_padding is None and output_shape is None:
            output_padding = self.output_padding
            output_shape = self.output_shape
        dim = len(x) - 2
        offset = make_list(self.offset, dim)
        stride = make_list(self.stride, dim)
        output_padding = make_list(output_padding, dim)

        if output_shape:
            output_shape = make_list(output_shape, dim)
        else:
            output_shape = [sz*st + o + p for sz, st, o, p in
                            zip(x[2:], stride, offset, output_padding)]
        return (*x[:2], *output_shape)

class BaseModel(nn.Module):
    def __init__(self, inshape):
        super().__init__()

        # ensure correct dimensionality
        self.ndims = len(inshape)
        assert self.ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % self.ndims

class SynthSeg_Encoder(BaseModel):

    def __init__(self,
                 inshape,
                 num_levels,
                 n_init_features,
                 feat_mult=1,
                 layer_feats=None,
                 activation='lrelu',
                 nb_conv_per_level=2,
                 batch_norm=True,
                 in_channels=1,
                 pool_size=2,
                 use_skip_connections=False,
                 model_name='enc'):

        super().__init__(inshape)

        if layer_feats is None:
            pass

        convL = globals()['ConvBlock%dD' % self.ndims]
        conv_kwargs = {'padding': 1, 'activation': activation}
        maxpool = getattr(nn, 'MaxPool%dd' % self.ndims)

        # configure encoder (down-sampling path)
        downarm = []
        prev_nfeats = in_channels
        for it_level in range(num_levels):
            nfeats = np.round(n_init_features * feat_mult ** it_level).astype(int)
            for it_conv in range(nb_conv_per_level):
                if layer_feats is not None:
                    nfeats = layer_feats[nb_conv_per_level * it_level + it_conv]

                name = '%s_conv_downarm_%d_%d' % (model_name, it_level, it_conv)
                if it_conv < (nb_conv_per_level - 1):
                    downarm.append([name, convL(prev_nfeats, nfeats, norm_layer='none', **conv_kwargs)])
                else:
                    downarm.append([name, convL(prev_nfeats, nfeats, norm_layer='none', **conv_kwargs)])
                    downarm.append(['%s_bn_down_%d' % (model_name, it_level),  nn.BatchNorm3d(nfeats, eps=1e-3, momentum=0.01)])

                prev_nfeats = nfeats

            if batch_norm:
                name = '%s_bn_down_%d' % (model_name, it_level)
                downarm.append([name, nn.BatchNorm3d(prev_nfeats)])

            if it_level < (num_levels - 1):
                name = '%s_maxpool_%d' % (model_name, it_level)
                downarm.append([name, maxpool(kernel_size=pool_size)])

        self.downarm = nn.ModuleDict(downarm)
        self.use_skip_connections = use_skip_connections
        self.nb_conv_per_level = nb_conv_per_level

    def forward(self, x):

        if self.use_skip_connections:
            x_skip = []
            x_enc = x
            lev = 0
            for l_name, layer in self.downarm.items():
                x_enc = layer(x_enc)
                if '_conv_downarm_' + str(lev) + '_' + str(self.nb_conv_per_level-1) in l_name:
                    x_skip.append(x_enc)
                    lev += 1

            x_skip.pop()
            x_skip.append(x_enc)
            return x_skip

        else:
            x_enc = x
            for l_name, layer in self.downarm.items():
                x_enc = layer(x_enc)
            return x_enc

class SynthSeg_Decoder(BaseModel):

    def __init__(self,
                 inshape,
                 num_levels,
                 n_init_features,
                 feat_mult=1,
                 layer_feats=None,
                 activation='elu',
                 nb_conv_per_level=2,
                 batch_norm=True,
                 in_channels=1,
                 pool_size=2,
                 use_skip_connections=False,
                 num_labels=2,
                 final_pred_activation='softmax',
                 deep_sup=False,
                 model_name='dec'):

        super().__init__(inshape)


        convL = globals()['ConvBlock%dD' % self.ndims]
        conv_kwargs = {'padding': 1, 'activation': activation}
        upsample = Upsample

        # configure encoder (down-sampling path)
        uparm = []
        deep_conv = []
        prev_nfeats = in_channels #np.round(n_init_features * feat_mult ** (num_levels - 1)).astype(int)
        for it_level in range(num_levels - 1):
            nfeats = np.round(n_init_features * feat_mult ** (num_levels - 2 - it_level)).astype(int)

            name = '%s_up_%d' % (model_name, num_levels + it_level)
            uparm.append([name, upsample(stride=pool_size)])

            if use_skip_connections:
                prev_nfeats += nfeats

            for it_conv in range(nb_conv_per_level):
                if layer_feats is not None:
                    nfeats = layer_feats[nb_conv_per_level * it_level + it_conv]
                name = '%s_conv_uparm_%d_%d' % (model_name, num_levels + it_level, it_conv)
                if it_conv < (nb_conv_per_level - 1):
                    uparm.append([name, convL(prev_nfeats, nfeats, norm_layer='none', **conv_kwargs)])
                else:
                    uparm.append([name, convL(prev_nfeats, nfeats, norm_layer='none', **conv_kwargs)])
                    uparm.append(['%s_bn_up_%d' % (model_name, it_level),  nn.BatchNorm3d(nfeats, eps=1e-3, momentum=0.01)])
                    if num_levels-2-it_level > 0:
                        deep_conv.append([str(2**(num_levels-2-it_level)), convL(nfeats, num_labels, norm_layer='none', kernel_size=1, activation='softmax')])
                prev_nfeats = nfeats

            if batch_norm:
                name = '%s_bn_up_%d' % (model_name, it_level)
                uparm.append([name, nn.BatchNorm3d(prev_nfeats)])

        # Compute likelyhood prediction (no activation yet)
        name = '%s_likelihood' % model_name
        uparm.append([name, convL(prev_nfeats, num_labels, kernel_size=1, activation='none')])

        # output prediction layer
        # we use a softmax to compute P(L_x|I) where x is each location
        if final_pred_activation == 'softmax':
            name = '%s_prediction' % model_name
            uparm.append([name, nn.Softmax(dim=1)])

        self.uparm = nn.ModuleDict(uparm)
        self.use_skip_connections = use_skip_connections
        self.deep_sup = deep_sup
        self.num_levels = num_levels
        self.final_pred_activation = final_pred_activation

        if deep_sup:
            self.deep_conv = nn.ModuleDict(deep_conv)


    def forward(self, x):
        if self.deep_sup:
            x_deep = {}

        x_dec = x.pop() if self.use_skip_connections else x
        for l_name, layer in self.uparm.items():
            if '_bn_up_' in l_name and self.deep_sup:
                it_level = self.num_levels - 2 - int(l_name.split('_')[-1])
                if str(2**it_level) in self.deep_conv.keys():
                    x_deep[str(2**it_level)] = self.deep_conv[str(2**it_level)](x_dec)
                    if self.final_pred_activation == 'softmax':
                        x_deep[str(2 ** it_level)] = F.softmax(x_deep[str(2**it_level)], dim=1)

            if '_up_' in l_name and not 'bn' in l_name:
                x_dec = layer(x_dec)
                if self.use_skip_connections:
                    x_dec = torch.cat([x.pop(), x_dec], dim=1)


            else:
                x_dec = layer(x_dec)


        if self.deep_sup:
            return x_dec, x_deep
        else:
            return x_dec

class SynthSeg_Unet(BaseModel):

    def __init__(self,
                 inshape,
                 num_levels,
                 n_init_features,
                 feat_mult=2,
                 layer_feats=None,
                 activation='elu',
                 nb_conv_per_level=2,
                 batch_norm=True,
                 in_channels=1,
                 pool_size=2,
                 use_skip_connections=False,
                 num_labels=2,
                 final_pred_activation='softmax',
                 model_name='unet',
                 deep_sup=False,
                 **kwargs
                 ):

        super().__init__(inshape)


        self.encoder = SynthSeg_Encoder(
            inshape=inshape,
            num_levels=num_levels,
            n_init_features=n_init_features,
            feat_mult=feat_mult,
            layer_feats=layer_feats,
            activation=activation,
            nb_conv_per_level=nb_conv_per_level,
            batch_norm=batch_norm,
            in_channels=in_channels,
            use_skip_connections=use_skip_connections,
            pool_size=pool_size,
            model_name=model_name
        )

        prev_nfeats = np.round(n_init_features * feat_mult ** (num_levels-1)).astype(int)

        self.decoder = SynthSeg_Decoder(
            inshape=inshape,
            num_levels=num_levels,
            n_init_features=n_init_features,
            feat_mult=feat_mult,
            layer_feats=list(reversed(layer_feats)) if layer_feats is not None else layer_feats,
            activation=activation,
            nb_conv_per_level=nb_conv_per_level,
            batch_norm=batch_norm,
            in_channels=prev_nfeats,
            pool_size=pool_size,
            use_skip_connections=use_skip_connections,
            num_labels=num_labels,
            final_pred_activation=final_pred_activation,
            model_name=model_name,
            deep_sup=deep_sup
        )



    def init_net(self, device, weightsfile=None, weightsfile_tf=None):
        if hasattr(self, 'clf_model'):
            device_last = 'cpu' if self.last_layer_to_cpu else device
            self.clf_model = init_net(self.clf_model, device=device_last)

        if hasattr(self, 'mheads'):
            device_last = 'cpu' if self.last_layer_to_cpu else device
            for roi, m in self.mheads.items():
                self.mheads[roi] = init_net(self.mheads[roi], device=device_last)

        if weightsfile is not None:
            self.encoder = self.encoder.to(device)
            self.decoder = self.decoder.to(device)
            checkpoint = torch.load(weightsfile, map_location=device)
            self.load_state_dict(checkpoint['state_dict'])

        elif weightsfile_tf is not None:
            self.encoder = self.encoder.to(device)
            self.decoder = self.decoder.to(device)
            self.load_tf_weights(path_to_h5=weightsfile)

        else:
            self.encoder = init_net(self.encoder, device=device)
            self.decoder = init_net(self.decoder, device=device)

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        # if hasattr(self, 'clf_model'):
        #     if self.last_layer_to_cpu:
        #         x = x.cpu()
        #     for l_name, layer in self.clf_model.items():
        #         x = layer(x)

        return x

    def load_tf_weights(self, path_to_h5=None):
        def make_list(x):
            if not isinstance(x, (list, tuple)):
                x = [x]
            return list(x)

        def backend(x):
            """Return the backend (dtype and device) of a tensor
            Parameters
            ----------
            x : tensor
            Returns
            -------
            dict with keys 'dtype' and 'device'
            """
            return dict(dtype=x.dtype, device=x.device)

        def movedim(input, source, destination):
            """Moves the position of one or more dimensions
            Other dimensions that are not explicitly moved remain in their
            original order and appear at the positions not specified in
            destination.
            Parameters
            ----------
            input : tensor
                Input tensor
            source : int or sequence[int]
                Initial positions of the dimensions
            destination : int or sequence[int]
                Output positions of the dimensions.
                If a single destination is provided:
                - if it is negative, the last source dimension is moved to
                  `destination` and all other source dimensions are moved to its left.
                - if it is positive, the first source dimension is moved to
                  `destination` and all other source dimensions are moved to its right.
            Returns
            -------
            output : tensor
                Tensor with moved dimensions.
            """
            input = torch.as_tensor(input)
            dim = input.dim()
            source = make_list(source)
            destination = make_list(destination)
            if len(destination) == 1:
                # we assume that the user wishes to keep moved dimensions
                # in the order they were provided
                destination = destination[0]
                if destination >= 0:
                    destination = list(range(destination, destination + len(source)))
                else:
                    destination = list(range(destination + 1 - len(source), destination + 1))
            if len(source) != len(destination):
                raise ValueError('Expected as many source as destination positions.')
            source = [dim + src if src < 0 else src for src in source]
            destination = [dim + dst if dst < 0 else dst for dst in destination]
            if len(set(source)) != len(source):
                raise ValueError(f'Expected source positions to be unique but got '
                                 f'{source}')
            if len(set(destination)) != len(destination):
                raise ValueError(f'Expected destination positions to be unique but got '
                                 f'{destination}')

            # compute permutation
            positions_in = list(range(dim))
            positions_out = [None] * dim
            for src, dst in zip(source, destination):
                positions_out[dst] = src
                positions_in[src] = None
            positions_in = filter(lambda x: x is not None, positions_in)
            for i, pos in enumerate(positions_out):
                if pos is None:
                    positions_out[i], *positions_in = positions_in

            return input.permute(*positions_out)

        def _set_weights(module, conv_keys, bn_keys, f, prefix='unet'):
            if isinstance(module, ConvBlock3D):
                if conv_keys:
                    key = conv_keys.pop(0)
                else:
                    # we might have reached the final "feat 2 class" conv
                    key = 'unet_likelihood'
                    #return # different number of classes
                    return

                kernel = torch.as_tensor(np.array(f[key][key]['kernel:0']), **backend(module.conv_layer.weight))
                kernel = movedim(kernel, [-1, -2], [0, 1])
                module.conv_layer.weight.copy_(kernel)
                bias = torch.as_tensor(f[key][key]['bias:0'], **backend(module.conv_layer.bias))
                module.conv_layer.bias.copy_(bias)

            elif isinstance(module, nn.BatchNorm3d): #prefix[-2:] == '_1':
                key = bn_keys.pop(0)
                beta = torch.as_tensor(f[key][key]['beta:0'], **backend(module.bias))
                module.bias.copy_(beta)
                gamma = torch.as_tensor(f[key][key]['gamma:0'], **backend(module.weight))
                module.weight.copy_(gamma)
                mean = torch.as_tensor(f[key][key]['moving_mean:0'], **backend(module.running_mean))
                module.running_mean.copy_(mean)
                var = torch.as_tensor(f[key][key]['moving_variance:0'],  **backend(module.running_var))
                module.running_var.copy_(var)
            else:
                for name, child in module.named_children():
                    _set_weights(child, conv_keys, bn_keys, f, f'{prefix}.{name}')


        try:
            import h5py
        except ImportError:
            h5py = None
        if not h5py:
            raise ImportError('h5py must be installed to load tf weights')
        with h5py.File(path_to_h5, 'r') as f, torch.no_grad():
            # The SynthSeg unet only has conv and batch norm weights
            down_conv_keys = sorted([key for key in f if 'conv_downarm' in key])
            up_conv_keys = sorted([key for key in f if 'conv_uparm' in key])
            down_bn_keys = sorted([key for key in f if 'bn_down' in key])
            up_bn_keys = sorted([key for key in f if 'bn_up' in key])
            conv_keys = [*down_conv_keys, *up_conv_keys]
            bn_keys = [*down_bn_keys, *up_bn_keys]

            _set_weights(self, conv_keys, bn_keys, f)

class InstanceModelClassic(nn.Module):

    def __init__(self, image_shape, ref_v2r, flo_v2r=None,batchsize=1, device='cpu', **kwargs):
        super().__init__()

        self.device = device
        self.batchsize = batchsize
        self.image_shape = image_shape
        if len(image_shape) > 3:
            self.image_shape = image_shape[2:]

        self.ndims = 3
        self.ref_v2r = torch.from_numpy(ref_v2r).to(device)
        self.flo_v2r = torch.from_numpy(flo_v2r).to(device) if flo_v2r is not None else self.ref_v2r

        vectors = [torch.arange(0, s) for s in self.image_shape]
        grids = torch.meshgrid(vectors, indexing='ij')
        self.grid = torch.stack(grids).to(device)  # y, x, z

    def _compute_ras_matrix(self, *args, **kwargs):
        raise NotImplementedError

    def set_params(self, *args, **kwargs):
        raise NotImplementedError

    def get_params(self):
        raise NotImplementedError

    def get_params_scaled(self):
        return self.get_params()

    def _compute_rotation(self, rotation):
        shape = rotation[..., 0].shape + (1,)

        Rx_row0 = torch.unsqueeze(torch.tile(torch.unsqueeze(torch.from_numpy(np.array([1., 0., 0.])), 0), shape),
                                  axis=1).to(self.device)
        Rx_row1 = torch.stack([torch.zeros(shape).to(self.device), torch.unsqueeze(torch.cos(rotation[..., 0]), -1),
                               torch.unsqueeze(-torch.sin(rotation[..., 0]), -1)], axis=-1)
        Rx_row2 = torch.stack([torch.zeros(shape).to(self.device), torch.unsqueeze(torch.sin(rotation[..., 0]), -1),
                               torch.unsqueeze(torch.cos(rotation[..., 0]), -1)], axis=-1)
        Rx = torch.cat([Rx_row0, Rx_row1, Rx_row2], axis=1)

        Ry_row0 = torch.stack([torch.unsqueeze(torch.cos(rotation[..., 1]), -1), torch.zeros(shape).to(self.device),
                               torch.unsqueeze(torch.sin(rotation[..., 1]), -1)], axis=-1)
        Ry_row1 = torch.unsqueeze(torch.tile(torch.unsqueeze(torch.from_numpy(np.array([0., 1., 0.])), 0), shape),
                                  axis=1).to(self.device)
        Ry_row2 = torch.stack([torch.unsqueeze(-torch.sin(rotation[..., 1]), -1), torch.zeros(shape).to(self.device),
                               torch.unsqueeze(torch.cos(rotation[..., 1]), -1)], axis=-1)
        Ry = torch.cat([Ry_row0, Ry_row1, Ry_row2], axis=1)

        Rz_row0 = torch.stack(
            [torch.unsqueeze(torch.cos(rotation[..., 2]), -1), torch.unsqueeze(-torch.sin(rotation[..., 2]), -1),
             torch.zeros(shape).to(self.device)], axis=-1)
        Rz_row1 = torch.stack(
            [torch.unsqueeze(torch.sin(rotation[..., 2]), -1), torch.unsqueeze(torch.cos(rotation[..., 2]), -1),
             torch.zeros(shape).to(self.device)], axis=-1)
        Rz_row2 = torch.unsqueeze(torch.tile(torch.unsqueeze(torch.from_numpy(np.array([0., 0., 1.])), 0), shape),
                                  axis=1).to(self.device)
        Rz = torch.cat([Rz_row0, Rz_row1, Rz_row2], axis=1)

        T_rot = torch.matmul(torch.matmul(Rx, Ry), Rz)

        return T_rot

    def _compute_matrix(self, *args, **kwargs):
        T_rig = self._compute_ras_matrix(**kwargs)
        T_rig_list = torch.unbind(T_rig, dim=0)
        T_rig_ras_list = [torch.linalg.inv(self.flo_v2r) @ T @ self.ref_v2r for T in T_rig_list]
        T = torch.stack(T_rig_ras_list, 0)

        return T.to(self.device)

    def get_ras_matrix(self):
        return self._compute_ras_matrix()

    def get_matrix(self):
        return self._compute_matrix()

    def forward(self, image_targ, **kwargs):
        T = self._compute_matrix()
        im = torch.permute(image_targ[0], (1, 2, 3, 0))

        T = T[0]
        di = T[0, 0] * self.grid[0] + T[0, 1] * self.grid[1] + T[0, 2] * self.grid[2] + T[0, 3]
        dj = T[1, 0] * self.grid[0] + T[1, 1] * self.grid[1] + T[1, 2] * self.grid[2] + T[1, 3]
        dk = T[2, 0] * self.grid[0] + T[2, 1] * self.grid[1] + T[2, 2] * self.grid[2] + T[2, 3]

        image_reg = fast_3D_interp_field_torch(im, di, dj, dk, mode='linear')

        return torch.unsqueeze(torch.permute(image_reg, (3, 0, 1, 2)), 0)

class InstanceAlignModelClassic(InstanceModelClassic):

    def __init__(self, image_shape, v2r, batchsize=1, device='cpu', cog=None,
                 tx_init=None, angle_init=None, tx_factor=1, angle_factor=1,
                 align_lr=True, align_ap=False, align_is=False):
        super().__init__(image_shape, ref_v2r=v2r, batchsize=batchsize, device=device)

        self.tx_factor = tx_factor
        self.angle_factor = angle_factor
        self.cog = cog
        self.align_lr = align_lr
        self.align_ap = align_ap
        self.align_is = align_is

        # Parameters
        if angle_init is not None:
            if torch.is_tensor(angle_init):
                self.angle = torch.nn.Parameter(angle_init)
            else:
                self.angle = torch.nn.Parameter(torch.from_numpy(angle_init))

        else:
            self.angle = torch.nn.Parameter(torch.zeros(self.batchsize, 3))

        if tx_init is not None:
            if torch.is_tensor(tx_init):
                self.translation = torch.nn.Parameter(tx_init)
            else:
                self.translation = torch.nn.Parameter(torch.from_numpy(tx_init))
        else:
            self.translation = torch.nn.Parameter(torch.zeros(self.batchsize, 3))

        self.angle.requires_grad = True
        self.translation.requires_grad = True

    def set_params(self, params):
        self.angle = torch.nn.Parameter(params[0]).to(self.device)
        self.translation = torch.nn.Parameter(params[1]).to(self.device)

    def get_params(self):
        return self.angle , self.translation

    def get_params_scaled(self):
        return self.angle * self.angle_factor, self.translation + self.tx_factor

    def _compute_matrix(self, flip_lr=False):
        T_rig = self._compute_ras_matrix(flip_lr=flip_lr)
        T_rig_list = torch.unbind(T_rig, dim=0)
        T_rig_ras_list = [torch.linalg.inv(self.flo_v2r) @ T @ self.ref_v2r for T in T_rig_list]
        T = torch.stack(T_rig_ras_list, 0)

        return T.to(self.device)

    def _compute_ras_matrix(self, flip_lr=False):

        angle, tx = self.get_params()

        T_center = torch.zeros((self.batchsize, 4, 4)).to(self.device)
        T_center[:, 0, 0] = 1
        T_center[:, 1, 1] = 1
        T_center[:, 2, 2] = 1
        T_center[:, 3, 3] = 1
        if self.cog is not None:
            T_center[:, :3, 3] = torch.from_numpy(-self.cog)

        T_center_inv = torch.zeros((self.batchsize, 4, 4)).to(self.device)
        T_center_inv[:, 0, 0] = 1
        T_center_inv[:, 1, 1] = 1
        T_center_inv[:, 2, 2] = 1
        T_center_inv[:, 3, 3] = 1
        if self.cog is not None:
            T_center_inv[:, :3, 3] = torch.from_numpy(self.cog)

        T_trans = torch.zeros((self.batchsize, 4, 4)).to(self.device)
        T_trans[:, :3, 3] = tx
        T_trans[:, 0, 0] = 1
        T_trans[:, 1, 1] = 1
        T_trans[:, 2, 2] = 1
        T_trans[:, 3, 3] = 1

        T_rot = torch.zeros((self.batchsize, 4, 4)).to(self.device)
        T_rot[:, :3, :3] = self._compute_rotation(angle)
        T_rot[:, 3, 3] = 1

        T_flip_lr = torch.zeros((self.batchsize, 4, 4)).to(self.device)
        T_flip_lr[:, 0, 0] = -1 if self.align_lr else 1
        T_flip_lr[:, 1, 1] = -1 if self.align_ap else 1
        T_flip_lr[:, 2, 2] = -1 if self.align_is else 1
        T_flip_lr[:, 3, 3] = 1

        if flip_lr:
            T_rig = T_center_inv @ T_trans @ T_rot @ T_flip_lr @ T_center
        else:
            T_rig = T_center_inv @ T_trans @ T_rot @ T_center

        return T_rig

    def forward(self, image_targ, **kwargs):
        T = self._compute_matrix(flip_lr=False)
        T_flip = self._compute_matrix(flip_lr=True)
        im = torch.permute(image_targ[0], (1, 2, 3, 0))

        T = T[0]
        di = T[0, 0] * self.grid[0] + T[0, 1] * self.grid[1] + T[0, 2] * self.grid[2] + T[0, 3]
        dj = T[1, 0] * self.grid[0] + T[1, 1] * self.grid[1] + T[1, 2] * self.grid[2] + T[1, 3]
        dk = T[2, 0] * self.grid[0] + T[2, 1] * self.grid[1] + T[2, 2] * self.grid[2] + T[2, 3]

        image_reg = fast_3D_interp_field_torch(im, di, dj, dk, mode='linear')

        T = T_flip[0]
        di = T[0, 0] * self.grid[0] + T[0, 1] * self.grid[1] + T[0, 2] * self.grid[2] + T[0, 3]
        dj = T[1, 0] * self.grid[0] + T[1, 1] * self.grid[1] + T[1, 2] * self.grid[2] + T[1, 3]
        dk = T[2, 0] * self.grid[0] + T[2, 1] * self.grid[1] + T[2, 2] * self.grid[2] + T[2, 3]

        image_flip_reg = fast_3D_interp_field_torch(im, di, dj, dk, mode='linear')

        return torch.unsqueeze(torch.permute(image_reg, (3, 0, 1, 2)), 0), torch.unsqueeze(torch.permute(image_flip_reg, (3, 0, 1, 2)), 0)


# Callbacks

class Callback(object):

    def on_train_init(self, model, **kwargs):
        pass

    def on_train_fi(self, model, **kwargs):
        pass

    def on_epoch_init(self, model, epoch, **kwargs):
        pass

    def on_epoch_fi(self, logs_dict, model, epoch, **kwargs):
        pass

    def on_step_init(self, logs_dict, model, epoch, **kwargs):
        pass

    def on_step_fi(self, logs_dict, model, epoch,**kwargs):
        pass


class History(Callback):

    def __init__(self, keys=None):
        self.logs = {}

        if keys is None:
            self.keys = []
        else:
            self.keys = keys

    def on_train_init(self, model, **kwargs):
        self.logs['Train'] = {}
        self.logs['Validation'] = {}


    def on_epoch_init(self, model, epoch, **kwargs):
        self.logs['Train'][epoch] = {}
        self.logs['Validation'][epoch] = {}


        for k in self.keys:
            self.logs['Train'][epoch][k] = []


    def on_step_fi(self, logs_dict, model, epoch, **kwargs):
        for k,v in logs_dict.items():
            self.logs['Train'][epoch][k].append(v)


    def on_epoch_fi(self, logs_dict, model, epoch, **kwargs):
        for k,v in logs_dict.items():
            self.logs['Validation'][epoch][k] = v


class ModelCheckpoint(Callback):

    def __init__(self, dirpath, save_model_frequency):

        self.dirpath = dirpath
        self.save_model_frequency = save_model_frequency

    def on_epoch_fi(self, logs_dict, model, epoch, **kwargs):

        optimizer = kwargs['optimizer']
        checkpoint = {
            'epoch': epoch + 1,
        }
        if isinstance(model, dict):
            for model_name, model_instance in model.items():
                try:
                    checkpoint['state_dict_' + model_name] = model_instance.state_dict()
                except:
                    continue

        else:
            checkpoint['state_dict'] = model.state_dict()

        if isinstance(optimizer, dict):
            for optimizer_name, optimizer_instance in optimizer.items():
                checkpoint['optimizer_' + optimizer_name] = optimizer_instance.state_dict()
        else:
            checkpoint['optimizer'] = optimizer.state_dict()

        filepath = os.path.join(self.dirpath, 'model_checkpoint.LAST.pth')
        torch.save(checkpoint, filepath)
        if np.mod(epoch, self.save_model_frequency) == 0:
            filepath = os.path.join(self.dirpath, 'model_checkpoint.' + str(epoch) + '.pth')
            torch.save(checkpoint, filepath)


    def on_train_fi(self, model, **kwargs):
        checkpoint = {}
        if isinstance(model, dict):
            for model_name, model_instance in model.items():
                checkpoint['state_dict_' + model_name] = model_instance.state_dict()
        else:
            checkpoint['state_dict'] = model.state_dict()

        filepath = os.path.join(self.dirpath, 'model_checkpoint.FI.pth')
        torch.save(checkpoint, filepath)


class PrinterCallback(Callback):

    def __init__(self, keys=None):
        self.keys = keys
        self.logs = {}

    def on_train_init(self, model, **kwargs):
        print('\n')
        print('    ####################')
        print('    # Training started #')
        print('    ####################')
        print('\n')

    def on_epoch_init(self, model, epoch, **kwargs):
        print('    * Epoch: ' + str(epoch))

    def on_step_fi(self, logs_dict, model, epoch, **kwargs):
        to_print = '      o Iteration: (' + str(kwargs['iteration']) + '/' + str(kwargs['N']) + '). ' + \
                   ', '.join([k + ': ' + str(round(v, 3)) for k, v in logs_dict.items()])
        print(to_print)
    def on_epoch_fi(self, logs_dict, model, epoch, **kwargs):
        to_print = '      Epoch summary: ' + ','.join([k + ': ' + str(round(v, 3)) for k, v in logs_dict.items()])
        print(to_print)
        print('\n')


    def on_train_fi(self, model, **kwargs):
        # print('#####################')
        print('\n')
        print('    #####################')
        print('    # Training finished #')
        print('    #####################')
        print('\n')
        # print('#####################')


class ToCSVCallback(Callback):

    def __init__(self, filepath, keys, attach=False):
        mode = 'a' if attach else 'w'
        fieldnames = ['Phase','epoch','iteration'] + keys
        write_header = True
        if os.path.exists(filepath) and attach:
            write_header = False
        csvfile = open(filepath, mode)
        self.csvwriter = csv.DictWriter(csvfile, fieldnames)

        if write_header:
            self.csvwriter.writeheader()

    def on_step_fi(self, logs_dict, model, epoch, **kwargs):
        write_dict = {**{'Phase': 'Train', 'epoch':epoch, 'iteration': kwargs['iteration']}, **logs_dict}
        self.csvwriter.writerow(write_dict)

    def on_epoch_fi(self, logs_dict, model, epoch, **kwargs):
        write_dict = {**{'Phase': 'Validation', 'epoch': epoch}, **logs_dict}
        self.csvwriter.writerow(write_dict)


