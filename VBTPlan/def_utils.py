import pdb

import torch
import numpy as np
import nibabel as nib


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

def sample_svf_transform(nonlin_std, nonlin_scale, image_shape, batchsize):
    # sample small field from normal distribution of specified std dev
    small_shape = [batchsize, len(image_shape)] + [int(s * nonlin_scale) for s in image_shape]
    elastic_trans = np.random.normal(scale=nonlin_std, size=small_shape)

    return elastic_trans

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
