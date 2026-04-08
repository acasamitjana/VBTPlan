# imports
from argparse import ArgumentParser

import numpy as np
import torch
import nibabel as nib
from monai.networks.nets import UNETR

# project imports
from VBTPlan.inference_utils import *
from VBTPlan.dcm_utils import *


print('\n\n\n')
print('# ------------------------ #')
print('# VBT Segmentation: Unet-R #')
print('# ------------------------ #')
print('\n')

parser = ArgumentParser(description="VBTPlan: SOTA methods", epilog='\n')
parser.add_argument("--i", help="Input filepath")
parser.add_argument("--o", default=None, help="Output filepath")
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--debug', action='store_true')

args = parser.parse_args()
use_gpu = torch.cuda.is_available() and not args.cpu
device = torch.device("cuda:1" if use_gpu else "cpu")

BUILD_RTSTRUCT = True

####################
# Data preparation #
####################
labels_lut = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
num_labels = len(np.unique(list(labels_lut.values())))

input_vagina_tf = [
    GaussianBlur(keys=['image'], sigma=np.array([1] * 3), normalize_area=True),
    MinMaxNormalization(keys=['image'], min_val=-800, max_val=500),
    Pad(keys=['image'], multp=2 ** 5, v2r_key='image_v2r'),
    AddBatchAxis(keys=['image']),
    ToTensor(keys=['image']),
]

output_full_tf = [
    Argmax(keys=['output_seg']),
    OneHot(keys=['output_seg'], lut=labels_lut),
    ToNumpy(keys=['output_seg']),
]

#########
# Model #
#########
print('Loading the model')
model = UNETR(
    in_channels=1,
    out_channels=num_labels,
    img_size=(256, 256, 64),
).to(device)

weightsfile = os.path.join(os.environ['PYTHONPATH'], 'models', 'model_unetr.pth')
checkpoint = torch.load(weightsfile, map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint['state_dict_seg'], strict=False)
model_dict = {'full': model}


###########
# Testing #
###########
print('Evaluating: ' + args.i)
if args.i.endswith('.nii') or args.i.endswith('.nii.gz'):
    BUILD_RTSTRUCT = False

# From DICOM to NIFTI
if args.o is not None:
    seg_dir = args.o

elif args.i.endswith('.nii') or args.i.endswith('.nii.gz'):
    seg_dir = os.path.dirname(args.i)

else:
    root_dir = os.path.dirname(args.i)
    sid = os.path.basename(args.i)
    seg_dir = os.path.join(root_dir, sid, 'seg')

if not os.path.exists(seg_dir):
    os.makedirs(seg_dir, exist_ok=True)

if os.path.exists(os.path.join(seg_dir, 'final_im.nii.gz')) and os.path.exists(os.path.join(seg_dir, 'RT.automatic.dcm')):
    print('[FILE FOUND] Subject is already processed.')
    exit()

if not os.path.exists(os.path.join(seg_dir, 'final_im.nii.gz')):
    print(' 1.- Check file format and transform DICOM to NIFTI (if needed).')
    if args.i.endswith('.nii') or args.i.endswith('.nii.gz'):
        import shutil

        shutil.copy(args.i, os.path.join(seg_dir, 'orig_im.nii.gz'))
        im_proxy = nib.load(args.i)
        image = np.array(im_proxy.dataobj)
        orig_vox2ras = im_proxy.affine

    else:
        lps2ras = np.eye(4)  # the library works with the LPS convention and NIFTI works with RAS.
        lps2ras[0, 0] = -1
        lps2ras[1, 1] = -1

        dicom_series_data = load_sorted_image_series(args.i)
        vox2lps = get_pixel_to_patient_transformation_matrix(dicom_series_data)
        orig_vox2ras = lps2ras @ vox2lps

        image_list = [(sd.pixel_array.T + sd.RescaleIntercept) * sd.RescaleSlope for sd in dicom_series_data if 'CT' in sd.filename]
        image = np.stack(image_list, -1)
        im_proxy = nib.Nifti1Image(image, orig_vox2ras)

    data_dict = {'image': image.astype('float32'), 'image_v2r': orig_vox2ras, 'orig_image': image, 'orig_v2r': orig_vox2ras}

    if args.debug:
        img = nib.Nifti1Image(image, orig_vox2ras)
        nib.save(img, 'orig_im.nii.gz')


    print(' 2.- Initial segmentation of the vagina.')

    testing_session = SegmentationInference(in_tf=input_vagina_tf, out_tf=output_full_tf, device=device)
    tensor_dict = testing_session.step_full(data_dict, {'seg': model_dict['full']}, sliding_window=True)

    img = nib.Nifti1Image(np.squeeze(tensor_dict['image'].cpu().numpy()), tensor_dict['image_v2r'])
    nib.save(img, os.path.join(seg_dir, 'orig_im.nii.gz'))

    seg_OAR_CTV = np.argmax(tensor_dict['output_seg'], 0).astype('int16')
    proxy_OAR_CTV = nib.Nifti1Image(seg_OAR_CTV, orig_vox2ras)
    nib.save(proxy_OAR_CTV, os.path.join(seg_dir, 'final_im.nii.gz'))

else:
    print('[FILE FOUND] Existing segmentation is used. Skipping 1-4 steps')
    proxy_OAR_CTV = nib.load(os.path.join(seg_dir, 'final_im.nii.gz'))
    seg_OAR_CTV = np.array(proxy_OAR_CTV.dataobj)

if BUILD_RTSTRUCT:
    rtstruct = RTStructBuilder.create_new(dicom_series_path=args.i)
    mask_whole = np.stack([seg_OAR_CTV[..., it_d].T for it_d in range(seg_OAR_CTV.shape[-1])], -1)

    mask = np.zeros_like(mask_whole, dtype='bool')
    mask[mask_whole == 1] = 1
    rtstruct.add_roi(
        mask=mask,
        color=[0, 255, 255],
        name='intesti',
        approximate_contours=False
    )

    mask = np.zeros_like(mask_whole, dtype='bool')
    mask[mask_whole == 2] = 1
    rtstruct.add_roi(
        mask=mask,
        color=[0, 255, 0],
        name='recte'
    )

    mask = np.zeros_like(mask_whole, dtype='bool')
    mask[mask_whole == 3] = 1
    rtstruct.add_roi(
        mask=mask,
        color=[0, 0, 255],
        name='bufeta'
    )

    mask = np.zeros_like(mask_whole, dtype='bool')
    mask[mask_whole == 4] = 1
    rtstruct.add_roi(
        mask=mask,
        color=[255, 0, 255],
        name='sigma'
    )

    print('vagina')
    mask = np.zeros_like(mask_whole, dtype='bool')
    mask[mask_whole == 5] = 1
    rtstruct.add_roi(
        mask=mask,
        color=[255, 255, 255],
        name='vagina'
    )

    rtstruct.save(os.path.join(seg_dir, 'RT.automatic.dcm'))


print('\n')
print('# ---- #')
print('# DONE #')
print('# ---- #')
print('\n\n')



