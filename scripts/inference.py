import os
import pdb
from os.path import join, exists, basename

from argparse import ArgumentParser
import numpy as np
import nibabel as nib
import torch

from VBTPlan.inference_utils import *
from VBTPlan.dcm_utils import *
from VBTPlan.fn_utils import one_hot_encoding
from VBTPlan.def_utils import vol_resample_fast

parser = ArgumentParser(description="VBTPlan: SOTA methods", epilog='\n')
parser.add_argument("--i", help="Input filepath")
parser.add_argument("--o", default=None, help="Output filepath")
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--debug', action='store_true')

args = parser.parse_args()
use_gpu = torch.cuda.is_available() and not args.cpu
device = torch.device("cuda:0" if use_gpu else "cpu")

default_nnunet_args = {
    '1': ['-d', '001', '-c', '3d_fullres', '-f', 'all', '-p', 'nnUNetResEncUNetMPlans'],
    '2': ['-d', '002', '-c', '3d_fullres', '-f', 'all', '-p', 'nnUNetResEncUNetMPlans']
}
subdirs = ['in', 'step_1', 'step_2', 'step_3', 'final', 'dicom', 'utils']

print('Evaluating: ' + args.i)
case_name = basename(args.i)
seg_fname = case_name + '.nii.gz'
if args.i.endswith('.nii') or args.i.endswith('.nii.gz'):
    image_fname = case_name.replace('.nii', '_0000.nii')
    BUILD_RTSTRUCT = False

else:
    image_fname = case_name + '_0000.nii.gz'
    BUILD_RTSTRUCT = True


# From DICOM to NIFTI
if args.o is not None:
    seg_dir = args.o

elif args.i.endswith('.nii') or args.i.endswith('.nii.gz'):
    seg_dir = os.path.dirname(args.i)

else:
    root_dir = os.path.dirname(args.i)
    sid = os.path.basename(args.i)
    seg_dir = os.path.join(root_dir, sid, 'seg')

create_dir(seg_dir, subdirs=subdirs)

print(' 1.- Check file format and transform DICOM to NIFTI (if needed).')
if args.i.endswith('.nii') or args.i.endswith('.nii.gz'):

    subprocess.call(['ln', '-s', args.i, os.path.join(seg_dir, 'in', image_fname)])

    im_proxy = nib.load(args.i)
    image = np.array(im_proxy.dataobj)
    orig_vox2ras = im_proxy.affine

else:
    lps2ras = np.eye(4)  # the library works with the LPS convetion and NIFTI works with RAS.
    lps2ras[0, 0] = -1
    lps2ras[1, 1] = -1

    dicom_series_data = load_sorted_image_series(args.i)
    vox2lps = get_pixel_to_patient_transformation_matrix(dicom_series_data)
    orig_vox2ras = lps2ras @ vox2lps

    image_list = [(sd.pixel_array.T + sd.RescaleIntercept) * sd.RescaleSlope for sd in dicom_series_data if 'CT' in sd.filename]
    image = np.stack(image_list, -1)
    im_proxy = nib.Nifti1Image(image, orig_vox2ras)
    nib.save(im_proxy, os.path.join(seg_dir, 'in', image_fname))

# ORIG_DIR = '/media/acasamitjana/Data/Data/RadioTH/VBT/PatientsUnsegmented'
# tmp_dir = '/tmp/RadioTH'
# in_dir = '/media/acasamitjana/HDD_Data_2/Results/RadioTH/nnunet/data/Dataset002_VBTaligned/imagesTs'
# out_dir = '/media/acasamitjana/HDD_Data_2/Results/RadioTH/nnunet/output/Dataset002_labelsTs'
#
# seg_file = sbj_file.replace('_0000', '')
# sid = sbj_file.split('_')[1]

data_path = join(os.environ['PYTHONPATH'], 'models', 'nnUnet', 'data')
preprocessed_path = join(os.environ['PYTHONPATH'], 'models', 'nnUnet', 'preprocessed')
results_path = join(os.environ['PYTHONPATH'], 'models', 'nnUnet', 'results')
os.environ['nnUNet_raw'] = data_path
os.environ['nnUNet_preprocessed'] = preprocessed_path
os.environ['nnUNet_results'] = results_path
run_flag = False
if not exists(join(seg_dir, 'step_1', seg_fname)):
    print(' 2.- Run first step: initialize the vagina model.\n')
    run_flag = True
    pdb.set_trace()
    subprocess.call(['nnUNetv2_predict',
                     '-i', join(seg_dir, 'in'),
                     '-o', join(seg_dir, 'step_1')] + default_nnunet_args['1'])
else:
    print(' 2.- Vagina model available. Reading from disk.\n')

if (not exists(join(seg_dir, 'step_2', image_fname)) or
        not exists(join(seg_dir, 'utils', 'affine_LR.npy')) or run_flag):
    print(' 3.- Create bounding box: align LR and define the bounding box.\n')
    run_flag = True

    im_proxy = nib.load(join(seg_dir, 'in', image_fname))
    image = np.array(im_proxy.dataobj)
    orig_vox2ras = im_proxy.affine

    mask_proxy = nib.load(join(seg_dir, 'step_1', seg_fname))
    mask_vagina = np.array(mask_proxy.dataobj) == 5

    # compute alignment with the applicator and not only the vagina segmentation to avoid segmentation errors.
    # the vagina seg is used to bbox the applicator.
    x, y, z = np.where(mask_vagina)
    x_min, x_max = np.min(x) - 10, np.max(x) + 10
    y_min, y_max = np.min(y) - 10, np.max(y) + 10
    T_crop = np.eye(4)
    T_crop[0, -1] = x_min
    T_crop[1, -1] = y_min

    mask_cropped = (np.array(im_proxy.dataobj) > 100)[x_min: x_max, y_min: y_max].astype('float')
    all_blobs, num_blobs = measure.label(mask_cropped, connectivity=2, return_num=True)
    all_blobs_count = np.bincount(all_blobs[mask_cropped>0])
    mask_cropped = all_blobs == np.argmax(all_blobs_count)

    mask_proxy = nib.Nifti1Image(mask_cropped.astype('float32'), mask_proxy.affine @ T_crop)
    mask_proxy, mask_cog = create_COG_space(mask_proxy)

    rig_matrix = align_LR(mask_proxy, w_reg=0.01, device='cpu', verbose=True)
    affine_LR = np.linalg.inv(rig_matrix) @ mask_cog

    # Resample vagina mask
    aligned_mask_proxy = nib.Nifti1Image(mask_cropped.astype('float32'), affine_LR @ im_proxy.affine @ T_crop)
    aligned_vagina_proxy = nib.Nifti1Image(mask_vagina.astype('float'), affine_LR @ orig_vox2ras)


    reg_array, aligned_v2r = create_template_space([aligned_vagina_proxy, aligned_mask_proxy])
    aligned_dim = np.sqrt(np.sum(aligned_v2r * aligned_v2r, axis=0))[:-1]
    mask_vagina_array = reg_array[0]
    mask_reg_array = reg_array[1]
    mask_reg_array = mask_reg_array > 0

    # get length of the first cylinder
    cylinder_sl = np.sum(mask_reg_array.reshape((-1, mask_reg_array.shape[2])), axis=0)
    percentile = np.percentile(cylinder_sl[np.where(cylinder_sl>0)[0]], 2)
    max_vagina_k = np.max(np.where(cylinder_sl > percentile)[0])
    min_vagina_k = int(max_vagina_k - 25/aligned_dim[2])

    min_image_RAS = aligned_v2r @ np.array([0, 0, min_vagina_k, 1]) - np.array([0, 0, 20, 0])
    max_image_RAS = aligned_v2r @ np.array([0, 0, max_vagina_k, 1]) + np.array([0, 0, 20, 0])
    min_image_k = int((np.linalg.inv(aligned_v2r) @ min_image_RAS)[2])
    max_image_k = int((np.linalg.inv(aligned_v2r) @ max_image_RAS)[2])
    tx_crop = np.eye(4)
    tx_crop[2, 3] = min_image_k

    bbox_shape = mask_vagina_array.shape[:2] + (max_image_k - min_image_k + 1,)
    bbox_v2r = aligned_v2r @ tx_crop
    bbox_proxy = nib.Nifti1Image(np.zeros(bbox_shape), bbox_v2r)

    # Resample image and segmentations
    image_bbox_proxy = nib.Nifti1Image(image, affine_LR @ orig_vox2ras)
    image_bbox_arr = vol_resample_fast(bbox_proxy, image_bbox_proxy, constant=-1000, return_np=True)

    img = nib.Nifti1Image(image_bbox_arr, bbox_v2r)
    nib.save(img, join(seg_dir, 'aligned_im.nii.gz'))


    img = nib.Nifti1Image(image_bbox_arr, bbox_v2r)
    nib.save(img, join(seg_dir, 'step_2', image_fname))
    np.save(join(seg_dir, 'utils', 'affine_LR.npy'), affine_LR)

else:
    print(' 3.- Bounding box already computed. Reading from disk.\n')
    affine_LR = np.load(join(seg_dir, 'utils', 'affine_LR.npy'))
    proxy_im_aligned = nib.load(join(seg_dir, 'step_2', image_fname))
    bbox_v2r = proxy_im_aligned.affine

if not exists(join(seg_dir, 'step_2', seg_fname)) or run_flag:
    print(' 4.- Running CTV and OAR segmentation within the bounding box.\n')
    run_flag = True
    subprocess.call(['nnUNetv2_predict',
                     '-i', join(seg_dir, 'step_2'),
                     '-o',  join(seg_dir, 'step_3')] + default_nnunet_args['2'])
else:
    print(' 4.- CTV and OAR segmentation already available. Reading from disk.\n')

if not exists(join(seg_dir, 'final', seg_fname)) or run_flag:
    print(' 5.- Registering segmentation back to original space.\n')
    run_flag = True
    proxyseg = nib.load(join(seg_dir, 'step_3', seg_fname))
    arrseg = np.array(proxyseg.dataobj)
    onehotseg = np.transpose(one_hot_encoding(arrseg, num_classes=6).astype('float'), axes=(1, 2, 3, 0))

    # Register to original image
    v2r = np.linalg.inv(affine_LR) @ bbox_v2r
    pred_proxy = nib.Nifti1Image(onehotseg, v2r)

    orig_proxy = nib.load(join(seg_dir, 'in', image_fname))
    proxy_onehot_pred = vol_resample_fast(orig_proxy, pred_proxy)
    onehot_pred = np.array(proxy_onehot_pred.dataobj)
    arr_pred = np.argmax(onehot_pred, axis=-1).astype('uint8')

    proxy_pred = nib.Nifti1Image(arr_pred, proxy_onehot_pred.affine)
    nib.save(proxy_pred, join(seg_dir, 'final', seg_fname))
else:
    print(' 5.- Registration is already computed. Reading from disk.\n')

if BUILD_RTSTRUCT and (not exists(join(seg_dir, 'dicom', seg_fname.replace('nii.gz', 'dcm'))) or run_flag):

    proxyseg = nib.load(join(seg_dir, 'final', seg_fname))
    arrseg = np.array(proxyseg.dataobj)

    rtstruct = RTStructBuilder.create_new(dicom_series_path=args.i)
    mask_whole = np.stack([arrseg[..., it_d].T for it_d in range(arrseg.shape[-1])], -1)

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

    mask = np.zeros_like(mask_whole, dtype='bool')
    mask[mask_whole == 5] = 1
    rtstruct.add_roi(
        mask=mask,
        color=[255, 255, 255],
        name='vagina'
    )

    rtstruct.save(join(seg_dir, 'dicom', seg_fname.replace('nii.gz', 'dcm')))
else:
    print(' 6.- DICOM file already there. Skipping.\n')

print(' **********')
print(' ** DONE **')
print(' **********\n\n')
