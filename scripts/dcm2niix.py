from os.path import join, exists
from os import makedirs
from argparse import ArgumentParser

import nibabel as nib
import numpy as np
from VBTPlan.dcm_utils import *


arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--i', required=True, help="Dicom folder.")
arg_parser.add_argument('--s', required=True, help="Segmentation file of interest (stored in the dicom folder)")
arg_parser.add_argument('--o', required=True, help="Output root directory")
arg_parser.add_argument('--sname', default="VAGINA OAR", help="Name of the region(s) of interest")

args = arg_parser.parse_args()

lps2ras = np.eye(4) #the library works with the LPS convetion and NIFTI works with RAS.
lps2ras[0, 0] = -1
lps2ras[1, 1] = -1

DICOM_PATH = args.i
SEG_PATH = join(args.i, args.s)
OUTPUT_DIR = args.o

rtstruct = RTStructBuilder.create_from(
    dicom_series_path=DICOM_PATH, rt_struct_path=SEG_PATH
)

vox2lps = get_pixel_to_patient_transformation_matrix(rtstruct.series_data)
vox2ras = lps2ras @ vox2lps
sid = rtstruct.ds.PatientID

if not exists(join(OUTPUT_DIR, 'sub-' + str(sid), 'ses-0', 'anat')): makedirs(join(OUTPUT_DIR, 'sub-' + str(sid), 'ses-0', 'anat'))
if not exists(join(OUTPUT_DIR, 'sub-' + str(sid), 'ses-0', 'seg')): makedirs(join(OUTPUT_DIR, 'sub-' + str(sid), 'ses-0', 'seg'))

if not exists(join(OUTPUT_DIR, 'sub-' + str(sid), 'ses-0', 'anat', 'sub-' + str(sid) + '_ses-0_CT.nii.gz')):
    image = np.stack([sd.pixel_array.T for sd in rtstruct.series_data], -1)
    img = nib.Nifti1Image(image, vox2ras)
    nib.save(img, join(OUTPUT_DIR, 'sub-' + str(sid), 'ses-0', 'anat', 'sub-' + str(sid) + '_ses-0_CT.nii.gz'))

mask_3d = rtstruct.get_roi_mask_by_name(args.sname)
mask_3d = np.stack([mask_3d[..., it_d].T for it_d in range(mask_3d.shape[-1])], -1)
img = nib.Nifti1Image(mask_3d.astype('uint8'), vox2ras)
nib.save(img, join(OUTPUT_DIR, 'sub-' + str(sid), 'ses-0', 'seg', 'sub-' + str(sid) + '_ses-0_desc-' + args.sname + '_dseg.nii.gz'))

