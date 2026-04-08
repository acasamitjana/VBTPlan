import os
import datetime
import pdb
from typing import List, Union
from enum import IntEnum
from dataclasses import dataclass
import warnings

import numpy as np
import cv2 as cv
from pydicom.uid import generate_uid
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ImplicitVRLittleEndian, PYDICOM_IMPLEMENTATION_UID
from pydicom.filereader import dcmread


COLOR_PALETTE = [
    [255, 0, 255],
    [0, 235, 235],
    [255, 255, 0],
    [255, 0, 0],
    [0, 132, 255],
    [0, 240, 0],
    [255, 175, 0],
    [0, 208, 255],
    [180, 255, 105],
    [255, 20, 147],
    [160, 32, 240],
    [0, 255, 127],
    [255, 114, 0],
    [64, 224, 208],
    [0, 178, 47],
    [220, 20, 60],
    [238, 130, 238],
    [218, 165, 32],
    [255, 140, 190],
    [0, 0, 255],
    [255, 225, 0],
]

ROI_GENERATION_ALGORITHMS = ["AUTOMATIC", "SEMIAUTOMATIC", "MANUAL"]


class SOPClassUID:
    RTSTRUCT_IMPLEMENTATION_CLASS = (
        PYDICOM_IMPLEMENTATION_UID  # TODO find out if this is ok
    )
    DETACHED_STUDY_MANAGEMENT = "1.2.840.10008.3.1.2.3.1"
    RTSTRUCT = "1.2.840.10008.5.1.4.1.1.481.3"


@dataclass
class ROIData:
    """Data class to easily pass ROI data to helper methods."""

    mask: str
    color: Union[str, List[int]]
    number: int
    name: str
    frame_of_reference_uid: int
    description: str = ""
    use_pin_hole: bool = False
    approximate_contours: bool = True
    roi_generation_algorithm: Union[str, int] = 0

    def __post_init__(self):
        self.validate_color()
        self.add_default_values()
        self.validate_roi_generation_algoirthm()

    def add_default_values(self):
        if self.color is None:
            self.color = COLOR_PALETTE[(self.number - 1) % len(COLOR_PALETTE)]

        if self.name is None:
            self.name = f"ROI-{self.number}"

    def validate_color(self):
        if self.color is None:
            return

        # Validating list eg: [0, 0, 0]
        if type(self.color) is list:
            if len(self.color) != 3:
                raise ValueError(f"{self.color} is an invalid color for an ROI")
            for c in self.color:
                try:
                    assert 0 <= c <= 255
                except:
                    raise ValueError(f"{self.color} is an invalid color for an ROI")

        else:
            self.color: str = str(self.color)
            self.color = self.color.strip("#")

            # fff -> ffffff
            if len(self.color) == 3:
                self.color = "".join([x * 2 for x in self.color])

            if not len(self.color) == 6:
                raise ValueError(f"{self.color} is an invalid color for an ROI")

            try:
                self.color = [int(self.color[i : i + 2], 16) for i in (0, 2, 4)]
            except Exception as e:
                raise ValueError(f"{self.color} is an invalid color for an ROI")

    def validate_roi_generation_algoirthm(self):

        if isinstance(self.roi_generation_algorithm, int):
            # for ints we use the predefined values in ROI_GENERATION_ALGORITHMS
            if self.roi_generation_algorithm > 2 or self.roi_generation_algorithm < 0:
                raise ValueError(
                    "roi_generation_algorithm must be either an int (0='AUTOMATIC', 1='SEMIAUTOMATIC', 2='MANUAL') "
                    "or a str (not recomended)."
                )
            else:
                self.roi_generation_algorithm = ROI_GENERATION_ALGORITHMS[
                    self.roi_generation_algorithm
                ]

        elif isinstance(self.roi_generation_algorithm, str):
            # users can pick a str if they want to use a value other than the three default values
            if self.roi_generation_algorithm not in ROI_GENERATION_ALGORITHMS:
                print(
                    "Got self.roi_generation_algorithm {}. Some viewers might complain about this option. "
                    "Better options might be 0='AUTOMATIC', 1='SEMIAUTOMATIC', or 2='MANUAL'.".format(
                        self.roi_generation_algorithm
                    )
                )

        else:
            raise TypeError(
                "Expected int (0='AUTOMATIC', 1='SEMIAUTOMATIC', 2='MANUAL') "
                "or a str (not recomended) for self.roi_generation_algorithm. Got {}.".format(
                    type(self.roi_generation_algorithm)
                )
            )

class Hierarchy(IntEnum):
    """
    Enum class for what the positions in the OpenCV hierarchy array mean
    """

    next_node = 0
    previous_node = 1
    first_child = 2
    parent_node = 3

class RTStruct:
    """
    Wrapper class to facilitate appending and extracting ROI's within an RTStruct
    """

    def __init__(self, series_data, ds: FileDataset, ROIGenerationAlgorithm=0):
        self.series_data = series_data
        self.ds = ds
        self.frame_of_reference_uid = ds.ReferencedFrameOfReferenceSequence[
            -1
        ].FrameOfReferenceUID  # Use last strucitured set ROI

    def set_series_description(self, description: str):
        """
        Set the series description for the RTStruct dataset
        """

        self.ds.SeriesDescription = description

    def add_roi(
        self,
        mask: np.ndarray,
        color: Union[str, List[int]] = None,
        name: str = None,
        description: str = "",
        use_pin_hole: bool = False,
        approximate_contours: bool = True,
        roi_generation_algorithm: Union[str, int] = 0,
    ):
        """
        Add a ROI to the rtstruct given a 3D binary mask for the ROI's at each slice
        Optionally input a color or name for the ROI
        If use_pin_hole is set to true, will cut a pinhole through ROI's with holes in them so that they are represented with one contour
        If approximate_contours is set to False, no approximation will be done when generating contour data, leading to much larger amount of contour data
        """

        # TODO test if name already exists
        self.validate_mask(mask)
        roi_number = len(self.ds.StructureSetROISequence) + 1
        roi_data = ROIData(
            mask,
            color,
            roi_number,
            name,
            self.frame_of_reference_uid,
            description,
            use_pin_hole,
            approximate_contours,
            roi_generation_algorithm,
        )

        self.ds.ROIContourSequence.append(
            create_roi_contour(roi_data, self.series_data)
        )
        self.ds.StructureSetROISequence.append(
            create_structure_set_roi(roi_data)
        )
        self.ds.RTROIObservationsSequence.append(
            create_rtroi_observation(roi_data)
        )

    def validate_mask(self, mask: np.ndarray) -> bool:
        if mask.dtype != bool:
            raise RTStruct.ROIException(
                f"Mask data type must be boolean. Got {mask.dtype}"
            )

        if mask.ndim != 3:
            raise RTStruct.ROIException(f"Mask must be 3 dimensional. Got {mask.ndim}")

        if len(self.series_data) != np.shape(mask)[2]:
            raise RTStruct.ROIException(
                "Mask must have the save number of layers (In the 3rd dimension) as input series. "
                + f"Expected {len(self.series_data)}, got {np.shape(mask)[2]}"
            )

        if np.sum(mask) == 0:
            print("[INFO]: ROI mask is empty")

        return True

    def get_roi_names(self) -> List[str]:
        """
        Returns a list of the names of all ROI within the RTStruct
        """

        if not self.ds.StructureSetROISequence:
            return []

        return [
            structure_roi.ROIName for structure_roi in self.ds.StructureSetROISequence
        ]

    def get_roi_mask_by_name(self, name) -> np.ndarray:
        """
        Returns the 3D binary mask of the ROI with the given input name
        """

        for structure_roi in self.ds.StructureSetROISequence:
            if structure_roi.ROIName == name:
                contour_sequence = get_contour_sequence_by_roi_number(
                    self.ds, structure_roi.ROINumber
                )
                return create_series_mask_from_contour_sequence(
                    self.series_data, contour_sequence
                )

        raise RTStruct.ROIException(f"ROI of name `{name}` does not exist in RTStruct")

    def save(self, file_path: str):
        """
        Saves the RTStruct with the specified name / location
        Automatically adds '.dcm' as a suffix
        """

        # Add .dcm if needed
        file_path = file_path if file_path.endswith(".dcm") else file_path + ".dcm"

        try:
            file = open(file_path, "w")
            # Opening worked, we should have a valid file_path
            print("Writing file to", file_path)
            self.ds.save_as(file_path)
            file.close()
        except OSError:
            raise Exception(f"Cannot write to file path '{file_path}'")

    class ROIException(Exception):
        """
        Exception class for invalid ROI masks
        """

        pass


class RTStructBuilder:
    """
    Class to help facilitate the two ways in one can instantiate the RTStruct wrapper
    """

    @staticmethod
    def create_new(dicom_series_path: str) -> RTStruct:
        """
        Method to generate a new rt struct from a DICOM series
        """

        series_data = load_sorted_image_series(dicom_series_path)
        ds = create_rtstruct_dataset(series_data)
        return RTStruct(series_data, ds)

    @staticmethod
    def create_from(dicom_series_path: str, rt_struct_path: str, warn_only: bool = False) -> RTStruct:
        """
        Method to load an existing rt struct, given related DICOM series and existing rt struct
        """

        series_data = load_sorted_image_series(dicom_series_path)
        ds = dcmread(rt_struct_path)
        RTStructBuilder.validate_rtstruct(ds)
        RTStructBuilder.validate_rtstruct_series_references(ds, series_data, warn_only)

        # TODO create new frame of reference? Right now we assume the last frame of reference created is suitable
        return RTStruct(series_data, ds)

    @staticmethod
    def validate_rtstruct(ds: Dataset):
        """
        Method to validate a dataset is a valid RTStruct containing the required fields
        """

        if (
            ds.SOPClassUID != SOPClassUID.RTSTRUCT
            or not hasattr(ds, "ROIContourSequence")
            or not hasattr(ds, "StructureSetROISequence")
            or not hasattr(ds, "RTROIObservationsSequence")
        ):
            raise Exception("Please check that the existing RTStruct is valid")

    @staticmethod
    def validate_rtstruct_series_references(ds: Dataset, series_data: List[Dataset], warn_only: bool = False):
        """
        Method to validate RTStruct only references dicom images found within the input series_data
        """
        for refd_frame_of_ref in ds.ReferencedFrameOfReferenceSequence:
            # Study sequence references are optional so return early if it does not exist
            if "RTReferencedStudySequence" not in refd_frame_of_ref:
                return

            for rt_refd_study in refd_frame_of_ref.RTReferencedStudySequence:
                for rt_refd_series in rt_refd_study.RTReferencedSeriesSequence:
                    for contour_image in rt_refd_series.ContourImageSequence:
                        RTStructBuilder.validate_contour_image_in_series_data(
                            contour_image, series_data, warn_only
                        )

    @staticmethod
    def validate_contour_image_in_series_data(
        contour_image: Dataset, series_data: List[Dataset], warning_only: bool = False
    ):
        """
        Method to validate that the ReferencedSOPInstanceUID of a given contour image exists within the series data
        """
        for series in series_data:
            if contour_image.ReferencedSOPInstanceUID == series.SOPInstanceUID:
                return

        # ReferencedSOPInstanceUID is NOT available
        msg = f"Loaded RTStruct references image(s) that are not contained in input series data. " \
              f"Problematic image has SOP Instance Id: {contour_image.ReferencedSOPInstanceUID}"
        if warning_only:
            warnings.warn(msg)
        else:
            raise Exception(msg)


def create_rtstruct_dataset(series_data) -> FileDataset:
    ds = generate_base_dataset()
    add_study_and_series_information(ds, series_data)
    add_patient_information(ds, series_data)
    add_refd_frame_of_ref_sequence(ds, series_data)
    return ds


def generate_base_dataset() -> FileDataset:
    file_name = "rt-utils-struct"
    file_meta = get_file_meta()
    ds = FileDataset(file_name, {}, file_meta=file_meta, preamble=b"\0" * 128)
    add_required_elements_to_ds(ds)
    add_sequence_lists_to_ds(ds)
    return ds


def get_file_meta() -> FileMetaDataset:
    file_meta = FileMetaDataset()
    file_meta.FileMetaInformationGroupLength = 202
    file_meta.FileMetaInformationVersion = b"\x00\x01"
    file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = SOPClassUID.RTSTRUCT
    file_meta.MediaStorageSOPInstanceUID = (
        generate_uid()
    )  # TODO find out random generation is fine
    file_meta.ImplementationClassUID = SOPClassUID.RTSTRUCT_IMPLEMENTATION_CLASS
    return file_meta


def add_required_elements_to_ds(ds: FileDataset):
    dt = datetime.datetime.now()
    # Append data elements required by the DICOM standarad
    ds.SpecificCharacterSet = "ISO_IR 100"
    ds.InstanceCreationDate = dt.strftime("%Y%m%d")
    ds.InstanceCreationTime = dt.strftime("%H%M%S.%f")
    ds.StructureSetLabel = "RTstruct"
    ds.StructureSetDate = dt.strftime("%Y%m%d")
    ds.StructureSetTime = dt.strftime("%H%M%S.%f")
    ds.Modality = "RTSTRUCT"
    ds.Manufacturer = "Qurit"
    ds.ManufacturerModelName = "rt-utils"
    ds.InstitutionName = "Qurit"
    # Set the transfer syntax
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    # Set values already defined in the file meta
    ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID

    ds.ApprovalStatus = "UNAPPROVED"


def add_sequence_lists_to_ds(ds: FileDataset):
    ds.StructureSetROISequence = Sequence()
    ds.ROIContourSequence = Sequence()
    ds.RTROIObservationsSequence = Sequence()


def add_study_and_series_information(ds: FileDataset, series_data):
    reference_ds = series_data[0]  # All elements in series should have the same data
    ds.StudyDate = reference_ds.StudyDate
    ds.SeriesDate = getattr(reference_ds, "SeriesDate", "")
    ds.StudyTime = reference_ds.StudyTime
    ds.SeriesTime = getattr(reference_ds, "SeriesTime", "")
    ds.StudyDescription = getattr(reference_ds, "StudyDescription", "")
    ds.SeriesDescription = getattr(reference_ds, "SeriesDescription", "")
    ds.StudyInstanceUID = reference_ds.StudyInstanceUID
    ds.SeriesInstanceUID = generate_uid()  # TODO: find out if random generation is ok
    ds.StudyID = reference_ds.StudyID
    ds.SeriesNumber = "1"  # TODO: find out if we can just use 1 (Should be fine since its a new series)


def add_patient_information(ds: FileDataset, series_data):
    reference_ds = series_data[0]  # All elements in series should have the same data
    ds.PatientName = getattr(reference_ds, "PatientName", "")
    ds.PatientID = getattr(reference_ds, "PatientID", "")
    ds.PatientBirthDate = getattr(reference_ds, "PatientBirthDate", "")
    ds.PatientSex = getattr(reference_ds, "PatientSex", "")
    ds.PatientAge = getattr(reference_ds, "PatientAge", "")
    ds.PatientSize = getattr(reference_ds, "PatientSize", "")
    ds.PatientWeight = getattr(reference_ds, "PatientWeight", "")


def add_refd_frame_of_ref_sequence(ds: FileDataset, series_data):
    refd_frame_of_ref = Dataset()
    refd_frame_of_ref.FrameOfReferenceUID =  getattr(series_data[0], 'FrameOfReferenceUID', generate_uid())
    refd_frame_of_ref.RTReferencedStudySequence = create_frame_of_ref_study_sequence(series_data)

    # Add to sequence
    ds.ReferencedFrameOfReferenceSequence = Sequence()
    ds.ReferencedFrameOfReferenceSequence.append(refd_frame_of_ref)


def create_frame_of_ref_study_sequence(series_data) -> Sequence:
    reference_ds = series_data[0]  # All elements in series should have the same data
    rt_refd_series = Dataset()
    rt_refd_series.SeriesInstanceUID = reference_ds.SeriesInstanceUID
    rt_refd_series.ContourImageSequence = create_contour_image_sequence(series_data)

    rt_refd_series_sequence = Sequence()
    rt_refd_series_sequence.append(rt_refd_series)

    rt_refd_study = Dataset()
    rt_refd_study.ReferencedSOPClassUID = SOPClassUID.DETACHED_STUDY_MANAGEMENT
    rt_refd_study.ReferencedSOPInstanceUID = reference_ds.StudyInstanceUID
    rt_refd_study.RTReferencedSeriesSequence = rt_refd_series_sequence

    rt_refd_study_sequence = Sequence()
    rt_refd_study_sequence.append(rt_refd_study)
    return rt_refd_study_sequence


def create_contour_image_sequence(series_data) -> Sequence:
    contour_image_sequence = Sequence()

    # Add each referenced image
    for series in series_data:
        contour_image = Dataset()
        contour_image.ReferencedSOPClassUID = series.SOPClassUID
        contour_image.ReferencedSOPInstanceUID = series.SOPInstanceUID
        contour_image_sequence.append(contour_image)

    return contour_image_sequence


def create_structure_set_roi(roi_data: ROIData) -> Dataset:
    # Structure Set ROI Sequence: Structure Set ROI 1
    structure_set_roi = Dataset()
    structure_set_roi.ROINumber = roi_data.number
    structure_set_roi.ReferencedFrameOfReferenceUID = roi_data.frame_of_reference_uid
    structure_set_roi.ROIName = roi_data.name
    structure_set_roi.ROIDescription = roi_data.description
    structure_set_roi.ROIGenerationAlgorithm = roi_data.roi_generation_algorithm
    return structure_set_roi


def create_roi_contour(roi_data: ROIData, series_data) -> Dataset:
    roi_contour = Dataset()
    roi_contour.ROIDisplayColor = roi_data.color
    roi_contour.ContourSequence = create_contour_sequence(roi_data, series_data)
    roi_contour.ReferencedROINumber = str(roi_data.number)
    return roi_contour


def create_contour_sequence(roi_data: ROIData, series_data) -> Sequence:
    """
    Iterate through each slice of the mask
    For each connected segment within a slice, create a contour
    """

    contour_sequence = Sequence()

    contours_coords = get_contours_coords(roi_data, series_data)

    for series_slice, slice_contours in zip(series_data, contours_coords):
        for contour_data in slice_contours:
            contour = create_contour(series_slice, contour_data)
            contour_sequence.append(contour)

    return contour_sequence

def create_contour_sequence_3D(roi_data: ROIData, series_data) -> Sequence:
    """
    Iterate through each slice of the mask
    For each connected segment within a slice, create a contour
    """

    contour_sequence = Sequence()

    contours_coords = get_contours_coords_3D(roi_data, series_data)

    for series_slice, slice_contours in zip(series_data, contours_coords):
        if slice_contours:
            contour = create_contour(series_slice, slice_contours)
            contour_sequence.append(contour)


    return contour_sequence

def create_contour(series_slice: Dataset, contour_data: np.ndarray) -> Dataset:
    contour_image = Dataset()
    contour_image.ReferencedSOPClassUID = series_slice.SOPClassUID
    contour_image.ReferencedSOPInstanceUID = series_slice.SOPInstanceUID

    # Contour Image Sequence
    contour_image_sequence = Sequence()
    contour_image_sequence.append(contour_image)

    contour = Dataset()
    contour.ContourImageSequence = contour_image_sequence
    contour.ContourGeometricType = (
        "CLOSED_PLANAR"  # TODO figure out how to get this value
    )
    contour.NumberOfContourPoints = (
        len(contour_data) / 3
    )  # Each point has an x, y, and z value
    contour.ContourData = contour_data

    return contour


def create_rtroi_observation(roi_data: ROIData) -> Dataset:
    rtroi_observation = Dataset()
    rtroi_observation.ObservationNumber = roi_data.number
    rtroi_observation.ReferencedROINumber = roi_data.number
    # TODO figure out how to get observation description
    rtroi_observation.ROIObservationDescription = "Type:Soft,Range:*/*,Fill:0,Opacity:0.0,Thickness:1,LineThickness:2,read-only:false"
    rtroi_observation.private_creators = "Qurit Lab"
    rtroi_observation.RTROIInterpretedType = ""
    rtroi_observation.ROIInterpreter = ""
    return rtroi_observation


def get_contour_sequence_by_roi_number(ds, roi_number):
    for roi_contour in ds.ROIContourSequence:
        # Ensure same type
        if str(roi_contour.ReferencedROINumber) == str(roi_number):
            return roi_contour.ContourSequence

    raise Exception(f"Referenced ROI number '{roi_number}' not found")


def get_contours_coords(roi_data: ROIData, series_data):
    transformation_matrix = get_pixel_to_patient_transformation_matrix(series_data)

    series_contours = []
    for i, series_slice in enumerate(series_data):
        mask_slice = roi_data.mask[:, :, i]

        # Do not add ROI's for blank slices
        if np.sum(mask_slice) == 0:
            series_contours.append([])
            continue

        # Create pin hole mask if specified
        if roi_data.use_pin_hole:
            mask_slice = create_pin_hole_mask(mask_slice, roi_data.approximate_contours)

        # Get contours from mask
        contours, _ = find_mask_contours(mask_slice, roi_data.approximate_contours)
        # from skimage.measure import marching_cubes
        # verts, faces, normals, values = marching_cubes(np.stack([mask_slice, mask_slice], axis=1), 0, step_size=1)
        # contours = [np.concatenate([verts[::2, 2:3],  verts[::2, 0:1]], axis=1).tolist()]
        # validate_contours(contours)

        # Format for DICOM
        formatted_contours = []
        for contour in contours:
            # Add z index
            contour = np.concatenate(
                (np.array(contour), np.full((len(contour), 1), i)), axis=1
            )

            transformed_contour = apply_transformation_to_3d_points(
                contour, transformation_matrix
            )

            # new_tf_contour = [transformed_contour[0], transformed_contour[1]]
            # added_c = [0, 1]
            # for _ in range(2, len(transformed_contour)):
            #     last_c = new_tf_contour[-1]
            #     list_it_new_c = np.argsort(np.sum((transformed_contour - last_c)**2, axis=1))
            #     list_it_new_c = [it for it in list_it_new_c if it not in added_c]
            #
            #     it_last_c = list_it_new_c[0]
            #     new_tf_contour.append(transformed_contour[it_last_c])
            #     added_c.append(it_last_c)
            #
            # new_tf_contour = np.array(new_tf_contour)
            # if np.sum(np.abs(new_tf_contour-transformed_contour)) > 0:
            #     print(np.sum(np.abs(new_tf_contour-transformed_contour)))
            #     # pdb.set_trace()
            #
            # transformed_contour = new_tf_contour
            dicom_formatted_contour = np.ravel(transformed_contour).tolist()
            formatted_contours.append(dicom_formatted_contour)

        series_contours.append(formatted_contours)

    return series_contours

def get_contours_coords_3D(roi_data: ROIData, series_data):
    transformation_matrix = get_pixel_to_patient_transformation_matrix(series_data)

    from skimage.measure import marching_cubes
    verts, faces, normals, values = marching_cubes(roi_data.mask, 0)
    coords = np.concatenate([verts[:, 1:2], verts[:, 2:3], verts[:, 0:1], np.ones((verts.shape[0], 1))], axis=1)
    transformed_contour =coords.dot(transformation_matrix.T)[:, :3]
    existing_i_contour = np.unique(coords[:, 2])
    pdb.set_trace()

    series_contours = []
    section_contour = []
    for i in range(len(series_data)):
        if i in existing_i_contour:
            valid_coords = transformed_contour[np.where(coords[:, 2] == i)]
            for v in valid_coords:
                section_contour.extend(v.tolist())

            series_contours.append(section_contour)

        else:
            series_contours.append([])

    return series_contours

def load_sorted_image_series(dicom_series_path: str):
    """
    File contains helper methods for loading / formatting DICOM images and contours
    """

    series_data = load_dcm_images_from_path(dicom_series_path)

    if len(series_data) == 0:
        raise Exception("No DICOM Images found in input path")

    # Sort slices in ascending order
    series_data.sort(key=get_slice_position, reverse=False)

    return series_data


def load_dcm_images_from_path(dicom_series_path: str) -> List[Dataset]:
    series_data = []
    for root, _, files in os.walk(dicom_series_path):
        for file in files:
            try:
                ds = dcmread(os.path.join(root, file))
                if hasattr(ds, "pixel_array"):
                    series_data.append(ds)

            except Exception:
                # Not a valid DICOM file
                continue

    return series_data



def find_mask_contours(mask: np.ndarray, approximate_contours: bool):
    approximation_method = (
        cv.CHAIN_APPROX_SIMPLE if approximate_contours else cv.CHAIN_APPROX_NONE
    )
    contours, hierarchy = cv.findContours(
        mask.astype(np.uint8), cv.RETR_LIST, approximation_method
    )
    contours = list(
        contours
    )  # Open-CV updated contours to be a tuple so we convert it back into a list here

    # Hierarcy format:
    #   rows = number of contours;
    #   columns:
    #       (1) next contour at the same level of hierarchy;
    #       (2) previous contour at the same level of hierarchy;
    #       (3) contour child;
    #       (4) contour parent;
    #
    # Need to filter only first-level contours:
    # print(hierarchy)
    # filter_flag = (hierarchy[0, :, -1] == -1) | (hierarchy[0, :, -1] == 0)
    # contours = [c for it_c, c in enumerate(contours) if filter_flag[it_c]]

    # Format extra array out of data
    for i, contour in enumerate(contours):
        contours[i] = [[pos[0][0], pos[0][1]] for pos in contour]
    hierarchy = hierarchy[0]  # Format extra array out of data

    return contours, hierarchy


def create_pin_hole_mask(mask: np.ndarray, approximate_contours: bool):
    """
    Creates masks with pin holes added to contour regions with holes.
    This is done so that a given region can be represented by a single contour.
    """

    contours, hierarchy = find_mask_contours(mask, approximate_contours)
    pin_hole_mask = mask.copy()

    # Iterate through the hierarchy, for child nodes, draw a line upwards from the first point
    for i, array in enumerate(hierarchy):
        parent_contour_index = array[Hierarchy.parent_node]
        if parent_contour_index == -1:
            continue  # Contour is not a child

        child_contour = contours[i]

        line_start = tuple(child_contour[0])

        pin_hole_mask = draw_line_upwards_from_point(
            pin_hole_mask, line_start, fill_value=0
        )
    return pin_hole_mask


def draw_line_upwards_from_point(
    mask: np.ndarray, start, fill_value: int
) -> np.ndarray:
    line_width = 2
    end = (start[0], start[1] - 1)
    mask = mask.astype(np.uint8)  # Type that OpenCV expects
    # Draw one point at a time until we hit a point that already has the desired value
    while mask[end] != fill_value:
        cv.line(mask, start, end, fill_value, line_width)

        # Update start and end to the next positions
        start = end
        end = (start[0], start[1] - line_width)
    return mask.astype(bool)


def validate_contours(contours: list):
    if len(contours) == 0:
        raise Exception(
            "Unable to find contour in non empty mask, please check your mask formatting"
        )


def get_pixel_to_patient_transformation_matrix(series_data):
    """
    https://nipy.org/nibabel/dicom/dicom_orientation.html
    """

    first_slice = series_data[0]

    offset = np.array(first_slice.ImagePositionPatient)
    row_spacing, column_spacing = first_slice.PixelSpacing
    slice_spacing = get_spacing_between_slices(series_data)
    row_direction, column_direction, slice_direction = get_slice_directions(first_slice)

    mat = np.identity(4, dtype=np.float32)
    mat[:3, 0] = row_direction * row_spacing
    mat[:3, 1] = column_direction * column_spacing
    mat[:3, 2] = slice_direction * slice_spacing
    mat[:3, 3] = offset

    return mat


def get_patient_to_pixel_transformation_matrix(series_data):
    first_slice = series_data[0]

    offset = np.array(first_slice.ImagePositionPatient)
    row_spacing, column_spacing = first_slice.PixelSpacing
    slice_spacing = get_spacing_between_slices(series_data)
    row_direction, column_direction, slice_direction = get_slice_directions(first_slice)

    # M = [ rotation&scaling   translation ]
    #     [        0                1      ]
    #
    # inv(M) = [ inv(rotation&scaling)   -inv(rotation&scaling) * translation ]
    #          [          0                                1                  ]

    linear = np.identity(3, dtype=np.float32)
    linear[0, :3] = row_direction / row_spacing
    linear[1, :3] = column_direction / column_spacing
    linear[2, :3] = slice_direction / slice_spacing

    mat = np.identity(4, dtype=np.float32)
    mat[:3, :3] = linear
    mat[:3, 3] = offset.dot(-linear.T)

    return mat


def apply_transformation_to_3d_points(
    points: np.ndarray, transformation_matrix: np.ndarray
):
    """
    * Augment each point with a '1' as the fourth coordinate to allow translation
    * Multiply by a 4x4 transformation matrix
    * Throw away added '1's
    """
    vec = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    return vec.dot(transformation_matrix.T)[:, :3]


def get_slice_position(series_slice: Dataset):
    _, _, slice_direction = get_slice_directions(series_slice)
    return np.dot(slice_direction, series_slice.ImagePositionPatient)


def get_slice_directions(series_slice: Dataset):
    orientation = series_slice.ImageOrientationPatient
    row_direction = np.array(orientation[:3])
    column_direction = np.array(orientation[3:])
    slice_direction = np.cross(row_direction, column_direction)

    if not np.allclose(
        np.dot(row_direction, column_direction), 0.0, atol=1e-3
    ) or not np.allclose(np.linalg.norm(slice_direction), 1.0, atol=1e-3):
        raise Exception("Invalid Image Orientation (Patient) attribute")

    return row_direction, column_direction, slice_direction


def get_spacing_between_slices(series_data):
    if len(series_data) > 1:
        zcoord_list = [get_slice_position(s) for s in series_data]
        zcoord_list = np.unique(zcoord_list)

        return (np.max(zcoord_list) - np.min(zcoord_list)) / (len(zcoord_list) - 1)
    # Return nonzero value for one slice just to make the transformation matrix invertible
    return 1.0


def create_series_mask_from_contour_sequence(series_data, contour_sequence: Sequence):
    mask = create_empty_series_mask(series_data)
    transformation_matrix = get_patient_to_pixel_transformation_matrix(series_data)

    # Iterate through each slice of the series, If it is a part of the contour, add the contour mask
    for i, series_slice in enumerate(series_data):
        slice_contour_data = get_slice_contour_data(series_slice, contour_sequence)
        if len(slice_contour_data):
            mask[:, :, i] = get_slice_mask_from_slice_contour_data(
                series_slice, slice_contour_data, transformation_matrix
            )
    return mask


def get_slice_contour_data(series_slice: Dataset, contour_sequence: Sequence):
    slice_contour_data = []

    # Traverse through sequence data and get all contour data pertaining to the given slice
    for contour in contour_sequence:
        for contour_image in contour.ContourImageSequence:
            if contour_image.ReferencedSOPInstanceUID == series_slice.SOPInstanceUID:
                slice_contour_data.append(contour.ContourData)

    return slice_contour_data


def get_slice_mask_from_slice_contour_data(
    series_slice: Dataset, slice_contour_data, transformation_matrix: np.ndarray
):
    # Go through all contours in a slice, create polygons in correct space and with a correct format
    # and append to polygons array (appropriate for fillPoly)
    polygons = []
    for contour_coords in slice_contour_data:
        reshaped_contour_data = np.reshape(contour_coords, [len(contour_coords) // 3, 3])
        translated_contour_data = apply_transformation_to_3d_points(reshaped_contour_data, transformation_matrix)
        polygon = [np.around([translated_contour_data[:, :2]]).astype(np.int32)]
        polygon = np.array(polygon).squeeze()
        polygons.append(polygon)
    slice_mask = create_empty_slice_mask(series_slice).astype(np.uint8)
    cv.fillPoly(img=slice_mask, pts=polygons, color=1)
    return slice_mask

def create_empty_series_mask(series_data):
    ref_dicom_image = series_data[0]
    mask_dims = (
        int(ref_dicom_image.Columns),
        int(ref_dicom_image.Rows),
        len(series_data),
    )
    mask = np.zeros(mask_dims).astype(bool)
    return mask


def create_empty_slice_mask(series_slice):
    mask_dims = (int(series_slice.Columns), int(series_slice.Rows))
    mask = np.zeros(mask_dims).astype(bool)
    return mask

