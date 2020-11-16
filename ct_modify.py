import SimpleITK as sitk
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import numpy as np


def get_liver_bb():
    # read portalvein label
    vessel_label_path = r"D:\Data\3DirCad\3Dircadb1\3Dircadb1.1\MASKS_DICOM\MASKS_DICOM\portalvein"
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(vessel_label_path)
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    vessel_image = series_reader.Execute()
    vessel_array = sitk.GetArrayFromImage(vessel_image)

    # read liver label
    liver_label_path = r"D:\Data\3DirCad\3Dircadb1\3Dircadb1.1\MASKS_DICOM\MASKS_DICOM\liver"
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(liver_label_path)
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    liver_image = series_reader.Execute()
    liver_array = sitk.GetArrayFromImage(liver_image)

    # bound box
    vessel_points = np.argwhere(vessel_array == vessel_array.max())
    vessel_bb = [np.min(vessel_points[:, 0]), np.max(vessel_points[:, 0]) + 1, np.min(vessel_points[:, 1]),
                np.max(vessel_points[:, 1]) + 1, np.min(vessel_points[:, 2]), np.max(vessel_points[:, 2]) + 1]

    liver_points = np.argwhere(liver_array == liver_array.max())
    liver_bb = [np.min(liver_points[:, 0]), np.max(liver_points[:, 0]) + 1, np.min(liver_points[:, 1]),
                np.max(liver_points[:, 1]) + 1, np.min(liver_points[:, 2]), np.max(liver_points[:, 2]) + 1]

    bb = [min(vessel_bb[0], liver_bb[0]), max(vessel_bb[1], liver_bb[1]),
          min(vessel_bb[2], liver_bb[2]), max(vessel_bb[3], liver_bb[3]),
          min(vessel_bb[4], liver_bb[4]), max(vessel_bb[5], liver_bb[5])]
    print(bb)
    return liver_array, vessel_array, bb


def save_stl(array, spacing, origin, name):
    # array to vtkImageData
    flatten_array = array.ravel()
    shape = np.array(array.shape)
    vtk_data_array = numpy_to_vtk(
        num_array=flatten_array,  # ndarray contains the fitting result from the points. It is a 3D array
        deep=True,
        array_type=vtk.VTK_FLOAT)

    # convert vessel array to poly and save as STL
    img_vtk = vtk.vtkImageData()
    img_vtk.SetDimensions(shape[::-1])
    img_vtk.SetSpacing(spacing)
    img_vtk.SetOrigin(origin)
    img_vtk.SetDirectionMatrix(-1, 0, 0,
                               0, -1, 0,
                               0, 0, 1)
    img_vtk.GetPointData().SetScalars(vtk_data_array)

    surf = vtk.vtkDiscreteMarchingCubes()
    surf.SetInputData(img_vtk)
    surf.SetValue(0, array.max())

    # smoothing the mesh
    smoother = vtk.vtkSmoothPolyDataFilter()
    smoother.SetInputConnection(surf.GetOutputPort())
    smoother.SetNumberOfIterations(50)
    smoother.SetRelaxationFactor(0.1)
    smoother.FeatureEdgeSmoothingOff()
    smoother.BoundarySmoothingOn()
    smoother.Update()

    stl_writer = vtk.vtkSTLWriter()
    stl_writer.SetFileName(name)
    stl_writer.SetInputData(smoother.GetOutput())
    stl_writer.Write()


def save_nii(array, spacing, origin, name):
    liver_image = sitk.GetImageFromArray(array)
    liver_image.SetSpacing(spacing)
    liver_image.SetOrigin(origin)
    sitk.WriteImage(liver_image, name)


def crop_livers(liver_array, vessel_array, bb):
    patient_image_path = r"D:\Data\3DirCad\3Dircadb1\3Dircadb1.1\PATIENT01.nii.gz"
    patient_image = sitk.ReadImage(patient_image_path)
    patient_array = sitk.GetArrayFromImage(patient_image)

    bb_liver = patient_array[bb[0]:bb[1] + 1, bb[2]:bb[3] + 1, bb[4]:bb[5] + 1]
    save_nii(bb_liver, patient_image.GetSpacing(), patient_image.GetOrigin(), "data/liver.nii.gz")

    cropped_liver_array = liver_array[bb[0]:bb[1] + 1, bb[2]:bb[3] + 1, bb[4]:bb[5] + 1]
    save_stl(cropped_liver_array, patient_image.GetSpacing(), patient_image.GetOrigin(), "data/liver.stl")

    cropped_vessel_array = vessel_array[bb[0]:bb[1] + 1, bb[2]:bb[3] + 1, bb[4]:bb[5] + 1]
    save_stl(cropped_vessel_array, patient_image.GetSpacing(), patient_image.GetOrigin(), "data/vessel.stl")
    save_nii(cropped_vessel_array, patient_image.GetSpacing(), patient_image.GetOrigin(), "data/vessel.nii.gz")

    # contrast
    patient_array[vessel_array == vessel_array.max()] *= 2
    bb_contrast_liver = patient_array[bb[0]:bb[1] + 1, bb[2]:bb[3] + 1, bb[4]:bb[5] + 1]
    save_nii(bb_contrast_liver, patient_image.GetSpacing(), patient_image.GetOrigin(), "data/contrast_liver.nii.gz")


if __name__ == '__main__':
    liver_array, vessel_array, bb = get_liver_bb()
    crop_livers(liver_array, vessel_array, bb)
