import vtk
import os
import numpy as np
import SimpleITK as sitk
from skimage import measure
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk


def arrayToPoly(array, origin, spacing, direction):
    # array to vtkImageData
    flatten_array = array.ravel()
    vtk_data_array = numpy_to_vtk(
        num_array=flatten_array,  # ndarray contains the fitting result from the points. It is a 3D array
        deep=True,
        array_type=vtk.VTK_UNSIGNED_CHAR)

    # Convert the VTK array to vtkImageData
    shape = np.array(array.shape[::-1])
    origin = np.array(origin)
    spacing = np.array(spacing)

    img_vtk = vtk.vtkImageData()
    img_vtk.SetDimensions(shape)

    img_vtk.SetSpacing(spacing)
    img_vtk.SetOrigin(origin)
    img_vtk.SetDirectionMatrix(*direction)
    img_vtk.GetPointData().SetScalars(vtk_data_array)

    surf = vtk.vtkDiscreteMarchingCubes()
    surf.SetInputData(img_vtk)
    surf.SetValue(0, 1)  # use surf.GenerateValues function if more than one contour is available in the file
    # surf.GenerateValues(2, 0, 1)  # use surf.GenerateValues function if more than one contour is available in the file
    surf.Update()

    # smoothing the mesh
    smoother = vtk.vtkSmoothPolyDataFilter()
    smoother.SetInputConnection(surf.GetOutputPort())
    smoother.SetNumberOfIterations(50)
    smoother.SetRelaxationFactor(0.1)
    smoother.FeatureEdgeSmoothingOff()
    smoother.BoundarySmoothingOn()
    smoother.Update()

    # Update normals on newly smoothed polydata
    normalGenerator = vtk.vtkPolyDataNormals()
    normalGenerator.SetInputConnection(smoother.GetOutputPort())
    normalGenerator.ComputePointNormalsOn()
    normalGenerator.ComputeCellNormalsOn()
    normalGenerator.Update()

    # transform
    transform = vtk.vtkTransform()
    transform.Scale(-1, -1, 1)
    transform_filter = vtk.vtkTransformFilter()
    transform_filter.SetTransform(transform)
    transform_filter.SetInputData(normalGenerator.GetOutput())
    transform_filter.TransformAllInputVectorsOn()
    transform_filter.Update()

    poly = transform_filter.GetOutput()
    return poly


def saveSTL(poly, save_path):
    stlWriter = vtk.vtkSTLWriter()
    stlWriter.SetFileName(save_path)
    stlWriter.SetFileType(vtk.VTK_BINARY)
    stlWriter.SetInputData(poly)
    stlWriter.Write()


def saveNII(array, origin, spacing, direction, save_path):
    image = sitk.GetImageFromArray(array)
    image.SetOrigin(origin)
    image.SetSpacing(spacing)
    image.SetDirection(direction)
    sitk.WriteImage(image, save_path)


def volumeToSTL(volume_path, stl_dir, stl_name):
    image = sitk.ReadImage(volume_path)
    origin = image.GetOrigin()
    spacing = image.GetSpacing()
    direction = image.GetDirection()
    array = sitk.GetArrayFromImage(image)

    labels, num = measure.label(array, return_num=True)
    assert num == 2

    l1 = np.sum(labels == 1)
    l2 = np.sum(labels == 2)

    if l1 > l2:
        left_array = (labels == 1) + 0
        right_array = (labels == 2) + 0
    else:
        left_array = (labels == 2) + 0
        right_array = (labels == 1) + 0
    left_array = np.array(left_array, dtype=np.uint8)
    right_array = np.array(right_array, dtype=np.uint8)

    saveNII(left_array, origin, spacing, direction, os.path.join(stl_dir, stl_name+"_left.nii"))
    saveNII(right_array, origin, spacing, direction, os.path.join(stl_dir, stl_name+"_right.nii"))

    left_poly = arrayToPoly(left_array, origin, spacing, direction)
    right_poly = arrayToPoly(right_array, origin, spacing, direction)
    saveSTL(left_poly, os.path.join(stl_dir, stl_name+"_left.stl"))
    saveSTL(right_poly, os.path.join(stl_dir, stl_name+"_right.stl"))


mask_dir = '/home/hja/Projects/3D2DRegister/ASOCA/Train_Masks/'
stl_dir = '/home/hja/Projects/3D2DRegister/ASOCA/Train_Masks_STL/'

from centerline_show import show_branches_3d
from centeline_tree_reader import construct_tree_from_txt
volumeToSTL(os.path.join(mask_dir, "0.nrrd"), stl_dir, "0")
# files = list(filter(lambda f: f.split(".")[-1] == "nrrd", os.listdir(mask_dir)))
# for f in files:
#     print(f)
#     volumeToSTL(os.path.join(mask_dir, f), stl_dir, f.split(".")[0])
