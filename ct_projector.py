import itk
import SimpleITK as sitk
import numpy as np
import pickle
import cv2
from skimage import measure

# parameters
tx = 0
ty = 0
tz = 0
rx = 0
ry = 0
rz = 0
cx = 0
cy = 0
cz = 0
size2D = [512, 512, 1]
spacing2D = [0.37, 0.37, 1.]
focalLength = 1500.
maxStepSize = 4.
minStepSize = 1.
o2Dx = 0
o2Dy = 0
threshold = 0


def projector(image_3d_path, save_image_name=None):
    # types
    image_type_3d = itk.Image[itk.F, 3]
    image_type_2d = itk.Image[itk.F, 3]
    inter_image_type = itk.Image[itk.F, 3]

    # read 3D image
    imageReader = itk.ImageFileReader[image_type_3d].New()
    imageReader.SetFileName(image_3d_path)

    imageReader.Update()
    image = imageReader.GetOutput()

    # transform 3D image according to given parameters
    transform = itk.Euler3DTransform[itk.D].New()
    transform.SetComputeZYX(True)
    transform.SetTranslation((tx, ty, tx))
    transform.SetRotation(np.pi / 180.0 * rx, np.pi / 180.0 * ry, np.pi / 180.0 * rz)

    spacing_3d = image.GetSpacing()
    origin_3d = image.GetOrigin()
    size_3d = image.GetBufferedRegion().GetSize()
    origin_volume = [origin_3d[i] + spacing_3d[i] * size_3d[i] / 2 for i in range(3)]
    rotate_center = [cx + origin_volume[0], cy + origin_volume[1], cz + origin_volume[2]]
    transform.SetCenter(rotate_center)

    # set attributes of projected plane (left upper)
    origin_2d = [
        origin_3d[0] + o2Dx - spacing2D[0] * (size2D[0] - 1) / 2.0,
        origin_3d[1] + o2Dy - spacing2D[1] * (size2D[1] - 1) / 2.0,
        origin_3d[2] + focalLength / 2.0
    ]
    focal_point = [origin_3d[0], origin_3d[1], origin_3d[2] - focalLength / 2.0]

    # Ray Cast Interpolate
    interpolator = itk.RayCastInterpolateImageFunction[inter_image_type, itk.D].New()
    interpolator.SetFocalPoint(focal_point)  # The focal point or position of the ray source.
    interpolator.SetThreshold(threshold)  # The threshold above which voxels along the ray path are integrated.
    interpolator.SetTransform(transform)  # This Transformation is used to calculate the new focal point position.

    # Write out the projection image
    filter = itk.ResampleImageFilter[inter_image_type, image_type_2d].New()
    filter.SetInput(image)
    filter.SetDefaultPixelValue(0)
    filter.SetTransform(transform)
    filter.SetInterpolator(interpolator)
    filter.SetSize(size2D)
    filter.SetOutputOrigin(origin_2d)
    filter.SetOutputSpacing(spacing2D)
    filter.Update()

    proj_raw_array = itk.GetArrayFromImage(filter.GetOutput())

    if save_image_name is not None:
        proj_neg_array = proj_raw_array.max() - proj_raw_array
        proj_neg_image = itk.GetImageFromArray(proj_neg_array)
        proj_neg_image.SetSpacing(spacing2D)

        writer = itk.ImageFileWriter[image_type_2d].New()
        writer.SetFileName("data/" + save_image_name + ".mhd")
        writer.SetInput(proj_neg_image)
        writer.Update()

        writer = itk.ImageFileWriter[image_type_2d].New()
        writer.SetFileName("data/" + save_image_name + ".tif")
        writer.SetInput(proj_neg_image)
        writer.Update()

        projection_parameters = {}
        projection_parameters['origin2D'] = origin_2d
        projection_parameters['origin3D'] = origin_3d
        projection_parameters['focalPoint'] = focal_point
        with open("data/%s_parameters.pkl" % save_image_name, 'wb') as fp:
            pickle.dump(projection_parameters, fp)

    return proj_raw_array


if __name__ == '__main__':
    # projection = projector("data/liver.nii.gz", "projection")
    # constrast_projection = projector("data/contrast_liver.nii.gz", "constrast_projection")
    #
    # # difference
    # diff_projection = projection - constrast_projection
    # diff_image = sitk.GetImageFromArray(diff_projection)
    # diff_image.SetSpacing(spacing2D)
    # sitk.WriteImage(diff_image, "data/diff_projection.mhd")
    mask_projection = projector("/home/hja/Projects/3D2DRegister/ASOCA/Train_Masks/0.nrrd", None)
    # mask_projection = projector("/home/hja/Projects/3D2DRegister/ASOCA/Train_Masks_STL/0_left.nii", None)
    mask_projection = mask_projection[0] + .0
    print(mask_projection.shape)
    cv2.imshow("projection", mask_projection)
    mask_projection[mask_projection >= 0.5] = 1.0
    mask_projection[mask_projection < 0.5] = .0
    cv2.imshow("projection1", mask_projection)
    cv2.waitKey(0)


