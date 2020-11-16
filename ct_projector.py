import itk
import SimpleITK as sitk
import numpy as np
import pickle

# parameters
# tx = -10
# ty = 0
# tz = -100
# rx = -90
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
spacing2D = [1., 1., 1.]
focalLength = 1500.
maxStepSize = 4.
minStepSize = 1.
o2Dx = 0
o2Dy = 0
threshold = 0


def projector(image_3d_path, save_image_name):
    # types
    image_type_3d = itk.Image[itk.F, 3]
    image_type_2d = itk.Image[itk.F, 3]
    inter_image_type = itk.Image[itk.F, 3]

    # read 3D image
    imageReader = itk.ImageFileReader[image_type_3d].New()
    imageReader.SetFileName(image_3d_path)

    imageReader.Update()
    image = imageReader.GetOutput()
    array = itk.GetArrayFromImage(image)
    print(array.shape)

    # transform 3D image according to given parameters
    transform = itk.Euler3DTransform[itk.D].New()
    transform.SetComputeZYX(True)
    transform.SetTranslation((tx, ty, tx))
    transform.SetRotation(np.pi / 180.0 * rx, np.pi / 180.0 * ry, np.pi / 180.0 * rz)

    spacing3D = image.GetSpacing()
    region3D = image.GetBufferedRegion()
    size3D = region3D.GetSize()
    origin3D = [spacing3D[0] * size3D[0] / 2., spacing3D[1] * size3D[1] / 2., spacing3D[2] * size3D[2] / 2.]
    center = [cx + origin3D[0], cy + origin3D[1], cz + origin3D[2]]
    transform.SetCenter(center)

    # set attributes of projected plane
    origin2D = [
        origin3D[0] + o2Dx - spacing2D[0] * (size2D[0] - 1) / 2.0,
        origin3D[1] + o2Dy - spacing2D[1] * (size2D[1] - 1) / 2.0,
        origin3D[2] + focalLength / 2.0
    ]
    focal_point = [origin3D[0], origin3D[1], origin3D[2] - focalLength / 2.0]

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
    filter.SetOutputOrigin(origin2D)
    filter.SetOutputSpacing(spacing2D)
    filter.Update()

    proj_raw_array = itk.GetArrayFromImage(filter.GetOutput())
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
    projection_parameters['origin2D'] = origin2D
    projection_parameters['origin3D'] = origin3D
    projection_parameters['focalPoint'] = focal_point
    with open("data/%s_parameters.pkl" % save_image_name, 'wb') as fp:
        pickle.dump(projection_parameters, fp)

    return proj_raw_array


if __name__ == '__main__':
    projection = projector("data/liver.nii.gz", "projection")
    constrast_projection = projector("data/contrast_liver.nii.gz", "constrast_projection")

    # difference
    diff_projection = projection - constrast_projection
    diff_image = sitk.GetImageFromArray(diff_projection)
    diff_image.SetSpacing(spacing2D)
    sitk.WriteImage(diff_image, "data/diff_projection.mhd")
