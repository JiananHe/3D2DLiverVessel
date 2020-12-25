import math
import numpy as np
import itk
import itkwidgets
from itkwidgets import view
from ipywidgets import interactive
import ipywidgets as widgets

# 加载原始体数据并显示
input_name = '/home/hja/Projects/3D2DRegister/ASOCA/Train_Masks/0.nrrd'
volume_lung = itk.imread(input_name, itk.ctype('float'))  # 读取影像文件，并将数据格式转换为float
print(volume_lung.GetLargestPossibleRegion().GetSize())
print(volume_lung.GetBufferedRegion().GetSize())
print(volume_lung.GetSpacing())
print(volume_lung.GetOrigin())
view(volume_lung, gradient_opacity=0.5, cmp=itkwidgets.cm.bone)

output_image_pixel_spacing = [0.37, 0.37, 1]
output_image_size = [512, 512, 1]  # [501, 501, 1]

InputImageType = type(volume_lung)
FilterType = itk.ResampleImageFilter[InputImageType, InputImageType]
filter = FilterType.New()
filter.SetInput(volume_lung)
filter.SetDefaultPixelValue(0)
filter.SetSize(output_image_size)
filter.SetOutputSpacing(output_image_pixel_spacing)

TransformType = itk.CenteredEuler3DTransform[itk.D]
transform = TransformType.New()
transform.SetComputeZYX(True)

InterpolatorType = itk.RayCastInterpolateImageFunction[InputImageType, itk.D]
interpolator = InterpolatorType.New()

viewer = None


def DigitallyReconstructedRadiograph(
        ray_source_distance=100,
        camera_tx=0.,
        camera_ty=0.,
        camera_tz=0.,
        rotation_x=0.,
        rotation_y=0.,
        rotation_z=0.,
        projection_normal_p_x=0.,
        projection_normal_p_y=0.,
        rotation_center_rt_volume_center_x=0.,
        rotation_center_rt_volume_center_y=0.,
        rotation_center_rt_volume_center_z=0.,
        threshold=0.,
):
    """
    Parameters description:

    ray_source_distance = 400                              # <-sid float>            Distance of ray source (focal point) focal point 400mm
    camera_translation_parameter = [0., 0., 0.]            # <-t float float float>  Translation parameter of the camera
    rotation_around_xyz = [0., 0., 0.]                     # <-rx float>             Rotation around x,y,z axis in degrees
    projection_normal_position = [0, 0]                    # <-normal float float>   The 2D projection normal position [default: 0x0mm]
    rotation_center_relative_to_volume_center = [0, 0, 0]  # <-cor float float float> The centre of rotation relative to centre of volume
    threshold = 10                                          # <-threshold float>      Threshold [default: 0]
    """

    dgree_to_radius_coef = 1. / 180. * math.pi
    camera_translation_parameter = [camera_tx, camera_ty, camera_tz]
    rotation_around_xyz = [rotation_x * dgree_to_radius_coef, rotation_y * dgree_to_radius_coef,
                           rotation_z * dgree_to_radius_coef]
    projection_normal_position = [projection_normal_p_x, projection_normal_p_y]
    rotation_center_relative_to_volume_center = [
        rotation_center_rt_volume_center_x,
        rotation_center_rt_volume_center_y,
        rotation_center_rt_volume_center_z
    ]

    imageOrigin = volume_lung.GetOrigin()
    imageSpacing = volume_lung.GetSpacing()
    imageRegion = volume_lung.GetBufferedRegion()
    imageSize = imageRegion.GetSize()
    imageCenter = [imageOrigin[i] + imageSpacing[i] * imageSize[i] / 2.0 for i in range(3)]

    transform.SetTranslation(camera_translation_parameter)
    transform.SetRotation(rotation_around_xyz[0], rotation_around_xyz[1], rotation_around_xyz[2])

    center = [c + imageCenter[i] for i, c in enumerate(rotation_center_relative_to_volume_center)]
    transform.SetCenter(center)

    interpolator.SetTransform(transform)
    interpolator.SetThreshold(threshold)
    focalPoint = [imageCenter[0], imageCenter[1], imageCenter[2] - ray_source_distance / 2.0]
    interpolator.SetFocalPoint(focalPoint)

    filter.SetInterpolator(interpolator)
    filter.SetTransform(transform)

    origin = [
        imageCenter[0] + projection_normal_position[0] - output_image_pixel_spacing[0] * (
                    output_image_size[0] - 1) / 2.,
        imageCenter[1] + projection_normal_position[1] - output_image_pixel_spacing[1] * (
                    output_image_size[1] - 1) / 2.,
        imageCenter[2] + imageSpacing[2] * imageSize[2]
    ]

    filter.SetOutputOrigin(origin)
    filter.Update()

    global viewer
    if viewer is None:
        viewer = view(filter.GetOutput(), mode='z')
    else:
        print("Update viewer image")
        viewer.image = filter.GetOutput()

    # print informations
    print("Volume image informations:")
    print("tvolume image origin : ", imageOrigin)
    print("tvolume image size   : ", imageSize)
    print("tvolume image spacing: ", imageSpacing)
    print("tvolume image center : ", imageCenter)
    print("Transform informations:")
    print("ttranslation         : ", camera_translation_parameter)
    print("trotation            : ", rotation_around_xyz)
    print("tcenter               : ", center)
    print("Interpolator informations: ")
    print("tthreshold           : ", threshold)
    print("tfocalPoint          : ", focalPoint)
    print("Filter informations:")
    print("toutput origin        : ", origin)


DigitallyReconstructedRadiograph()

slider = interactive(
    DigitallyReconstructedRadiograph,
    ray_source_distance=(0, 800, 50),
    camera_tx=(-400, 400,10),
    camera_ty=(-400, 400,10),
    camera_tz=(-400, 400,10),
    rotation_x=(-45,45,1),
    rotation_y=(-45,45,1),
    rotation_z=(-45,45,1),
    projection_normal_p_x=(-100,100,1),
    projection_normal_p_y=(-100,100,1),
    rotation_center_rt_volume_center_x=(-100,100,1),
    rotation_center_rt_volume_center_y=(-100,100,1),
    rotation_center_rt_volume_center_z=(-100,100,1),
)

widgets.VBox([viewer, slider])
