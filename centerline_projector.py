import numpy as np
from centeline_tree_reader import construct_tree_from_txt, get_branches_points
from centerline_show import show_branches
import SimpleITK as sitk
import pickle


# def get_projection_parameters():
#     with open("data/projection_parameters.pkl", 'rb') as fp:
#         parameters = pickle.load(fp)
#     focal_point = np.array(parameters['focalPoint'])
#     origin_2d = np.array(parameters['origin2D'])
#     origin_3d = np.array(parameters['origin3D'])
#     # plane_origin = [origin_3d[0], origin_3d[1], origin_2d[2]]
#     plane_normal = origin_3d - focal_point
#     plane_normal /= np.sqrt(np.sum(plane_normal ** 2))
#     print(parameters)
#     return focal_point, origin_2d, plane_normal


def get_projection_parameters(single_dsa_dcm, ct_dcm_folder):
    """
    Get distance of source to detector according to the single dicom of DSA.
    Calculate origin of 3D CT in image coordinate system.
    Then we simulate the position of focal point and detector plane according to above two parameters.
    (Note: The simulation should be improved with more geometry information about C-arm !)
    :param single_dsa_dcm:
    :param ct_dcm_folder:
    :return:
    """
    # get distance of source to detector according to the single dicom of DSA
    from dicom_parser import Image
    image = Image(single_dsa_dcm)
    source_to_detector = float(image.header.raw['DistanceSourceToDetector'].value)

    # get origin of 3D CT
    reader = sitk.ImageSeriesReader()
    ct_dcm_series = reader.GetGDCMSeriesFileNames(ct_dcm_folder)
    reader.SetFileNames(ct_dcm_series)
    image = reader.Execute()
    img_origin = image.GetOrigin()
    img_spacing = image.GetSpacing()
    array = sitk.GetArrayFromImage(image)
    img_size = np.array(array.shape)[[1, 2, 0]]

    # may need improved ???
    origin_3d = [img_spacing[0] * img_size[0] / 2., img_spacing[1] * img_size[1] / 2., img_spacing[2] * img_size[2] / 2.]
    source_point = np.array([origin_3d[0] - (source_to_detector / 2) * 1 / 2,
                             origin_3d[1] + (source_to_detector / 2) * 1 / 2,
                             origin_3d[2] - (source_to_detector / 2) * np.sqrt(2) / 2])
    origin_2d = np.array([origin_3d[0] + (source_to_detector / 2) * 1 / 2,
                          origin_3d[1] - (source_to_detector / 2) * 1 / 2,
                          origin_3d[2] + (source_to_detector / 2) * np.sqrt(2) / 2])
    plane_normal = origin_3d - source_point
    plane_normal /= np.sqrt(np.sum(plane_normal ** 2))

    return source_point, origin_2d, origin_3d, plane_normal, source_to_detector


def project_points(points, source_point, plane_origin, plane_normal):
    """"
    :param points: (N, 3)
    :param source_point: source point
    :param plane_origin: plane origin
    :param plane_normal: plane normal
    :return: (N, 3)
    """
    source_point = np.array(source_point)
    plane_origin = np.array(plane_origin)
    plane_normal = np.array(plane_normal)
    numerator = np.sum((plane_origin - source_point) * plane_normal)
    denominator = np.matmul((points - source_point), plane_normal)
    t = numerator / denominator
    t = np.tile(t, (3, 1)).T
    return source_point + t * (points - source_point)


def project_centerline_points(branches_points, single_dsa_dcm, ct_dcm_folder, tx, ty, tz, rx, ry, rz):
    source_point, origin_2d, origin_3d, plane_normal, source_to_detector = get_projection_parameters(single_dsa_dcm, ct_dcm_folder)
    projected_points = []
    for branch_points in branches_points:
        # pp = trans_rotate_points(np.array(branch_points), source_point, origin_3d, plane_normal, tx, ty, tz, rx, ry, rz)
        pp = rotate_points(branch_points, -30.1, -20.6, 765)
        pp = project_points(pp, [0, 0, 0], [0, 0, source_to_detector], [0, 0, 1])
        # pp = project_points(np.array(branch_points), source_point, origin_2d, plane_normal)
        projected_points.append(pp)

    return projected_points, source_point, origin_2d, plane_normal


def trans_rotate_points(points, source_point, origin_3d, plane_normal, tx, ty, tz, rx, ry, rz):
    """
    move the points coordinate (CT coordinate) to the detector coordinate in which the origin is the focal point
    and the z axis is from the origin to the detector plane (parallel to the plane normal), the coordinate system of
    focal point and plane normal is same as the points coordinate system. Then we transform and rotate points in new CS.
    Note that rotation is around the origin_3d in new CS.
    :param points:
    :param source_point:
    :param origin_3d:
    :param plane_normal:
    :param tx:
    :param ty:
    :param tz:
    :param rx: angle (not radians)
    :param ry:
    :param rz:
    :return:
    """
    # define new coordinate system
    z_normal = plane_normal / np.sqrt(np.sum(plane_normal ** 2))
    # set a random x axis and then set y axis

    x_normal = np.array([1, -z_normal[0] / z_normal[1], 0]) if z_normal[1] != 0 else np.array([0, 1, 0])
    x_normal = x_normal / np.sqrt(np.sum(x_normal ** 2))

    y_normal = np.cross(z_normal, x_normal)
    y_normal = y_normal / np.sqrt(np.sum(y_normal ** 2))

    # move to new origin
    points = np.array(points - source_point)
    origin_3d = np.array(origin_3d - source_point)
    # new coordinate of points
    new_cs = np.array([x_normal, y_normal, z_normal]).T
    points = np.matmul(points, new_cs)
    origin_3d = np.matmul(origin_3d, new_cs)

    # rotate around origin_3d
    rx = rx * np.pi / 180; ry = ry * np.pi / 180; rz = rz * np.pi / 180
    points = (points - origin_3d).T
    points = np.matmul(np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]]), points)  # x
    points = np.matmul(np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]]), points)  # y
    points = np.matmul(np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]]), points)  # z
    points = points.T + origin_3d

    # translate
    points += np.array([tx, ty, tz])

    return points


def rotate_points(points, primary, secondary, SOD):
    R_p = np.array([[np.cos(primary), 0, np.sin(primary)], [0, 1, 0], [-np.sin(primary), 0, np.cos(primary)]])

    new_axis = np.matmul(R_p.T, np.array([[1], [0], [0]]))
    new_axis = new_axis / np.sqrt(np.sum(new_axis ** 2))
    new_coord = np.matmul(new_axis, new_axis.T)

    R_s = new_coord + np.cos(secondary) * (np.eye(3) - new_coord) + \
          np.sin(secondary) * np.array(
        [[0, -new_axis[2], new_axis[1]], [new_axis[2], 0, -new_axis[0]], [-new_axis[1], new_axis[0], 0]])
    R = np.hstack([np.matmul(R_p, R_s), np.array([[0], [0], [-SOD]])])

    bp = np.array(points)
    bp = np.hstack([bp, np.ones((bp.shape[0], 1))]).T
    rp = np.matmul(R, bp)
    return rp.T


def projector(branches_points, primary, secondary, SOD, SID, n_u, n_v):
    primary = primary * np.pi / 180
    secondary = secondary * np.pi / 180

    P = np.array([[-SID, 0, -n_u], [0, -SID, -n_v], [0, 0, 1]])

    R_p = np.array([[np.cos(primary), 0, np.sin(primary)], [0, 1, 0], [-np.sin(primary), 0, np.cos(primary)]])

    new_axis = np.matmul(R_p.T, np.array([[1], [0], [0]]))
    new_axis = new_axis / np.sqrt(np.sum(new_axis ** 2))
    new_coord = np.matmul(new_axis, new_axis.T)

    R_s = new_coord + np.cos(secondary) * (np.eye(3) - new_coord) + \
          np.sin(secondary) * np.array([[0, -new_axis[2], new_axis[1]], [new_axis[2], 0, -new_axis[0]], [-new_axis[1], new_axis[0], 0]])
    R = np.hstack([np.matmul(R_p, R_s), np.array([[0], [0], [-SOD]])])

    projected_points = []
    for branch_points in branches_points:
        bp = np.array(branch_points)
        bp = np.hstack([bp, np.ones((bp.shape[0], 1))]).T
        rp = np.matmul(R, bp)
        pp = np.matmul(P, rp)
        pp = pp / pp[2, :]
        projected_points.append(pp.T)

    return projected_points


if __name__ == '__main__':
    # root, _ = construct_tree_from_txt(r"data/vessel_centerline.txt")
    # root, _ = construct_tree_from_txt("../Data/coronary/CAI_TIE_ZHU/CTA/CAI TIE ZHU_Left.txt", 1, 2, [9])
    root, _ = construct_tree_from_txt("../Data/coronary/CAI_TIE_ZHU/CTA/CAI TIE ZHU_Right.txt", 3, 2, [4])
    branches_points = get_branches_points(root)

    # projected_points = projector(branches_points, 1.5, -22.401, 765, 999, 256, 256)
    # projected_points = projector(branches_points, -30.1, -20.6, 765, 1113, 256, 256)
    # projected_points = projector(branches_points, 41.8, -0.3, 765, 944, 256, 256)
    # show_branches(projected_points, fix_color=False, show_window=True)

    tx = 0; ty = 0; tz = 0; rx = -90; ry = -90; rz = 180
    # for ry in range(90, 91, 30):
    projected_points, source_point, origin_2d, plane_normal = project_centerline_points(branches_points,
                                                                                        r"../LiverVesselData/Cao ShenFu/Cao ShenFu-DSA1  20200225/1/Cao ShenFu-DSA1_10.DCM",
                                                                                        r"../LiverVesselData/Cao ShenFu/CT CaoShenFu  20190307/vein",
                                                                                        tx, ty, tz, rx, ry, rz)
    show_branches_3d(projected_points, fix_color=False, show_window=True,
                  window_save_name="%d-%d-%d-%d-%d-%d.png" % (tx, ty, tz, rx, ry, rz))
