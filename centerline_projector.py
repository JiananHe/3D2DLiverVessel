import numpy as np
from centeline_tree_reader import construct_tree_from_txt, get_branches_points
from centerline_show import show_branches_3d, show_branches_2d
import SimpleITK as sitk
import cv2
from dicom_parser import Image


"""
    以所有的point的坐标的重心为3d原点origin_3d，然后从DSA的头文件中读取SOD距离，SID距离与探测平板的长宽(nw, nh);
    (origin_3d)_z - SOD设为X射线源的坐标source；建立新的坐标系：以source为原点，source指向origin_3d为z轴，根据z轴随机建立x轴与y轴；
    把points的坐标变化到新的坐标系内，然后进行旋转平移和投影
    旋转：把3D point以origin_3d为旋转中心，绕着新坐标系的xyz轴旋转；平移：沿着新坐标系的xyz轴平移；投影：根据SID与探测平板的长宽构建投影矩阵。
"""


def trans_rotate_points(points, origin_3d, tx, ty, tz, rx, ry, rz):
    # rotate around origin_3d
    rx = rx * np.pi / 180
    ry = ry * np.pi / 180
    rz = rz * np.pi / 180
    points = (points - origin_3d).T
    points = np.matmul(np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]]), points)  # x
    points = np.matmul(np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]]), points)  # y
    points = np.matmul(np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]]), points)  # z
    points = points.T + origin_3d

    # translate
    points += np.array([tx, ty, tz])
    return points


def project_points(points, source, SID):
    vector_to_source = points - source
    t = np.tile(SID / vector_to_source[:, 2], (3, 1)).T
    points = source + t * vector_to_source

    return points


def get_plane_centerline(branches_points, branches_index, plane_centre, plane_size, plane_spacing):
    # plane_centre --> plane_size//2
    points_coord = (branches_points - plane_centre)[:, :2] / plane_spacing + np.array(plane_size) // 2
    points_coord = np.array(np.round(points_coord), dtype=np.int)
    points_coord[:, 0][points_coord[:, 0] < 0] = 0
    points_coord[:, 0][points_coord[:, 0] > plane_size[0] - 1] = plane_size[0] - 1
    points_coord[:, 1][points_coord[:, 1] < 0] = 0
    points_coord[:, 1][points_coord[:, 1] > plane_size[1] - 1] = plane_size[1] - 1

    plane_centerline = np.zeros(plane_size, dtype=np.uint8)
    sid = 0
    points_coord[:, [0, 1]] = points_coord[:, [1, 0]]  # exchange x and y to suit opencv
    for bid in branches_index:
        branch_coord = points_coord[sid: sid+bid]
        sid += bid

        pid = 0
        while pid < len(branch_coord) - 1:
            cv2.line(plane_centerline, tuple(branch_coord[pid]), tuple(branch_coord[pid+1]), 255)
            pid += 1

    return plane_centerline


def projector_main(branches_points, branches_index, tx, ty, tz, rx, ry, rz, SID, SOD, nw, nh, sw, sh):
    """
    Project 3d centerline points onto the detector plane, the coordinate system is the origin 3d cs from CT.
    The rotate centre is the origin 3d (mean coordinate of all 3d points).
    :param branches_index:
    :param branches_points: points in 3D coordinate system
    :param tx: translation along x axis
    :param ty: translation along y axis
    :param tz: translation along z axis
    :param rx: rotate round x axis
    :param ry: rotate round y axis
    :param rz: rotate round z axis
    :param SID: the distance from the X-ray source to the detector plane
    :param SOD: the distance from the X-ray source to the patient (origin 3d along the z axis)
    :param nw: plane width
    :param nh: plane height
    :param sw: plane width spacing
    :param sh: plane height spacing
    :return: the centerline in the detector plane
    """
    origin_3d = np.mean(branches_points, axis=0)
    source_point = origin_3d - np.array([0, 0, SOD])

    tr_points = trans_rotate_points(branches_points, origin_3d, tx, ty, tz, rx, ry, rz)
    pr_points = project_points(tr_points, source_point, SID)
    pr_source = source_point + np.array([0, 0, SID])
    plane_centerline = get_plane_centerline(pr_points, branches_index, pr_source, (nh, nw), (sh, sw))

    return plane_centerline


if __name__ == '__main__':
    # root, _ = construct_tree_from_txt(r"data/vessel_centerline.txt")
    root1, _ = construct_tree_from_txt("../Data/coronary/CAI_TIE_ZHU/CTA/CAI TIE ZHU_Left.txt", 1, 2, [9])
    root2, _ = construct_tree_from_txt("../Data/coronary/CAI_TIE_ZHU/CTA/CAI TIE ZHU_Right.txt", 3, 2, [4])
    branches_points1, branches_index1 = get_branches_points(root1)
    branches_points2, branches_index2 = get_branches_points(root2)

    # for rx in range(0, 360, 30):
    #     for ry in range(0, 360, 30):
    #         projected = projector_main(branches_points1, 0, 0, 0, rx, ry, rz, 1000, 765, 512, 512)
    #         show_branches(projected, branches_index1, (512, 512), fix_color=True, show_window=False,
    #                       window_save_name="%d-%d-%d.png" % (rx, ry, rz),
    #                       save_dir="../Data/coronary/CAI_TIE_ZHU/CTA/left_project")

    tx=0; ty=0; tz=0; rx=1.5; ry=0; rz=90
    # for rx in [-30.1, 90-30.1, 180-30.1, 30.1, 90+30.1, 180+30.1]:
    #     for ry in [-20.6, 90-20.6, 180-20.6, 20.6, 90+20.6, 180+20.6]:

    for rx in [1.5, 90+1.5, 180+1.5, -1.5, 90-1.5, 180-1.5]:
        for ry in [-22.4, 90-22.4, 180-22.4, 22.4, 90+22.4, 180+22.4]:
            print(rx, ry)
            plane_centerline = projector_main(branches_points1, branches_index1, tx, ty, tz, rx, ry, rz,
                                              1000, 765, 512, 512, 0.37, 0.37)
            show_branches_2d(plane_centerline, dsa_image="../Data/coronary/CAI_TIE_ZHU/DSA/IM000001_1.jpg",
                             dsa_segment="../Data/coronary/CAI_TIE_ZHU/DSA/IM000001_1_seg.jpg")
