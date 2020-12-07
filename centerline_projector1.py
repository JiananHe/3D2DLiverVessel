import numpy as np
from centeline_tree_reader import construct_tree_from_txt, get_branches_points
from centerline_show import show_branches
import SimpleITK as sitk
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


def projector_main(all_points, tx, ty, tz, rx, ry, rz, SID, SOD, nw, nh):
    origin_3d = np.mean(all_points, axis=0)
    source_point = origin_3d - np.array([0, 0, SOD])

    tr_points = trans_rotate_points(all_points, origin_3d, tx, ty, tz, rx, ry, rz)
    pr_points = project_points(tr_points, source_point, SID)

    return pr_points


if __name__ == '__main__':
    # root, _ = construct_tree_from_txt(r"data/vessel_centerline.txt")
    root1, _ = construct_tree_from_txt("../Data/coronary/CAI_TIE_ZHU/CTA/CAI TIE ZHU_Left.txt", 1, 2, [9])
    root2, _ = construct_tree_from_txt("../Data/coronary/CAI_TIE_ZHU/CTA/CAI TIE ZHU_Right.txt", 3, 2, [4])
    branches_points1, branches_index1 = get_branches_points(root1)
    branches_points2, branches_index2 = get_branches_points(root2)

    rx = 0; ry = 0; rz = 0
    # for rx in range(0, 360, 30):
    #     for ry in range(0, 360, 30):
    #         projected = projector_main(branches_points1, 0, 0, 0, rx, ry, rz, 1000, 765, 512, 512)
    #         show_branches(projected, branches_index1, (512, 512), fix_color=True, show_window=False,
    #                       window_save_name="%d-%d-%d.png" % (rx, ry, rz),
    #                       save_dir="../Data/coronary/CAI_TIE_ZHU/CTA/left_project")
    for rx in range(0, 360, 30):
        for ry in range(0, 360, 30):
            projected = projector_main(branches_points2, 0, 0, 0, rx, ry, rz, 1000, 765, 512, 512)
            show_branches(projected, branches_index2, (512, 512), fix_color=True, show_window=False,
                          save_name="%d-%d-%d.png" % (rx, ry, rz),
                          save_dir="../Data/coronary/CAI_TIE_ZHU/CTA/right_project")
