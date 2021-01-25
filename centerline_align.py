import numpy as np
import cv2
import os
from dsa_segmentation.main import skeletonize_image, show_image
from centerline_projector import construct_tree_from_txt, get_branches_points, projector_main, show_branches_2d


def centre_centerline(image):
    line_points = np.argwhere(image == 255)
    lh = line_points[:, 0].max() - line_points[:, 0].min() + 1
    lw = line_points[:, 1].max() - line_points[:, 1].min() + 1

    shape = image.shape
    centroid = np.mean(line_points, axis=0)
    offset = np.array(shape) / 2 - centroid
    offset_line_points = (line_points + offset).astype(np.int)
    offset_line_points[:, 0][offset_line_points[:, 0] < 0] = 0
    offset_line_points[:, 0][offset_line_points[:, 0] > shape[0]-1] = shape[0]-1
    offset_line_points[:, 1][offset_line_points[:, 1] < 0] = 0
    offset_line_points[:, 1][offset_line_points[:, 1] > shape[1]-1] = shape[1]-1

    offset_image = np.ones(shape, dtype=np.uint8) * 255
    offset_image[(offset_line_points[:, 0], offset_line_points[:, 1])] = 0
    return offset_image, (lh, lw), offset


def scale_centerline(image, lh, lw, height, width):
    """
    scale the centerline image after move its centroid to the image center
    :param image:
    :param lh: height of the centerline before centring
    :param lw: width of the centerline before centring
    :param height: height of the DSA centerline before centring
    :param width:  height of the DSA centerline before centring
    :return:
    """
    scale = min(height/lh, width/lw)
    s1 = image.shape

    image = cv2.resize(image, (int(s1[0] * scale + 0.5), int(s1[1] * scale + 0.5)),
                       interpolation=cv2.INTER_NEAREST)
    s2 = image.shape

    scaled_image = np.zeros(s1, dtype=np.uint8)
    if scale >= 1:
        scaled_image = image[(s2[0]-s1[0])//2:(s2[0]-s1[0])//2+s1[0], (s2[1]-s1[1])//2:(s2[1]-s1[1])//2+s1[1]]
    else:
        scaled_image[(s1[0]-s2[0])//2:(s1[0]-s2[0])//2+s2[0], (s1[1]-s2[1])//2:(s1[1]-s2[1])//2+s2[1]] = image

    return scaled_image, scale



# DSA
# image = cv2.imread("../Data/coronary/CAI_TIE_ZHU/DSA/IM000001_1_seg.jpg", 0)
image = cv2.imread("../Data/coronary/CAI_TIE_ZHU/DSA/IM000008_1_seg.jpg", 0)
# image = cv2.imread("../Data/coronary/CAI_TIE_ZHU/DSA/IM000012_1_seg.jpg", 0)
image[image >= 200] = 255
image[image < 200] = 0
skeleton_image = skeletonize_image(image, dilate=False, show=False)

centre_skeleton, (dsa_height, dsa_width), dsa_offset = centre_centerline(skeleton_image)
dist_image = cv2.distanceTransform(centre_skeleton, cv2.DIST_L2, cv2.DIST_MASK_3)
show_image(centre_skeleton, "centered skeleton")
show_image(dist_image, "distance")

# CTA
root1, _ = construct_tree_from_txt("../Data/coronary/CAI_TIE_ZHU/CTA/CAI TIE ZHU_Left.txt", 1, 2, [9])
root2, _ = construct_tree_from_txt("../Data/coronary/CAI_TIE_ZHU/CTA/CAI TIE ZHU_Right.txt", 3, 2, [4])
branches_points1, branches_index1 = get_branches_points(root1)
branches_points2, branches_index2 = get_branches_points(root2)

min_score = 10000
opt_trans = None
tx = 0; ty = 0; tz = 0; rx = 0; ry = 0; rz = 0
for rx in range(0, 360, 30):
    for ry in range(0, 360, 30):
        for rz in range(0, 360, 30):
            plane_centerline = projector_main(branches_points1, branches_index1, tx, ty, tz, rx, ry, rz,
                                              1000, 765, 512, 512, 0.37, 0.37)
            # show_branches_2d(plane_centerline, dsa_image="../Data/coronary/CAI_TIE_ZHU/DSA/IM000001_1.jpg",
            #                  dsa_segment="../Data/coronary/CAI_TIE_ZHU/DSA/IM000001_1_seg.jpg")

            centered_image, (lh, lw), lf = centre_centerline(plane_centerline)
            scaled_image, scale = scale_centerline(centered_image, lh, lw, dsa_height, dsa_width)
            score = np.sum(dist_image[scaled_image == 0]) / len(np.argwhere(scaled_image == 0))

            print("[%d, %d, %d] - scale:%.3f, score:%.3f" % (rx, ry, rz, scale, score))
            # show_image(scaled_image, "centered and scaled image")
            # cv2.waitKey(0)

            if min_score > score:
                min_score = score
                opt_trans = [rx, ry, rz, (lf[0] - dsa_offset[0])*0.37, (lf[1] - dsa_offset[1])*0.37]

print("min score:", min_score, "opt: ", opt_trans)
plane_centerline = projector_main(branches_points1, branches_index1, opt_trans[3], opt_trans[4], tz,
                                  opt_trans[0], opt_trans[1], opt_trans[2], 1000, 765, 512, 512, 0.37, 0.37)
show_branches_2d(plane_centerline)
show_branches_2d(plane_centerline, dsa_image="../Data/coronary/CAI_TIE_ZHU/DSA/IM000008_1.jpg",
                 dsa_segment="../Data/coronary/CAI_TIE_ZHU/DSA/IM000008_1_seg.jpg")

import scipy.io as sio
sio.savemat("./data/IM000008_1_optimal_projection.mat", {"points": np.argwhere(plane_centerline == 255)})
