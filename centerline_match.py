import numpy as np
import cv2
import os
from dsa_segmentation.main import skeletonize_image, show_image
from centerline_projector1 import construct_tree_from_txt, get_branches_points, projector_main, show_branches


def centre_image(image, shape):
    line_points = np.argwhere(image == 255)
    centroid = np.mean(line_points, axis=0)
    offset_line_points = (line_points + np.array(shape) / 2 - centroid).astype(np.int)
    offset_line_points[:, 0][offset_line_points[:, 0] < 0] = 0
    offset_line_points[:, 0][offset_line_points[:, 0] > shape[0]-1] = shape[0]-1
    offset_line_points[:, 1][offset_line_points[:, 1] < 0] = 0
    offset_line_points[:, 1][offset_line_points[:, 1] > shape[1]-1] = shape[1]-1

    offset_image = np.ones(shape, dtype=np.uint8) * 255
    offset_image[(offset_line_points[:, 0], offset_line_points[:, 1])] = 0
    return offset_image


# DSA
# image = cv2.imread("../Data/coronary/CAI_TIE_ZHU/DSA/IM000001_1_seg.jpg", 0)
# image = cv2.imread("../Data/coronary/CAI_TIE_ZHU/DSA/IM000008_1_seg.jpg", 0)
image = cv2.imread("../Data/coronary/CAI_TIE_ZHU/DSA/IM000012_1_seg.jpg", 0)
image[image >= 200] = 255
image[image < 200] = 0

skeleton_image = skeletonize_image(image, dilate=False, show=True)
new_skeleton = centre_image(skeleton_image, (512, 512))

dist_image = cv2.distanceTransform(new_skeleton, cv2.DIST_L2, cv2.DIST_MASK_3)
show_image(image, "origin")
show_image(new_skeleton, "new skeleton")
show_image(dist_image, "distance")

# CTA
# root_dir = "../Data/coronary/CAI_TIE_ZHU/CTA/left_project"
root_dir = "../Data/coronary/CAI_TIE_ZHU/CTA/right_project"

match_distance = {}
for img_file in os.listdir(root_dir):
    img = cv2.imread(os.path.join(root_dir, img_file), 0)
    img[img != 255] = 0
    img[img == 255] = 255
    img = 255 - img

    sk_img = skeletonize_image(img, dilate=False, show=False)
    cr_img = centre_image(sk_img, (512, 512))
    score = np.sum(dist_image[cr_img == 0])
    match_distance[img_file] = score

min_score_file = min(match_distance, key=lambda x: match_distance[x])
show_image(cv2.imread(os.path.join(root_dir, min_score_file), 0), "min score: %s" % min_score_file)
cv2.waitKey(0)
