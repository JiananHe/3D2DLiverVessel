import cv2
import SimpleITK as sitk
import numpy as np
from dicom_parser import Image
from skimage import morphology
from skimage.measure import label, regionprops
from dsa_segmentation.vesselness2D import *


def read_dicom(path):
    dcm_img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(dcm_img).squeeze()

    image = Image(path)
    window_center = image.header.raw['WindowCenter'].value
    window_width = image.header.raw['WindowWidth'].value
    min_grey = window_center-window_width//2
    max_grey = window_center+window_width//2
    arr = np.clip(arr, a_min=min_grey, a_max=max_grey)
    return arr


def bothat_filter(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    closed_img = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    show_image(closed_img, "closed")
    bothat_img = closed_img - image
    show_image(bothat_img, "bothat")
    return bothat_img


def smooth_image(image):
    smooth_image = cv2.GaussianBlur(image, (7, 7), 5)
    show_image(smooth_image, "smooth")
    return smooth_image


def thresh_vesselness(vesselness, thresh):
    _, threshed = cv2.threshold(vesselness, thresh, 1.0, cv2.THRESH_BINARY)
    return threshed


def largest_connect_component(image, area_thresh):
    lbl_img, num = label(image, return_num=True)
    props = regionprops(lbl_img)

    areas = np.array([prop.area for prop in props])
    sort_label = np.argsort(-areas)

    lbl_img[lbl_img != sort_label[1] + 1] = 0
    lbl_img[lbl_img == sort_label[1] + 1] = 255
    lbl_img = np.array(lbl_img, np.uint8)

    # max_label = 0
    # max_area = 0
    # for prop in props:
    #     if prop.area < area_thresh:
    #         lbl_img[lbl_img == prop.label] = 0
    #     if prop.area > max_area:
    #         max_area = prop.area
    #         max_label = prop.label
    #
    # # 去掉面积最大的白色外边框
    # lbl_img[lbl_img == max_label] = 0

    return lbl_img


def skeletonize_image(image, dilate=True, show=True):
    # dilate
    if dilate:
        image = cv2.dilate(image, np.ones((3, 3), np.uint8), iterations=1)

    # skeletonize
    dilated_image = np.array(image / 255, np.uint8)
    skeletonized_image = morphology.skeletonize(dilated_image) + .0

    if show:
        cv2.imshow("skeletonized", skeletonized_image)

    return np.array(skeletonized_image*255, np.uint8)


def show_image(image, name):
    image = (image - image.min()) / (image.max() - image.min())
    cv2.imshow(name, image)


def save_as_bmp(image, name):
    bmp_image = (image - image.min()) / (image.max() - image.min())
    bpm_image = np.array(bmp_image * 255, np.uint8)
    cv2.imwrite(PROCESS_ROOT+name+".bmp", bpm_image)


def show_image_with_centerline(image, centerline, name):
    image = (image - image.min()) / (image.max() - image.min())
    rgb_image = cv2.cvtColor(np.array(image * 255, np.uint8), cv2.COLOR_GRAY2RGB)
    rgb_image[centerline == 255] = np.array([255, 255, 0])
    cv2.imshow(name, rgb_image)


if __name__ == "__main__":
    # 读取图片
    # I = imread(IMAGE_ROOT)
    # I = mpimg.imread(IMAGE_ROOT)  # 读取和代码处于同一目录下的 lena.png

    I = read_dicom(r'/home/hja/Projects/3D2DRegister/LiverVesselData/Cao ShenFu/Cao ShenFu-DSA2/1/Cao ShenFu-DSA2_40.DCM')
    # I0 = np.reshape(I, I.size)
    I0 = np.copy(I)
    show_image(I, "original")

    # 均衡和归一化（这里的阈值在后面不用）
    thr = np.percentile(I0[I0 > 0], 1) * 0.9  # 【妈的这个结果为啥和matlab不一样啊搞不懂】
    Ip = np.copy(I)  # 用深复制，不然原图也改了
    Ip[Ip < thr] = thr
    print(Ip.min())
    Ip = Ip - Ip.min()
    Ip = Ip / Ip.max()
    show_image(Ip, "normalized")
    save_as_bmp(Ip, "normalized")

    # Ip = bothat_filter(Ip)
    # Ip = smooth_image(Ip)

    # 计算
    V1 = vesselness2D(Ip, np.arange(2, 6.0, 1), [1, 1], 1, False)

    show_image(V1, "tau=1")

    V1_thresh = thresh_vesselness(V1, 0.6)
    show_image(V1_thresh, "tau=1 thresh")

    lcc_img = largest_connect_component(V1_thresh, 1200)
    show_image(lcc_img, "lcc")

    centerline = skeletonize_image(lcc_img)
    show_image_with_centerline(I, centerline, "overlay")

    cv2.waitKey(0)
