from dsa_segmentation.vesselness2D import read_dicom, show_image
import cv2


arr1 = read_dicom(r"/home/hja/Projects/3D2DRegister/LiverVesselData/Cao ShenFu/Cao ShenFu-DSA1  20200225/1/Cao ShenFu-DSA1_15.DCM")
arr2 = read_dicom(r"/home/hja/Projects/3D2DRegister/LiverVesselData/Cao ShenFu/Cao ShenFu-DSA1  20200225/1/Cao ShenFu-DSA1_18.DCM")

show_image(arr1, "arr1")
show_image(arr2, "arr2")
show_image(arr2 - arr1, "arr2-arr1")
cv2.waitKey(0)
