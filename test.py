from dicom_parser import Image
import numpy as np

image = Image(r'E:\Data\SEU_Hospital\DICOM\Li Jian Guo\6\15_14.dcm')
print(image.header.raw)
print(np.array(image.header.raw['ImagePositionPatient'].value))

import SimpleITK as sitk
reader = sitk.ImageSeriesReader()
ct_dcm_series = reader.GetGDCMSeriesFileNames(r'D:\Projects\3D2DRegister\LiverVesselData\Cao ShenFu\CT CaoShenFu  20190307\dongmai\7-20201022111548')
reader.SetFileNames(ct_dcm_series)
image = reader.Execute()
print(image.GetOrigin())
print(image.GetSpacing())
print(sitk.GetArrayFromImage(image).shape)
