import os
import shutil
import numpy as np
import SimpleITK as sitk


data_root = r'/home/hja/Projects/3D2DRegister/LiverVesselData'


def translation_stages(str):
    ct_stages = {"动脉": "artery", "静脉": "vein", "平扫": "plain scan", "延迟": "delay"}
    for (k, v) in ct_stages.items():
        if str.__contains__(k):
            return v
        if v == str:
            return v


if __name__ == '__main__':
    for case in os.listdir(data_root):
        print(case)
        case_dir = os.path.join(data_root, case)
        case_ct_folder = list(filter(lambda f: f.__contains__("CT"), os.listdir(case_dir)))[0]
        case_ct_path = os.path.join(case_dir, case_ct_folder)
        for stage in os.listdir(case_ct_path):
            print(stage)
            # rename folder
            stage_eng = translation_stages(stage)
            if os.path.exists(os.path.join(case_ct_path, stage_eng+".nii.gz")):
                continue

            shutil.move(os.path.join(case_ct_path, stage), os.path.join(case_ct_path, stage_eng))

            stage_path = os.path.join(case_ct_path, stage_eng)
            if len(os.listdir(stage_path)) == 1:
                stage_path = os.path.join(stage_path, os.listdir(stage_path)[0])

            # rename slices
            for slice_name in os.listdir(stage_path):
                shutil.move(os.path.join(stage_path, slice_name),
                            os.path.join(stage_path, slice_name.split("_")[-1]))

            # dicom series to nii
            reader = sitk.ImageSeriesReader()
            series_names = reader.GetGDCMSeriesFileNames(stage_path)
            reader.SetFileNames(series_names)
            image = reader.Execute()
            sitk.WriteImage(image, os.path.join(case_dir, stage_eng+".nii.gz"))
