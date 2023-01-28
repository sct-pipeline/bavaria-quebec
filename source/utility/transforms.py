
from monai.transforms import (AddChanneld, Compose, CropForegroundd, LoadImaged, RandFlipd, 
            RandCropByPosNegLabeld, Spacingd, RandRotate90d, ToTensord, NormalizeIntensityd, 
            EnsureType, RandWeightedCropd, HistogramNormalized, EnsureTyped, Invertd, SaveImaged,
            EnsureChannelFirstd, CenterSpatialCropd, RandSpatialCropSamplesd, )


def train_transforms(crop_size, num_samples_pv, lbl_key="label"):
    return Compose([   
            LoadImaged(keys=["image", lbl_key]),
            EnsureChannelFirstd(keys=["image", lbl_key]),
            CropForegroundd(keys=["image", lbl_key], source_key="image"),     # crops >0 values with a bounding box
            CenterSpatialCropd(keys=["image", lbl_key], roi_size=crop_size),  # crops the center of the image
            # RandSpatialCropSamplesd(keys=["image", lbl_key], roi_size=crop_size, num_samples=num_samples_pv, random_center=True, random_size=False),
            RandFlipd(keys=["image", lbl_key], spatial_axis=[0], prob=0.50,),
            RandFlipd(keys=["image", lbl_key], spatial_axis=[1], prob=0.50,),
            RandFlipd(keys=["image", lbl_key],spatial_axis=[2],prob=0.50,),
            RandRotate90d(keys=["image", lbl_key], prob=0.10, max_k=3,),
            HistogramNormalized(keys=["image"], mask=None),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
            ToTensord(keys=["image", lbl_key]), 
        ])

def val_transforms(lbl_key="label"):
    return Compose([
            LoadImaged(keys=["image", lbl_key]),
            EnsureChannelFirstd(keys=["image", lbl_key]),
            CropForegroundd(keys=["image", lbl_key], source_key="image"),
            HistogramNormalized(keys=["image"], mask=None),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
            ToTensord(keys=["image", lbl_key]),
        ])

def test_transforms(lbl_key="label"):
    return Compose([
            LoadImaged(keys=["image", lbl_key]),
            EnsureChannelFirstd(keys=["image", lbl_key]),
            AddChanneld(keys=["image", lbl_key]),
            HistogramNormalized(keys=["image"], mask=None),  
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
            ToTensord(keys=["image", lbl_key]),
        ])