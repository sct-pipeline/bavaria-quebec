## nn-unet based ms spine lesion segmentation

This directory features small utility scripts to allow to train a nn-unet for lesion segmentation,
based on the conventions of the [official nn-unet repository](https://github.com/MIC-DKFZ/nnUNet#run-inference) by Isensee et al.

### Setup the environment

Please setup a virtual environment as described [here](https://github.com/MIC-DKFZ/nnUNet#installation).

### Prepare training data

Input Directory Format:

```
├── CHANGES
├── Correction_segmentation.xlsx
├── dataset_description.json
├── derivatives
│   └── labels
│       ├── sub-m012474
│       │   └── ses-20080925
│       │       └── anat
│       │           ├── sub-m012474_ses-20080925_acq-ax_lesion-manual_T2w.json
│       │           ├── sub-m012474_ses-20080925_acq-ax_lesion-manual_T2w.nii.gz
│       │           ├── sub-m012474_ses-20080925_acq-ax_seg-manual_T2w.json
│       │           └── sub-m012474_ses-20080925_acq-ax_seg-manual_T2w.nii.gz
│       ├── sub-m023917
│       │   └── ses-20130506
│       │       └── anat
│       │           ├── sub-m023917_ses-20130506_acq-ax_lesion-manual_T2w.json
│       │           ├── sub-m023917_ses-20130506_acq-ax_lesion-manual_T2w.nii.gz
│       │           ├── sub-m023917_ses-20130506_acq-ax_seg-manual_T2w.json
│       │           └── sub-m023917_ses-20130506_acq-ax_seg-manual_T2w.nii.gz
...
├── sub-m012474
│   └── ses-20080925
│       └── anat
│       ├── sub-m012474_ses-20080925_acq-ax_lesion-manual_T2w.json
│       ├── sub-m012474_ses-20080925_acq-ax_lesion-manual_T2w.nii.gz
│       ├── sub-m012474_ses-20080925_acq-ax_seg-manual_T2w.json
│       └── sub-m012474_ses-20080925_acq-ax_seg-manual_T2w.nii.gz
...
```

### Preparing the train/test sets for nn-UNet

To prepare training and test set data, please run the following command. Please be aware that this only works for the tested structure,
where we utilize the axial T2w (and optionally) the sagittal T2w as input channels. Other modalities will likely fail.

```
python3 create_sets.py --image_directory /home/datasets/20221104_NeuroPoly_Cohort/ --label_directory /home/datasets/20221104_NeuroPoly_Cohort/ derivatives/labels/ --output_directory . --split_dict ivado-split.json --use_sag_channel

```
Afterwards, we can expect the following data directory structure:

```
conversion_dict_train.json  dataset_train.json  imagesTr  imagesTs  labelsTr labelsTs
```

Please place the folder `Task500_MSBrainLesion` into the raw database:

e.g. /mnt/Drive4/nnunet_paths/nnUNet_raw_data_base/nnUNet_raw_data

### Prepare test data

To create the test set, we have to follow a similar approach:

```
python3 create_testset.py --input_directory /mnt/Drive4/julian/nnunet_ms_lesion/data/achieva/ --output_directory dataset -tn MSBrain_Lesion -tno 500
```

Afterwards, we can expect the following data directory structure:

```
conversion_dict_test.json  dataset_test.json  imagesTr  imagesTs  labelsTr
```

Please place/merge the folder `Task500_MSBrainLesion` into the raw database (same location where the training samples live):

e.g. /mnt/Drive4/nnunet_paths/nnUNet_raw_data_base/nnUNet_raw_data

Now, we can finally run inference on the testset - we trained a 3d_fullres, so please use this model as input model:

```
nnUNet_predict -i $nnUNet_raw_data_base/nnUNet_raw_data/Task500_MSBrainLesion/imagesTs -o /mnt/Drive4/julian/out -t 500 -m 3d_fullres
```
### Evaluation of test set samples

Evaluation of DICE score: (c.f. [url](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/inference_example_Prostate.md))

nnUNet_evaluate_folder -ref FOLDER_WITH_GT -pred FOLDER_WITH_PREDICTIONS -l 1