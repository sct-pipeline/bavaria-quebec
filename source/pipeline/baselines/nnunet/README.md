## nn-unet based ms spine lesion segmentation

This directory features small utility scripts to allow to train a nn-unet for lesion segmentation,
based on the naming conventions of the [official nn-unet repository](https://github.com/MIC-DKFZ/nnUNet#run-inference) by Isensee et al.

### Setup the environment and paths

Please setup a conda environment as described [here](https://github.com/MIC-DKFZ/nnUNet#installation). 
For this project, we use option 3ii) which installs the framework from the cloned repository. Furthermore, we recommend to install the hiddenlayer python library to visualize the network topologies.

Moreover, to function properly, nnunet requires certain environment variables to be set. Follow the [guide](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md) to set these.

### Prepare training data structure

Our current database / cohort adheres to the BIDS standard. The following structure is generated after running the [SCT-based preprocessing script](https://github.com/sct-pipeline/bavaria-quebec/blob/main/preprocessing/preprocess_data.sh).

Input Directory Format:

```
├── dataset_description.json
├── derivatives
│   ├── dataset_description.json
│   └── labels
│       ├── sub-m012474
│       │   └── ses-20080925
│       │       └── anat
│       │           ├── sub-m012474_ses-20080925_acq-ax_lesion-manual_T2w.json
│       │           └── sub-m012474_ses-20080925_acq-ax_lesion-manual_T2w.nii.gz
│       ├── sub-m023917
│       │   └── ses-20130506
│       │       └── anat
│       │           ├── sub-m023917_ses-20130506_acq-ax_lesion-manual_T2w.json
│       │           └── sub-m023917_ses-20130506_acq-ax_lesion-manual_T2w.nii.gz
...
|── participants.json
├── participants.tsv
├── README
├── sub-m012474
│   └── ses-20080925
│       └── anat
│           ├── sub-m012474_ses-20080925_acq-ax_T2w.json
│           └── sub-m012474_ses-20080925_acq-ax_T2w.nii.gz
├── sub-m023917
│   └── ses-20130506
│       └── anat
│           ├── sub-m023917_ses-20130506_acq-ax_T2w.json
│           └── sub-m023917_ses-20130506_acq-ax_T2w.nii.gz
...
```
However, nn-unet requires the following dataset structure, that we need to place in: \
`nnunet_paths/nnUNet_raw_data_base/nnUNet_raw_data/Task501_MSSpineLesionPreprocessedAxialOnly` \

It looks like this:

```
├── conversion_dict_sagittal_channel_False.json
├── dataset.json
├── imagesTr
│   ├── MSSpineLesionPreprocessedAxialOnly_001_0000.nii.gz
│   ├── MSSpineLesionPreprocessedAxialOnly_002_0000.nii.gz
...
├── imagesTs
│   ├── MSSpineLesionPreprocessedAxialOnly_001_0000.nii.gz
│   ├── MSSpineLesionPreprocessedAxialOnly_002_0000.nii.gz
...
├── labelsTr
│   ├── MSSpineLesionPreprocessedAxialOnly_001.nii.gz
│   ├── MSSpineLesionPreprocessedAxialOnly_002.nii.gz
...
├── labelsTs
│   ├── MSSpineLesionPreprocessedAxialOnly_001.nii.gz
│   ├── MSSpineLesionPreprocessedAxialOnly_002.nii.gz
...
```

### Preparing the train/test sets for nn-UNet

To prepare training and test set data, please run the following command. Please be aware that this only works for the tested (preprocessed) structure,
where we utilize the axial T2w (and optionally) the sagittal T2w as input channels. Other modalities will likely fail.

```
python3 create_sets.py --image_directory /home/datasets/20221104_NeuroPoly_Cohort/ --label_directory /home/datasets/20221104_NeuroPoly_Cohort/ derivatives/labels/ --output_directory . --split_dict ivado-split.json --use_sag_channel
```
Afterwards, we need to copy the generated directory `Task501_MSSpineLesionPreprocessedAxialOnly/` to the raw dataset directory of nnunet's working directory.

### Starting the training

nn-unet provides different options for training: [2d, 3d_lowres, 3d_fullres, 3d_cascade]. Prepare the training by running the following command:

`nnUNet_plan_and_preprocess -t 501 -tl 32 -tf 32 --verify_dataset_integrity`, and specify the number of workers via `-tl` and `-tf`.

In this case, we use 2d and 3d_fullres options, which need to be independently trained for all (5) folds.

`
CUDA_VISIBLE_DEVICES=6 nnUNet_train 3d_fullres nnUnetTrainV2 501 3 --npz
`

Here 501 corresponds to the task id you assigned during the train/test set generation, and 3 to the fold id (run the same instructions for all folds in [0,1,2,3,4]).

### Resuming the training after N (50, 100, ...) epochs

As nn-unet takes a while to train, it's very practical that nnunet automatically saves checkpoints every 50 epochs. To continue training, simply add the `-c` option. E.g.

`nnUNet_plan_and_preprocess -t 501 -tl 32 -tf 32 --verify_dataset_integrity -c`

### Choosing the best model

Run `nnUNet_find_best_configuration` to identify the best model based on five-fold cross validation. However, this also requires you having trained all five folds! You can also disable ensembling via `--disable_ensembling`.


### Prepare test data

Now, we can finally run inference on the testset - we trained a 3d_fullres, so e.g. we can use this model as input model:

```
nnUNet_predict -i $nnUNet_raw_data_base/nnUNet_raw_data/Task500_MSBrainLesion/imagesTs -o /mnt/Drive4/julian/out -t 500 -m 3d_fullres
```
### Evaluation of test set 
As we will compare the performance of the nn-unet to other models, and want to use the same evaluation script on all generated masks,
we omit the nn-unet specific evaluation and analysis.

If you are curious how nn-unet does it, please refer to the following command and link:
`nnUNet_evaluate_folder -ref FOLDER_WITH_GT -pred FOLDER_WITH_PREDICTIONS -l 1`
Evaluation of DICE score: (c.f. [url](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/inference_example_Prostate.md))
