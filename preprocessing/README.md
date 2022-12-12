This folder contains the files required for preprocessing the data. The preprocessing script (using `Spinal Cord Toolbox v5.4`) is `preprocess_data.sh`. The Quality Control (QC) file is `qc_preprocess.py`. The file `exclude.yml` contains the list of subjects excluded from the dataset along with the reason for doing so. 

## Preprocessing Steps

The flow of `preprocess_data.sh` is as follows:

1. A variable for the contrast to be preprocessed is defined (currently only axial scans `acq-ax` are preprocessed)
2. Checks whether the spinal cord (SC) segmentation mask exists within the dataset, if yes, uses the manual SC mask, if not then automatically segments the SC using `sct_deepseg_sc`.
3. The SC mask is dilated and then the original image is cropped with `sct_crop_image` using the dilated SC mask as the reference.
4. The (now) cropped axial image is resampled to an isotropic resolution `0.75 mm`.
5. The same process is repeated for cropping the SC lesion mask. 

Note, the cropping is done around the entire region of the SC (so as to reduce size of the input images) and not just around the SC lesion. 

### Running the Preprocessing Script
The `sct_run_batch` is used to perform the preprocessing on all subjects with following command:
```
sct_run_batch -script preprocess_data.sh -path-data PATH_DATA -path-output PATH_OUT -jobs 64
```

If you intend to run the preprocessing only a few subjects (in this case, 4), then use the following command:
```
sct_run_batch -script preprocess_data.sh -path-data PATH_DATA -path-output PATH_OUT -jobs 64 -include-list sub-mXXXXXX/ses-XXXXXXXX sub-mXXXXXX/ses-XXXXXXXX sub-mXXXXXX/ses-XXXXXXXX sub-mXXXXXX/ses-XXXXXXXX
```
where `PATH_DATA` is the path to the BIDS dataset folder, and `PATH_OUT` is where the output of preprocessing will be saved to. `PATH_OUT` will contain the folders `data_processed/` and `qc/` among others after a successful run.

### Running the QC Script
After a successful preprocessing run, the next step is to do the QC to check for errors:
```
python qc_preprocess.py -s PATH_OUT
```
The `PATH_OUT` is the same output path used while running `sct_run_batch`.

This QC script checks whether:

1. the resolutions match between the two original sessions,
2. the image and GT sizes are equivalent for each subject,
3. the isotropic-resampling worked as expected,
4. and most importantly, whether cropping of the original image with the SC mask erroneously crops out any SC lesions.

NOTE: A mistake in the annotation process can result in lesions appearing _outside_ the SC. In such cases, those subjects were excluded from the dataset. More information on those subjects can be found in the `exclude.yml` file.