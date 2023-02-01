#!/bin/bash
#
# Preprocess data.
#
# Dependencies (versions):
# - SCT (5.4.0)
#
# Usage:
# sct_run_batch -script preprocess_data.sh -path-data <PATH-TO-DATASET> -path-output <PATH-TO-OUTPUT> -jobs <num-cpu-cores>

# Manual segmentations or labels should be located under:
# PATH_DATA/derivatives/labels/SUBJECT/ses-0X/anat/

# The following global variables are retrieved from the caller sct_run_batch
# but could be overwritten by uncommenting the lines below:
# PATH_DATA_PROCESSED="~/data_processed"
# PATH_RESULTS="~/results"
# PATH_LOG="~/log"
# PATH_QC="~/qc"

# Global variables
CENTERLINE_METHOD="svm"  # method sct_deepseg_sc uses for centerline extraction: 'svm', 'cnn'

# Uncomment for full verbose
set -x

# Immediately exit if error
set -e -o pipefail

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT


# CONVENIENCE FUNCTIONS
# ======================================================================================================================

segment_if_does_not_exist() {
  ###
  #  This function checks if a manual spinal cord segmentation file already exists, then:
  #    - If it does, copy it locally.
  #    - If it doesn't, perform automatic spinal cord segmentation.
  #  This allows you to add manual segmentations on a subject-by-subject basis without disrupting the pipeline.
  ###
  local file="$1"
  local contrast="$2"
  local centerline_method="$3"
  # Update global variable with segmentation file name
  FILESEG="${file}_seg"
  FILESEGMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${FILESEG}-manual.nii.gz"
  echo
  echo "Looking for manual segmentation: $FILESEGMANUAL"
  if [[ -e $FILESEGMANUAL ]]; then
    echo "Found! Using manual segmentation."
    rsync -avzh $FILESEGMANUAL ${FILESEG}.nii.gz
    sct_qc -i ${file}_T2w.nii.gz -s ${FILESEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}
  else
    echo "Not found. Proceeding with automatic segmentation."
    # Segment spinal cord based on the specified centerline method
    if [[ $centerline_method == "cnn" ]]; then
      sct_deepseg_sc -i ${file}_T2w.nii.gz -c $contrast -brain 1 -centerline cnn -qc ${PATH_QC} -qc-subject ${SUBJECT}
    elif [[ $centerline_method == "svm" ]]; then
      sct_deepseg_sc -i ${file}_T2w.nii.gz -c $contrast -centerline svm -qc ${PATH_QC} -qc-subject ${SUBJECT}
    else
      echo "Centerline extraction method = ${centerline_method} is not recognized!"
      exit 1
    fi
  fi
}

# Retrieve input params and other params
SUBJECT=$1

# get starting time:
start=`date +%s`

# SCRIPT STARTS HERE
# ==============================================================================
# Display useful info for the log, such as SCT version, RAM and CPU cores available
sct_check_dependencies -short

# Go to folder where data will be copied and processed
cd $PATH_DATA_PROCESSED

# Copy BIDS-required files to processed data folder (e.g. list of participants)
if [[ ! -f "participants.tsv" ]]; then
  rsync -avzh $PATH_DATA/participants.tsv .
fi
if [[ ! -f "participants.json" ]]; then
  rsync -avzh $PATH_DATA/participants.json .
fi
if [[ ! -f "dataset_description.json" ]]; then
  rsync -avzh $PATH_DATA/dataset_description.json .
fi
if [[ ! -f "README" ]]; then
  rsync -avzh $PATH_DATA/README .
fi

# Copy source images
# Note: we use '/./' in order to include the sub-folder 'ses-0X'
rsync -Ravzh $PATH_DATA/./$SUBJECT .

# Copy segmentation ground truths (GT)
mkdir -p derivatives/labels
rsync -Ravzh $PATH_DATA/derivatives/labels/./$SUBJECT derivatives/labels/.

# Go to subject folder for source images
cd ${SUBJECT}/anat

# Define variables
# We do a substitution '/' --> '_' in case there is a subfolder 'ses-0X/'
file="${SUBJECT//[\/]/_}"

# Add suffix corresponding to the view
# NOTE: removed contrast from here because image filenames have _T2w but labels do not
file=${file}_acq-ax

# Make sure the image metadata is a valid JSON object
if [[ ! -s ${file}_T2w.json ]]; then
  echo "{}" >> ${file}_T2w.json
fi

# Spinal cord segmentation using the T2w contrast
# NOTE: cannot add _T2w suffix here as it corrupts the local 'file' variable. 
# Hence, need to manually add the suffix inside the function (only to the images)
segment_if_does_not_exist ${file} t2 ${CENTERLINE_METHOD}
file_seg="${FILESEG}"

echo "Current Directory:"
echo "$PWD"

# Resample the cropped image to 0.75mm isotropic
sct_resample -i ${file}_T2w.nii.gz -mm 0.75x0.75x0.75 -o ${file}_T2w_res.nii.gz

# sct_deep_seg for non-cropped, but resampled axial images with thick slices using the t2_ax setting
sct_deepseg_lesion -i ${file}_T2w_res.nii.gz -c t2_ax -o lesion_t2ax

# Go to subject folder for segmentation GTs
cd $PATH_DATA_PROCESSED/derivatives/labels/$SUBJECT/anat

# Define variables
file_gt="${file}_lesion-manual"

# Redefine variable for final SC segmentation mask as path changed
file_seg_dil=${PATH_DATA_PROCESSED}/${SUBJECT}/anat/${file_seg}_dilate

# Make sure the first rater metadata is a valid JSON object
if [[ ! -s ${file_gt}.json ]]; then
  echo "{}" >> ${file_gt}.json
fi

# Resample the manual seg to 0.75mm isotropic
sct_resample -i ${file_gt}.nii.gz -mm 0.75x0.75x0.75 -o ${file_gt}_res.nii.gz

# Re-binarize the output segmentation due to the resampling function
sct_maths -i ${file_gt}_res.nii.gz -bin 1e-12 -o ${file_gt}_res.nii.gz

# Go back to the root output path
cd $PATH_OUTPUT

# Create and populate clean data processed folder for training
PATH_DATA_PROCESSED_CLEAN="${PATH_DATA_PROCESSED}_clean"

# Copy over required BIDs files
mkdir -p $PATH_DATA_PROCESSED_CLEAN $PATH_DATA_PROCESSED_CLEAN/${SUBJECT} $PATH_DATA_PROCESSED_CLEAN/${SUBJECT}/anat
rsync -avzh $PATH_DATA_PROCESSED/dataset_description.json $PATH_DATA_PROCESSED_CLEAN/
rsync -avzh $PATH_DATA_PROCESSED/participants.* $PATH_DATA_PROCESSED_CLEAN/
rsync -avzh $PATH_DATA_PROCESSED/README $PATH_DATA_PROCESSED_CLEAN/
rsync -avzh $PATH_DATA_PROCESSED/dataset_description.json $PATH_DATA_PROCESSED_CLEAN/derivatives/

mkdir -p $PATH_DATA_PROCESSED_CLEAN/derivatives $PATH_DATA_PROCESSED_CLEAN/derivatives/sct-5.8-anisotropic $PATH_DATA_PROCESSED_CLEAN/derivatives/sct-5.8-anisotropic/labels 
mkdir -p $PATH_DATA_PROCESSED_CLEAN/derivatives/sct-5.8-anisotropic/labels/${SUBJECT} $PATH_DATA_PROCESSED_CLEAN/derivatives/sct-5.8-anisotropic/labels/${SUBJECT}/anat/

rsync -avzh $PATH_DATA_PROCESSED/${SUBJECT}/anat/${file}_T2w_res.nii.gz $PATH_DATA_PROCESSED_CLEAN/${SUBJECT}/anat/${file}_T2w.nii.gz
rsync -avzh $PATH_DATA_PROCESSED/${SUBJECT}/anat/${file}_T2w.json $PATH_DATA_PROCESSED_CLEAN/${SUBJECT}/anat/${file}_T2w.json
rsync -avzh $PATH_DATA_PROCESSED/derivatives/labels/${SUBJECT}/anat/${file_gt}_res.nii.gz $PATH_DATA_PROCESSED_CLEAN/derivatives/sct-5.8-anisotropic/labels/${SUBJECT}/anat/${file_gt}.nii.gz
rsync -avzh $PATH_DATA_PROCESSED/derivatives/labels/${SUBJECT}/anat/${file_gt}.json $PATH_DATA_PROCESSED_CLEAN/derivatives/sct-5.8-anisotropic/labels/${SUBJECT}/anat/${file_gt}.json
rsync -avzh $PATH_DATA_PROCESSED/${SUBJECT}/anat/lesion_t2ax/${file}_T2w_res_lesionseg.nii.gz $PATH_DATA_PROCESSED_CLEAN/derivatives/sct-5.8-anisotropic/labels/${SUBJECT}/anat/${file}_lesion-processed_T2w.nii.gz

# Display useful info for the log
end=`date +%s`
runtime=$((end-start))
echo
echo "~~~"
echo "SCT version: `sct_version`"
echo "Ran on:      `uname -nsr`"
echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "~~~"
