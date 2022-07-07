#!/bin/bash
#
# Process data. This script is designed to be run in the folder for a single subject, however 'sct_run_batch' can be
# used to run this script multiple times in parallel across a multi-subject BIDS dataset.
#
# Usage:
#   ./process_data.sh <SUBJECT>
#
# Example:
#   ./process_data.sh sub-03
#
# Author: Julien Cohen-Adad

# The following global variables are retrieved from the caller sct_run_batch
# but could be overwritten by uncommenting the lines below:
# PATH_DATA_PROCESSED="~/data_processed"
# PATH_RESULTS="~/results"
# PATH_LOG="~/log"
# PATH_QC="~/qc"


# BASH SETTINGS
# ======================================================================================================================

# Uncomment for full verbose
set -x

# Immediately exit if error
set -e

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT


# CONVENIENCE FUNCTIONS
# ======================================================================================================================

label_if_does_not_exist() {
  ###
  #  This function checks if a manual label file already exists, then:
  #     - If it does, copy it locally.
  #     - If it doesn't, perform automatic labeling.
  #   This allows you to add manual labels on a subject-by-subject basis without disrupting the pipeline.
  ###
  local file="$1"
  local file_seg="$2"
  # Update global variable with segmentation file name
  FILELABEL="${file}_seg_labeled"
  FILELABELMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECTSESSION}/anat/${FILELABEL}-manual.nii.gz"
  echo "Looking for manual label: $FILELABELMANUAL"
  if [[ -e $FILELABELMANUAL ]]; then
    echo "Found! Using manual labels."
    rsync -avzh $FILELABELMANUAL ${FILELABEL}.nii.gz
  else
    echo "Not found. Proceeding with automatic labeling."
    # Generate labeled segmentation
    #sct_image -i ${file}.nii.gz -header
    #sct_image -i ${file_seg}.nii.gz -header
    # sct_image -i 
    sct_image -i ${file}.nii.gz -set-sform-to-qform
    sct_image -i ${file_seg}.nii.gz -set-sform-to-qform
    sct_label_vertebrae -i ${file}.nii.gz -s ${file_seg}.nii.gz -c t1 -qc "${PATH_QC}" -qc-subject "${SUBJECTSESSION}"
    # Create labels in the cord at C3 and C5 mid-vertebral levels
    # sct_label_utils -i ${file_seg}_labeled.nii.gz -vert-body 3,5 -o ${FILELABEL}.nii.gz
  fi
}

segment_if_does_not_exist() {
  ###
  #  This function checks if a manual spinal cord segmentation file already exists, then:
  #    - If it does, copy it locally.
  #    - If it doesn't, perform automatic spinal cord segmentation.
  #  This allows you to add manual segmentations on a subject-by-subject basis without disrupting the pipeline.
  ###
  local file="$1"
  local contrast="$2"
  # Update global variable with segmentation file name
  FILESEG="${file}_seg"
  FILESEGMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECTSESSION}/anat/${FILESEG}-manual.nii.gz"
  echo
  echo "Looking for manual segmentation: $FILESEGMANUAL"
  if [[ -e $FILESEGMANUAL ]]; then
    echo "Found! Using manual segmentation."
    rsync -avzh $FILESEGMANUAL ${FILESEG}.nii.gz
    sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECTSESSION}
  else
    echo "Not found. Proceeding with automatic segmentation."
    # Segment spinal cord
    sct_deepseg_sc -i ${file}.nii.gz -c $contrast -qc ${PATH_QC} -qc-subject ${SUBJECTSESSION}
  fi
}

# SCRIPT STARTS HERE
# ======================================================================================================================

# Retrieve input params
SUBJECTSESSION=$1
echo $SUBJECTSESSION

# get starting time:
start=`date +%s`

# Display useful info for the log, such as SCT version, RAM and CPU cores available
sct_check_dependencies -short

# Go to folder where data will be copied and processed
cd $PATH_DATA_PROCESSED

# Copy source images
# We use '/./' in order to include the sub-folder 'ses-0X'
rsync -Ravzh $PATH_DATA/./$SUBJECTSESSION .

# Go to subject folder for source images
cd ${SUBJECTSESSION}/anat

# Define variables
# We do a substitution '/' --> '_' in case there is a subfolder 'ses-0X/'
file="${SUBJECTSESSION//[\/]/_}"


# T1w
# ======================================================================================================================
file_t1w="${file}_T1w"
# Segment spinal cord (only if it does not exist)
segment_if_does_not_exist "${file_t1w}" "t1"
file_t1w_seg="${FILESEG}"

# Create labels in the cord 
label_if_does_not_exist "${file_t1w}" "${file_t1w_seg}"
file_label="${FILELABEL}"
# Register to template
sct_register_to_template -i "${file_t1w}.nii.gz" -s "${file_t1w_seg}.nii.gz" -l "${file_label}.nii.gz" -c t1 \
#                         -param step=1,type=seg,algo=centermassrot:step=2,type=im,algo=syn,iter=5,slicewise=1,metric=CC,smooth=0 \
                         -qc "${PATH_QC}"
# Warp template
# Note: we don't need the white matter atlas at this point, therefore use flag "-a 0"
sct_warp_template -d "${file_t1w}.nii.gz" -w warp_template2anat.nii.gz -a 0 -ofolder label_t1w -qc "${PATH_QC}"
# Compute average CSA between C1 and C2 levels (append across subjects)
sct_process_segmentation -i "${file_t1w_seg}.nii.gz" -vert 1:3 -vertfile label_t1w/template/PAM50_levels.nii.gz \
                         -o "${PATH_RESULTS}/CSA.csv" -append 1 -qc "${PATH_QC}"



FILES_TO_CHECK=(
  "$file_t1w_seg.nii.gz"
)
for file in "${FILES_TO_CHECK[@]}"; do
  if [ ! -e "${file}" ]; then
    echo "${SUBJECTSESSION}/${file} does not exist" >> "${PATH_LOG}/error.log"
  fi
done

# Display useful info for the log
end=`date +%s`
runtime=$((end-start))
echo
echo "~~~"
echo "SCT version: `sct_version`"
echo "Ran on:      `uname -nsr`"
echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "~~~"
