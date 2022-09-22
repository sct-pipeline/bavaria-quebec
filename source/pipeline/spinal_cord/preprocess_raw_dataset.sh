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
# Author: Julian McGinnis

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

# get a list of all sagittal chunks
var = grep -R "*ax_chunk*"
echo $var
# stitch all sagittal stacks together, to one common space
# the resulting file may be named:
# sub-123456_ses-20220101_acq-ax_T2w.nii.gz
# while the chunks may be named:
# sub-123456_ses-20220101_acq-ax_chunk-1_T2w.nii.gz


# get a list of all axial chunks
# here we stitch both, chunks and masks together


# sagittal chunks
## determine number of sagittal chunks
## https://stackoverflow.com/questions/69903324/how-to-count-the-number-of-files-each-pattern-of-a-group-appears-in-a-file


ls $search_path | grep *.txt > filename.txt

# Make sure q/sform are the same
sct_image -i ${file_t1w}.nii.gz -set-sform-to-qform



# T1w
# ======================================================================================================================
file_t1w="${file}_T1w"

# Make sure q/sform are the same
sct_image -i ${file_t1w}.nii.gz -set-sform-to-qform

# Segment spinal cord (only if it does not exist)
segment_if_does_not_exist "${file_t1w}" "t1"
file_t1w_seg="${FILESEG}"

# Create labels in the cord 
label_if_does_not_exist "${file_t1w}" "${file_t1w_seg}"
file_t1w_seg_labeled="${file_t1w_seg}_labeled"
# Compute average CSA between C1 and C2 levels (append across subjects)
sct_process_segmentation -i "${file_t1w_seg}.nii.gz" -vert 1:3 -vertfile ${file_t1w_seg_labeled}.nii.gz \
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
