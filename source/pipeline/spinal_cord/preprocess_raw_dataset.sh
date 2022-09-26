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

# Process images
# ======================================================================================================================

# echo current directory
echo $PWD

# get list of all nifti files except jsons
files=(*)
files=( $( for i in ${files[@]} ; do echo $i ; done | grep '.*T2w.nii.gz'))

for i in "${files[@]}"
do
   sct_image -i $i -set-sform-to-qform
done

# obtain seperate lists for axial chunks, sagittal chunks and axial lesions
sag_files=( $( for i in ${files[@]} ; do echo $i ; done | grep 'acq-sag_chunk.*T2w.nii.gz'))
axial_files=( $( for i in ${files[@]} ; do echo $i ; done | grep 'acq-ax_chunk.*T2w.nii.gz'))
ax_lesion_files=( $( for i in ${files[@]} ; do echo $i ; done | grep 'acq-ax_chunk.*dseg.nii.gz'))
echo ${sag_files[@]}
echo ${axial_files[@]}
echo ${ax_lesion_files[@]}

sct_image -i $sag_files -o stitched.nii.gz -stitch 

# Display useful info for the log
end=`date +%s`
runtime=$((end-start))
echo
echo "~~~"
echo "SCT version: `sct_version`"
echo "Ran on:      `uname -nsr`"
echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "~~~"
