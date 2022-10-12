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
echo $file

# in case there are no matching files
shopt -s nullglob

# get list of all nifti files except jsons
files=(*nii.gz)

# use qform
for i in "${files[@]}"
do
   sct_image -i $i -set-sform-to-qform
done

sag_files=(*acq-sag_chunk*.nii.gz)
axial_files=(*acq-ax_chunk*T2w.nii.gz)
ax_lesion_files=(*_dseg.nii.gz)

sag_files_json=(*acq-sag_chunk*.json)
axial_files_json=(*acq-ax_chunk*T2w.json)
ax_lesion_files_json=(*_dseg.json)

# convert lesion labels to {0,1} range
# apparently they were not labelled consistently
# they vary e.g. {0,2.47081804} , {0, 2.157}, {0, 2.0} or {0,1}
# probably because different people used different colors?

for i in "${ax_lesion_files[@]}"
do
   sct_image -i $i -type float32 -o $i
   sct_maths -i $i -bin 1e-12 -o $i
done

# obtain the directory where the script is located
# Save script path
#PATH_SCRIPT=$PWD
#SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
#echo $SCRIPT_DIR

SCRIPT_DIR=$2

if (( ${#sag_files[@]} > 1))
then
    sct_image -i ${sag_files[@]} -o "${file}_acq-sag_T2w.nii.gz" -stitch -qc "${PATH_QC}"
    echo $PWD
    python3 ${SCRIPT_DIR}/merge_jsons.py -i ${sag_files_json[@]} -o "${file}_acq-sag_T2w.json"
fi

if (( ${#axial_files[@]} > 1))
then
    sct_image -i ${axial_files[@]} -o "${file}_acq-ax_T2w.nii.gz" -stitch -qc "${PATH_QC}"
    python3 ${SCRIPT_DIR}/merge_jsons.py -i ${axial_files_json[@]} -o "${file}_acq-ax_T2w.json"
    # lesion files
    sct_image -i ${ax_lesion_files[@]} -o "${file}_acq-ax_dseg.nii.gz" -stitch 
    python3 ${SCRIPT_DIR}/merge_jsons.py -i ${ax_lesion_files_json[@]} -o "${file}_acq-ax_dseg.json"

    # stitching introduces interpolation due to resampling function, binarize output
    sct_maths -i "${file}_acq-ax_dseg.nii.gz" -bin 1e-12 -o "${file}_acq-ax_dseg.nii.gz"
fi 

# Display useful info for the log
end=`date +%s`
runtime=$((end-start))
echo
echo "~~~"
echo "SCT version: `sct_version`"
echo "Ran on:      `uname -nsr`"
echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "~~~"
