#!/bin/bash

# Uncomment for full verbose
# set -x

# Immediately exit if error
set -e -o pipefail

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# Go to the folder where original data is stored
ROOT_DIR=/home/GRAMES.POLYMTL.CA/u114716/tum-poly/auto3dseg
PATH_DEST_DIR=${ROOT_DIR}/data_tmpdir/bavaria-spine-ms-new
PATH_DATALIST=${ROOT_DIR}/datalists

# input arguments
PATH_DATA_DIR=$1
PATH_AUTOSEG_OUT=$2

# check if path_dest_dir exists
if [ ! -d "$PATH_DEST_DIR" ]; then
    echo "Directory $PATH_DEST_DIR does not exist. Creating it."
    mkdir -p $PATH_DEST_DIR
    mkdir -p $PATH_DEST_DIR/derivatives/labels
fi

# iterate over all folders
for subject in $PATH_DATA_DIR/*; do

    # get the subject name
    subject_name=$(basename $subject)
    echo "Processing subject: $subject_name"

    # create the output folder
    mkdir -p $PATH_DEST_DIR/$subject_name

    # iterate over all session subfolders
    for session in $PATH_DATA_DIR/$subject_name/*; do

        # get the session name
        session_name=$(basename $session)
        echo "      Processing session: $session_name"

        # create the output folder
        mkdir -p $PATH_DEST_DIR/$subject_name/$session_name/anat

        # enter the session folder
        cd $PATH_DEST_DIR/$subject_name/$session_name/anat

        # get the file names
        file="${subject_name}_${session_name}_acq-ax_T2w.nii.gz"
        file_sc_seg="${subject_name}_${session_name}_seg-manual_T2w.nii.gz"
        file_lesion_seg="${subject_name}_${session_name}_lesions-manual_T2w.nii.gz"

        # check if the file exists else echo error
        if [ -f "$PATH_DATA_DIR/$subject_name/$session_name/anat/$file" ]; then
            # using rsync to copy the file
            rsync -az $PATH_DATA_DIR/$subject_name/$session_name/anat/$file $PATH_DEST_DIR/$subject_name/$session_name/anat/$file
        else
            echo "File $file does not exist. Skipping subject."
        fi

        # create the output folder
        mkdir -p $PATH_DEST_DIR/derivatives/labels/$subject_name/$session_name/anat

        # check if the lesion file exists else echo error
        if [ -f "$PATH_DATA_DIR/$subject_name/$session_name/anat/$file_lesion_seg" ]; then
            # copy lesion and sc seg files
            rsync -az $PATH_DATA_DIR/$subject_name/$session_name/anat/$file_sc_seg $PATH_DEST_DIR/derivatives/labels/$subject_name/$session_name/anat/${file_sc_seg/_seg-manual_T2w/_acq-ax_T2w_seg-manual}
            rsync -az $PATH_DATA_DIR/$subject_name/$session_name/anat/$file_lesion_seg $PATH_DEST_DIR/derivatives/labels/$subject_name/$session_name/anat/${file_lesion_seg/_lesions-manual_T2w/_acq-ax_T2w_lesion-manual}
        else
            echo "File $file_lesion_seg does not exist. Copying only the SC Segmentation."
            if [ ! -f "$PATH_DATA_DIR/$subject_name/$session_name/anat/$file_sc_seg" ]; then
                echo "File $file_sc_seg does not exist. Skipping subject."
            else
                # copy only the sc seg file
                rsync -az $PATH_DATA_DIR/$subject_name/$session_name/anat/$file_sc_seg $PATH_DEST_DIR/derivatives/labels/$subject_name/$session_name/anat/${file_sc_seg/_seg-manual_T2w/_acq-ax_T2w_seg-manual}
            fi
        fi
    done

    cd $PATH_DEST_DIR/$subject_name
done

echo "----------------------------------------"
echo "Copied dataset to $PATH_DEST_DIR. "
echo "Creating MSD-style datalist..."
echo "----------------------------------------"

# run the python script to create the datalist
python $ROOT_DIR/dataset_conversion/create_msd_datalist_bavaria-spine-ms.py --dataroot $PATH_DEST_DIR --out $PATH_DATALIST --multiclass

echo "----------------------------------------"
echo "Datalist created at $PATH_DATALIST."
echo "Starting Auto3DSeg training..."
echo "----------------------------------------"

# run the training script
# inputs to the training script: path to dataroot, path to datalist, path to output folder
python $ROOT_DIR/run_autoseg.py --dataroot $PATH_DEST_DIR --datalist $PATH_DATALIST/msd_bavaria_new.json --out $PATH_AUTOSEG_OUT

echo "----------------------------------------"
echo "Auto3DSeg training completed."
echo "----------------------------------------"

# clear the tmp directory
rm -r $PATH_DEST_DIR

