#!/bin/bash
#
# Reorient sagittally scanned MRI images to axial orientation, complying to
#
# the stitching convention of the stitching algorithm by Ben Glocker.
#
# Usage:
#   ./reorient.sh <SCAN>
#
# Authors: Julian McGinnis

# Retrieve input params
SCAN=$1

# assemble name
SCAN_NAME=${SCAN%%.*}
FILE_ENDING="_axial_ornt.nii.gz"
FILE_PATH="$SCAN_NAME$FILE_ENDING"

echo "Creating $FILE_PATH"

# get starting time:
start=`date +%s`

echo "Swaping Dimensions"
fslswapdim ${SCAN} LR AP IS ${FILE_PATH}
echo "Forcing neurological orientation"
fslorient -forceneurological ${FILE_PATH}

# Display useful info for the log
end=`date +%s`
runtime=$((end-start))
echo "Duration: $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
