"""
This script evaluates the reference segmentations and model predictions 
using the "animaSegPerfAnalyzer" command

****************************************************************************************
SegPerfAnalyser (Segmentation Performance Analyzer) provides different marks, metrics 
and scores for segmentation evaluation.
3 categories are available:
    - SEGMENTATION EVALUATION:
        Dice, the mean overlap
        Jaccard, the union overlap
        Sensitivity
        Specificity
        NPV (Negative Predictive Value)
        PPV (Positive Predictive Value)
        RVE (Relative Volume Error) in percentage
    - SURFACE DISTANCE EVALUATION:
        Hausdorff distance
        Contour mean distance
        Average surface distance
    - DETECTION LESIONS EVALUATION:
        PPVL (Positive Predictive Value for Lesions)
        SensL, Lesion detection sensitivity
        F1 Score, a F1 Score between PPVL and SensL

Results are provided as follows: 
Jaccard;    Dice;   Sensitivity;    Specificity;    PPV;    NPV;    RelativeVolumeError;    
HausdorffDistance;  ContourMeanDistance;    SurfaceDistance;  PPVL;   SensL;  F1_score;       

NbTestedLesions;    VolTestedLesions;  --> These metrics are computed for images that 
                                            have no lesions in the GT
****************************************************************************************

Mathematical details on how these metrics are computed can be found here:
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6135867/pdf/41598_2018_Article_31911.pdf

and in Section 4 of this paper (for how the subjects with no lesions are handled):
https://portal.fli-iam.irisa.fr/files/2021/06/MS_Challenge_Evaluation_Challengers.pdf

INSTALLATION:
##### STEP 0: Install git lfs via apt if you don't already have it.
##### STEP 1: Install ANIMA #####
cd ~
mkdir anima/
cd anima/
wget -q https://github.com/Inria-Visages/Anima-Public/releases/download/v4.2/Anima-Ubuntu-4.2.zip   (change version to latest)
unzip Anima-Ubuntu-4.2.zip
git lfs install
git clone --depth 1 https://github.com/Inria-Visages/Anima-Scripts-Public.git
git clone --depth 1 https://github.com/Inria-Visages/Anima-Scripts-Data-Public.git
cd ~
mkdir .anima/
touch .anima/config.txt
nano .anima/config.txt

##### STEP 2: Configure directories #####
# Variable names and section titles should stay the same
# Put this file in your HomeFolder/.anima/config.txt
# Make the anima variable point to your Anima public build
# Make the extra-data-root point to the data folder of Anima-Scripts
# The last folder separator for each path is crucial, do not forget them
# Use full paths, nothing relative or using tildes 

[anima-scripts]
anima = /home/<your-user-name>/anima/Anima-Binaries-4.2/
anima-scripts-public-root = /home/<your-user-name>/anima/Anima-Scripts-Public/
extra-data-root = /home/<your-user-name>/anima/Anima-Scripts-Data-Public/

USAGE:
python nnUNet_compute_test_metrics_anima.py --pred_folder <path_to_predictions_folder> 
--gt_folder <path_to_gt_folder> -t_id <task_id> -t_name <task_name> -o <output_folder>

"""

import os
import glob
import subprocess
import argparse
from collections import defaultdict
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import nibabel as nib
from pathlib import Path

# get the ANIMA binaries path
cmd = r'''grep "^anima = " ~/.anima/config.txt | sed "s/.* = //"'''
anima_binaries_path = subprocess.check_output(cmd, shell=True).decode('utf-8').strip('\n')
print('ANIMA Binaries Path:', anima_binaries_path)

# Define arguments
parser = argparse.ArgumentParser(description='Compute test metrics using animaSegPerfAnalyzer')

# Arguments for model, data, and training
parser.add_argument('--pred_folder', required=True, type=str,
                    help='Path to the folder containing nifti images of test predictions')
parser.add_argument('--gt_folder', required=True, type=str,
                    help='Path to the folder containing nifti images of GT labels')                
parser.add_argument('-o', '--output_folder', required=True, type=str,
                    help='Path to the output folder to save the test metrics results')

args = parser.parse_args()

pred_folder, gt_folder = args.pred_folder, args.gt_folder
num_predictions = len(glob.glob(os.path.join(pred_folder, "*.nii.gz")))
num_gts = len(glob.glob(os.path.join(gt_folder, "*.nii.gz"))) 

if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder, exist_ok=True)

# basic checks
assert num_gts == num_predictions, 'Number of predictions and GTs do not match. Please check the folders.'
print(num_gts, "\t", num_predictions)

def get_test_metrics(pred_folder, gt_folder, num_predictions):
    """
    Computes the test metrics given folders containing nifti images of test predictions 
    and GT images by running the "animaSegPerfAnalyzer" command
    """

    pred = sorted(Path(pred_folder).rglob("*.nii.gz"))
    pred = [str(x) for x in pred]
    gt = Path(gt_folder).rglob("*.nii.gz")
    gt = sorted([str(x) for x in gt])

    for idx in range(num_predictions):
        
        # Load the predictions and GTs        
        #pred_file = os.path.join(pred_folder, f"{args.task_name}_{(idx+1):03d}.nii.gz")
        pred_file = pred[idx]
        pred_npy = nib.load(pred_file).get_fdata()
        # make sure the predictions are binary because ANIMA accepts binarized inputs only
        pred_npy = np.array(pred_npy > 0.5, dtype=float)

        #gt_file = os.path.join(gt_folder, f"{args.task_name}_{(idx+1):03d}.nii.gz")
        gt_file = gt[idx]
        gt_npy = nib.load(gt_file).get_fdata()
        # make sure the GT is binary because ANIMA accepts binarized inputs only
        gt_npy = np.array(gt_npy > 0.5, dtype=float)
        # print(((gt_npy==0.0) | (gt_npy==1.0)).all())

        # Save the binarized predictions and GTs
        pred_nib = nib.Nifti1Image(pred_npy, affine=np.eye(4))
        gtc_nib = nib.Nifti1Image(gt_npy, affine=np.eye(4))
        nib.save(img=pred_nib, filename=pred[idx].replace(".nii.gz", "_binarized.nii.gz"))
        nib.save(img=gtc_nib, filename=gt[idx].replace(".nii.gz", "_binarized.nii.gz"))

        # Run ANIMA segmentation performance metrics on the predictions
        # NOTE 1: For checking all the available options run the following command from your terminal: 
        #       <anima_binaries_path>/animaSegPerfAnalyzer -h
        # NOTE 2: We use certain additional arguments below with the following purposes:
        #       -i -> input image, -r -> reference image, -o -> output folder
        #       -d -> evaluates surface distance, -l -> evaluates the detection of lesions
        #       -a -> intra-lesion evalulation (advanced), -s -> segmentation evaluation, 
        #       -X -> save as XML file  -A -> prints details on output metrics and exits
        
        seg_perf_analyzer_cmd = '%s -i %s -r %s -o %s -d -l -a -s -X'
        os.system(seg_perf_analyzer_cmd %
                    (os.path.join(anima_binaries_path, 'animaSegPerfAnalyzer'),
                    pred[idx].replace(".nii.gz", "_binarized.nii.gz"),
                    gt[idx].replace(".nii.gz", "_binarized.nii.gz"),
                    os.path.join(args.output_folder, f"{(idx+1)}")))

        # Delete temporary binarized NIfTI files
        os.remove(pred[idx].replace(".nii.gz", "_binarized.nii.gz"))
        os.remove(gt[idx].replace(".nii.gz", "_binarized.nii.gz"))

    # Get all XML filepaths where ANIMA performance metrics are saved for each hold-out subject
    subject_filepaths = [os.path.join(args.output_folder, f) for f in
                            os.listdir(args.output_folder) if f.endswith('.xml')]
    
    return subject_filepaths
    

# Get all XML filepaths where ANIMA performance metrics are saved for each hold-out subject
subject_filepaths = get_test_metrics(pred_folder, gt_folder, num_predictions)

test_metrics = defaultdict(list)

# Update the test metrics dictionary by iterating over all subjects
for subject_filepath in subject_filepaths:
    # print(subject_filepath)
    subject = os.path.split(subject_filepath)[-1].split('_')[0]
    root_node = ET.parse(source=subject_filepath).getroot()

    # if GT is empty then metrics aren't calculated, hence the only entries in the XML file 
    # NbTestedLesions and VolTestedLesions, both of which are zero. Hence, we can skip subjects
    # with empty GTs by checked if the length of the .xml file is 2
    if len(root_node) == 2:
        print(f"Skipping Subject={int(subject):03d} ENTIRELY Due to Empty GT!")
        continue

    # # Check if RelativeVolumeError is INF -> means the GT is empty and should be ignored
    # rve_metric = list(root_node)[6]
    # assert rve_metric.get('name') == 'RelativeVolumeError'
    # if np.isinf(float(rve_metric.text)):
    #     print('Skipping Subject=%s ENTIRELY Due to Empty GT!' % subject)
    #     continue

    for metric in list(root_node):
        name, value = metric.get('name'), float(metric.text)

        if np.isinf(value) or np.isnan(value):
            print(f'Skipping Metric={name} for Subject={int(subject):03d} Due to INF or NaNs!')
            continue

        test_metrics[name].append(value)


# Print aggregation of each metric via mean and standard dev.
print('Test Phase Metrics [ANIMA]: ')
for key in test_metrics:
    print('\t%s -> Mean: %0.4f Std: %0.2f' % (key, np.mean(test_metrics[key]), np.std(test_metrics[key])))
    
    # save the metrics to a log file
    with open(os.path.join(args.output_folder, 'log.txt'), 'a') as f:
                print("\t%s --> Mean: %0.3f, Std: %0.3f" % 
                        (key, np.mean(test_metrics[key]), np.std(test_metrics[key])), file=f)
