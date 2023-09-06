"""
Converts the nnUNet-format bavaria-quebec-spine-ms and sct_deepseg_lesion datasets to MSD-style JSON format for MONAI training. 

Usage example:
    python create_msd_datalist.py -pd ~/nnunet-v2/nnUNet_raw/Dataset301_MSspineBQDeepSegRPI -po ../datalists
                    --contrast ax_t2w --seed 42
"""


import os
import json
from tqdm import tqdm
import numpy as np
import argparse
from loguru import logger
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser(description='Code for creating k-fold splits of the spine-generic dataset.')

parser.add_argument('-pd', '--path-data', required=True, type=str, 
                    help='Path to the data set directory. Must contain the imagesTr, labelsTr, imagesTs and labelsTs folders.')
# parser.add_argument('-pj', '--path-joblib', help='Path to joblib file from ivadomed containing the dataset splits.',
#                     default=None, type=str)
parser.add_argument('-po', '--path-out', type=str, help='Path to the output directory where dataset json is saved')
parser.add_argument("--contrast", default="ax_t2w", type=str, help="Contrast to use for training", 
                    choices=["ax_t2w", "sag_t2w"])
# parser.add_argument('--label-type', default='soft', type=str, help="Type of labels to use for training",
#                     choices=['hard', 'soft'])
parser.add_argument('--seed', default=42, type=int, help="Seed for reproducibility")
args = parser.parse_args()


root = args.path_data
seed = args.seed
contrast = args.contrast

# global variables
# PATH_DERIVATIVES = os.path.join(root, "derivatives", "labels")
NNUNET_DATASET_PREFIX = root.split('/')[-1].split("_")[1]

# Get all subjects from the nnUNet_raw folder
# NOTE: we're taking subjects from labelsXX folders because they don't have the annoying _0000 suffix. 
# those will be add later in the code
train_subjects = [subject for subject in os.listdir(os.path.join(root, "labelsTr")) if subject.endswith('.nii.gz')]
test_subjects_bq = [subject for subject in os.listdir(os.path.join(root, "labelsTsBQ")) if subject.endswith('.nii.gz')]
test_subjects_ds = [subject for subject in os.listdir(os.path.join(root, "labelsTsDeepSeg")) if subject.endswith('.nii.gz')]

# remove the nnunet dataset name prefix from the subject names
train_subjects = [sub.replace(f"{NNUNET_DATASET_PREFIX}_",'') for sub in train_subjects]
test_subjects_bq = [sub.replace(f"{NNUNET_DATASET_PREFIX}_",'') for sub in test_subjects_bq]
test_subjects_ds = [sub.replace(f"{NNUNET_DATASET_PREFIX}_",'') for sub in test_subjects_ds]

# logger.info(f"Total number of training subjects: {len(train_subjects)}")
# logger.info(f"Total number of testing subjects (Bavaria): {len(test_subjects_bq)}")
# logger.info(f"Total number of testing subjects (DeepSeg Lesion): {len(test_subjects_ds)}")

# split the training subjects into train/val splits
train_ratio, val_ratio = 0.8, 0.2
train_subjects, val_subjects = train_test_split(train_subjects, test_size=val_ratio, random_state=args.seed)
# # Use the training split to further split into training and validation splits
# train_subjects, val_subjects = train_test_split(train_subjects, test_size=val_ratio / (train_ratio + val_ratio),
#                                                 random_state=args.seed, )

# logger.info(f"Number of training SUBJECTS: {len(train_subjects)}")
# logger.info(f"Number of validation SUBJECTS: {len(val_subjects)}")

# keys to be defined in the dataset_0.json
params = {}
params["description"] = "bavaria-quebec__deepseg-lesion"
params["labels"] = {
    "0": "background",
    "1": "sc-seg",
    "2": "lesion-seg"
    }
params["license"] = "nk"
params["modality"] = {
    "0": "MRI"
    }
params["name"] = "bavaria-quebec__deepseg-lesion"
params["numTestBQ"] = len(test_subjects_bq)
params["numTestDeepSeg"] = len(test_subjects_ds)
params["numTraining"] = len(train_subjects)
params["numValidation"] = len(val_subjects)
params["seed"] = args.seed
params["reference"] = "TUM/Polytechnique-Montreal"
params["tensorImageSize"] = "3D"

train_subjects_dict = {"train": train_subjects}
val_subjects_dict = {"validation": val_subjects}
test_subjects_dict_bq =  {"test_bavaria": test_subjects_bq}
test_subjects_dict_ds =  {"test_deepseg": test_subjects_ds}

all_subjects_list = [train_subjects_dict, val_subjects_dict, test_subjects_dict_bq, test_subjects_dict_ds]

for subjects_dict in tqdm(all_subjects_list, desc="Iterating through train/val/test splits"):

    for name, subs_list in subjects_dict.items():

        if name in ["train", "validation"]:
            
            temp_list = []
            for subject_no, subject in enumerate(subs_list):
            
                temp_data = {}
                subject = subject.replace(".nii.gz", "")
                temp_data["image"] = os.path.join(root, "imagesTr", f"{NNUNET_DATASET_PREFIX}_{subject}_0000.nii.gz")
                temp_data["label"] = os.path.join(root, "labelsTr", f"{NNUNET_DATASET_PREFIX}_{subject}.nii.gz")
                if os.path.exists(temp_data["label"]) and os.path.exists(temp_data["image"]):
                    temp_list.append(temp_data)
                else:
                    logger.info(f"Subject {subject} does not have the image or label.")
        
        elif name == "test_bavaria":

            temp_list = []
            for subject_no, subject in enumerate(subs_list):
            
                temp_data = {}
                subject = subject.replace(".nii.gz", "")
                temp_data["image"] = os.path.join(root, "imagesTsBQ", f"{NNUNET_DATASET_PREFIX}_{subject}_0000.nii.gz")
                temp_data["label"] = os.path.join(root, "labelsTsBQ", f"{NNUNET_DATASET_PREFIX}_{subject}.nii.gz")
                if os.path.exists(temp_data["label"]) and os.path.exists(temp_data["image"]):
                    temp_list.append(temp_data)
                else:
                    logger.info(f"Subject {subject} does not have the image or label.")
            
        elif name == "test_deepseg":

            temp_list = []
            for subject_no, subject in enumerate(subs_list):
            
                temp_data = {}
                subject = subject.replace(".nii.gz", "")
                temp_data["image"] = os.path.join(root, "imagesTsDeepSeg", f"{NNUNET_DATASET_PREFIX}_{subject}_0000.nii.gz")
                temp_data["label"] = os.path.join(root, "labelsTsDeepSeg", f"{NNUNET_DATASET_PREFIX}_{subject}.nii.gz")
                if os.path.exists(temp_data["label"]) and os.path.exists(temp_data["image"]):
                    temp_list.append(temp_data)
                else:
                    logger.info(f"Subject {subject} does not have image or label.")

        
        params[name] = temp_list
        logger.info(f"Number of IMAGES in {name} set: {len(temp_list)}")

final_json = json.dumps(params, indent=4, sort_keys=True)
if not os.path.exists(args.path_out):
    os.makedirs(args.path_out, exist_ok=True)

jsonFile = open(args.path_out + "/" + f"dataset_bavaria-deepseg_seed{seed}.json", "w")
jsonFile.write(final_json)
jsonFile.close()



    


