import os
import json
import pandas as pd
import argparse
from tqdm import tqdm
import random
import numpy as np
from fold_generator import FoldGenerator
from loguru import logger
import nibabel as nib

root = "/home/GRAMES.POLYMTL.CA/u114716/duke/temp/muena/bavaria_quebec_cleaned"

parser = argparse.ArgumentParser(description='Code for creating k-fold splits of the bavaria-quebec dataset.')

parser.add_argument('-se', '--seed', default=42, type=int, help="Seed for reproducibility")
parser.add_argument('-ncv', '--num_cv_folds', default=5, type=int, help="To create a k-fold dataset for cross validation")
# NOTE: if cross-validation folds are not desired then use -ncv 0
parser.add_argument('-dr', '--data_root', default=root, type=str, help='Path to the data set directory')

args = parser.parse_args()

save_path = "/home/GRAMES.POLYMTL.CA/u114716/tum-poly/custom_model_3D/datalists"
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)

# Get all subjects
subjects_df = pd.read_csv(os.path.join(args.data_root, 'participants.tsv'), sep='\t')
subjects = subjects_df['participant_id'].values.tolist()
logger.info(f"Found {len(subjects)} subjects in total!")

seed = args.seed
num_cv_folds = args.num_cv_folds    # for 100 subjects, performs a 60-20-20 split with num_cv_plots

if args.num_cv_folds != 0:
    # create k-fold CV datasets as usual

    # returns a nested list of length (num_cv_folds), each element (again, a list) consisting of 
    # train, val, test indices and the fold number
    names_list = FoldGenerator(seed, num_cv_folds, len_data=len(subjects)).get_fold_names()

    for fold in range(num_cv_folds):

        train_ix, val_ix, test_ix, fold_num = names_list[fold]
        training_subjects = [subjects[tr_ix] for tr_ix in train_ix]
        validation_subjects = [subjects[v_ix] for v_ix in val_ix]
        test_subjects = [subjects[te_ix] for te_ix in test_ix]
        # print(training_subjects, "\n",validation_subjects, "\n",test_subjects)

        # keys to be defined in the dataset_0.json
        params = {}
        params["description"] = "Bavaria-Quebec Spine MS Lesion Segmentation"
        params["labels"] = {
            "0": "background",
            "1": "ms-lesion"
            }
        params["license"] = "nk"
        params["modality"] = {
            "0": "MRI"
            }
        params["name"] = f"bavaria-quebec-spine-ms data fold-{fold_num}"
        params["numTest"] = len(test_subjects)
        params["numTraining"] = len(training_subjects) 
        params["numValidation"] = len(validation_subjects)
        params["reference"] = ""
        params["tensorImageSize"] = "3D"

        # train_val_subjects_dict = {
        #     "training": training_subjects,
        #     "validation": validation_subjects,
        # } 
        train_subjects_dict = {"training": training_subjects} 
        val_subjects_dict = {"validation": validation_subjects}
        test_subjects_dict =  {"test": test_subjects}

        all_list = [train_subjects_dict, val_subjects_dict, test_subjects_dict]

        # iterate over all train/val/test dicts
        for subs_dict in all_list:
            logger.info(f"Going over '{list(subs_dict.keys())[0]}' subjects - Fold {fold}")

            temp_shapes_list = []
            for name, subs_list in subs_dict.items():

                temp_list = []
                for subject_no, subject in enumerate(tqdm(subs_list, desc='Loading Volumes')):
                
                    # Another for loop for going through sessions
                    temp_subject_path = os.path.join(root, subject)
                    num_sessions_per_subject = sum(os.path.isdir(os.path.join(temp_subject_path, pth)) for pth in os.listdir(temp_subject_path))
                    session_name = os.listdir(temp_subject_path)[0]

                    for ses_idx in range(1, num_sessions_per_subject+1):
                        temp_data = {}
                        # Get paths with session numbers
                        session = session_name      # 'ses-0' + str(ses_idx)
                        subject_images_path = os.path.join(root, subject, session, 'anat')
                        subject_labels_path = os.path.join(root, 'derivatives', 'labels', subject, session, 'anat')
                        
                        # e.g. file sub-m012474_ses-20080925_acq-ax_T2w.nii.gz
                        subject_image_file = os.path.join(subject_images_path, '%s_%s_acq-ax_T2w.nii.gz' % (subject, session))
                        # e.g. file sub-m012474_ses-20080925_acq-ax_T2w_lesion-manual.nii.gz
                        subject_label_file = os.path.join(subject_labels_path, '%s_%s_acq-ax_T2w_lesion-manual.nii.gz' % (subject, session))

                        # get shapes of each subject to calculate median later
                        temp_shapes_list.append(np.shape(nib.load(subject_image_file).get_fdata()))

                        # store in a temp dictionary
                        temp_data["image"] = subject_image_file #.replace(root+"/", '') # .strip(root)
                        temp_data["label"] = subject_label_file #.replace(root+"/", '') # .strip(root)
                        
                        temp_list.append(temp_data)
                
                params[name] = temp_list
                params[f"{list(subs_dict.keys())[0]}_median_shape"] = np.median(temp_shapes_list, axis=0).tolist()
                params[f"{list(subs_dict.keys())[0]}_min_shape"] = np.min(temp_shapes_list, axis=0).tolist()
                params[f"{list(subs_dict.keys())[0]}_max_shape"] = np.max(temp_shapes_list, axis=0).tolist()

        final_json = json.dumps(params, indent=4, sort_keys=True)
        jsonFile = open(os.path.join(save_path, f"bavaria-qc_fold-{fold_num}.json"), "w")
        # jsonFile = open(args.data_root + "/" + f"dataset_fold-{fold_num}.json", "w")
        jsonFile.write(final_json)
        jsonFile.close()

else:
    # create one json file with 80-20 train-test split
    # Hold-out a fraction of subjects for test phase
    random.seed(seed)
    random.shuffle(subjects)
    fraction_test = 0.2
    test_subjects = subjects[:int(len(subjects) * fraction_test)]
    # print('Hold-out Subjects: ', test_subjects)

    # The rest of the subjects will be used for the train and validation phases
    training_subjects = subjects[int(len(subjects) * fraction_test):]
    # keys to be defined in the dataset_0.json
    params = {}
    params["description"] = "Bavaria-Quebec Spine MS Lesion Segmentation"
    params["labels"] = {
        "0": "background",
        "1": "ms-lesion"
        }
    params["license"] = "nk"
    params["modality"] = {
        "0": "MRI"
        }
    params["name"] = f"bavaria-quebec-spine-ms data"
    params["numTest"] = len(test_subjects)
    params["numTraining"] = len(training_subjects)
    params["reference"] = ""
    params["tensorImageSize"] = "3D"
    
    train_subjects_dict = {"training": training_subjects,} 
    test_subjects_dict =  {"test": test_subjects}

    all_list = [train_subjects_dict, test_subjects_dict]

    # iterate over all train/val/test dicts
    for subs_dict in all_list:
        logger.info(f"Going over '{list(subs_dict.keys())[0]}' subjects")

        # run loop for subjects
        temp_shapes_list = []
        for name, subs_list in subs_dict.items():

            temp_list = []
            for subject_no, subject in enumerate(tqdm(subs_list, desc='Loading Volumes')):
            
                # Another for loop for going through sessions
                temp_subject_path = os.path.join(root, subject)
                num_sessions_per_subject = sum(os.path.isdir(os.path.join(temp_subject_path, pth)) for pth in os.listdir(temp_subject_path))
                session_name = os.listdir(temp_subject_path)[0]

                for ses_idx in range(1, num_sessions_per_subject+1):
                    temp_data = {}
                    # Get paths with session numbers
                    session = session_name      # 'ses-0' + str(ses_idx)
                    subject_images_path = os.path.join(root, subject, session, 'anat')
                    subject_labels_path = os.path.join(root, 'derivatives', 'labels', subject, session, 'anat')

                    subject_image_file = os.path.join(subject_images_path, '%s_%s_acq-ax_T2w.nii.gz' % (subject, session))
                    subject_label_file = os.path.join(subject_labels_path, '%s_%s_acq-ax_T2w_lesion-manual.nii.gz' % (subject, session))

                    # get shapes of each subject to calculate median later
                    temp_shapes_list.append(np.shape(nib.load(subject_image_file).get_fdata()))

                    # store in a temp dictionary
                    temp_data["image"] = subject_image_file #.replace(root+"/", '') # .strip(root)
                    temp_data["label"] = subject_label_file #.replace(root+"/", '') # .strip(root)
                    
                    temp_list.append(temp_data)
            
            params[name] = temp_list


    final_json = json.dumps(params, indent=4, sort_keys=True)
    jsonFile = open(os.path.join(save_path, f"bavaria-qc_0.json"), "w")
    # jsonFile = open(args.data_root + "/" + f"dataset_0.json", "w")
    jsonFile.write(final_json)
    jsonFile.close()


    


