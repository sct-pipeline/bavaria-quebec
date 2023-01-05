import argparse
import pathlib
from pathlib import Path
import json
import os
import shutil
from collections import OrderedDict

import nibabel as nib
import numpy as np

# this script is employed to generate the nn-Unet based dataset format
# as described in this readme: 
# https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md

# we assume a BIDS-compliant directory structure:
# e.g. 
# image directory:
# 20221104_NeuroPoly_Cohort_work_in_progress/sub-m052556/ses-20130710/anat/
# sub-m052556_ses-20130710_acq-ax_T2w.nii.gz
# 
# 20221104_NeuroPoly_Cohort_work_in_progress/sub-m052556/ses-20130710/anat/
# sub-m052556_ses-20130710_acq-sag_T2w.nii.gz
# 
# label directory: 
# 20221104_NeuroPoly_Cohort_work_in_progress/derivatives/labels/sub-m052556/ses-20130710/anat/
# sub-m052556_ses-20130710_acq-ax_lesion-manual_T2w.nii.gz
# 
# it will likely fail for all other structures!


# parse command line arguments
parser = argparse.ArgumentParser(description='Convert BIDS-structured database to nn-unet format.')
parser.add_argument('--image_directory', help='Path to BIDS structured database.', required=True)
parser.add_argument('--label_directory', help='Path to BIDS structured database, the _label_ directory should be included.', required=True)
parser.add_argument('--output_directory', help='Path to output directory.', required=True)
parser.add_argument('--taskname', help='Specify the task name, e.g. Hippocampus', default='MSSpineLesion', type=str)
parser.add_argument('--tasknumber', help='Specify the task number, has to be greater than 500', default=501,type=int)
parser.add_argument('--split_dict', help='Specify the splits using ivadomed dict, expecting a json file.', required=True)
parser.add_argument('--use_sag_channel', action='store_true', help='Use sagittal image (no label) as a second input channel.')

args = parser.parse_args()

path_in_images = Path(args.image_directory)
path_in_labels = Path(args.label_directory)
path_out = Path(os.path.join(os.path.abspath(args.output_directory), f'Task{args.tasknumber}_{args.taskname}'))
path_out_imagesTr = Path(os.path.join(path_out, 'imagesTr'))
path_out_imagesTs = Path(os.path.join(path_out, 'imagesTs'))
path_out_labelsTr = Path(os.path.join(path_out, 'labelsTr'))
path_out_labelsTs = Path(os.path.join(path_out, 'labelsTs'))

# we load both train and validation set into the train images as nnunet uses cross-fold-validation
train_image_ax = []
train_image_sag = []
train_image_labels = []
test_image_ax = []
test_image_sag = []
test_image_labels = []

if __name__ == '__main__':

    # make the directories
    pathlib.Path(path_out).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_imagesTr).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_imagesTs).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_labelsTr).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_labelsTs).mkdir(parents=True, exist_ok=True)

    conversion_dict = {}
    dirs = sorted(list(path_in_images.glob('*/')))
    dirs = [str(x) for x in dirs]
    
    # filter out derivatives directory for raw images
    # ignore MAC .DS_Store files

    dirs = [k for k in dirs if 'sub' in k]
    dirs = [k for k in dirs if 'derivatives' not in k]
    dirs = [k for k in dirs if '.DS' not in k]

    scan_cnt_train = 0
    scan_cnt_test = 0

    with open(args.split_dict) as f:
        splits = json.load(f)

    valid_train_imgs = []
    valid_test_imgs = []
    valid_train_imgs.append(splits["train"])
    valid_train_imgs.append(splits["valid"])
    valid_test_imgs.append(splits["test"])

    # flatten the lists
    valid_train_imgs =[item for sublist in valid_train_imgs for item in sublist] 
    valid_test_imgs =[item for sublist in valid_test_imgs for item in sublist] 

    # assert number of training set images is equivalent to ivadomed
    for dir in dirs:  # iterate over subdirs
        # glob the session directories
        subdirs = sorted(list(Path(dir).glob('*')))
        for subdir in subdirs:
            ax_file = sorted(list(subdir.rglob('*acq-ax_T2w.nii.gz')))[0]

            if args.use_sag_channel:
                sag_file = sorted(list(subdir.rglob('*acq-sag_T2w.nii.gz')))[0]
                common = os.path.commonpath([ax_file, sag_file])

            else:
                common = os.path.dirname(ax_file)

            common = os.path.relpath(common, args.image_directory)

            # find the corresponding segmentation file
            seg_path = os.path.join(args.label_directory, common)
            seg_file = sorted(list(Path(seg_path).rglob('*lesion*.nii.gz')))[0]

            assert os.path.isfile(seg_file), 'No segmentation mask with this name!'

            if any(str(Path(ax_file).name) in word for word in valid_train_imgs) or any(str(Path(ax_file).name) in word for word in valid_test_imgs):

                if any(str(Path(ax_file).name) in word for word in valid_train_imgs):

                    scan_cnt_train+= 1
                    # create the new convention names
                    ax_file_nnunet = os.path.join(path_out_imagesTr,f'{args.taskname}_{scan_cnt_train:03d}_0000.nii.gz')
                    seg_file_nnunet = os.path.join(path_out_labelsTr,f'{args.taskname}_{scan_cnt_train:03d}.nii.gz')

                    train_image_ax.append(str(ax_file_nnunet))
                    train_image_labels.append(str(seg_file_nnunet))

                    # copy the files to new structure
                    shutil.copyfile(ax_file, ax_file_nnunet)
                    shutil.copyfile(seg_file, seg_file_nnunet)

                    # TO DO - put this in preprocessing routines!
                    # replace the label header with the image header, and binarize label!
                    image = nib.load(seg_file_nnunet)
                    data = image.get_fdata()
                    threshold = 1e-12
                    data = np.where(data > threshold, 1, 0)
                    ref = nib.load(ax_file_nnunet)
                    new_image = nib.Nifti1Image(data, ref.affine, ref.header)
                    nib.save(new_image, seg_file_nnunet)


                    conversion_dict[str(os.path.abspath(ax_file))] = ax_file_nnunet
                    conversion_dict[str(os.path.abspath(seg_file))] = seg_file_nnunet

                    if args.use_sag_channel:
                        sag_file_nnunet = os.path.join(path_out_imagesTr,f'{args.taskname}_{scan_cnt_train:03d}_0001.nii.gz')
                        train_image_sag.append(str(sag_file_nnunet))
                        shutil.copyfile(sag_file, sag_file_nnunet)
                        conversion_dict[str(os.path.abspath(sag_file))] = sag_file_nnunet

                else:
                    scan_cnt_test += 1
                    # create the new convention names
                    ax_file_nnunet = os.path.join(path_out_imagesTs,f'{args.taskname}_{scan_cnt_test:03d}_0000.nii.gz')
                    seg_file_nnunet = os.path.join(path_out_labelsTs,f'{args.taskname}_{scan_cnt_test:03d}.nii.gz')

                    test_image_ax.append(str(ax_file_nnunet))
                    test_image_labels.append(str(seg_file_nnunet))

                    # copy the files to new structure
                    shutil.copyfile(ax_file, ax_file_nnunet)
                    shutil.copyfile(seg_file, seg_file_nnunet)

                    conversion_dict[str(os.path.abspath(ax_file))] = ax_file_nnunet
                    conversion_dict[str(os.path.abspath(seg_file))] = seg_file_nnunet

                    if args.use_sag_channel:
                        sag_file_nnunet = os.path.join(path_out_imagesTs,f'{args.taskname}_{scan_cnt_test:03d}_0001.nii.gz')
                        test_image_sag.append(str(sag_file_nnunet))
                        shutil.copyfile(sag_file, sag_file_nnunet)
                        conversion_dict[str(os.path.abspath(sag_file))] = sag_file_nnunet
           
            else:
                print("Skipping file, could not be located in the specified split.", ax_file)

    assert scan_cnt_train == len(valid_train_imgs), 'No. of train/val images does not correspond to ivadomed dict.'
    assert scan_cnt_test == len(valid_test_imgs), 'No. of test images does not correspond to ivadomed dict.'

    # create dataset_description.json
    json_object = json.dumps(conversion_dict, indent=4)
    # write to dataset description
    conversion_dict_name = f"conversion_dict_sagittal_channel_{args.use_sag_channel}.json"
    with open(os.path.join(path_out, conversion_dict_name), "w") as outfile:
        outfile.write(json_object)


    # c.f. dataset json generation
    # general info : https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/dataset_conversion/utils.py
    # example: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/dataset_conversion/Task055_SegTHOR.py

    json_dict = OrderedDict()
    json_dict['name'] = args.taskname
    json_dict['description'] = args.taskname
    json_dict['tensorImageSize'] = "3D"
    json_dict['reference'] = "TBD"
    json_dict['licence'] = "TBD"
    json_dict['release'] = "0.0"

    if args.use_sag_channel:
        json_dict['modality'] = {
            "0": "ax",
            "1": "sag",
        }
    else:
        json_dict['modality'] = {
            "0": "ax",
        }
    
    json_dict['labels'] = {
        "0": "background",
        "1": "lesion",

   }
    json_dict['numTraining'] = scan_cnt_train
    json_dict['numTest'] = scan_cnt_test

    json_dict['training'] = [{'image': str(train_image_labels[i]).replace("labelsTr", "imagesTr") , "label": train_image_labels[i] }
                                 for i in range(len(train_image_ax))]
    # Note: See https://github.com/MIC-DKFZ/nnUNet/issues/407 for how this should be described
    json_dict['test'] = [str(test_image_labels[i]).replace("labelsTs", "imagesTs") for i in range(len(test_image_ax))]

    # create dataset_description.json
    json_object = json.dumps(json_dict, indent=4)
    # write to dataset description
    # nn-unet requires it to be "dataset.json"
    dataset_dict_name = f"dataset.json"
    with open(os.path.join(path_out, dataset_dict_name), "w") as outfile:
        outfile.write(json_object)



