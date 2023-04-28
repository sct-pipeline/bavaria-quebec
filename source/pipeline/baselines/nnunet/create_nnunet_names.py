import argparse
import pathlib
from pathlib import Path
import json
import os
import shutil
from collections import OrderedDict

import sys
import nibabel as nib
import numpy as np
from tqdm import tqdm

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


# stolen from here: https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")

def binarize_segmentation(ax_file_nnunet, seg_file_nnunet, threshold):
    image = nib.load(seg_file_nnunet)
    data = image.get_fdata()
    data = np.where(data > threshold, 1, 0)
    ref = nib.load(ax_file_nnunet)
    return nib.Nifti1Image(data, ref.affine, ref.header)

if __name__ == '__main__':

    # parse command line arguments
    parser = argparse.ArgumentParser(description='Convert BIDS-structured database to nn-unet format.')
    parser.add_argument('--image_directory', help='Path to BIDS structured database.', required=True)
    parser.add_argument('--label_directory', help='Path to derivatives directory in  a BIDS structured database, used to provide flexibility.', required=True)
    parser.add_argument('--output_directory', help='Path to output directory.', required=True)
    parser.add_argument('--label_str', type=str, nargs="+", help="String included in label file of the NIFTI. Must be unique!", required=True)

    parser.add_argument('--taskname', help='Specify the task name, e.g. Hippocampus', default='MSSpineLesion', type=str)
    parser.add_argument('--tasknumber', help='Specify the task number, has to be greater than 500', default=501,type=int)
    parser.add_argument('--split_dict', help='Specify the splits using an a json file.', required=True)

    parser.add_argument('--use_sag_channel', action='store_true', help='Use sagittal image (no label) as a second input channel.')
    parser.add_argument('--use_region_based', action='store_true', help='Select this option if you want to do region based training (i.e. lesions within spinal cord masks).')
    parser.add_argument('--binarize_labels', action='store_true', help="Binarize the label for nn-unet.")
    parser.add_argument('--threshold', type=float, default=1e-12, help="Binarizeation threshold for the label(s) for nn-unet.")

    args = parser.parse_args()

    path_in_images = Path(args.image_directory)
    path_in_labels = Path(args.label_directory)
    path_out = Path(os.path.join(os.path.abspath(args.output_directory), f'Dataset{args.tasknumber}_{args.taskname}'))
    path_out_imagesTr = Path(os.path.join(path_out, 'imagesTr'))
    path_out_imagesTs = Path(os.path.join(path_out, 'imagesTs'))
    path_out_labelsTr = Path(os.path.join(path_out, 'labelsTr'))
    path_out_labelsTs = Path(os.path.join(path_out, 'labelsTs'))

    # make the directories
    pathlib.Path(path_out).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_imagesTr).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_imagesTs).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_labelsTr).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_labelsTs).mkdir(parents=True, exist_ok=True)

    # we load both train and validation set into the train images as nnunet uses cross-fold-validation
    train_image_ax = []
    train_image_sag = []
    test_image_ax = []
    test_image_sag = []
    train_image_labels = []
    test_image_labels = []
    conversion_dict = {}

    # check if region based training is desired
    if args.use_region_based:
        assert len(args.label_str) > 1, "Failed to provide multiple segmentation labels."
        print(f"Using {args.label_str[0]} as global region.")
        print(f"Using {args.label_str[1]} as local region.")

        retval = query_yes_no("Do you want to proceed?", None)
        if retval != True:
            sys.exit("Failed to provide multiple segmentation labels.")


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

    for dir in tqdm(dirs):
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

            # find the corresponding segmentation file(s)
            if args.use_region_based:
                # multi-task (cord AND lesions)
                seg_path = os.path.join(args.label_directory, common)
                region_global = sorted(list(Path(seg_path).rglob(f'*{args.label_str[0]}*.nii.gz')))[0]
                region_local = sorted(list(Path(seg_path).rglob(f'*{args.label_str[1]}*.nii.gz')))[0]
                assert os.path.isfile(region_global), 'No segmentation mask with this name!'
                assert os.path.isfile(region_local), 'No segmentation mask with this name!'

            else:
                # single-task (i.e. cord OR lesions)
                seg_path = os.path.join(args.label_directory, common)
                seg_file = sorted(list(Path(seg_path).rglob(f'*{args.label_str[0]}*.nii.gz')))[0]
                assert os.path.isfile(seg_file), 'No segmentation mask with this name!'


            # only proceed if sub/session-id is included in the sets
            if any(str(Path(ax_file).name) in word for word in valid_train_imgs) or any(str(Path(ax_file).name) in word for word in valid_test_imgs):

                if any(str(Path(ax_file).name) in word for word in valid_train_imgs):

                    scan_cnt_train+= 1
                    # create the new convention names
                    ax_file_nnunet = os.path.join(path_out_imagesTr,f'{args.taskname}_{scan_cnt_train:03d}_0000.nii.gz')
                    train_image_ax.append(str(ax_file_nnunet))

                    # create a system link (instead of copying)
                    os.symlink(os.path.abspath(ax_file), ax_file_nnunet)
                    conversion_dict[str(os.path.abspath(ax_file))] = ax_file_nnunet

                    if args.use_sag_channel:
                        sag_file_nnunet = os.path.join(path_out_imagesTr,f'{args.taskname}_{scan_cnt_train:03d}_0001.nii.gz')
                        train_image_sag.append(str(sag_file_nnunet))
                        os.symlink(os.path.abspath(sag_file), sag_file_nnunet)
                        conversion_dict[str(os.path.abspath(sag_file))] = sag_file_nnunet

                    # segmentation files
                    if not args.use_region_based:

                        # TO DO - put this in preprocessing routines!
                        # replace the label header with the image header, and binarize label!
                        # FOR NOW we are COPYING, and NOT providing symlinks so the original labels are not affected
                        seg_file_nnunet = os.path.join(path_out_labelsTr,f'{args.taskname}_{scan_cnt_train:03d}.nii.gz')

                        if args.binarize_labels:
                            # we copy the original label and binarize it 
                            shutil.copyfile(seg_file, seg_file_nnunet)
                            # overwrite the label file
                            new_image = binarize_segmentation(ax_file_nnunet, seg_file_nnunet, args.threshold)
                            nib.save(new_image, seg_file_nnunet)

                        else:
                            # we only create a symlink
                            os.symlink(os.path.abspath(seg_file), seg_file_nnunet)

                    else:

                        # read both files
                        region_global_data = nib.load(region_global).get_fdata()
                        region_local_data = nib.load(region_local).get_fdata()

                        if args.binarize_labels:
                            # binarize with threshold
                            region_global_data = np.where(region_global_data > args.threshold, 1, 0)
                            region_local_data = np.where(region_local_data > args.threshold, 1, 0)

                        data = region_local_data + region_global_data
                        data = np.rint(data)

                        # create new nifti
                        ref = nib.load(ax_file_nnunet)
                        mask = nib.Nifti1Image(data, ref.affine, ref.header)

                        seg_file_nnunet = os.path.join(path_out_labelsTr,f'{args.taskname}_{scan_cnt_train:03d}.nii.gz')
                        nib.save(mask, seg_file_nnunet)

                    train_image_labels.append(str(seg_file_nnunet))
                    conversion_dict[str(os.path.abspath(seg_file))] = seg_file_nnunet
  
                else:

                    scan_cnt_test+= 1
                    # create the new convention names
                    ax_file_nnunet = os.path.join(path_out_imagesTs,f'{args.taskname}_{scan_cnt_test:03d}_0000.nii.gz')
                    test_image_ax.append(str(ax_file_nnunet))

                    # create a system link (instead of copying)
                    os.symlink(os.path.abspath(ax_file), ax_file_nnunet)
                    conversion_dict[str(os.path.abspath(ax_file))] = ax_file_nnunet

                    if args.use_sag_channel:
                        sag_file_nnunet = os.path.join(path_out_imagesTs,f'{args.taskname}_{scan_cnt_test:03d}_0001.nii.gz')
                        test_image_sag.append(str(sag_file_nnunet))
                        os.symlink(os.path.abspath(sag_file), sag_file_nnunet)
                        conversion_dict[str(os.path.abspath(sag_file))] = sag_file_nnunet

                    # segmentation files

                    if not args.use_region_based:

                        # TO DO - put this in preprocessing routines!
                        # replace the label header with the image header, and binarize label!
                        # FOR NOW we are COPYING, and NOT providing symlinks so the original labels are not affected
                        seg_file_nnunet = os.path.join(path_out_labelsTs,f'{args.taskname}_{scan_cnt_test:03d}.nii.gz')

                        if args.binarize_labels:
                            # we copy the original label and binarize it 
                            shutil.copyfile(seg_file, seg_file_nnunet)
                            # overwrite the label file
                            new_image = binarize_segmentation(ax_file_nnunet, seg_file_nnunet, args.threshold)
                            nib.save(new_image, seg_file_nnunet)

                        else:
                            # we only create a symlink
                            os.symlink(os.path.abspath(seg_file), seg_file_nnunet)
                        
                    else:

                        # read both files
                        region_global_data = nib.load(region_global).get_fdata()
                        region_local_data = nib.load(region_local).get_fdata()

                        if args.binarize_labels:
                            # binarize with threshold
                            region_global_data = np.where(region_global_data > args.threshold, 1, 0)
                            region_local_data = np.where(region_local_data > args.threshold, 1, 0)

                        data = region_local_data + region_global_data
                        data = np.rint(data)

                        # create new nifti
                        ref = nib.load(ax_file_nnunet)
                        mask = nib.Nifti1Image(data, ref.affine, ref.header)

                        seg_file_nnunet = os.path.join(path_out_labelsTs,f'{args.taskname}_{scan_cnt_test:03d}.nii.gz')
                        nib.save(mask, seg_file_nnunet)

                    test_image_labels.append(str(seg_file_nnunet))
                    conversion_dict[str(os.path.abspath(seg_file))] = seg_file_nnunet
           
            else:
                print("Skipping file, could not be located in the specified split.", ax_file)

    assert scan_cnt_train == len(valid_train_imgs), 'No. of train/val images does not correspond to ivadomed dict.'
    assert scan_cnt_test == len(valid_test_imgs), 'No. of test images does not correspond to ivadomed dict.'

    # create conversion dictionary so we can retrieve the original file names
    json_object = json.dumps(conversion_dict, indent=4)
    # write to dataset description
    conversion_dict_name = f"conversion_dict.json"
    with open(os.path.join(path_out, conversion_dict_name), "w") as outfile:
        outfile.write(json_object)

    # c.f. dataset json generation
    # general info : https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/dataset_conversion/utils.py
    # example: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/dataset_conversion/Task055_SegTHOR.py

    json_dict = OrderedDict()
    json_dict['name'] = args.taskname
    json_dict['description'] = args.taskname
    json_dict['file_ending'] = ".nii.gz"
    json_dict['tensorImageSize'] = "3D"
    json_dict['reference'] = "TBD"
    json_dict['licence'] = "TBD"
    json_dict['release'] = "0.0"

    if args.use_sag_channel:
        json_dict['channel_names'] = {
            "0": "ax",
            "1": "sag",
        }
    else:
        json_dict['channel_names'] = {
            "0": "ax",
        }

    if not args.use_region_based:
    
        json_dict['labels'] = {
            "background": 0,
            f"{args.label_str}": 1,
        }


    else:
        json_dict['labels'] = {
            "background": 0,
            "spinal_cord": 1,
            "spinal_lesions": 2,
        }

    json_dict['numTraining'] = scan_cnt_train
    json_dict['numTest'] = scan_cnt_test

    # NOTE - even with two modalities, there is only one label or image file in the dict - no need to panic ;)
    json_dict['training'] = [{'image': str(train_image_labels[i]).replace("labelsTr", "imagesTr") , "label": train_image_labels[i] }
                                for i in range(len(train_image_ax))]
    # Note: See https://github.com/MIC-DKFZ/nnUNet/issues/407 for how this should be described
    json_dict['test'] = [str(test_image_labels[i]).replace("labelsTs", "imagesTs") for i in range(len(test_image_ax))]

    # create dataset_description.json
    # copied from nnUnet: 
    '''
    IMPORTANT Because the conversion back to a segmentation map is sensitive to the order in which the regions are declared ("place label X in the first region") you need to make sure that this order is not perturbed! When automatically generating the dataset.json, make sure the dictionary keys do not get sorted alphabetically! Set sort_keys=False in json.dump()!!!
    nnU-Net will perform the evaluation + model selection also on the regions, not the individual labels!
    '''

    json_object = json.dumps(json_dict, indent=4, sort_keys=False) # sort_keys!!!!!
    # write to dataset description
    # nn-unet requires it to be "dataset.json"
    dataset_dict_name = f"dataset.json"
    with open(os.path.join(path_out, dataset_dict_name), "w") as outfile:
        outfile.write(json_object)



