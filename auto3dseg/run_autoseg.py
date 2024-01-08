import os
import argparse

from monai.bundle.config_parser import ConfigParser
from monai.apps import download_and_extract

from monai.apps.auto3dseg import AutoRunner
from monai.config import print_config

def get_parser():

    parser = argparse.ArgumentParser(description='Code for running MONAI Auto3DSeg.')

    parser.add_argument('--dataroot', required=True, type=str, 
                        help='Path to the data set directory. Must contain the imagesTr, labelsTr, imagesTs and labelsTs folders.')
    parser.add_argument('--datalist', required=True, type=str, help='Path to the MSD-style datalist.')    
    parser.add_argument('--out', type=str, help='Path to the output directory to store results')

    return parser


def main():

    # get arguments
    args = get_parser().parse_args()

    # results directory
    if not os.path.exists(args.out):
        os.makedirs(args.out, exist_ok=True)

    # prepare input yaml config
    input_cfg = {
    "name": "Whole-spine MS Lesion Seg",  # optional, it is only for your own record
    "task": "segmentation",  # optional, it is only for your own record
    "modality": "MRI",  # required
    "datalist": args.datalist,  # required
    "dataroot": args.dataroot,  # required
    }

    # input = "./input.yaml"
    # ConfigParser.export_config_file(input_cfg, input)

    # define autorunner class
    runner = AutoRunner(
        input=input_cfg,
        work_dir=args.out
    )

    # for debugging purposes use only 1
    runner.set_num_fold(num_fold=1)

    # customize training parameters by override the default values
    max_epochs = 2
    train_params = {
        "num_epochs_per_validation": 1,
        "num_images_per_batch": 2,
        "num_epochs": max_epochs,
        "num_warmup_epochs": 1,
    }
    runner.set_training_params(params=train_params)

    # customize the inference parameters by override the default values
    # set model ensemble method
    pred_params = {
        "mode": "vote",  # use majority vote instead of mean to ensemble the predictions
        "sigmoid": True,  # when to use sigmoid to binarize the prediction and output the label
    }
    runner.set_prediction_params(params=pred_params)

    # RUN!
    runner.run()


if __name__ == "__main__":
    main()