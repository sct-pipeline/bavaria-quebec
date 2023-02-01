import argparse
from pathlib import Path
from batchgenerators.utilities.file_and_folder_operations import *

# parse command line arguments
parser = argparse.ArgumentParser(description='Visualize pickle file of nnnunet.')
parser.add_argument('--image', help='Path to nnunet pickle.', required=True)
args = parser.parse_args()

a = load_pickle(args.image)
print(a)
# print(a['plans_per_stage'])
