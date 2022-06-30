import argparse
import json
import os
import json
import gzip
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import shutil

parser = argparse.ArgumentParser(description='BIDSify the MS brain database.')
parser.add_argument('-i','--input_directory', help='Folder of top directory non-bids database.', required=True)
parser.add_argument('-d','--dataset_description', help='Provide pandas csv of database.')
parser.add_argument('-o','--output_directory', help='Folder of bids database', required=True)

args = parser.parse_args()
mode = 0o777

# create a new folder in the top directory
raw_database_dir = os.path.dirname(args.input_directory)
raw_database_name = os.path.basename(os.path.normpath(args.input_directory))
bids_database_name = f'{raw_database_name}_bids'
bids_database_path = os.path.join(os.path.abspath(args.output_directory),bids_database_name)

try:
    Path(bids_database_path).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(bids_database_path,"derivatives")).mkdir(parents=True, exist_ok=True)

except:
    raise("Could not create top directory of BIDS database.")

# BIDS file-format structure
t2w_label = 'T2w'
t2w_raw_label = 't2.nii'
t2w_raw_label_zipped = 't2.nii.gz'

# define LICENSE
license = "MIT"

try:
    # create BIDS top directory structure
    os.mkdir(bids_database_path, mode)

except OSError as exc:
    print("Failed creating the top directory of BIDS datastructure.")
    pass

# @TODO: create extensive dataset description
#  https://bids-specification.readthedocs.io/en/stable/03-modality-agnostic-files.html

dataset={
  "Name": "TUMNIC-MS_SpinalCord_Dataset",
  "BIDSVersion": "1.7.0",
  "DatasetType": "raw",
  "License": "CC0",
  "Authors": [
    "Mark Muehlau",
    "Jan Kirschke",
    "Julian McGinnis",
  ],
  "Acknowledgements": "",
  "HowToAcknowledge": "",
  "Funding": [
    "",
    ""
  ],
  "EthicsApprovals": [
    ""
  ],
  "ReferencesAndLinks": [
    "",
    ""
  ],
  "DatasetDOI": "",
  "HEDVersion": "",
  "GeneratedBy": [
    {
      "Name": "jqmcginnis",
      "Version": "0.0.1",
    }
  ],
  "SourceDatasets": [
    {
      "URL": "https://aim-lab.io/",
      "Version": "June 29 2022"
    }
  ]
}

# read information from csv

dataset_description = pd.read_csv(args.dataset_description)
groups = dataset_description.groupby(['SubjectID','SessionDate']).size()

participant_ids = list(dataset_description["SubjectID"].unique())
combinations = list(groups.to_dict().keys())


for c in combinations:
    (id, session_date) = c
    print(f'Patient ID {id}, Session Date {session_date}')
    try:
        Path(os.path.join(bids_database_path, f'sub-{id:06}',f'ses-{session_date:06}')).mkdir(parents=True, exist_ok=True)
    except:
        raise("Lacking permisions?")

# for in each row of the dataset description
# 1) extract the filename


for index, row in dataset_description.iterrows():
    #acq_tag = row["s"]
    #chunk_tag = row["chunk"]
    id = row["SubjectID"]
    session_date = row["SessionDate"]
    ornt = row["ornt"]
    chunk_nr = row["partName"]

    img_path = row["ImgPath"]
    json_path = img_path.replace("nii.gz","json")

    path = os.path.join(bids_database_path, f'sub-{id:06d}',f'ses-{session_date:06d}', f'sub-{id:06d}_ses-{session_date:06d}_acq-{ornt}_chunk-{chunk_nr}')
    path_t2w = f'{path}_T1w.nii.gz'
    path_json = f'{path}_T1w.json'
    print(path_t2w)

    try:

        shutil.copy(json_path,path_json)
        shutil.copy(img_path,path_t2w)

    except:
        pass




# Serializing json
json_object = json.dumps(dataset, indent=4)
# write to dataset description
with open(os.path.join(bids_database_path,"dataset_description.json"), "w") as outfile:
    outfile.write(json_object)

# create participants.tsv

# create README
with open(os.path.join(bids_database_path,"README"), "a") as outfile:
    outfile.write("This will be the Readme.")

# create LICENSE
# choose from a variety of licenses
with open(os.path.join(bids_database_path,"LICENSE"), "w") as outfile:
    outfile.write("This is the license file.")


d = {'participant_id': participant_ids}#, 'age': participant_age, 'sex': participant_sex}

df = pd.DataFrame(data=d)
df.to_csv(os.path.join(bids_database_path,"participants.tsv"), sep="\t",index=False)

participant_description = {
    "age": {
        "Description": "age of the participant",
        "Units": "years"
    },
    "sex": {
        "Description": "sex of the participant as reported by the participant",
        "Levels": {
            "M": "male",
            "F": "female"
        }
    },
    "handedness": {
        "Description": "handedness of the participant as reported by the participant",
        "Levels": {
            "left": "left",
            "right": "right"
        }
    },
    "group": {
        "Description": "experimental group the participant belonged to",
        "Levels": {
            "read": "participants who read an inspirational text before the experiment",
            "write": "participants who wrote an inspirational text before the experiment"
        }
    }
}

# Serializing json
json_object = json.dumps(participant_description, indent=4)
# write to dataset description
with open(os.path.join(bids_database_path,"participants.json"), "w") as outfile:
    outfile.write(json_object)

