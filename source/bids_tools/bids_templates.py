import json

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

def get_dataset_description():
    return dataset

def get_readme():
    return 0
def get_license_file():
    return 0


