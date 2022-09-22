This Readme gives an overview of the complete spinal cord pipeline:

- preprocess_raw_dataset.sh 

Transform the raw database:

├── dataset_description.json 
|── LICENSE
├── participants.json
├── participants.tsv
├── README.md
└── sub-123456
    ├── ses-20220101
    │   ├── anat
    │   │   ├── sub-123456_ses-20220101_acq-ax_chunk-1_T2w.json
    │   │   ├── sub-123456_ses-20220101_acq-ax_chunk-1_T2w.nii.gz
    │   │   ├── sub-123456_ses-20220101_acq-ax_chunk-2_T2w.json
    │   │   ├── sub-123456_ses-20220101_acq-ax_chunk-2_T2w.nii.gz
    │   │   ├── sub-123456_ses-20220101_acq-ax_chunk-3_T2w.json
    │   │   ├── sub-123456_ses-20220101_acq-ax_chunk-3_T2w.nii.gz
    │   │   ├── sub-123456_ses-20220101_acq-ax_chunk-1_dseg.nii.gz
    │   │   ├── sub-123456_ses-20220101_acq-ax_chunk-2_dseg.nii.gz
    │   │   ├── sub-123456_ses-20220101_acq-ax_chunk-3_dseg.nii.gz
    │   │   ├── sub-123456_ses-20220101_acq-sag_chunk-1_T2w.json
    │   │   ├── sub-123456_ses-20220101_acq-sag_chunk-1_T2w.nii.gz
    │   │   ├── sub-123456_ses-20220101_acq-sag_chunk-2_T2w.json
    │   │   ├── sub-123456_ses-20220101_acq-sag_chunk-2_T2w.nii.gz
    │   │   ├── sub-123456_ses-20220101_acq-sag_chunk-3_T2w.json
    │   │   └── sub-123456_ses-20220101_acq-sag_chunk-3_T2w.nii.gz
    │   └── sub-123456_ses-20220101_scans.tsv

To:

├── derivatives
└── sub-123456
    ├── ses-20220101
    │   ├── anat
    │   │   ├── sub-123456_ses-20220101_acq-ax_T2w.json
    │   │   ├── sub-123456_ses-20220101_acq-ax_T2w.nii.gz
    │   │   ├── sub-123456_ses-20220101_acq-ax_dseg.nii.gz
    │   │   ├── sub-123456_ses-20220101_acq-ax_dseg.json
    │   │   ├── sub-123456_ses-20220101_acq-sag_T2w.json
    │   │   └── sub-123456_ses-20220101_acq-sag_T2w.nii.gz
    │   └── sub-123456_ses-20220101_scans.tsv

- 