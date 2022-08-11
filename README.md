# Artificial intelligence for an integrative approach to analyze the brain and spinal cord in multiple sclerosis
### Cooperation between NeuroPoly (Polytechnique Montreal,Quebec) and TUM (Munich, Bavaria)

This project investigates both brain and spine MRI scans for Multiple Sclerosis Research, with a focus on spinal cord lesion detection.

## Table of contents

T.B.D.

### Notes:

```
As all studied data are kept privately, this repository does not (and should not) feature patient-related data and/or scans. These are shared in a privacy-preserving manner between the NeuroPoly and TUM labs.
```

### 1. Data Set Curation 

In a first step, the clinical practitioners perform a low-resolution scout scan to place the patient in the MRI scanner in order to locate the spinal cord. Subsequently, the spinal cord is imaged in two to three sagittal stacks. Taking the sagittaly scanned images into consideration, axial imaging of the spinal cord is planned and conducted. Axial images are obtained in order to verify lesions from a different perspective and to detect (smaller) lesions that are only visible in axial slices \cite{breckwoldt2017increasing} \cite{weier2012biplanar}. 

Finally, the dataset consists of a large-scale, multi-scanner patient cohort comprising multiple (longitudinal) sessions each consisting of multiple axially and sagitally aquired 2D MRI scans for each patient.  All results are stored as DICOMs to the clinical PACS severs, and are subsequently extracted as one NIFTI+JSON per scan.

#### 2. Raw Dataset Structure

The current, raw database is not structered in a 100% BIDS-compliant way, as depicted in this example:

```
...
── m_013874
│   ├── ses-20141127
│   │   ├── sub-m_013874_ses-20141127_sequ-11_t2.json
│   │   ├── sub-m_013874_ses-20141127_sequ-11_t2.nii.gz
│   │   ├── sub-m_013874_ses-20141127_sequ-18_t2.json
│   │   ├── sub-m_013874_ses-20141127_sequ-18_t2.nii.gz
│   │   ├── sub-m_013874_ses-20141127_sequ-21_t2.json
│   │   ├── sub-m_013874_ses-20141127_sequ-21_t2.nii.gz
│   │   ├── sub-m_013874_ses-20141127_sequ-25_t2.json
│   │   ├── sub-m_013874_ses-20141127_sequ-25_t2.nii.gz
│   │   ├── sub-m_013874_ses-20141127_sequ-5_t2.json
│   │   ├── sub-m_013874_ses-20141127_sequ-5_t2.nii.gz
│   │   ├── sub-m_013874_ses-20141127_sequ-7_t2.json
│   │   ├── sub-m_013874_ses-20141127_sequ-7_t2.nii.gz
│   └── ses-20190411
│       ├── sub-m_013874_ses-20190411_sequ-301_t2.json
│       ├── sub-m_013874_ses-20190411_sequ-301_t2.nii.gz
│       ├── sub-m_013874_ses-20190411_sequ-302_t2.json
│       ├── sub-m_013874_ses-20190411_sequ-302_t2.nii.gz
│       ├── sub-m_013874_ses-20190411_sequ-401_t2.json
│       ├── sub-m_013874_ses-20190411_sequ-401_t2.nii.gz
│       ├── sub-m_013874_ses-20190411_sequ-402_t2.json
│       ├── sub-m_013874_ses-20190411_sequ-402_t2.nii.gz
│       ├── sub-m_013874_ses-20190411_sequ-403_t2.json
│       └── sub-m_013874_ses-20190411_sequ-403_t2.nii.gz
├── m_016399
│   ├── ses-20190129
│   │   ├── sub-m_016399_ses-20190129_sequ-11_t2.json
│   │   ├── sub-m_016399_ses-20190129_sequ-11_t2.nii.gz
...

```

To establish a basis for subsequent processing, machine learning using ivadomed and integration with the spinal cord toolbox, we convert the database to a BIDS-compliant database, facilitating the integration with existing tools.

As it is not completely clear, to which acquisition type (axial or sagittal) and chunk (cervical, thoracic and lumbar), each of the scans belongs to, this information has to be gathered and infered from the scans and sidecar jsons.

For this, we use two scripts:

```
python3 parse_database.py -i{path_to_raw_db} -f dataset_description.csv -o .
```

Having gathered all required information, we can use the `dataset_description.csv` to create the BIDS-compliant database:

```
python3 create_bids_db.py -i {path_to_raw_db} -d dataset_description.csv -o .
```

The newly generated, BIDS compliant database adheres to the following structure

```
.
├── CHANGES.md
├── code
│   ├── sct-v5.5
│   └── sct-v5.6
├── dataset_description.json
├── derivatives
│   └── labels
│       └── sub-123456
│           ├── ses-20220101
│           │   └── anat
│           │       ├── sub-123456_ses-20220101_acq-ax_lesions-manual.json
│           │       ├── sub-123456_ses-20220101_acq-ax_lesions-manual.nii.gz
│           │       ├── sub-123456_ses-20220101_acq-sag_lesions-manual.json
│           │       └── sub-123456_ses-20220101_acq-sag_lesions-manual.nii.gz
│           └── ses-20220202
│               └── anat
│                   ├── sub-123456_ses-20220202_acq-ax_lesions-manual.json
│                   ├── sub-123456_ses-20220202_acq-ax_lesions-manual.nii.gz
│                   ├── sub-123456_ses-20220202_acq-sag_lesions-manual.json
│                   └── sub-123456_ses-20220202_acq-sag_lesions-manual.nii.gz
├── LICENSE
├── participants.json
├── participants.tsv
├── README.md
└── sub-123456
    ├── ses-20220101
    │   ├── anat
    │   │   ├── sub-123456_ses-20220101_acq-ax_T2w.json
    │   │   ├── sub-123456_ses-20220101_acq-ax_T2w.nii.gz
    │   │   ├── sub-123456_ses-20220101_axq-sag_T2w.json
    │   │   └── sub-123456_ses-20220101_axq-sag_T2w.nii.gz
    │   └── sub-123456_ses-20220101_scans.tsv
    └── ses-20220202
        ├── anat
        │   ├── sub-123456_ses-20220202_acq-ax_chunk-1_T2w.json
        │   ├── sub-123456_ses-20220202_acq-ax_chunk-1_T2w.nii.gz
        │   ├── sub-123456_ses-20220202_acq-ax_chunk-2_T2w.json
        │   ├── sub-123456_ses-20220202_acq-ax_chunk-2_T2w.nii.gz
        │   ├── sub-123456_ses-20220202_acq-ax_chunk-3_T2w.json
        │   ├── sub-123456_ses-20220202_acq-ax_chunk-3_T2w.nii.gz
        │   ├── sub-123456_ses-20220202_acq-sag_chunk-1_T2w.json
        │   ├── sub-123456_ses-20220202_acq-sag_chunk-1_T2w.nii.gz
        │   ├── sub-123456_ses-20220202_acq-sag_chunk-2_T2w.json
        │   ├── sub-123456_ses-20220202_acq-sag_chunk-2_T2w.nii.gz
        │   ├── sub-123456_ses-20220202_acq-sag_chunk-3_T2w.json
        │   └── sub-123456_ses-20220202_acq-sag_chunk-3_T2w.nii.gz
        └── sub-123456_ses-20220202_scans.tsv
```


### Preprocessing and Stitching

To run the stitching scripts on the bids database, we need to ensure the data is 





