## Cooperation between NeuroPoly (Polytechnique Montreal,Quebec) and TUM (Munich, Bavaria)

This project aims to investigate both brain and spine MRI scans for Multiple Sclerosis Research. 

Note:

As all studied data are kept privately, this repository does not (and should not) feature patient-related data and/or scans. These are shared in a privacy-preserving manner between the NeuroPoly and TUM labs.

## Table of Contents

- [Data collection and organization](#data-collection-and-organization)
  * [Background Information](#background-information)
  * [Brain Dataset](#brain-dataset)
  * [Spinal Cord Dataset](#spinal-cord-dataset)
- [Analysis Pipeline](#analysis-pipeline)
  * [General Information](#general-information)
  * [Preprocessing](#preprocessing)
  * [Processing](#processing)
  * [Quality Check](#quality-check)

- - -

## Data collection and organization
### Background Information 

In this project, we use the [Spinal Cord Toolbox](https://github.com/spinalcordtoolbox/spinalcordtoolbox) to analyze both, brain and spinal cord for multiple sclerosis research questions. The MS brain database has been curated by various PhD students of the lab at TUM and various pipelines, such as CAT12, LST and Freesurfer have been employed to analyze the data. Similarly, this project aims to create a spinal cord database to provide the possibility of large-scale analysis of spinal cord lesions, and the joined analysis with the brain.

### BIDS compliance

Originally, the datasets and cohort have not complied with the BIDS standard. The conversion to BIDS is specific to each cohort and dataset, as the workflow for different sequences differs and due to the fact that medical doctors and students have segmented lesions in different ways. We attempt to gather all of the scripts related to the conversion in [BIDS Tools](https://github.com/jqmcginnis/bids_tools) and describe the newly introduced BIDS conventions in this repository.

### Brain Dataset

The brain database has been curated by various PhD students at the TUM lab, and adheres to the BIDS convention. It consists of T1w, T2w, Flair Scans of the brain, scanned with isotropic resolution.

### Spinal Cord Dataset 

In contrast to the brain database, the imaging of spinal cord is much more complex, and requires a more sophisticated protocol.

In a first step, the clinical practitioners perform a low-resolution scout scan to place the patient in the MRI scanner in order to locate the spinal cord. Subsequently, the spinal cord is imaged in two to three sagittal stacks. Taking the sagittaly scanned images into consideration, axial imaging of the spinal cord is planned and conducted. Axial images are obtained in order to verify lesions from a different perspective and to detect (smaller) lesions that are only visible in axial slices (Breckwoldt et al., Weier et al.). 

Finally, the dataset consists of a large-scale, multi-scanner patient cohort comprising multiple (longitudinal) sessions each consisting of multiple axially and sagitally aquired 2D MRI scans for each patient.  ll results are stored as DICOMs to the clinical PACS severs, and are subsequently extracted as one NIFTI+JSON per scan.

#### BIDS conform Raw Dataset Structure

To establish a basis for subsequent processing, machine learning using ivadomed and integration with the spinal cord toolbox, we convert the database to a BIDS-compliant database, facilitating the integration with existing tools. As it is not completely clear, to which acquisition type (axial or sagittal) and chunk (cervical, thoracic and lumbar), each of the scans belongs to, this information has to be gathered and infered from the scans and sidecar jsons.

The newly generated, BIDS compliant database adheres to the following structure:

```
.
├── CHANGES.md
├── code
│   ├── sct-v5.5
│   └── sct-v5.6
├── dataset_description.json
├── derivatives
│   └── labels
│       └── sub-m123456
│           ├── ses-20220101
│           │   └── anat
│           │       ├── sub-m123456_ses-20220101_acq-ax_lesions-manual.json
│           │       ├── sub-m123456_ses-20220101_acq-ax_lesions-manual.nii.gz
│           │       ├── sub-m123456_ses-20220101_acq-sag_lesions-manual.json
│           │       └── sub-m123456_ses-20220101_acq-sag_lesions-manual.nii.gz
│           └── ses-20220202
│               └── anat
│                   ├── sub-m123456_ses-20220202_acq-ax_lesions-manual.json
│                   ├── sub-m123456_ses-20220202_acq-ax_lesions-manual.nii.gz
│                   ├── sub-m123456_ses-20220202_acq-sag_lesions-manual.json
│                   └── sub-m123456_ses-20220202_acq-sag_lesions-manual.nii.gz
├── LICENSE
├── participants.json
├── participants.tsv
├── README.md
└── sub-m123456
    ├── ses-20220101
    │   ├── anat
    │   │   ├── sub-m123456_ses-20220101_acq-ax_T2w.json
    │   │   ├── sub-m123456_ses-20220101_acq-ax_T2w.nii.gz
    │   │   ├── sub-m123456_ses-20220101_axq-sag_T2w.json
    │   │   └── sub-m123456_ses-20220101_axq-sag_T2w.nii.gz
    │   └── sub-m123456_ses-20220101_scans.tsv
    └── ses-20220202
        ├── anat
        │   ├── sub-m123456_ses-20220202_acq-ax_chunk-1_T2w.json
        │   ├── sub-m123456_ses-20220202_acq-ax_chunk-1_T2w.nii.gz
        │   ├── sub-m123456_ses-20220202_acq-ax_chunk-2_T2w.json
        │   ├── sub-m123456_ses-20220202_acq-ax_chunk-2_T2w.nii.gz
        │   ├── sub-m123456_ses-20220202_acq-ax_chunk-3_T2w.json
        │   ├── sub-m123456_ses-20220202_acq-ax_chunk-3_T2w.nii.gz
        │   ├── sub-m123456_ses-20220202_acq-sag_chunk-1_T2w.json
        │   ├── sub-m123456_ses-20220202_acq-sag_chunk-1_T2w.nii.gz
        │   ├── sub-m123456_ses-20220202_acq-sag_chunk-2_T2w.json
        │   ├── sub-m123456_ses-20220202_acq-sag_chunk-2_T2w.nii.gz
        │   ├── sub-m123456_ses-20220202_acq-sag_chunk-3_T2w.json
        │   └── sub-m123456_ses-20220202_acq-sag_chunk-3_T2w.nii.gz
        └── sub-m123456_ses-20220202_scans.tsv
```

## Analysis Pipeline
### General Information

Pipeline scripts can be found in [directory](https://github.com/sct-pipeline/bavaria-quebec/tree/main/source/pipeline).

### Preprocessing

To use SCT's capabilities w.r.t. statistical analysis and registration to the PAM50 template, it is necessary to stitch the axial and sagittal slices to whole-spine images.

For this, we evaluated multiple stitching algorithms. After thorough examination of the results, we decided to use the approach by [Lavdas et al.](https://github.com/biomedia-mira/stitching), which is now integrated in SCT v.5.8. 

To obtain the stitched database, please run:

```
sct_run_batch -script preprocess_raw_dataset.sh -path-data /path/to/db -path-output result -jobs 4
```

### Processing

To use SCT on a BIDS compliant brain database (i.e. on the cervical spine regions), we implemented a bash-based multi-subject script for the entire workflow in SCT.

Currently, it consists of the following steps:

1. Segmentation of spinal cord
2. Labeling of vertebrae
3. Computation of CSA in cervical spine

Link to [scripts](https://github.com/sct-pipeline/bavaria-quebec/tree/main/source/pipeline/brain/).

### Quality Check

To run the quality check w.r.t. SCT's results, please launch the `index.html` in the qc folder of the processed scans. 

The quality check will be detailed once examples are readily available.





