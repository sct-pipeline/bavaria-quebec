### sct_deepseg_lesion baseline

Fortunately, there already exists a well-established spinal cord lesion segmentation algorithm by Gros et al. which is part of sct.

To provide a for comparision, we compute the following baselines:

1. Spinal Cord Lesion Segmentation on the cropped, and preprocessed axial images using the `t2_ax`contrast.
2. Spinal Cord Lesion Segmentation on the uncropped, but preprossed axial images using the `t2_ax`contrast.
3. Spinal Cord Lesion Segmentation on the cropped, and preprocessed axial images using the `t2`contrast (isotropic contrast).
4. Spinal Cord Lesion Segmentation on the uncropped, but preprossed axial images using the `t2`contrast (siotropic contrast).
