import numpy as np
import argparse
import os
import nibabel as nib
from nibabel.affines import apply_affine
import SimpleITK as sitk

parser = argparse.ArgumentParser(description='identify coordinate span of a nifti file.')
parser.add_argument('-i','--image', type=str, help='Name of the Nifti file..', required=True)

args = parser.parse_args()

## Nibabel based calculation using affine transformation

print("\nNibabel based calculation using affine transformation")
img = nib.load(args.image)
dims = img.get_fdata().shape

print(f'Dimensions of the image{dims}')

x_min = apply_affine(img.affine, [0,0,0])[0]
x_max = apply_affine(img.affine, [dims[0],0,0])[0]
y_min = apply_affine(img.affine, [0,0,0])[1]
y_max = apply_affine(img.affine, [0,dims[1],0])[1]
z_min = apply_affine(img.affine, [0,0,0])[2]
z_max = apply_affine(img.affine, [0,0, dims[2]])[2]

print(f"xmin: {x_min},xmax: {x_max}")
print(f"ymin: {y_min},ymax: {y_max}")
print(f"zmin: {z_min},zmax: {z_max}")

print(f"X-Dimension spans space from {x_min} to {x_max}, occupying a physical space of {np.abs (x_max - x_min)}")
print(f"Y-Dimension spans space from {y_min} to {y_max}, occupying a physical space of {np.abs (y_max - y_min)}")
print(f"Z-Dimension spans space from {z_min} to {z_max}, occupying a physical space of {np.abs (z_max - z_min)}")


## Simple-ITK based calculation using affine transformation
print("\nSimple-ITK based calculation using affine transformation")

# without resampling
sitk_img = sitk.ReadImage(args.image)

print(f'Dimensions of the image{sitk_img.GetSize()}')

x_min = sitk_img.TransformIndexToPhysicalPoint((0,0,0))[0]
x_max = sitk_img.TransformIndexToPhysicalPoint((dims[0],0,0))[0]
y_min = sitk_img.TransformIndexToPhysicalPoint((0,0,0))[1]
y_max = sitk_img.TransformIndexToPhysicalPoint((0,dims[1],0))[1]
z_min = sitk_img.TransformIndexToPhysicalPoint((0,0,0))[2]
z_max = sitk_img.TransformIndexToPhysicalPoint((0,0, dims[2]))[2]

print(f"xmin: {x_min},xmax: {x_max}")
print(f"ymin: {y_min},ymax: {y_max}")
print(f"zmin: {z_min},zmax: {z_max}")

print(f"X-Dimension spans space from {x_min} to {x_max}, occupying a physical space of {np.abs (x_max - x_min)}")
print(f"Y-Dimension spans space from {y_min} to {y_max}, occupying a physical space of {np.abs (y_max - y_min)}")
print(f"Z-Dimension spans space from {z_min} to {z_max}, occupying a physical space of {np.abs (z_max - z_min)}")

# Custom calculation

## Simple-ITK based calculation using affine transformation
print("\nSimple-ITK based calculation using meta-data only (origin, size and spacing)")

print(f'Dimensions of the image{sitk_img.GetSize()}')

x_origin = sitk_img.GetOrigin()[0]
y_origin = sitk_img.GetOrigin()[1]
z_origin = sitk_img.GetOrigin()[2]

x_min = x_origin
y_min = y_origin
z_min = z_origin

x_max = x_origin+sitk_img.GetSize()[0]*sitk_img.GetSpacing()[0]
y_max = y_origin+sitk_img.GetSize()[1]*sitk_img.GetSpacing()[1]
z_max = z_origin+sitk_img.GetSize()[2]*sitk_img.GetSpacing()[2]

print(f"xmin: {x_min},xmax: {x_max}")
print(f"ymin: {y_min},ymax: {y_max}")
print(f"zmin: {z_min},zmax: {z_max}")

print(f"X-Dimension spans space from {x_min} to {x_max}, occupying a physical space of {np.abs (x_max - x_min)}")
print(f"Y-Dimension spans space from {y_min} to {y_max}, occupying a physical space of {np.abs (y_max - y_min)}")
print(f"Z-Dimension spans space from {z_min} to {z_max}, occupying a physical space of {np.abs (z_max - z_min)}")

print("\nAffines:")

print("QForm and SForm coded=True\n")
qform, qform_code = img.header.get_qform(coded=True)
sform, sform_code = img.header.get_sform(coded=True)
print("qform ({!s})\n{!s}".format(qform_code, qform))
print("sform ({!s})\n{!s}".format(sform_code, sform))

print("QForm and SForm coded=False \n")
qform = img.header.get_qform(coded=False)
sform = img.header.get_sform(coded=False)
print("qform \n({!s})\n".format(qform))
print("sform \n({!s})\n".format(sform))