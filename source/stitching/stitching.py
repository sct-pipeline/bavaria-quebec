import os
import numpy as np
import argparse
import imageio
import SimpleITK as sitk

def generate_snapshot(im, file_path, orient='ax'):
    """
    :param im: numpy image array
    :param file_path: file_path used to store the PNG image.
    :param orient: orientation string, either 'sag' or'ax'
    :return: N/A
    """
    (j,k,l) = im.shape
    if orient=='ax':
       im = im[0:j, 0:k, int (l/2)]
       imageio.imwrite(file_path, np.rot90(im, 2))
    elif orient=='sag':
        im = im[0:j, 0:k, int(l / 2)]
        imageio.imwrite(file_path, np.rot90(im, 1))
    else:
        im = im[0:j, 0:k, int(l / 2)]
        imageio.imwrite(file_path, np.rot90(im,2))

def resample_itk(im, desired_spacing, interp_type):
    """
    :param im: sitk image to be resampled
    :param desired_spacing: list of [x,y,z]-resolution
    :param interp_type: str of interpolation filter
    :return: resampled image
    """
    algorithms = ['linear','spline','nearest']
    assert interp_type in algorithms, "Interpolation not supported"

    # desired spacing corresponds to the [x,y,z] resolution we would like to obtain
    desired_spacing = np.asarray(desired_spacing, dtype=float)
    # old dimensions of the file
    old_size = np.asarray(im.GetSize(), dtype=float)
    # old spacing
    old_spacing = np.asarray(im.GetSpacing(), dtype=float)
    print(f'Current spacing{old_spacing}')
    print(f'Resampling to desired Spacing{desired_spacing}')

    physical_size = old_size * old_spacing
    new_size = physical_size / desired_spacing  # mm / (mm / voxel) = voxel
    new_size = np.round(new_size)

    # Scale the image volume using given interpolation
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(im)
    resampler.SetOutputSpacing(list(desired_spacing))

    resampler.SetSize(np.array(new_size, dtype='int').tolist())

    if interp_type == 'linear':
        resampler.SetInterpolator(sitk.sitkLinear)
    elif interp_type == 'spline':
        resampler.SetInterpolator(sitk.sitkBSpline)
    elif interp_type == 'nearest':
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        raise("Interpolation not supported.")

    im_resampled = resampler.Execute(im)
    return im_resampled

def convert_path_to_sitk_img(img_path, resample=False):
    """
    :param img_path: path to image
    :return: sitk-image
    """
    sitk_img = sitk.ReadImage(img_path)
    # cast images to sitk Float
    sitk_img = sitk.Cast(sitk_img, sitk.sitkFloat32)
    
    if resample:

        # resample the images to obtain a resolution of 0.5, 0.5, 0.5 using the linear filter
        sitk_img = resample_itk(sitk_img, [0.5, 0.5, 0.5], sampling_method)

    padf = sitk.ConstantPadImageFilter()
    padf.SetConstant(0)
    padf.SetPadLowerBound((10, 20, 80))
    padf.SetPadUpperBound((10, 20, 80))
    sitk_img = padf.Execute(sitk_img)

    return sitk_img


def stitch_image(sitk_img1,sitk_img2):
    """
    :param sitk_img1: reference image
    :param sitk_img2: 2nd image that is registered to reference
    :return: stitched image stack
    """

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=24)
    R.SetOptimizerAsRegularStepGradientDescent(learningRate=1,
                                               minStep=1e-4,
                                               numberOfIterations=200,
                                               estimateLearningRate=R.Once)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.01)
    R.SetInterpolator(sitk.sitkLinear)
    outTx = R.Execute(sitk_img1, sitk_img2)

    s1 = (int(sitk_img1.GetSize()[0]) + int(sitk_img2.GetSize()[0])) / 2
    s2 = (int(sitk_img1.GetSize()[1]) + int(sitk_img2.GetSize()[1]))
    s3 = (int(sitk_img1.GetSize()[2]) + int(sitk_img2.GetSize()[2])) / 2
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk_img2)

    resampler.SetSize(np.array([s1, s2, s3], dtype='int').tolist())
    re2 = resampler.Execute(sitk_img2)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk_img2)

    resampler.SetSize(np.array([s1, s2, s3], dtype='int').tolist())
    resampler.SetTransform(outTx)
    re1 = resampler.Execute(sitk_img1)

    rwe = re1 + re2
    return rwe


parser = argparse.ArgumentParser(description='Stitch two sagitally acquired images into a common, interpolated image.')
parser.add_argument('-i', '--img', help='2 or more image stacks.', required=True, nargs='+')
parser.add_argument('-o', '--output_directory', help='Directory to store the result.', required=True)
parser.add_argument('-f', '--file_name', type=str, help='Filename of histgram including filename ending.', default="stitched.nii.gz")
parser.add_argument('-s','--sampling_method', type=str, help='Sampling strategy for interpolation.', default='linear')
parser.add_argument('-ornt','--orientation', type=str, help='Specify the orientation for the snapshot.', default='sag')
parser.add_argument('--resample', default=False, action='store_true')

# parse variables
args = parser.parse_args()
img_paths = args.img
assert len(img_paths) >= 2, "Please provide 2 or more image stacks"
sampling_method = args.sampling_method
output_dir = args.output_directory
file_name = args.file_name
resample_flag = bool(args.resample)

# without resampling
sitk_img_1 = sitk.ReadImage(img_paths[0])
sitk_img_2 = sitk.ReadImage(img_paths[1])
# cast images to sitk Float
sitk_img_1 = sitk.Cast(sitk_img_1, sitk.sitkFloat32)
sitk_img_2 = sitk.Cast(sitk_img_2, sitk.sitkFloat32)

res_img = stitch_image(sitk_img_1, sitk_img_2)

# cast images to sitk Float
sitk_img1 = convert_path_to_sitk_img(img_paths[0],resample_flag)
sitk_img2 = convert_path_to_sitk_img(img_paths[1],resample_flag)

res_img = stitch_image(sitk_img1, sitk_img2)
# helper vars
iterations = len(img_paths)
i = 2
print(f'Executed stitching round no. {i - 1}.')

while iterations > i:
    # stitch next image to stack
    res_img = stitch_image(res_img,convert_path_to_sitk_img(img_paths[i]))
    i = i+1
    print(f'Executed stitching round no. {i - 1}.')

sitk.WriteImage(res_img, os.path.join(output_dir,file_name))

# create snapshot
snp_name = file_name.replace("nii.gz","png")
generate_snapshot(sitk.GetArrayViewFromImage(res_img), os.path.join(output_dir,snp_name),args.orientation)

# helper output
print(f'itksnap {os.path.join(output_dir,file_name)}')
