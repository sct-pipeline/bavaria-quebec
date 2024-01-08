import os
import nibabel as nib
import numpy as np
import logging
from copy import deepcopy
import subprocess

logger = logging.getLogger(__name__)


def create_empty_mask(fname, fname_label):
    """
    Create empty mask from reference image
    :param fname: absolute path to reference image
    :param fname_label: absolute path to output mask under derivatives
    """
    img = nib.load(fname)
    data = np.zeros(img.shape)
    img_mask = nib.Nifti1Image(data, affine=img.affine, header=img.header)
    nib.save(img_mask, fname_label)
    print("No label file found, creating an empty mask: {}".format(fname_label))


def create_multiclass_label(subject_labels_path, image_file, sub_ses_name, thr=0.5):
    """
    Creates multi-class label training. The classes are: 
    0: background
    1: spinal cord seg
    2: lesion seg
    """

    # define path for sc seg and lesion seg files
    subject_seg_file = os.path.join(subject_labels_path, f"{sub_ses_name}_acq-ax_T2w_seg-manual.nii.gz")
    subject_label_file = os.path.join(subject_labels_path, f"{sub_ses_name}_acq-ax_T2w_lesion-manual.nii.gz")

    # check if the seg file exists
    if not os.path.exists(subject_seg_file):
        logger.info(f"Segmentation file for subject {sub_ses_name} does not exist. Skipping.")
        return None    
    
    # check if the lesion label file exists else create an empty mask
    if not os.path.exists(subject_label_file):
        create_empty_mask(image_file, subject_label_file)
        logger.info(f"Label file for subject {sub_ses_name} does not exist. Creating an empty mask.")

    # load the labels
    lesion_label_npy = nib.load(subject_label_file).get_fdata()
    seg_label_npy = nib.load(subject_seg_file).get_fdata()

    # binarize the labels
    lesion_label_npy = np.where(lesion_label_npy > thr, 1, 0)
    seg_label_npy = np.where(seg_label_npy > thr, 1, 0)

    # check if the shapes of the labels match
    assert lesion_label_npy.shape == seg_label_npy.shape, \
          f'Shape mismatch between lesion label and segmentation label for subject {sub_ses_name}. Check the labels.'

    # create a new label array with the same shape as the original labels
    label_npy = np.zeros(lesion_label_npy.shape, dtype=np.int16)
    # spinal cord
    label_npy[seg_label_npy == 1] = 1
    # lesion seg
    label_npy[lesion_label_npy == 1] = 2
    # TODO: what happens when the subject has no lesion?

    # print unique values in the label array
    # print(f'Unique values in the label array for subject {sub_ses_name}: {np.unique(label_npy)}')
    
    # save the new label file
    ref = nib.load(image_file)
    seg_lesion_nii = nib.Nifti1Image(label_npy, ref.affine, ref.header)
    nib.save(seg_lesion_nii, os.path.join(subject_labels_path, f"{sub_ses_name}_seg-lesion-manual.nii.gz"))

    # overwrite the original label file with the region-based label
    subject_label_file = os.path.join(subject_labels_path, f"{sub_ses_name}_seg-lesion-manual.nii.gz")

    return subject_label_file


def get_git_branch_and_commit(dataset_path=None):
    """
    :return: git branch and commit ID, with trailing '*' if modified
    Taken from: https://github.com/spinalcordtoolbox/spinalcordtoolbox/blob/master/spinalcordtoolbox/utils/sys.py#L476 
    and https://github.com/spinalcordtoolbox/spinalcordtoolbox/blob/master/spinalcordtoolbox/utils/sys.py#L461
    """

    # branch info
    b = subprocess.Popen(["git", "rev-parse", "--abbrev-ref", "HEAD"], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, cwd=dataset_path)
    b_output, _ = b.communicate()
    b_status = b.returncode

    if b_status == 0:
        branch = b_output.decode().strip()
    else:
        branch = "!?!"

    # commit info
    p = subprocess.Popen(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=dataset_path)
    output, _ = p.communicate()
    status = p.returncode
    if status == 0:
        commit = output.decode().strip()
    else:
        commit = "?!?"

    p = subprocess.Popen(["git", "status", "--porcelain"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=dataset_path)
    output, _ = p.communicate()
    status = p.returncode
    if status == 0:
        unclean = True
        for line in output.decode().strip().splitlines():
            line = line.rstrip()
            if line.startswith("??"):  # ignore ignored files, they can't hurt
                continue
            break
        else:
            unclean = False
        if unclean:
            commit += "*"

    return branch, commit


class Image(object):
    """
    Compact version of SCT's Image Class (https://github.com/spinalcordtoolbox/spinalcordtoolbox/blob/master/spinalcordtoolbox/image.py#L245)
    Create an object that behaves similarly to nibabel's image object. Useful additions include: dims, change_orientation and getNonZeroCoordinates.
    Taken from: https://github.com/ivadomed/utilities/blob/main/scripts/image.py
    Changed default verbosity to 0.
    """

    def __init__(self, param=None, hdr=None, orientation=None, absolutepath=None, dim=None):
        """
        :param param: string indicating a path to a image file or an `Image` object.
        """

        # initialization of all parameters
        self.affine = None
        self.data = None
        self._path = None
        self.ext = ""

        if absolutepath is not None:
            self._path = os.path.abspath(absolutepath)
        
        # Case 1: load an image from file
        if isinstance(param, str):
            self.loadFromPath(param)
        # Case 2: create a copy of an existing `Image` object
        elif isinstance(param, type(self)):
            self.copy(param)
        # Case 3: create a blank image from a list of dimensions
        elif isinstance(param, list):
            self.data = np.zeros(param)
            self.hdr = hdr.copy() if hdr is not None else nib.Nifti1Header()
            self.hdr.set_data_shape(self.data.shape)
        # Case 4: create an image from an existing data array
        elif isinstance(param, (np.ndarray, np.generic)):
            self.data = param
            self.hdr = hdr.copy() if hdr is not None else nib.Nifti1Header()
            self.hdr.set_data_shape(self.data.shape)
        else:
            raise TypeError('Image constructor takes at least one argument.')
    
        # Fix any mismatch between the array's datatype and the header datatype
        self.fix_header_dtype()

    @property
    def dim(self):
        return get_dimension(self)
    
    @property
    def orientation(self):
        return get_orientation(self)
    
    @property
    def absolutepath(self):
        """
        Storage path (either actual or potential)

        Notes:

        - As several tools perform chdir() it's very important to have absolute paths
        - When set, if relative:

          - If it already existed, it becomes a new basename in the old dirname
          - Else, it becomes absolute (shortcut)

        Usually not directly touched (use `Image.save`), but in some cases it's
        the best way to set it.
        """
        return self._path
    
    @absolutepath.setter
    def absolutepath(self, value):
        if value is None:
            self._path = None
            return
        elif not os.path.isabs(value) and self._path is not None:
            value = os.path.join(os.path.dirname(self._path), value)
        elif not os.path.isabs(value):
            value = os.path.abspath(value)
        self._path = value
    
    @property
    def header(self):
        return self.hdr

    @header.setter
    def header(self, value):
        self.hdr = value

    def __deepcopy__(self, memo):
        return type(self)(deepcopy(self.data, memo), deepcopy(self.hdr, memo), deepcopy(self.orientation, memo), deepcopy(self.absolutepath, memo), deepcopy(self.dim, memo))

    def copy(self, image=None):
        if image is not None:
            self.affine = deepcopy(image.affine)
            self.data = deepcopy(image.data)
            self.hdr = deepcopy(image.hdr)
            self._path = deepcopy(image._path)
        else:
            return deepcopy(self)

    def loadFromPath(self, path):
        """
        This function load an image from an absolute path using nibabel library

        :param path: path of the file from which the image will be loaded
        :return:
        """

        self.absolutepath = os.path.abspath(path)
        im_file = nib.load(self.absolutepath, mmap=True)
        self.affine = im_file.affine.copy()
        self.data = np.asanyarray(im_file.dataobj)
        self.hdr = im_file.header.copy()
        if path != self.absolutepath:
            logger.debug("Loaded %s (%s) orientation %s shape %s", path, self.absolutepath, self.orientation, self.data.shape)
        else:
            logger.debug("Loaded %s orientation %s shape %s", path, self.orientation, self.data.shape)

    def change_orientation(self, orientation, inverse=False):
        """
        Change orientation on image (in-place).

        :param orientation: orientation string (SCT "from" convention)

        :param inverse: if you think backwards, use this to specify that you actually\
                        want to transform *from* the specified orientation, not *to*\
                        it.

        """
        change_orientation(self, orientation, self, inverse=inverse)
        return self
        
    def change_type(self, dtype):
        """
        Change data type on image.

        Note: the image path is voided.
        """
        change_type(self, dtype, self)
        return self
    
    def fix_header_dtype(self):
        """
        Change the header dtype to the match the datatype of the array.
        """
        # Using bool for nibabel headers is unsupported, so use uint8 instead:
        # `nibabel.spatialimages.HeaderDataError: data dtype "bool" not supported`
        dtype_data = self.data.dtype
        if dtype_data == bool:
            dtype_data = np.uint8

        dtype_header = self.hdr.get_data_dtype()
        if dtype_header != dtype_data:
            logger.warning(f"Image header specifies datatype '{dtype_header}', but array is of type "
                           f"'{dtype_data}'. Header metadata will be overwritten to use '{dtype_data}'.")
            self.hdr.set_data_dtype(dtype_data)
    
    def save(self, path=None, dtype=None, verbose=0, mutable=False):
        """
        Write an image in a nifti file

        :param path: Where to save the data, if None it will be taken from the\
                     absolutepath member.\
                     If path is a directory, will save to a file under this directory\
                     with the basename from the absolutepath member.

        :param dtype: if not set, the image is saved in the same type as input data\
                      if 'minimize', image storage space is minimized\
                        (2, 'uint8', np.uint8, "NIFTI_TYPE_UINT8"),\
                        (4, 'int16', np.int16, "NIFTI_TYPE_INT16"),\
                        (8, 'int32', np.int32, "NIFTI_TYPE_INT32"),\
                        (16, 'float32', np.float32, "NIFTI_TYPE_FLOAT32"),\
                        (32, 'complex64', np.complex64, "NIFTI_TYPE_COMPLEX64"),\
                        (64, 'float64', np.float64, "NIFTI_TYPE_FLOAT64"),\
                        (256, 'int8', np.int8, "NIFTI_TYPE_INT8"),\
                        (512, 'uint16', np.uint16, "NIFTI_TYPE_UINT16"),\
                        (768, 'uint32', np.uint32, "NIFTI_TYPE_UINT32"),\
                        (1024,'int64', np.int64, "NIFTI_TYPE_INT64"),\
                        (1280, 'uint64', np.uint64, "NIFTI_TYPE_UINT64"),\
                        (1536, 'float128', _float128t, "NIFTI_TYPE_FLOAT128"),\
                        (1792, 'complex128', np.complex128, "NIFTI_TYPE_COMPLEX128"),\
                        (2048, 'complex256', _complex256t, "NIFTI_TYPE_COMPLEX256"),

        :param mutable: whether to update members with newly created path or dtype
        """
        if mutable:  # do all modifications in-place
            # Case 1: `path` not specified
            if path is None:
                if self.absolutepath:  # Fallback to the original filepath
                    path = self.absolutepath
                else:
                    raise ValueError("Don't know where to save the image (no absolutepath or path parameter)")
            # Case 2: `path` points to an existing directory
            elif os.path.isdir(path):
                if self.absolutepath:  # Use the original filename, but save to the directory specified by `path`
                    path = os.path.join(os.path.abspath(path), os.path.basename(self.absolutepath))
                else:
                    raise ValueError("Don't know where to save the image (path parameter is dir, but absolutepath is "
                                     "missing)")
            # Case 3: `path` points to a file (or a *nonexistent* directory) so use its value as-is
            #    (We're okay with letting nonexistent directories slip through, because it's difficult to distinguish
            #     between nonexistent directories and nonexistent files. Plus, `nibabel` will catch any further errors.)
            else:
                pass

            if os.path.isfile(path) and verbose:
                logger.warning("File %s already exists. Will overwrite it.", path)
            if os.path.isabs(path):
                logger.debug("Saving image to %s orientation %s shape %s",
                             path, self.orientation, self.data.shape)
            else:
                logger.debug("Saving image to %s (%s) orientation %s shape %s",
                             path, os.path.abspath(path), self.orientation, self.data.shape)

            # Now that `path` has been set and log messages have been written, we can assign it to the image itself
            self.absolutepath = os.path.abspath(path)

            if dtype is not None:
                self.change_type(dtype)

            if self.hdr is not None:
                self.hdr.set_data_shape(self.data.shape)
                self.fix_header_dtype()

            # nb. that copy() is important because if it were a memory map, save() would corrupt it
            dataobj = self.data.copy()
            affine = None
            header = self.hdr.copy() if self.hdr is not None else None
            nib.save(nib.nifti1.Nifti1Image(dataobj, affine, header), self.absolutepath)
            if not os.path.isfile(self.absolutepath):
                raise RuntimeError(f"Couldn't save image to {self.absolutepath}")
        else:
            # if we're not operating in-place, then make any required modifications on a throw-away copy
            self.copy().save(path, dtype, verbose, mutable=True)
        return self




def get_dimension(im_file, verbose=1):
    """
    Copied from https://github.com/spinalcordtoolbox/spinalcordtoolbox/

    Get dimension from Image or nibabel object. Manages 2D, 3D or 4D images.

    :param: im_file: Image or nibabel object
    :return: nx, ny, nz, nt, px, py, pz, pt
    """
    if not isinstance(im_file, (nib.nifti1.Nifti1Image, Image)):
        raise TypeError("The provided image file is neither a nibabel.nifti1.Nifti1Image instance nor an Image instance")
    # initializating ndims [nx, ny, nz, nt] and pdims [px, py, pz, pt]
    ndims = [1, 1, 1, 1]
    pdims = [1, 1, 1, 1]
    data_shape = im_file.header.get_data_shape()
    zooms = im_file.header.get_zooms()
    for i in range(min(len(data_shape), 4)):
        ndims[i] = data_shape[i]
        pdims[i] = zooms[i]
    return *ndims, *pdims


def change_orientation(im_src, orientation, im_dst=None, inverse=False):
    """
    Copied from https://github.com/spinalcordtoolbox/spinalcordtoolbox/

    :param im_src: source image
    :param orientation: orientation string (SCT "from" convention)
    :param im_dst: destination image (can be the source image for in-place
                   operation, can be unset to generate one)
    :param inverse: if you think backwards, use this to specify that you actually
                    want to transform *from* the specified orientation, not *to* it.
    :return: an image with changed orientation

    .. note::
        - the resulting image has no path member set
        - if the source image is < 3D, it is reshaped to 3D and the destination is 3D
    """

    if len(im_src.data.shape) < 3:
        pass  # Will reshape to 3D
    elif len(im_src.data.shape) == 3:
        pass  # OK, standard 3D volume
    elif len(im_src.data.shape) == 4:
        pass  # OK, standard 4D volume
    elif len(im_src.data.shape) == 5 and im_src.header.get_intent()[0] == "vector":
        pass  # OK, physical displacement field
    else:
        raise NotImplementedError("Don't know how to change orientation for this image")

    im_src_orientation = im_src.orientation
    im_dst_orientation = orientation
    if inverse:
        im_src_orientation, im_dst_orientation = im_dst_orientation, im_src_orientation

    perm, inversion = _get_permutations(im_src_orientation, im_dst_orientation)

    if im_dst is None:
        im_dst = im_src.copy()
        im_dst._path = None

    im_src_data = im_src.data
    if len(im_src_data.shape) < 3:
        im_src_data = im_src_data.reshape(tuple(list(im_src_data.shape) + ([1] * (3 - len(im_src_data.shape)))))

    # Update data by performing inversions and swaps

    # axes inversion (flip)
    data = im_src_data[::inversion[0], ::inversion[1], ::inversion[2]]

    # axes manipulations (transpose)
    if perm == [1, 0, 2]:
        data = np.swapaxes(data, 0, 1)
    elif perm == [2, 1, 0]:
        data = np.swapaxes(data, 0, 2)
    elif perm == [0, 2, 1]:
        data = np.swapaxes(data, 1, 2)
    elif perm == [2, 0, 1]:
        data = np.swapaxes(data, 0, 2)  # transform [2, 0, 1] to [1, 0, 2]
        data = np.swapaxes(data, 0, 1)  # transform [1, 0, 2] to [0, 1, 2]
    elif perm == [1, 2, 0]:
        data = np.swapaxes(data, 0, 2)  # transform [1, 2, 0] to [0, 2, 1]
        data = np.swapaxes(data, 1, 2)  # transform [0, 2, 1] to [0, 1, 2]
    elif perm == [0, 1, 2]:
        # do nothing
        pass
    else:
        raise NotImplementedError()

    # Update header

    im_src_aff = im_src.hdr.get_best_affine()
    aff = nib.orientations.inv_ornt_aff(
        np.array((perm, inversion)).T,
        im_src_data.shape)
    im_dst_aff = np.matmul(im_src_aff, aff)

    im_dst.header.set_qform(im_dst_aff)
    im_dst.header.set_sform(im_dst_aff)
    im_dst.header.set_data_shape(data.shape)
    im_dst.data = data

    return im_dst


def _get_permutations(im_src_orientation, im_dst_orientation):
    """
    Copied from https://github.com/spinalcordtoolbox/spinalcordtoolbox/

    :param im_src_orientation str: Orientation of source image. Example: 'RPI'
    :param im_dest_orientation str: Orientation of destination image. Example: 'SAL'
    :return: list of axes permutations and list of inversions to achieve an orientation change
    """

    opposite_character = {'L': 'R', 'R': 'L', 'A': 'P', 'P': 'A', 'I': 'S', 'S': 'I'}

    perm = [0, 1, 2]
    inversion = [1, 1, 1]
    for i, character in enumerate(im_src_orientation):
        try:
            perm[i] = im_dst_orientation.index(character)
        except ValueError:
            perm[i] = im_dst_orientation.index(opposite_character[character])
            inversion[i] = -1

    return perm, inversion


def get_orientation(im):
    """
    Copied from https://github.com/spinalcordtoolbox/spinalcordtoolbox/

    :param im: an Image
    :return: reference space string (ie. what's in Image.orientation)
    """
    res = "".join(nib.orientations.aff2axcodes(im.hdr.get_best_affine()))
    return orientation_string_nib2sct(res)


def orientation_string_nib2sct(s):
    """
    Copied from https://github.com/spinalcordtoolbox/spinalcordtoolbox/

    :return: SCT reference space code from nibabel one
    """
    opposite_character = {'L': 'R', 'R': 'L', 'A': 'P', 'P': 'A', 'I': 'S', 'S': 'I'}
    return "".join([opposite_character[x] for x in s])


def change_type(im_src, dtype, im_dst=None):
    """
    Change the voxel type of the image

    :param dtype:    if not set, the image is saved in standard type\
                    if 'minimize', image space is minimize\
                    if 'minimize_int', image space is minimize and values are approximated to integers\
                    (2, 'uint8', np.uint8, "NIFTI_TYPE_UINT8"),\
                    (4, 'int16', np.int16, "NIFTI_TYPE_INT16"),\
                    (8, 'int32', np.int32, "NIFTI_TYPE_INT32"),\
                    (16, 'float32', np.float32, "NIFTI_TYPE_FLOAT32"),\
                    (32, 'complex64', np.complex64, "NIFTI_TYPE_COMPLEX64"),\
                    (64, 'float64', np.float64, "NIFTI_TYPE_FLOAT64"),\
                    (256, 'int8', np.int8, "NIFTI_TYPE_INT8"),\
                    (512, 'uint16', np.uint16, "NIFTI_TYPE_UINT16"),\
                    (768, 'uint32', np.uint32, "NIFTI_TYPE_UINT32"),\
                    (1024,'int64', np.int64, "NIFTI_TYPE_INT64"),\
                    (1280, 'uint64', np.uint64, "NIFTI_TYPE_UINT64"),\
                    (1536, 'float128', _float128t, "NIFTI_TYPE_FLOAT128"),\
                    (1792, 'complex128', np.complex128, "NIFTI_TYPE_COMPLEX128"),\
                    (2048, 'complex256', _complex256t, "NIFTI_TYPE_COMPLEX256"),
    :return:

    Copied from https://github.com/spinalcordtoolbox/spinalcordtoolbox/
    """

    if im_dst is None:
        im_dst = im_src.copy()
        im_dst._path = None

    if dtype is None:
        return im_dst

    # get min/max from input image
    min_in = np.nanmin(im_src.data)
    max_in = np.nanmax(im_src.data)

    # find optimum type for the input image
    if dtype in ('minimize', 'minimize_int'):
        # warning: does not take intensity resolution into account, neither complex voxels

        # check if voxel values are real or integer
        isInteger = True
        if dtype == 'minimize':
            for vox in im_src.data.flatten():
                if int(vox) != vox:
                    isInteger = False
                    break

        if isInteger:
            if min_in >= 0:  # unsigned
                if max_in <= np.iinfo(np.uint8).max:
                    dtype = np.uint8
                elif max_in <= np.iinfo(np.uint16):
                    dtype = np.uint16
                elif max_in <= np.iinfo(np.uint32).max:
                    dtype = np.uint32
                elif max_in <= np.iinfo(np.uint64).max:
                    dtype = np.uint64
                else:
                    raise ValueError("Maximum value of the image is to big to be represented.")
            else:
                if max_in <= np.iinfo(np.int8).max and min_in >= np.iinfo(np.int8).min:
                    dtype = np.int8
                elif max_in <= np.iinfo(np.int16).max and min_in >= np.iinfo(np.int16).min:
                    dtype = np.int16
                elif max_in <= np.iinfo(np.int32).max and min_in >= np.iinfo(np.int32).min:
                    dtype = np.int32
                elif max_in <= np.iinfo(np.int64).max and min_in >= np.iinfo(np.int64).min:
                    dtype = np.int64
                else:
                    raise ValueError("Maximum value of the image is to big to be represented.")
        else:
            # if max_in <= np.finfo(np.float16).max and min_in >= np.finfo(np.float16).min:
            #    type = 'np.float16' # not supported by nibabel
            if max_in <= np.finfo(np.float32).max and min_in >= np.finfo(np.float32).min:
                dtype = np.float32
            elif max_in <= np.finfo(np.float64).max and min_in >= np.finfo(np.float64).min:
                dtype = np.float64

        dtype = to_dtype(dtype)
    else:
        dtype = to_dtype(dtype)

        # if output type is int, check if it needs intensity rescaling
        if "int" in dtype.name:
            # get min/max from output type
            min_out = np.iinfo(dtype).min
            max_out = np.iinfo(dtype).max
            # before rescaling, check if there would be an intensity overflow

            if (min_in < min_out) or (max_in > max_out):
                # This condition is important for binary images since we do not want to scale them
                logger.warning(f"To avoid intensity overflow due to convertion to +{dtype.name}+, intensity will be rescaled to the maximum quantization scale")
                # rescale intensity
                data_rescaled = im_src.data * (max_out - min_out) / (max_in - min_in)
                im_dst.data = data_rescaled - (data_rescaled.min() - min_out)

    # change type of data in both numpy array and nifti header
    im_dst.data = getattr(np, dtype.name)(im_dst.data)
    im_dst.hdr.set_data_dtype(dtype)
    return im_dst


def to_dtype(dtype):
    """
    Take a dtypeification and return an np.dtype

    :param dtype: dtypeification (string or np.dtype or None are supported for now)
    :return: dtype or None

    Copied from https://github.com/spinalcordtoolbox/spinalcordtoolbox/
    """
    # TODO add more or filter on things supported by nibabel

    if dtype is None:
        return None
    if isinstance(dtype, type):
        if isinstance(dtype(0).dtype, np.dtype):
            return dtype(0).dtype
    if isinstance(dtype, np.dtype):
        return dtype
    if isinstance(dtype, str):
        return np.dtype(dtype)

    raise TypeError("data type {}: {} not understood".format(dtype.__class__, dtype))
