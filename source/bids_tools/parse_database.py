
from pathlib import Path
import json
import argparse
import re
import os
import nibabel as nib
import pandas as pd

# originally developed by J.Kirschke, modified to spinal cord data by Julian McGinnis

# set debugging options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# parse command line arguments
parser = argparse.ArgumentParser(description='BIDSify the MS brain database.')
parser.add_argument('-i', '--input_directory', help='Folder of database.', required=True)
parser.add_argument('-o', '--output_directory', help='Folder of bids database', required=True)
parser.add_argument('-f', '--filename', type=str, default="dataset_description.csv")

args = parser.parse_args()

data_root = Path(args.input_directory)
frame = pd.DataFrame()

dcm_keys = ['AcquisitionDate', 'AcquisitionTime', 'AcquisitionMatrix', 'DeviceSerialNumber',
          'MagneticFieldStrength', 'NumberOfAverages', 'EchoNumbers', 'PixelBandwidth',
          'ImageComments', 'EchoTime', 'Manufacturer', 'ManufacturerModelName', 'PatientAge', 'PatientID',
          'PatientName', 'PatientSex', 'PatientWeight', 'Rows', 'AcquisitionDuration',
          'PixelSpacing', 'SliceThickness', 'EchoTrainLength', 'NumberOfPhaseEncodingSteps', 'FlipAngle',
          'PercentPhaseFieldOfView', 'PercentSampling', 'SeriesDescription',
          'SpacingBetweenSlices', 'RepetitionTime', 'ImagePositionPatient']

def load_json(json_pth):
    """
    :param json_pth: path to jsostringn as
    :return: dictionary list
    """
    with open(json_pth) as json_data:
        dict_list = json.load(json_data)
        json_data.close()
    return dict_list

def get_value_from_json_key(myjson,key):
    """
    :param myjson: json dict
    :param key: key to be extracted
    :return: value
    """
    try:
        value=myjson.get(key,"")
    except:
        value=''
    return value

def get_orientation(aff):
    """
    :param aff:  affine
    :return: string of orientation {sag,ax,cor,iso}
    """
    pixdim = nib.affines.voxel_sizes(aff)
    nii_axes = nib.aff2axcodes(aff)
    if max(pixdim) <= 1:
        orient = "iso"
    else:
        thickest_slice_ind = pixdim.argmax()
        if nii_axes[thickest_slice_ind] == "L" or nii_axes[thickest_slice_ind] == "R":
            orient = "sag"
        elif nii_axes[thickest_slice_ind] == "A" or nii_axes[thickest_slice_ind] == "P":
            orient = "cor"
        elif nii_axes[thickest_slice_ind] == "I" or nii_axes[thickest_slice_ind] == "S":
            orient = "ax"
        else:
            orient = "non"
    return orient

def getSubjectID(path):
    """
    :param path: path to data file
    :return: return the BIDS-compliant subject ID
    """
    stringList = str(path).split("/")
    indices = [i for i, s in enumerate(stringList) if 'sub-' in s]
    text = stringList[indices[0]]
    try:
        found = re.search('sub-m_(.+?)_ses', text).group(1)
    except AttributeError:
        found = ''
    return found

def getSessionID(path):
    """
    :param path: path to data file
    :return: return the BIDS-compliant session ID
    """
    stringList = str(path).split("/")
    indices = [i for i, s in enumerate(stringList) if '_ses-' in s]
    text = stringList[indices[0]]
    try:
        found = re.search('_ses-(.+?)_', text).group(1)
    except AttributeError:
        found = ''
    return found

def assignChunkId(df_frame,df_selection):
    """
    :param df_frame: pandas dataframe of complete database
    :param df_frame: pandas dataframe of selection that has changed
    :return: N/A
    """
    id = 0
    for index, row in df_selection.iterrows():
        id += 1
        df_frame.at[index, 'chunkID'] = int(id)

if __name__ == '__main__':

    dirs = sorted(list(data_root.glob('*')))
    print(f'Number of directories: {len(dirs)}')
    i = 0
    for dir in dirs:  # iterate over subdirs
        files = sorted(list(dir.rglob('*.nii.gz')))  # within raw data dir
        files = [x for x in files if "t2" in str(x)] # only t2 sequences
        if len(files) > 0:  # skip root_dirs and empty dirs
            i += 1
            print(f' Progress: {i}/{len(dirs)}, current_directory: {dir}')
            for ses, img_path in enumerate(files):
                hdr_path = img_path.with_name(img_path.name.replace('.nii.gz', '.json'))
                if not img_path.is_file():
                    print('error', img_path.name)
                    tmp_frame = pd.DataFrame(index=[img_path.name.replace('.nii.gz', '')],dtype=object)
                    tmp_frame['error'] = 'missing_img'
                else:
                    tmp_frame = pd.DataFrame(index=[img_path.name.replace('.nii.gz', '')])
                    # try:
                    tmp_frame['Session'] = str(ses)
                    img = nib.load(img_path)
                    aff = img.affine

                    tmp_frame['SubjectID'] = getSubjectID(img_path)
                    tmp_frame['SessionDate'] = getSessionID(img_path)

                    pixdim = nib.affines.voxel_sizes(aff)
                    pixshape = img.header.get_data_shape()
                    tmp_frame['ornt'] = get_orientation(aff)
                    for dim, val in enumerate(pixdim):
                        tmp_frame['res_dim_' + str(dim)] = val
                    for dim, val in enumerate(pixshape):
                        tmp_frame['n_pix_' + str(dim)] = val
                    if hdr_path.is_file():
                        dcm_dict = load_json(hdr_path)
                        for x in dcm_keys:
                            a = get_value_from_json_key(dcm_dict, x)#
                            if type(a) is not list:
                                tmp_frame[x] =a
                            else:
                                y = 0
                                for j in a:
                                    tmp_frame[f'{x}_{y}'] = j
                                    y += 1

                    else:
                        tmp_frame['error'] = 'missing_hdr'

                tmp_frame['FileID'] = img_path.name.replace('.nii.gz', '')
                tmp_frame['ImgPath'] = img_path
                frame = frame.append(tmp_frame, ignore_index=True, sort=False)
                del tmp_frame

    # perform computations on patient and session_ids

    # get all patient ids
    frame['index'] = frame.index
    frame['chunkID'] = 0
    frame['stitched'] = False
    patient_ids = list(frame['PatientID'].unique())

    for id in patient_ids:
        df = frame.loc[frame['PatientID'] == id]
        # obtain unique session dates
        session_dates = list(df['SessionDate'].unique())
        for date in session_dates:
            i = 0
            # to create chunk ids, we seperate the patient df into smaller dfs
            ax = df.loc[(df["ornt"] == 'ax') & (df["SessionDate"] == date)].sort_values(by=['ImagePositionPatient_2']) # z-axis
            sag = df.loc[(df["ornt"] == 'sag') & (df["SessionDate"] == date)].sort_values(by=['ImagePositionPatient_2']) # z-axis
            cor = df.loc[(df["ornt"] == 'cor') & (df["SessionDate"] == date)].sort_values(by=['ImagePositionPatient_2']) # z-axis
            iso = df.loc[(df["ornt"] == 'iso') & (df["SessionDate"] == date)].sort_values(by=['ImagePositionPatient_2']) # z-axis
            # assign chunk ids to files
            assignChunkId(frame,ax)
            assignChunkId(frame,sag)
            assignChunkId(frame,cor)
            assignChunkId(frame,iso)

    # label as stitch if dim_0 != dim_1
    for index, row in frame.iterrows():
        if frame.at[index,'n_pix_0'] != frame.at[index,'n_pix_1']:
            frame.at[index, 'stitched'] = True
        else:
            frame.at[index, 'stitched'] = False

    frame.to_csv(os.path.join(args.output_directory,args.filename))