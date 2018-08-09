from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

import os
import utils
from tqdm import tqdm_notebook
from sklearn.model_selection import train_test_split
import multiprocessing
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision

from tabulate import tabulate


# Binary brain mask used to cut out the skull.
mask = utils.load_nifti('data/binary_brain_mask.nii.gz')


# ------------------------- ADNI data tables -----------------------------------

# Ritter/Haynes lab file system at BCCN Berlin. 
#ADNI_DIR = '/analysis/share/ADNI'

# Local.
ADNI_DIR = 'data/ADNI'

# Filepaths for 3 Tesla scans.
table_3T = os.path.join(ADNI_DIR, 'ADNI_tables/customized/DxByImgClean_CompleteAnnual2YearVisitList_3T.csv')
image_dir_3T = os.path.join(ADNI_DIR, 'ADNI_2Yr_3T_preprocessed')
corrupt_images_3T = ['037_S_0501/Baseline', '037_S_0501/Month12', '037_S_0501/Month24', '051_S_1123/Baseline', '051_S_1123/Month12', '051_S_1123/Month24', '116_S_0649/Month12', '116_S_0649/Month24', '116_S_1232/Baseline', '027_S_1387/Baseline', '027_S_1387/Month12', '027_S_1387/Month24', '116_S_0382/Baseline', '027_S_0404/Baseline', '027_S_0404/Month24', '027_S_1385/Month12', '023_S_0376/Month12', '023_S_0030/Baseline', '023_S_0030/Month24', '023_S_1247/Baseline', '023_S_1247/Month12', '027_S_1082/Month24', '018_S_0450/Baseline', '005_S_0572/Baseline', '005_S_0572/Month12', '005_S_0572/Month24']

# Filepaths for 1.5 Tesla scans.
table_15T = os.path.join(ADNI_DIR, 'ADNI_tables/customized/DxByImgClean_CompleteAnnual2YearVisitList_1_5T.csv')
image_dir_15T = os.path.join(ADNI_DIR, 'ADNI_2Yr_15T_quick_preprocessed')
corrupt_images_15T = ['067_S_0077/Screening']


# TODO: Maybe rename to load_table or load_adni_table
def load_data_table(table, image_dir, corrupt_images=None):
    """Read data table, find corresponding images, filter out corrupt, missing and MCI images, and return the samples as a pandas dataframe."""
    
    # Read table into dataframe.
    print('Loading dataframe for', table)
    df = pd.read_csv(table)
    print('Found', len(df), 'images in table')
    
    # Add column with filepaths to images.
    df['filepath'] = df.apply(lambda row: get_image_filepath(row, image_dir), axis=1)
    
    # Filter out corrupt images (i.e. images where the preprocessing failed).
    len_before = len(df)
    if corrupt_images is not None:
        df = df[df.apply(lambda row: '{}/{}'.format(row['PTID'], row['Visit']) not in corrupt_images, axis=1)]
    print('Filtered out', len_before - len(df), 'of', len_before, 'images because of failed preprocessing')
    
    # Filter out images where files are missing.
    len_before = len(df)
    #print(df[~np.array(map(os.path.exists, df['filepath']))]['filepath'].values)
    df = df[map(os.path.exists, df['filepath'])]
    print('Filtered out', len_before - len(df), 'of', len_before, 'images because of missing files')
    
    # Filter out images with MCI.
    len_before = len(df)
    df = df[df['DX'] != 'MCI']
    print('Filtered out', len_before - len(df), 'of', len_before, 'images that were MCI')
    
    print('Final dataframe contains', len(df), 'images from', len(df['PTID'].unique()), 'patients')
    print()
    
    return df


def load_data_table_3T():
    """Load the data table for all 3 Tesla images."""
    return load_data_table(table_3T, image_dir_3T, corrupt_images_3T)
    
def load_data_table_15T():
    """Load the data table for all 1.5 Tesla images."""
    return load_data_table(table_15T, image_dir_15T, corrupt_images_15T)
    
def load_data_table_both():
    """Load the data tables for all 1.5 Tesla and 3 Tesla images and combine them."""
    df_15T = load_data_table(table_15T, image_dir_15T, corrupt_images_15T)
    df_3T = load_data_table(table_3T, image_dir_3T, corrupt_images_3T)
    df = pd.concat([df_15T, df_3T])
    return df
    

def get_image_filepath(df_row, root_dir=''):
    """Return the filepath of the image that is described in the row of the data table."""
    # Current format for the image filepath is: 
    # <PTID>/<Visit (spaces removed)>/<PTID>_<Scan.Date (/ replaced by -)>_<Visit (spaces removed)>_<Image.ID>_<DX>_Warped.nii.gz
    filedir = os.path.join(df_row['PTID'], df_row['Visit'].replace(' ', ''))
    filename = '{}_{}_{}_{}_{}_Warped.nii.gz'.format(df_row['PTID'], df_row['Scan.Date'].replace('/', '-'), df_row['Visit'].replace(' ', ''), df_row['Image.ID'], df_row['DX'])
    return os.path.join(root_dir, filedir, filename)

    
    
    
# ------------------------ PyTorch datasets and loaders ----------------------

class ADNIDataset(Dataset):
    """
    PyTorch dataset that consists of MRI images and labels.
    
    Args:
        filenames (iterable of strings): The filenames fo the MRI images.
        labels (iterable): The labels for the images.
        mask (array): If not None (default), images are masked by multiplying with this array.
        transform: Any transformations to apply to the images.
    """

    def __init__(self, filenames, labels, mask=None, transform=None):
        self.filenames = filenames
        self.labels = torch.LongTensor(labels)
        self.mask = mask
        self.transform = transform

        # Required by torchsample.
        self.num_inputs = 1
        self.num_targets = 1

        # Default values. Should be set via fit_normalization.
        self.mean = 0
        self.std = 1

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        """Return the image as a numpy array and the label."""
        label = self.labels[idx]

        struct_arr = utils.load_nifti(self.filenames[idx], mask=self.mask)
        # TDOO: Try normalizing each image to mean 0 and std 1 here.
        #struct_arr = (struct_arr - struct_arr.mean()) / (struct_arr.std() + 1e-10)
        struct_arr = (struct_arr - self.mean) / (self.std + 1e-10)  # prevent 0 division by adding small factor
        struct_arr = struct_arr[None]  # add (empty) channel dimension
        struct_arr = torch.FloatTensor(struct_arr)

        if self.transform is not None:
            struct_arr = self.transform(struct_arr)

        return struct_arr, label

    def image_shape(self):
        """The shape of the MRI images."""
        return utils.load_nifti(self.filenames[0], mask=mask).shape

    def fit_normalization(self, num_sample=None, show_progress=False):
        """
        Calculate the voxel-wise mean and std across the dataset for normalization.
        
        Args:
            num_sample (int or None): If None (default), calculate the values across the complete dataset, 
                                      otherwise sample a number of images.
            show_progress (bool): Show a progress bar during the calculation."
        """
            
        if num_sample is None:
            num_sample = len(self)

        image_shape = self.image_shape()
        all_struct_arr = np.zeros((num_sample, image_shape[0], image_shape[1], image_shape[2]))

        sampled_filenames = np.random.choice(self.filenames, num_sample, replace=False)
        if show_progress:
            sampled_filenames = tqdm_notebook(sampled_filenames)

        for i, filename in enumerate(sampled_filenames):
            struct_arr = utils.load_nifti(filename, mask=mask)
            all_struct_arr[i] = struct_arr

        self.mean = all_struct_arr.mean(0)
        self.std = all_struct_arr.std(0)

    def get_raw_image(self, idx):
        """Return the raw image at index idx (i.e. not normalized, no color channel, no transform."""
        return utils.load_nifti(self.filenames[idx], mask=self.mask)
    

def print_df_stats(df, df_train, df_val):
    """Print some statistics about the patients and images in a dataset."""
    headers = ['Images', '-> AD', '-> CN', 'Patients', '-> AD', '-> CN']

    def get_stats(df):
        df_ad = df[df['DX'] == 'Dementia']
        df_cn = df[df['DX'] == 'CN']
        return [len(df), len(df_ad), len(df_cn), len(df['PTID'].unique()), len(df_ad['PTID'].unique()), len(df_cn['PTID'].unique())]

    stats = []
    stats.append(['All'] + get_stats(df))
    stats.append(['Train'] + get_stats(df_train))
    stats.append(['Val'] + get_stats(df_val))

    print(tabulate(stats, headers=headers))
    print()


# TODO: Rename *_val to *_test.
def build_datasets(df, patients_train, patients_val, print_stats=True, normalize=True):
    """
    Build PyTorch datasets based on a data table and a patient-wise train-test split.
    
    Args:
        df (pandas dataframe): The data table from ADNI.
        patients_train (iterable of strings): The patients to include in the train set.
        patients_val (iterable of strings): The patients to include in the val set.
        print_stats (boolean): Whether to print some statistics about the datasets.
        normalize (boolean): Whether to caluclate mean and std across the dataset for later normalization.
        
    Returns:
        The train and val dataset.
    """
    # Compile train and val dfs based on patients.
    df_train = df[df.apply(lambda row: row['PTID'] in patients_train, axis=1)]
    df_val = df[df.apply(lambda row: row['PTID'] in patients_val, axis=1)]

    if print_stats:
        print_df_stats(df, df_train, df_val)

    # Extract filenames and labels from dfs.
    train_filenames = np.array(df_train['filepath'])
    val_filenames = np.array(df_val['filepath'])
    train_labels = np.array(df_train['DX'] == 'Dementia', dtype=int)#[:, None]
    val_labels = np.array(df_val['DX'] == 'Dementia', dtype=int)#[:, None]

    train_dataset = ADNIDataset(train_filenames, train_labels, mask=mask)
    val_dataset = ADNIDataset(val_filenames, val_labels, mask=mask)

    # TODO: Maybe normalize each scan first, so that they are on a common scale.
    # TODO: Save these values to file together with the model.
    # TODO: Sample over more images.
    if normalize:
        print('Calculating mean and std for normalization:')
        train_dataset.fit_normalization(200, show_progress=True)
        val_dataset.mean, val_dataset.std = train_dataset.mean, train_dataset.std
    else:
        print('Dataset is not normalized, this could dramatically decrease performance')

    return train_dataset, val_dataset


def build_loaders(train_dataset, val_dataset):
    """Build PyTorch data loaders from the datasets."""
    
    # In contrast to Korolev et al. 2017, we do not enforce one sample per class in each batch.
    # TODO: Maybe change batch size to 3 or 4. Check how this affects memory and accuracy.
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=multiprocessing.cpu_count(), pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False, num_workers=multiprocessing.cpu_count(), pin_memory=torch.cuda.is_available())

    return train_loader, val_loader
