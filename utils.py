from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

import nibabel as nib
from scipy.ndimage.interpolation import zoom


def plot_learning_curve(history):
    """
    Plot loss and accuracy over epochs, as recorded in a History object
    from training with keras or torchsample.
    """

    fig, axes = plt.subplots(2, sharex=True, figsize=(10, 7))

    epochs = range(1, len(history['loss'])+1)

    plt.sca(axes[0])
    plt.plot(epochs, history['loss'], 'b-', label='Train')
    plt.plot(epochs, history['val_loss'], 'b--', label='Val')
    plt.ylabel('Loss')
    plt.ylim(0, 1.5)
    plt.legend()

    plt.sca(axes[1])
    plt.plot(epochs, history['acc_metric'], 'r-', label='Train')
    plt.plot(epochs, history['val_acc_metric'], 'r--', label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy / %')
    plt.legend()



# TODO: Does z_factor in combination with mask make sense (because it wasn't included in load_masked_nifti)?
# TODO: If the masking operation is to performance intenstive, move it to the gpu via pytorch.
def load_nifti(file_path, mask=None, z_factor=None, remove_nan=True):
    """Load a 3D array from a NIFTI file."""
    img = nib.load(file_path)
    struct_arr = np.array(img.get_data())
    if remove_nan:
        struct_arr = np.nan_to_num(struct_arr)#.nan_to_num(copy=False)
        # TODO: If this ever gives an error again, change it to the alternative below.
            #    struct_arr[np.where(np.isnan(struct_arr))] = 0
    if mask is not None:
        # TODO: Maybe use numpy's masked array here? Check out if it is faster.
        struct_arr *= mask
    if z_factor is not None:
        struct_arr = np.around(zoom(struct_arr, z_factor), 0)

    return struct_arr


# Transparent colormap (alpha to red), that is used for plotting an overlay.
# See https://stackoverflow.com/questions/37327308/add-alpha-to-an-existing-matplotlib-colormap
cmap = plt.cm.Reds
alpha_cmap = cmap(np.arange(cmap.N))
alpha_cmap[:,-1] = np.linspace(0, 1, cmap.N)
alpha_cmap = mpl.colors.ListedColormap(alpha_cmap)


# TODO: Calculating the slice numbers gives index error for some values of num_slices.
# TODO: Show figure colorbar.
# TODO: Calculate vmin and vmax automatically, maybe by log-scaling the overlay.

def plot_slices(struct_arr, num_slices=10, overlay=None, overlay_vmin=0, overlay_vmax=0.005):
    """
    Plot equally spaced slices of a 3D image along each dimension.
    """
    fig, axes = plt.subplots(3, 8, figsize=(15, 6))

    dimensions = np.asarray(struct_arr.shape)
    intervals = dimensions / (num_slices - 1)

    def plot_slice(arr_slice, i_slice, dimension_label, ax):
        plt.sca(ax)
        plt.axis('off')
        plt.imshow(arr_slice, cmap='gray', interpolation=None)
        plt.text(0.03, 0.97, '{}={}'.format(dimension_label, i_slice), color='white', horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)


    for i, ax in enumerate(axes[0]):
        i_slice = int(intervals[0] / 2 + i * intervals[0])
        #if channels_first:
        #    arr_slice = struct_arr[0, i_slice, :, :]
        #else:
        arr_slice = struct_arr[i_slice, :, :]
        plot_slice(arr_slice, i_slice, 'x', ax)

        if overlay is not None:
            plt.imshow(overlay[i_slice, :, :], cmap=alpha_cmap, vmin=overlay_vmin, vmax=overlay_vmax, interpolation=None)
            #plt.colorbar()

    for i, ax in enumerate(axes[1]):
        i_slice = int(intervals[1] / 2 + i * intervals[1])
        #if channels_first:
        #    arr_slice = struct_arr[0, :, i_slice, :]
        #else:
        arr_slice = struct_arr[:, i_slice, :]
        plot_slice(arr_slice, i_slice, 'y', ax)

        if overlay is not None:
            plt.imshow(overlay[:, i_slice, :], cmap=alpha_cmap, vmin=overlay_vmin, vmax=overlay_vmax, interpolation=None)
            #plt.colorbar()


    for i, ax in enumerate(axes[2]):
        i_slice = int(intervals[2] / 2 + i * intervals[2])
        #if channels_first:
        #    arr_slice = struct_arr[0, :, :, i_slice]
        #else:
        arr_slice = struct_arr[:, :, i_slice]
        plot_slice(arr_slice, i_slice, 'z', ax)

        if overlay is not None:
            plt.imshow(overlay[:, :, i_slice], cmap=alpha_cmap, vmin=overlay_vmin, vmax=overlay_vmax, interpolation=None)
            #plt.colorbar()

#plot_slices(struct_arr)#, channels_first=False)
