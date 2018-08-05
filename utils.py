from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation
import os
import json

import nibabel as nib
from scipy.ndimage.interpolation import zoom
from scipy import ndimage


def save_history(filename, trainer):
    """Save the history from a torchsample trainer to file."""
    with open(filename, 'w+') as f:
        json.dump(trainer.history.epoch_metrics, f)
        
def load_history(filename):
    """Load the history from a torchsample trainer from file."""
    with open(filename) as f:
        return json.load(f)


def plot_learning_curve(history):
    """
    Plot loss and accuracy over epochs, as recorded in a History object
    from training with keras or torchsample.
    """

    fig, axes = plt.subplots(2, sharex=True, figsize=(10, 7))

    epochs = range(1, len(history['loss'])+1)

    plt.sca(axes[0])
    plt.grid()
    plt.plot(epochs, history['loss'], 'b-', label='Train')
    try:
        plt.plot(epochs, history['val_loss'], 'b--', label='Val')
    except KeyError:
        pass
    plt.ylabel('Loss')
    plt.ylim(0, 1.5)
    plt.legend()

    plt.sca(axes[1])
    plt.grid()
    plt.plot(epochs, history['acc_metric'], 'r-', label='Train')
    try:
        plt.plot(epochs, history['val_acc_metric'], 'r--', label='Val')
    except KeyError:
        pass
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy / %')
    plt.legend()



def load_nifti(file_path, mask=None, z_factor=None, remove_nan=True):
    """Load a 3D array from a NIFTI file."""
    img = nib.load(file_path)
    struct_arr = np.array(img.get_data())
    
    if remove_nan:
        struct_arr = np.nan_to_num(struct_arr)
    if mask is not None:
        struct_arr *= mask
    if z_factor is not None:
        struct_arr = np.around(zoom(struct_arr, z_factor), 0)

    return struct_arr


def save_nifti(file_path, struct_arr):
    """Save a 3D array to a NIFTI file."""
    img = nib.Nifti1Image(struct_arr, np.eye(4))
    nib.save(img, file_path)


# Transparent colormap (alpha to red), that is used for plotting an overlay.
# See https://stackoverflow.com/questions/37327308/add-alpha-to-an-existing-matplotlib-colormap
alpha_to_red_cmap = np.zeros((256, 4))
alpha_to_red_cmap[:, 0] = 0.8
alpha_to_red_cmap[:, -1] = np.linspace(0, 1, 256)#cmap.N-20)  # alpha values
alpha_to_red_cmap = mpl.colors.ListedColormap(alpha_to_red_cmap)

red_to_alpha_cmap = np.zeros((256, 4))
red_to_alpha_cmap[:, 0] = 0.8
red_to_alpha_cmap[:, -1] = np.linspace(1, 0, 256)#cmap.N-20)  # alpha values
red_to_alpha_cmap = mpl.colors.ListedColormap(red_to_alpha_cmap)

def plot_slices(struct_arr, num_slices=7, cmap='gray', vmin=None, vmax=None, overlay=None, overlay_cmap=alpha_to_red_cmap, overlay_vmin=None, overlay_vmax=None):
    """
    Plot equally spaced slices of a 3D image (and an overlay) along every axis
    
    Args:
        struct_arr (3D array or tensor): The 3D array to plot (usually from a nifti file).
        num_slices (int): The number of slices to plot for each dimension.
        cmap: The colormap for the image (default: `'gray'`).
        vmin (float): Same as in matplotlib.imshow. If `None`, take the global minimum of `struct_arr`.
        vmax (float): Same as in matplotlib.imshow. If `None`, take the global maximum of `struct_arr`.
        overlay (3D array or tensor): The 3D array to plot as an overlay on top of the image. Same size as `struct_arr`.
        overlay_cmap: The colomap for the overlay (default: `alpha_to_red_cmap`). 
        overlay_vmin (float): Same as in matplotlib.imshow. If `None`, take the global minimum of `overlay`.
        overlay_vmax (float): Same as in matplotlib.imshow. If `None`, take the global maximum of `overlay`.
    """
    if vmin is None:
        vmin = struct_arr.min()
    if vmax is None:
        vmax = struct_arr.max()
    if overlay_vmin is None and overlay is not None:
        overlay_vmin = overlay.min()
    if overlay_vmax is None and overlay is not None:
        overlay_vmax = overlay.max()
    print(vmin, vmax, overlay_vmin, overlay_vmax)
        
    fig, axes = plt.subplots(3, num_slices, figsize=(15, 6))
    intervals = np.asarray(struct_arr.shape) / num_slices

    for axis, axis_label in zip([0, 1, 2], ['x', 'y', 'z']):
        for i, ax in enumerate(axes[axis]):
            i_slice = int(np.round(intervals[axis] / 2 + i * intervals[axis]))
            #print(axis_label, 'plotting slice', i_slice)
            
            plt.sca(ax)
            plt.axis('off')
            plt.imshow(sp.ndimage.rotate(np.take(struct_arr, i_slice, axis=axis), 90), vmin=vmin, vmax=vmax, 
                       cmap=cmap, interpolation=None)
            plt.text(0.03, 0.97, '{}={}'.format(axis_label, i_slice), color='white', 
                     horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
            
            if overlay is not None:
                plt.imshow(sp.ndimage.rotate(np.take(overlay, i_slice, axis=axis), 90), cmap=overlay_cmap, 
                           vmin=overlay_vmin, vmax=overlay_vmax, interpolation=None)

def animate_slices(struct_arr, overlay=None, axis=0, reverse_direction=False, interval=40, vmin=None, vmax=None, overlay_vmin=None, overlay_vmax=None):
    """
    Create a matplotlib animation that moves through a 3D image along a specified axis.
    """
    
    if vmin is None:
        vmin = struct_arr.min()
    if vmax is None:
        vmax = struct_arr.max()
    if overlay_vmin is None and overlay is not None:
        overlay_vmin = overlay.min()
    if overlay_vmax is None and overlay is not None:
        overlay_vmax = overlay.max()
        
    fig, ax = plt.subplots()
    axis_label = ['x', 'y', 'z'][axis]

    # TODO: If I select slice 50 here at the beginning, the plots look different.
    im = ax.imshow(np.take(struct_arr, 0, axis=axis), vmin=vmin, vmax=vmax, cmap='gray', interpolation=None, animated=True)
    if overlay is not None:
        im_overlay = ax.imshow(np.take(overlay, 0, axis=axis), vmin=overlay_vmin, vmax=overlay_vmax, 
                               cmap=alpha_to_red_cmap, interpolation=None, animated=True)
    text = ax.text(0.03, 0.97, '{}={}'.format(axis_label, 0), color='white', 
                   horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
    ax.axis('off')

    def update(i):
        im.set_array(np.take(struct_arr, i, axis=axis))
        if overlay is not None:
            im_overlay.set_array(np.take(overlay, i, axis=axis))
        text.set_text('{}={}'.format(axis_label, i))
        return im, text

    num_frames = struct_arr.shape[axis]
    if reverse_direction:
        frames = np.arange(num_frames-1, 0, -1)
    else:
        frames = np.arange(0, num_frames)
    
    return mpl.animation.FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)


def resize_image(img, size, interpolation=0):
    """Resize img to size. Interpolation between 0 (no interpolation) and 5 (maximum interpolation)."""
    zoom_factors = np.asarray(size) / np.asarray(img.shape)
    return sp.ndimage.zoom(img, zoom_factors, order=interpolation)
