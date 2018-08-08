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

from utils import resize_image


# ---------------------------- Interpretation methods --------------------------------
def sensitivity_analysis(model, image_tensor, target_class=None, postprocess='abs', apply_softmax=True, cuda=False, verbose=False):
    """
    Perform sensitivity analysis (via backpropagation; Simonyan et al. 2014) to determine the relevance of each image pixel 
    for the classification decision. Return a relevance heatmap over the input image.
    
    Args:
        model (torch.nn.Module): The pytorch model. Should be set to eval mode. 
        image_tensor (torch.Tensor or numpy.ndarray): The image to run through the `model` (channels first!).
        target_class (int): The target output class for which to produce the heatmap. 
                      If `None` (default), use the most likely class from the `model`s output.
        postprocess (None or 'abs' or 'square'): The method to postprocess the heatmap with. `'abs'` is used 
                                                 in Simonyan et al. 2014, `'square'` is used in Montavon et al. 2018.
        apply_softmax (boolean): Whether to apply the softmax function to the output. Useful for models that are trained 
                                 with `torch.nn.CrossEntropyLoss` and do not apply softmax themselves.
        appl (None or 'binary' or 'categorical'): Whether the output format of the `model` is binary 
                                                         (i.e. one output neuron with sigmoid activation) or categorical 
                                                         (i.e. multiple output neurons with softmax activation). 
                                                         If `None` (default), infer from the shape of the output. 
        cuda (boolean): Whether to run the computation on a cuda device.
        verbose (boolean): Whether to display additional output during the computation.
        
    Returns:
        A numpy array of the same shape as image_tensor, indicating the relevance of each image pixel. 
    """
    if postprocess not in [None, 'abs', 'square']:
        raise ValueError("postprocess must be None, 'abs' or 'square'")
    
    # Forward pass.
    image_tensor = torch.Tensor(image_tensor)  # convert numpy or list to tensor
    if cuda:
        image_tensor = image_tensor.cuda()
    X = Variable(image_tensor[None], requires_grad=True)  # add dimension to simulate batch
    output = model(X)
    if apply_softmax:
        output = F.softmax(output)
    
    # Backward pass.
    model.zero_grad()
    output_class = output.max(1)[1].data[0]
    if verbose: print('Image was classified as', output_class, 'with probability', output.max(1)[0].data[0])
    one_hot_output = torch.zeros(output.size())
    if target_class is None:
        one_hot_output[0, output_class] = 1
    else:
        one_hot_output[0, target_class] = 1
    if cuda:
        one_hot_output = one_hot_output.cuda()
    output.backward(gradient=one_hot_output)
        
    relevance_map = X.grad.data[0].cpu().numpy()
    
    # Postprocess the relevance map.
    if postprocess == 'abs':  # as in Simonyan et al. (2014)
        return np.abs(relevance_map)
    elif postprocess == 'square':  # as in Montavon et al. (2018)
        return relevance_map**2
    elif postprocess is None:
        return relevance_map
        
        
def guided_backprop(model, image_tensor, target_class=None, postprocess='abs', apply_softmax=True, cuda=False, verbose=False):
    """
    Perform guided backpropagation (Springenberg et al. 2015) to determine the relevance of each image pixel 
    for the classification decision. Return a relevance heatmap over the input image.
    
    Note: The `model` MUST implement any ReLU layers via `torch.nn.ReLU` (i.e. it needs to have an instance 
    of torch.nn.ReLU as an attribute). Models that use `torch.nn.functional.relu` instead will not work properly!
    
    Args:
        model (torch.nn.Module): The pytorch model. Should be set to eval mode. 
        image_tensor (torch.Tensor or numpy.ndarray): The image to run through the `model` (channels first!).
        target (int): The target output class for which to produce the heatmap. 
                      If `None` (default), use the most likely class from the `model`s output.
        postprocess (None or 'abs' or 'square'): The method to postprocess the heatmap with. `'abs'` is used 
                                                 in Simonyan et al. 2013, `'square'` is used in Montavon et al. 2018.
        apply_softmax (boolean): Whether to apply the softmax function to the output. Useful for models that are trained 
                                 with `torch.nn.CrossEntropyLoss` and do not apply softmax themselves.
        cuda (boolean): Whether to run the computation on a cuda device.
        verbose (boolean): Whether to display additional output during the computation.
        
    Returns:
        A numpy array of the same shape as image_tensor, indicating the relevance of each image pixel. 
    """
    layer_to_hook = nn.ReLU
    
    def relu_hook_function(module, grad_in, grad_out):
        """
        If there is a negative gradient, change it to zero.
        """
        return (torch.clamp(grad_in[0], min=0.0),)
        
    hook_handles = []
    
    try:
        # Loop through layers, hook up ReLUs with relu_hook_function, store handles to hooks.
        for module in model.children():
            # TODO: Maybe hook on ELU layers as well (or on others?).
            if isinstance(module, layer_to_hook):
                # TODO: Add a warning if no activation layers have been hooked, so that the user does not forget 
                #       to invoke the activation via nn.ReLU instead of F.relu.
                if verbose: print('Registered hook for layer:', module)
                hook_handle = module.register_backward_hook(relu_hook_function)
                hook_handles.append(hook_handle)

        # Calculate backprop with modified ReLUs.
        relevance_map = sensitivity_analysis(model, image_tensor, target_class=target_class, postprocess=postprocess, apply_softmax=apply_softmax, cuda=cuda, verbose=verbose)
        
    finally:
        # Remove hooks from model.
        # The finally clause re-raises any possible exceptions.
        if verbose: print('Removing {} hook(s)'.format(len(hook_handles)))
        for hook_handle in hook_handles:
            hook_handle.remove()
            del hook_handle
        
    return relevance_map


def occlusion(model, image_tensor, target_class=None, size=50, stride=25, occlusion_value=0, apply_softmax=True, three_d=None, resize=True, cuda=False, verbose=False):
    """
    Perform occlusion (Zeiler & Fergus 2014) to determine the relevance of each image pixel 
    for the classification decision. Return a relevance heatmap over the input image.
    
    Note: The current implementation can only handle 2D and 3D images. 
    It usually infers the correct image dimensions, otherwise they can be set via the `three_d` parameter. 
    
    Args:
        model (torch.nn.Module): The pytorch model. Should be set to eval mode. 
        image_tensor (torch.Tensor or numpy.ndarray): The image to run through the `model` (channels first!).
        target_class (int): The target output class for which to produce the heatmap. 
                      If `None` (default), use the most likely class from the `model`s output.
        size (int): The size of the occlusion patch.
        stride (int): The stride with which to move the occlusion patch across the image.
        occlusion_value (int): The value of the occlusion patch.
        apply_softmax (boolean): Whether to apply the softmax function to the output. Useful for models that are trained 
                                 with `torch.nn.CrossEntropyLoss` and do not apply softmax themselves.
        three_d (boolean): Whether the image is 3 dimensional (e.g. MRI scans). 
                           If `None` (default), infer from the shape of `image_tensor`. 
        resize (boolean): The output from the occlusion method is usually smaller than the original `image_tensor`. 
                          If `True` (default), the output will be resized to fit the original shape (without interpolation). 
        cuda (boolean): Whether to run the computation on a cuda device.
        verbose (boolean): Whether to display additional output during the computation.
        
    Returns:
        A numpy array of the same shape as image_tensor, indicating the relevance of each image pixel. 
    """
    
    # TODO: Try to make this better, i.e. generalize the method to any kind of input.
    if three_d is None:
        three_d = (len(image_tensor.shape) == 4)  # guess if input image is 3D
    
    image_tensor = torch.Tensor(image_tensor)  # convert numpy or list to tensor
    if cuda:
        image_tensor = image_tensor.cuda()
    output = model(Variable(image_tensor[None], requires_grad=False)).cpu()
    if apply_softmax:
        output = F.softmax(output)
 
    output_class = output.max(1)[1].data.numpy()[0]
    if verbose: print('Image was classified as', output_class, 'with probability', output.max(1)[0].data[0])
    if target_class is None:
        target_class = output_class
    unoccluded_prob = output.data[0, target_class]
        
    width = image_tensor.shape[1]
    height = image_tensor.shape[2]
    
    xs = range(0, width, stride)
    ys = range(0, height, stride)
    
    # TODO: Maybe use torch tensor here.
    if three_d:
        depth = image_tensor.shape[3]
        zs = range(0, depth, stride)
        relevance_map = np.zeros((len(xs), len(ys), len(zs)))
    else:
        relevance_map = np.zeros((len(xs), len(ys)))
    
    if verbose:
        xs = tqdm_notebook(xs, desc='x')
        ys = tqdm_notebook(ys, desc='y', leave=False)
        if three_d:
            zs = tqdm_notebook(zs, desc='z', leave=False)
            
    image_tensor_occluded = image_tensor.clone()  # TODO: Check how long this takes.
    
    if cuda:
        image_tensor_occluded = image_tensor_occluded.cuda()
    
    for i_x, x in enumerate(xs):
        x_from = max(x - int(size/2), 0)
        x_to = min(x + int(size/2), width)
        
        for i_y, y in enumerate(ys):
            y_from = max(y - int(size/2), 0)
            y_to = min(y + int(size/2), height)
            
            if three_d:
                for i_z, z in enumerate(zs):
                    z_from = max(z - int(size/2), 0)
                    z_to = min(z + int(size/2), depth)

                    #if verbose: print('Occluding from x={} to x={} and y={} to y={} and z={} to z={}'.format(x_from, x_to, y_from, y_to, z_from, z_to))

                    image_tensor_occluded.copy_(image_tensor)
                    image_tensor_occluded[:, x_from:x_to, y_from:y_to, z_from:z_to] = occlusion_value
            
                    # TODO: Maybe run this batched.
                    output = model(Variable(image_tensor_occluded[None], requires_grad=False))
                    if apply_softmax:
                        output = F.softmax(output)
                    
                    occluded_prob = output.data[0, target_class]
                    relevance_map[i_x, i_y, i_z] = unoccluded_prob - occluded_prob
                    
            else:
                #if verbose: print('Occluding from x={} to x={} and y={} to y={}'.format(x_from, x_to, y_from, y_to, z_from, z_to))
                image_tensor_occluded.copy_(image_tensor)
                image_tensor_occluded[:, x_from:x_to, y_from:y_to] = occlusion_value
                
                # TODO: Maybe run this batched.
                output = model(Variable(image_tensor_occluded[None], requires_grad=False))
                if apply_softmax:
                    output = F.softmax(output)
                
                occluded_prob = output.data[0, target_class]
                relevance_map[i_x, i_y] = unoccluded_prob - ocluded_prob
                
    relevance_map = np.maximum(relevance_map, 0)
                    
    if resize:
        relevance_map = utils.resize_image(relevance_map, image_tensor.shape[1:])
                
    return relevance_map


def area_occlusion(model, image_tensor, area_masks, target_class=None, occlusion_value=0, apply_softmax=True, cuda=False, verbose=False):
    """
    Perform brain area occlusion to determine the relevance of each image pixel 
    for the classification decision. Return a relevance heatmap over the input image.
    
    Args:
        model (torch.nn.Module): The pytorch model. Should be set to eval mode. 
        image_tensor (torch.Tensor or numpy.ndarray): The image to run through the `model` (channels first!).
        target_class (int): The target output class for which to produce the heatmap. 
                      If `None` (default), use the most likely class from the `model`s output.
        occlusion_value (int): The value of the occlusion patch.
        apply_softmax (boolean): Whether to apply the softmax function to the output. Useful for models that are trained 
                                 with `torch.nn.CrossEntropyLoss` and do not apply softmax themselves.
        cuda (boolean): Whether to run the computation on a cuda device.
        verbose (boolean): Whether to display additional output during the computation.
        
    Returns:
        A numpy array of the same shape as image_tensor, indicating the relevance of each image pixel. 
    """
    
    image_tensor = torch.Tensor(image_tensor)  # convert numpy or list to tensor
    if cuda:
        image_tensor = image_tensor.cuda()
    output = model(Variable(image_tensor[None], requires_grad=False))
    if apply_softmax:
        output = F.softmax(output)
   
    output_class = output.max(1)[1].data.cpu().numpy()[0]
    if verbose: print('Image was classified as', output_class, 'with probability', output.max(1)[0].data[0])
    if target_class is None:
        target_class = output_class
    unoccluded_prob = output.data[0, target_class]
        
    relevance_map = torch.zeros(image_tensor.shape[1:])
    if cuda:
        relevance_map = relevance_map.cuda()
        
    for area_mask in tqdm_notebook(area_masks) if verbose else area_masks:
        # TODO: Maybe have area_mask as tensor in the first place.
        area_mask = torch.FloatTensor(area_mask)
        if cuda:
            area_mask = area_mask.cuda()
        image_tensor_occluded = image_tensor * (1 - area_mask).view(image_tensor.shape)
        
        output = model(Variable(image_tensor_occluded[None], requires_grad=False))
        if apply_softmax:
            output = F.softmax(output)
            
        occluded_prob = output.data[0, target_class]
        relevance_map[area_mask.view(image_tensor.shape) == 1] = (unoccluded_prob - occluded_prob)

    relevance_map = relevance_map.cpu().numpy()
    relevance_map = np.maximum(relevance_map, 0)
    return relevance_map



def all_children(model):
    """Return a list of all child modules of the model, and their children, and their children's children, ..."""
    children = list(model.children())
    for child in model.children():
        children.extend(all_children(child))
    return children

def grad_cam(model, image_tensor, target_class=None, last_conv_layer=None, resize=True, interpolation=1, apply_softmax=True, cuda=False, verbose=False):
    """
    Perform Grad-CAM (Selvaraju et al. 2017) to determine the relevance of each image pixel 
    for the classification decision. Return a relevance heatmap over the input image.
    
    NOTE: THIS IMPLEMENTATION IS BUGGY AND IS ONLY PROVIDED FOR COMPLETENESS.
    
    Args:
        model (torch.nn.Module): The pytorch model. Should be set to eval mode. 
        image_tensor (torch.Tensor or numpy.ndarray): The image to run through the `model` (channels first!).
        target_class (int): The target output class for which to produce the heatmap. 
                      If `None` (default), use the most likely class from the `model`s output.
        last_conv_layer (torch.nn.Module): The last conv layer of the `model`. If `None` (default), 
                                           infer from the structure of the model itself.
        resize (boolean): The output from the occlusion method is usually smaller than the original `image_tensor`. 
                          If `True` (default), the output will be resized to fit the original shape.
        interpolation (int): The interpolation to use for the resizing (0-5).
        apply_softmax (boolean): Whether to apply the softmax function to the output. Useful for models that are trained 
                                 with `torch.nn.CrossEntropyLoss` and do not apply softmax themselves.
        cuda (boolean): Whether to run the computation on a cuda device.
        verbose (boolean): Whether to display additional output during the computation.
        
    Returns:
        A numpy array of the same shape as image_tensor, indicating the relevance of each image pixel. 
    """
    
    image_tensor = torch.Tensor(image_tensor)
    if cuda:
        image_tensor = image_tensor.cuda()
    X = Variable(image_tensor[None], requires_grad=False)  # add dimension to simulate batch
    
    # Step 1: Find the last conv layer if it is not set manually.
    if last_conv_layer is None:
        for module in all_children(model)[::-1]:  # traverse model backwards
            if type(module) == nn.Conv1d or type(module) == nn.Conv2d or type(module) == nn.Conv3d:
                last_conv_layer = module
                if verbose: print('Found last conv layer:', last_conv_layer)
                break
        if last_conv_layer is None:
            raise ValueError('Could not find last conv layer automatically, please set it manually')
    
    
    # Step 2: Register hooks on the last conv layer to store the (forward) feature maps and the gradient.
    store = {}  # hacky way to store values in this function from hooks
    
    def feature_maps_hook(module, input, output):
        store['feature_maps'] = output.data.cpu().numpy()[0]
        print('Stored feature maps')
        
    def gradient_hook(module, grad_in, grad_out):
        store['gradient'] = grad_out[0].data.cpu().numpy()[0]
        print('Stored gradient')
        
        
    hook_handle_forward = last_conv_layer.register_forward_hook(feature_maps_hook)
    hook_handle_backward = last_conv_layer.register_backward_hook(gradient_hook)
    
    try:
        # Step 3: Run model forward (will store feature maps of the last conv layer).
        output = model(X)
        if apply_softmax:
            output = F.softmax(output)
        
        # Step 4: Run model backwards (will store gradient of the last conv layer).
        model.zero_grad()
        output_class = output.max(1)[1].cpu().data.numpy()[0]
        if verbose: print('Image was classified as', output_class, 'with probability', output.max(1)[0].cpu().data.numpy()[0])
        one_hot_output = torch.zeros(output.shape)
        if cuda: 
            one_hot_output = one_hot_output.cuda()
        if target_class is None:
            one_hot_output[0, output_class] = 1
        else:
            one_hot_output[0, target_class] = 1
        print(one_hot_output)
        output.backward(gradient=one_hot_output)
        #print(one_hot_output)
        
        # TODO: Maybe implement this in native pytorch and return a tensor.
        # Step 5: Calculate CAM as linear combination of each feature map and its averaged gradient.
        coefficients = store['gradient'].mean(axis=tuple(range(1, store['gradient'].ndim)), keepdims=True)
        print(np.min(coefficients), np.mean(coefficients), np.max(coefficients))
        print(coefficients.shape)
        print(store['feature_maps'].shape)
        #print(coefficients)
        #store['feature_maps'] = np.array([resize_image(fm, mask.shape)*val_dataset.std+val_dataset.mean for fm in store['feature_maps']])
        cam = (coefficients * store['feature_maps']).sum(axis=0)
        print(cam.shape)
        print(np.min(cam), np.mean(cam), np.max(cam))
        
        vmin = np.min(coefficients * store['feature_maps'])
        vmax = np.max(coefficients * store['feature_maps'])
        #for fm, g, c in zip(store['feature_maps'], store['gradient'], coefficients.flatten()):
        #    if c in sorted(coefficients, reverse=True)[:5]:
                #print(c)
                #utils.plot_slices(fm*c, num_slices=3, vmin=vmin, vmax=vmax)
                #utils.plot_slices(mask, overlay=resize_image(fm, mask.shape)*val_dataset.std+val_dataset.mean, num_slices=3)
        
        
        cam = np.maximum(cam, 0)
        print(np.min(cam), np.mean(cam), np.max(cam))
        
        #set_trace()
        
        if resize:
            cam = resize_image(cam, image_tensor.shape[1:], interpolation=interpolation)
        
        return cam
        
    finally:
        hook_handle_forward.remove()
        hook_handle_backward.remove()
        
        
        
# ----------------------------------- Averages over datasets ---------------

def average_over_dataset(interpretation_method, model, dataset, num_samples=None, seed=None, show_progress=False, **kwargs):
    """Apply an interpretation method to each sample of a dataset, and average separately over AD and NC samples."""
    
    if seed is not None:
        np.random.seed(seed)
        
    # Run once to figure out shape of the relevance map. Cannot be inferred 
    # from image shape alone, because some interpretation methods contain
    # a channel dimension and some not.
    struct_arr, label = dataset[0]
    relevance_map = interpretation_method(model, struct_arr, **kwargs)
        
    avg_relevance_map_AD = np.zeros_like(relevance_map)
    avg_relevance_map_NC = np.zeros_like(relevance_map)
    
    count_AD = 0
    count_NC = 0
    
    if num_samples is None:
        idx = range(len(dataset))
    else: 
        idx = np.random.choice(len(dataset), num_samples, replace=False)
    if show_progress:
        idx = tqdm_notebook(idx)
        
    for i in idx:
        struct_arr, label = dataset[i]

        relevance_map = interpretation_method(model, struct_arr, **kwargs)

        if label == 1:
            #print(label, 'adding to AD', relevance_map.shape)
            avg_relevance_map_AD += relevance_map#[0]
            count_AD += 1
        else:
            #print(label, 'adding to NC', relevance_map.shape)
            avg_relevance_map_NC += relevance_map#[0]
            count_NC += 1
            
    avg_relevance_map_AD /= count_AD
    avg_relevance_map_NC /= count_NC
    
    avg_relevance_map_all = (avg_relevance_map_AD * count_AD + avg_relevance_map_NC * count_NC) / (count_AD + count_NC)
    
    return avg_relevance_map_AD, avg_relevance_map_NC, avg_relevance_map_all
    
    
    
# ------------------------------ Distance between heatmaps ----------------------------

def heatmap_distance(a, b):
    """Calculate the Euclidean distance between two n-dimensional arrays."""
    
    def preprocess(arr):
        """Preprocess an array for use in Euclidean distance."""
        #arr = arr * mask
        arr = arr.flatten()
        arr = arr / arr.sum()  # normalize to sum 1
        #arr = arr.clip(1e-20, None)  # avoid 0 division in KL divergence
        return arr
    
    a, b = preprocess(a), preprocess(b)
    
    # Euclidean distance.
    return np.sqrt(np.sum((a - b)**2))

    # Wasserstein Distance.
    #return sp.stats.wasserstein_distance(a, b)
    
    # Symmetric KL Divergence (ill-defined for arrays with 0 values!).
    #return sp.stats.entropy(a, b) + sp.stats.entropy(b, a)