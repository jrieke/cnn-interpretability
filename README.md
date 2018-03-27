# cnn-interpretability
Interpreting CNN classifications on MRI images

# Requirements


# Quickstart


# I'm not using pytorch
If your model is not in pytorch, but you still want to use these interpretation methods, you can try to transform the model to pytorch ([overview of conversion tools](https://github.com/ysh329/deep-learning-model-convertor)).

For keras to pytorch, I can recommend [nn-transfer](https://github.com/gzuidhof/nn-transfer). If you use it, keep in mind that by default, pytorch uses channels-first format and keras channels-last format for images. Even though nn-transfer takes care of this difference for the orientation of the convolution kernels, you may still need to permute your dimensions in the pytorch model between the convolutional and fully-connected stage (for 3D images, I did `x = x.permute(0, 2, 3, 4, 1).contiguous()`). The safest bet is to switch keras to use channels-first as well, then nn-transfer should handle everything by itself.
