# Visualizing Convolutional Networks for MRI-based Diagnosis of Alzheimer’s Disease

This is the code for a paper to be presented at [MLCN 2018](https://mlcn2018.com/) by Johannes Rieke, Fabian Eitel, Martin Weygandt, John-Dylan Haynes and Kerstin Ritter.

**Abstract:** Visualizing and interpreting convolutional neural networks (CNNs) is an important task to increase trust in automatic medical decision making systems. In this study, we train a 3D CNN to detect Alzheimer’s disease based on structural MRI scans of the brain. Then, we apply four different gradient-based and occlusion-based visualization methods that explain the network’s classification decisions by highlight- ing relevant areas in the input image. We compare the methods qualita- tively and quantitatively. We find that all four methods focus on brain regions known to be involved in Alzheimer’s disease, such as inferior and middle temporal gyrus. While the occlusion-based methods focus more on specific regions, the gradient-based methods pick up distributed rel- evance patterns. Additionally, we find that the distribution of relevance varies across patients, with some having a stronger focus on the temporal lobe, whereas for others more cortical areas are relevant. In summary, we show that applying different visualization methods is important to understand the decisions of a CNN, a step that is crucial to increase clinical impact and trust in computer-based decision support systems.

![Heatmaps](figures/heatmaps-ad.png)


## Code Structure

The codebase uses PyTorch and Jupyter notebooks. The main files for the paper are:

- `training.ipynb` is the notebook to train the model and perform cross validation.
- `interpretation-mri.ipynb` contains the code to create relevance heatmaps with different visualization methods. It also includes the code to reproduce all figures and tables from the paper.
- All `*.py` files contain methods that are imported in the notebooks above.

Additionally, there are two other notebooks:
- `interpretation-photos.ipynb` uses the same visualization methods as in the paper but applies them to 2D photos. This might be an easier introduction to the topic. 
- `small-dataset.ipynb` contains some old code to run a similar experiment on a smaller dataset.


## Data

The MRI scans used for training are from the [Alzheimer Disease Neuroimaging Initiative (ADNI)](http://adni.loni.usc.edu/). The data is free but you need to apply for access on http://adni.loni.usc.edu/. Once you have an account, go [here](http://adni.loni.usc.edu/data-samples/access-data/) and log in. You need to download two things:

- The data table: Go to "Download" -> "Study Data" -> "Study Info" and download the file "ADNI 1.5T MRI Standardized Lists". Unpack the zip file and move the file "ADNI_CompleteAnnual2YearVisitList_8_22_12.csv" to the folder `data` in this repo.
- The images: You need to download all scans from the table above. TODO: Add more detailled description. 

These are the 1.5 Tesla scans that we used to train the model in the paper. This repo also contains code to train on the 3 Tesla scans or on both types of scans. 



## Requirements

- Python 2 (mostly compatible with Python 3 syntax, but not tested)
- Scientific packages (included with anaconda): numpy, scipy, matplotlib, pandas, jupyter, scikit-learn
- Other packages: tqdm, tabulate
- PyTorch: torch, torchvision (tested with 0.3.1, but mostly compatible with 0.4)
- torchsample: I made a custom fork of torchsample which fixes some bugs. You can download it from https://github.com/jrieke/torchsample or install directly via `pip install git+https://github.com/jrieke/torchsample`. Please use this fork instead of the original package, otherwise the code will break. 



## Non-pytorch models
If your model is not in pytorch, but you still want to use the visualization methods, you can try to transform the model to pytorch ([overview of conversion tools](https://github.com/ysh329/deep-learning-model-convertor)).

For keras to pytorch, I can recommend [nn-transfer](https://github.com/gzuidhof/nn-transfer). If you use it, keep in mind that by default, pytorch uses channels-first format and keras channels-last format for images. Even though nn-transfer takes care of this difference for the orientation of the convolution kernels, you may still need to permute your dimensions in the pytorch model between the convolutional and fully-connected stage (for 3D images, I did `x = x.permute(0, 2, 3, 4, 1).contiguous()`). The safest bet is to switch keras to use channels-first as well, then nn-transfer should handle everything by itself.
