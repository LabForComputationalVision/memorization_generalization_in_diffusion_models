# Generalization in diffusion models arises from geometry-adaptive harmonic representations
ICLR 2024 (accepted as oral):\
[paper on openreview](https://openreview.net/forum?id=ANvmVS2Yr0&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2024%2FConference%2FAuthors%23your-submissions))\
[paper on arXiv (v1: 4 Oct 2023 , v2: 15 Mar 2024)](https://arxiv.org/pdf/2310.02557.pdf) \
Zahra Kadkhodaie, Florentin Guth, Eero P. Simoncelli, Stephane Mallat<br>


## Contents of this repository:
### 1. Pre-trained denoisers
The [denoisers](denoisers) directory contains several denoisers, trained for removing Gaussian noise from images with the objective of minimizing mean square error. 
All denoisers are universal and "blind": they can remove noise of any standard deviation, and this standard deviation does not need to be specified. 
The [denoisers](denoisers) directory contains a separate folder for each architecture (UNet, BF_CNN), with code specified in [code/network.py](code/network.py). 
Within each architecure directory, there are multiple folders containing variants of that denoiser trained on different datasets.

### 2. Code
The <var>code</var> directory contains the python code for 
- *training* a universal blind denoiser used in a diffusion framework:
  * run celebA_to_torch.py to preprocess images before training 
  * run main.py which calls the trainer.py to train a denoiser. 
- *sampling* algorithm to sample from the prior embedded in the denoiser:
  * see the notebooks
- and all the required helper functions such as
  * code for generating synthetic $C^{\alpha}$ and Disc images
  * code to compute the eigen basis of the denoiser operation
  * helper functions to support above and demos
### 3. Notebooks
The notebooks folder contains demo code for generating results and figures shown in the paper. 

### Requirements 
You'll need python version 3.9.13 and pytorch 1.13.1 and the following packages to execute the code: \\

os \
time \
sys \
gzip \
skimage 0.19.2 \
matplotlib 3.5.2 \
argparse 1.1 \
scipy 1.9.1 \
PIL 9.2.0 \
pywt 1.3.0

## Summary 
Deep neural networks (DNNs) trained for image denoising are able to generate high-quality samples with score-based reverse diffusion algorithms. But how do they acheive this feat? There are two possible candidate strategies: 
1. They might memorize the training set images or patches of those images and reproducing them when sampling new images (Somepalli et al., 2023; Carlini et al., 2023). This is a form of overfitting or high model variance. The problem with this strategy is that it highly depends on the images in the training set. Changing the training set results in a different solution. 
3. Or they might actually learn an estimate of the "true" underlying distribution of images through continuous interpolation between training images. This is a better solution, because it is not so dependent on the data (low model variance), and generated samples are distinct from the images in the training set. In other words, they are generalizing outside the training set. 

In this work, we first ask which one of these strategies diffusion models adopt? Are they memorizing or generalizing? 

We confirm that when trained on small data sets (relative to the capacity of the network) these network memorize the training set, but we also demonstrate that these same models stop memorizing and transition to generalization when trained on sufficiently large sets. Specifically, we show that two denoisers trained on sufficiently large non-overlapping sets converge to essentially the same denoising function. That is, the learned model becomes independent of the training set (i.e., model variance falls to zero). As a result, when used for image generation, these networks produce nearly identical samples. 
<p align="center">
  <img src="results/github_fig1.png" width="1000">
</p>

These results provide stronger and more direct evidence of generalization than standard comparisons of average performance on train and test sets. 

But how is this generalization possible despite the curse of dimensionality? In the absence of all inductive biases, to learn a density of 8-bit images of resolution $80\times80$ the size of the required data set is $N = 256 ^ {80\times80}$, which is larger than the number of atoms in the universe.  
Our experiments show that generalization is achieved with a much smaller and realizable training set (roughly $10^5$ images suffices), reflecting powerful inductive biases of these networks. What are the inductive biases of these networks which give rise to such strong generalization? 

We analyze the learned denoising functions and show that the inductive biases give rise to a shrinkage operation in a basis adapted to the underlying image. Examination of these bases reveals oscillating harmonic structures along contours and in homogeneous regions. We demonstrate that trained denoisers are inductively biased towards these geometry-adaptive harmonic bases since they arise not only when the network is trained on photographic images, but also when it is trained on image classes supported on low-dimensional manifolds for which the harmonic basis is suboptimal. Finally, we show that when trained on regular image classes for which the optimal basis is known to be geometry-adaptive and harmonic, the denoising performance of the networks is near-optimal.
 
<p align="center">
  <img src="results/github_fig2.png" width="1000">
</p>



## Cite 
@inproceedings{ \
kadkhodaie2024generalization, \
title={Generalization in diffusion models arises from geometry-adaptive harmonic representation}, \
author={Zahra Kadkhodaie and Florentin Guth and Eero P Simoncelli and St{\'e}phane Mallat}, \
booktitle={The Twelfth International Conference on Learning Representations}, \
year={2024}, \
url={https://openreview.net/forum?id=ANvmVS2Yr0} \
}
