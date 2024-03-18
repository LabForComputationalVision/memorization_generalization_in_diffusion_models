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
- training a universal blind denoiser used in a diffusion framework
- sampling algorithm to sample from the prior embedded in the denoiser
- all the required helper functions such as
  * code for generating synthetic $C^{\alpha}$ and Disc images
  * code to compute the eigen basis of the denoiser operation
  * etc
