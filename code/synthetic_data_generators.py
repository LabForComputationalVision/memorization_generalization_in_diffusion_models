import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import torch.nn as nn
import torch
import os
from PIL import Image
import gzip
from matplotlib.patches import Circle, Polygon
import random
from wavelets import OneLevelWaveletTransform
from dataloader_func import rescale_image_range
 

######################################################################################
################################ data generators  ################################
######################################################################################




################ c alpha ################
def make_fft_filter_1d(N):
    filt = abs(torch.fft.fftshift(torch.fft.fftfreq(N)))
    filt = 1/filt/N
    i = torch.where(filt == torch.inf)[0]
    filt[i.item()] = filt[i.item()+1]

    return filt



def make_fft_filter_2d(N):
    filt = torch.fft.fftshift(torch.fft.fftfreq(N))
    X, Y = torch.meshgrid(filt, filt)
    filt2 = (X**2 + Y**2).sqrt()
    filt2 = 1/filt2/N
    i, j = torch.where(filt2 == torch.inf)
    filt2[i.item(), j.item()] = filt2[i.item()+1, j.item()]
    return filt2


def make_fft_filter_2d_seprable(N):  
    filt = torch.fft.fftshift(torch.fft.fftfreq(N))
    i = torch.where(filt == 0)[0]
    filt[i.item()] = filt[i.item()+1]
    filt2 = abs(torch.matmul(filt.reshape(-1,1), filt.reshape(1,-1))).sqrt()
    filt2 = 1/filt2/N
    return filt2 

def make_C_alpha_contour(alpha, filt, N = 43): 
    ders = torch.rand(size = (N,), dtype=torch.float32)*2-1
    integrated = torch.fft.ifft(torch.fft.ifftshift( ( torch.fft.fftshift(torch.fft.fft(ders)) *( filt** (alpha))) )).real
    return integrated


def make_C_beta_background(beta, filt2, N = 43): 
    ders2 = torch.rand(size = (N,N), dtype=torch.float32)*2-1
    integrated =torch.fft.ifft2(torch.fft.ifftshift( ( torch.fft.fftshift(torch.fft.fft2(ders2)) * (filt2**(beta))) )).real 
    return integrated
 
    
# def make_C_alpha_images(alpha, beta, separable=False, N=43,num_samples=1, constant_background=False, factor=1 , antialiasing=0, wavelet="db2", mode="reflect"):
#     '''
#     N: image size. 
#     K: number of images
#     for vertical blurring: (factor, 1)
#     '''
#     all_im = []


#     if antialiasing > 0: 
#         N = N * (2**antialiasing) - 2 ** antialiasing
    

#     # filter for 2d
#     if separable:
#         filt2 = make_fft_filter_2d_seprable(N)
#     else: 
#         filt2 = make_fft_filter_2d(N)
        
#     #filter for 1d 
#     filt = make_fft_filter_1d(N* factor)
    
#     #filter for averaging the mask 
#     ave_filt = nn.Conv2d(1,1,factor, stride = factor, padding = 0
#                          , padding_mode='reflect', bias = None)
#     ave_filt.weight = torch.nn.Parameter(torch.ones((1,1,factor,factor))/(factor**2))
    
#     ### genearte images 
#     while len(all_im)<num_samples:
#         ## make the backgrounds
#         if constant_background: 
#             background1 = np.random.rand(1)[0] * torch.ones(size = (N,N))
#             background2 = np.random.rand(1)[0] * torch.ones(size = (N,N))
#         else: 
#             background1 = make_C_beta_background(beta,filt2, N)
#             background2 = make_C_beta_background(beta,filt2, N)        

#             ## rescale background to create an intensity difference between the two parts 
#             # thresh = torch.rand(1).item() # change to this to enfornce a larger gap between the mean of the two backgrounds 
#             if torch.randint(low=0, high=2, size=(1,)).item() == 0:
#                 background1 = rescale_image_range(background1,max_I=1, min_I=torch.rand(1).item())
#                 background2 = rescale_image_range(background2,max_I=torch.rand(1).item(), min_I=0)
#             else:
#                 background2 = rescale_image_range(background2,max_I=1, min_I=torch.rand(1).item())
#                 background1 = rescale_image_range(background1,max_I=torch.rand(1).item(), min_I=0)
                
#         ## make the contours
#         contour = make_C_alpha_contour(alpha ,filt, N*factor )
#         contour = torch.round(contour*N*factor).type(torch.int) + int((N*factor)/2)      
        
#         #### mask and replace   
#         mask = torch.ones((N*factor , N*factor))
#         for i in range(N*factor):
#             mask[0:contour[i],i]=0
            
#         mask_down = ave_filt(mask.unsqueeze(0).unsqueeze(0)).squeeze().detach()
#         im = background1 * mask_down + background2 * (1-mask_down) 

            
#         all_im.append(im)

#     all_im = torch.stack(all_im).unsqueeze(1) 
#     all_im = downsample(all_im.detach(), num_times=antialiasing, wavelet=wavelet, mode=mode)# (N, 1, H/2^j, W/2^j)
#     if antialiasing >0: 
#         all_im = rescale_image_range(all_im, max_I=1, min_I=0)
#     return all_im



def make_C_alpha_images(alpha, beta, separable=False, im_size=43,num_samples=1, constant_background=False, factor=(1,1) , antialiasing=0, wavelet="db2", mode="reflect"):
    '''
    im_size: image size. 
    num_samples: number of images
    for vertical blurring: (factor, 1)
    '''
    all_im = []


    if antialiasing > 0: 
        im_size = im_size * (2**antialiasing) - 2 ** antialiasing

    filt = make_fft_filter_1d(im_size)
    
    if separable:
        filt2 = make_fft_filter_2d_seprable(im_size)
    else: 
        filt2 = make_fft_filter_2d(im_size)

    ave_filt = nn.Conv2d(1,1,(factor[0],factor[1]), stride = (1,1), padding = (int(factor[0]/2),int(factor[1]/2)) , padding_mode='reflect', bias = None)
    ave_filt.weight = torch.nn.Parameter(torch.ones((1,1,factor[0],factor[1]))/(factor[0]* factor[1]))
    while len(all_im)<num_samples:
        contour = make_C_alpha_contour(alpha ,filt, im_size )
        contour = torch.round(contour*im_size).type(torch.int) + int((im_size)/2)    
        
        if constant_background: 
            background1 = np.random.rand(1)[0] * torch.ones(size = (im_size,im_size))
            background2 = np.random.rand(1)[0] * torch.ones(size = (im_size,im_size))

        else: 
            background1 = make_C_beta_background(beta,filt2, im_size)
            background2 = make_C_beta_background(beta,filt2, im_size)        

            ## rescale background to create an intensity difference between the two parts 
            # thresh = torch.rand(1).item() # change to this to enfornce a larger gap between the mean of the two backgrounds 
            if torch.randint(low=0, high=2, size=(1,)).item() == 0:
                background1 = rescale_image_range(background1,max_I=1, min_I=torch.rand(1).item())
                background2 = rescale_image_range(background2,max_I=torch.rand(1).item(), min_I=0)
            else:
                background2 = rescale_image_range(background2,max_I=1, min_I=torch.rand(1).item())
                background1 = rescale_image_range(background1,max_I=torch.rand(1).item(), min_I=0)
        #### mask and replace         
        mask = torch.ones((im_size , im_size))
        for i in range(im_size):
            mask[0:contour[i],i]=0
        if factor[0] >1 or factor[1]>1:
            mask_down = ave_filt(mask.unsqueeze(0).unsqueeze(0)).squeeze().detach()
            if factor[0]%2==0: 
                mask_down = mask_down[0:-1, :]
            if factor[1]%2==0: 
                mask_down = mask_down[:,0:-1]
                
            im = background1 * mask_down + background2 * (1-mask_down) 
        else: 
            im = background1 * mask + background2 * (1-mask) 
            
        all_im.append(im)

    all_im = torch.stack(all_im).unsqueeze(1) 
    all_im = downsample(all_im.detach(), num_times=antialiasing, wavelet=wavelet, mode=mode)# (N, 1, H/2^j, W/2^j)
    if antialiasing >0: 
        all_im = rescale_image_range(all_im, max_I=1, min_I=0)
    return all_im 



def downsample(x, num_times, wavelet="db2", mode="periodization"):
    """ Downsample an (*, H, W) image `num_times` times using the given wavelet filter. """
    transform = OneLevelWaveletTransform(wavelet=wavelet, mode=mode)

    for _ in range(num_times):
        x = transform.decompose(x)[..., 0, :, :]  # (*, H/2^j, W/2^j)

    return x


################ disks ################

def generate_circles(N):
    '''N:number of generated samples'''
    
    images = []
    
    for j in range(N):
        c = np.random.rand(1)[0]
        background_color = (c,c,c)
        fig, ax = plt.subplots(figsize = (4,4), dpi =25,facecolor=background_color)
        ax.set_xlim(0,80)
        ax.set_ylim(0,80)
        # fig, ax = plt.subplots(figsize = (4,4), dpi =13,facecolor=background_color)
        # ax.set_xlim(0,43)
        # ax.set_ylim(0,43)
        
        d = np.random.rand(1)[0]
        face_color =(d,d,d)
        center = np.random.randint( low = 10, high = 70, size = (2,) )
        max_r = min(min(center[0], 80-center[0]), min(center[1], 80-center[1]))
        # center = np.random.randint( low = 5, high = 37, size = (2,) )
        # max_r = min(min(center[0], 41-center[0]), min(center[1], 41-center[1]))        
        r = np.random.randint(low = 5, high = max_r)
        disc = Circle(center, radius=r , color= face_color )

        ax.add_patch(disc)
        ax.axis('off')
        plt.savefig('myfig.png', bbox_inches='tight' )
        plt.close()


        im = torch.tensor(plt.imread('myfig.png').mean(axis=2)).unsqueeze(0)
        images.append(im)
    return torch.stack(images)







def circle_dataset(patch_size, num_patches, translate=False, scale=False, foreground=False, background=False,
                   wrap=False, continuous=True, antialiasing=0, wavelet="db2", mode="reflect"):
    """ Returns (N, 1, H, W) images of circles.
    :param patch_size: (H, W) tuple
    :param num_patches: number of samples N
    :param translate: whether to translate the circles
    :param scale: whether to scale the circles
    :param foreground: whether to randomize the foreground color
    :param background: whether to randomize the background color
    :param wrap: whether to generate wrapping circles at the boundaries
    :param continuous: whether consider continuous positions and sizes as opposed to integers
    :param antialiasing: generate larger images and downsample them this number of times
    :param wavelet: wavelet used for downsampling
    :param mode: padding mode for downsampling
    """
    assert patch_size[0] == patch_size[1]  # XXX too lazy to implement rectangular images
    size = patch_size[0] * 2 ** antialiasing  # Generate large images by the aliasing factor

    # Use numpy for randint because pytorch doesn't support low and high arguments as tensors
    rand_np = np.random.uniform if continuous else np.random.randint
    rand = lambda *args, **kwargs: torch.from_numpy(rand_np(*args, **kwargs))

    min_r = 3
    radius = rand(size=(num_patches,), low=min_r, high=(size - min_r) // 2) if scale \
        else torch.full(size=(num_patches,), fill_value=size // 4)  # (N,)

    r = 0 if wrap else radius[:, None]
    center = rand(size=(num_patches, 2), low=r, high=size - r) if translate \
        else torch.full(size=(num_patches, 2), fill_value=size // 2, dtype=torch.int32)  # (N, 2)

    foreground = torch.rand((num_patches,)) if foreground else torch.ones((num_patches,))  # (N,)
    background = torch.rand((num_patches,)) if background else torch.zeros((num_patches,))  # (N,)

    y, x = torch.meshgrid(*(torch.arange(s) for s in (size, size)), indexing="ij")  # (H, W) both
    u = torch.stack((x, y), dim=-1)  # (H, W, 2)

    # Compute modulo in [-(size // 2), (size - 1) // 2]
    diff_mod = lambda diff: (diff + size / 2) % size - size / 2
    squared_dist = torch.sum(diff_mod(u[None] - center[:, None, None]) ** 2, dim=-1)  # (N, H, W)
    circle = squared_dist <= radius[:, None, None].float() ** 2  # (N, H, W)

    data = background[:, None, None] + (foreground - background)[:, None, None] * circle.float()  # (N, H, W)
    data = data[:, None]  # (N, 1, H, W)
    data = downsample(data, num_times=antialiasing, wavelet=wavelet, mode=mode)  # (N, 1, H/2^j, W/2^j)

    return data


