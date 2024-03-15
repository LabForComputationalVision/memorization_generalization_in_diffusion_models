import numpy as np
import torch
from dataloader_func import add_noise_torch, rescale_image_range
######################################################################################
################################ quality metrics #####################################
######################################################################################

def sig_to_psnr(std, I_max = 255): 
    '''std for im in range 0 to 255'''
    return -10*np.log10( (std/255)**2  )   



def calc_psnr(denoiser,loader, sigma_range, device,max_I=1., rescale=False):
    '''
    Takes denoiser, clean data, and a range of noise, 
    returns ave psnr for all sigma in the noise range
    '''    
    psnr_range = {}
    for sigma in sigma_range:
        psnr_range[str(round(sigma.item(),2))] = []
        psnr = 0
        for i, batch in enumerate(loader,0):
            denoiser.eval()
            clean = batch.to(device)
            if rescale:
                clean = rescale_image_range(clean, 1.,0.)
            noisy , noise = add_noise_torch(clean, sigma, device)
            with torch.no_grad():
                output = denoiser(noisy)
                denoised = noisy - output

            psnr += batch_ave_psnr_torch(clean, denoised ,max_I).item()
        psnr_range[str(round(sigma.item(),2))] = psnr/(i+1)
    return psnr_range



def cos_similarity(im1,im2):
    return torch.matmul(((im1/im1.norm(dim=(2,3), keepdim=True).norm(dim=1, keepdim=True)).flatten(start_dim=1)),
                 (im2/im2.norm(dim=(2,3), keepdim=True).norm(dim=1, keepdim=True)).flatten(start_dim=1).T)




remove_im_mean = lambda data : data - data.mean(dim=(1,2,3),keepdims=True )

def im_set_corr(set1, set2, remove_mean=True):
    '''
    im_set: tensor of size N,C,H, W
    '''

    if len(set1.shape) != 4 or len(set2.shape) != 4 :
        raise ValueError('Input shape error')
    if remove_mean: 
        set1 = remove_im_mean(set1)
        set2 = remove_im_mean(set2)

    norms1 = set1.norm(dim=(2,3), keepdim=True).norm(dim=1, keepdim=True)
    norms1[norms1 == 0 ] = .001 # to avoid dividing by 0 for blank images 
    norms2 = set2.norm(dim=(2,3), keepdim=True).norm(dim=1, keepdim=True)
    norms2[norms2 == 0 ] = .001 # to avoid dividing by 0 for blank images 
    
    return torch.matmul(((set1/norms1).flatten(start_dim=1)),
             (set2/norms2).flatten(start_dim=1).T)




def remove_repeats(dataset, threshold=.95): 
    
    remove_im_mean = lambda data : data - data.mean(dim=(2,3),keepdims=True )
    # downsample the dataset to 20x20 images to save computation 
    pool = torch.nn.AvgPool2d(int(dataset.shape[2]/20))
    dataset_down = pool(dataset)
    #compute correlatopn between all images in the set
    corrs = im_set_corr(remove_im_mean(dataset_down), remove_im_mean(dataset_down) )
    #remove the upper diagonal of the correlation matrix
    corr_data_diag = torch.tril(corrs, diagonal=-1)
    #pick a treshold
    # find images with correlations above the threshold 
    rep_IDs_1 , rep_IDs_2 = torch.where(abs(corr_data_diag )>threshold)
    #drop one in each pair of repeated images 
    data_cleaned = np.delete(dataset,rep_IDs_1 , axis = 0)
    return data_cleaned


