import numpy as np
import torch
from dataloader_func import add_noise_torch, rescale_image_range
######################################################################################
################################ quality metrics #####################################
######################################################################################

def MSE(ref_im, im ):
    return ((ref_im - im)**2).mean()


def sig_to_psnr(std, I_max = 255): 
    '''std for im in range 0 to 255'''
    return -10*np.log10( (std/255)**2  )   

def batch_psnr_numpy(ref_ims, images ,max_I):
    '''
    batch of ref_im and im are tensors of dims N, W, H
    '''
    if len(ref_ims.shape)==3:
        mse = ((ref_ims - images)**2).mean(axis=(1,2))
    elif len(ref_ims.shape)==4:
        mse = ((ref_ims - images)**2).mean(axis=(1,2,3))
    psnr_all =  10*(np.log10(max_I**2) - np.log10(mse))
    return psnr_all


def batch_ave_psnr_torch(ref_ims, images ,max_I):
    '''
    batch of ref_im and im are tensors of dims N,C, W, H
    returns ave psnr of a batch of images 
    '''
    mse = ((ref_ims - images)**2).mean(dim=(1,2,3))
    psnr_all =  10*(np.log10(max_I**2) - torch.log10(mse))
    return psnr_all.mean()


def psnr(denoiser,loader, sigma,device,skip=True): 
    '''
    Takes denoiser, clean data, noise level, 
    and returns ave psnr of denoised images for that specific sigma
    '''
    psnr_sum = 0
    for i, batch in enumerate(loader,0):
        denoiser.eval()
        clean = batch.to(device)
        noisy , noise = add_noise_torch(clean, sigma, device)
        with torch.no_grad():
            denoised = denoiser(noisy)
        if skip: 
            denoised = noisy - denoised
            
        psnr_sum += batch_ave_psnr_torch(clean, denoised ,1.).item()
    return psnr_sum/(i+1)
 
# def calc_psnr_range(clean, noise_range, denoiser): 
#     ### not good. Give rid of and replace with the next function 
#     '''
#     Takes denoiser, clean dat, and a range of noise, 
#     returns ave psnr for all sigma in the noise range
#     '''
#     psnrs = {}
#     denoiser.eval()
#     for sigma in noise_range:     
#         noise = torch.randn_like(clean)*sigma/255
#         noisy = clean + noise.cuda()

#         with torch.no_grad(): 
#             denoised = noisy - denoiser(noisy).detach()    
#         psnrs[sigma.item()] = batch_ave_psnr_torch(clean.cpu() , denoised.cpu(), 1)            

#     return psnrs


  

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


def calc_denoised_bias_var(denoiser,loader, sigma_range, device):
    bias = {}
    var ={}
    mse = {}
    mean = {}
    for sigma in sigma_range:
        all_denoised = []
        for i, batch in enumerate(loader,0):
            denoiser.eval()
            clean = batch.to(device)
            noisy , noise = add_noise_torch(clean, sigma, device)
            with torch.no_grad():
                output = denoiser(noisy)
                denoised = noisy - output.detach()
                all_denoised.append(denoised )

        all_denoised = torch.hstack(all_denoised)
        denoised_mean = all_denoised.mean(dim=0 , keepdims= True)
        denoised_bias = clean.mean(dim=0 , keepdims= True) - denoised_mean
        denoised_var = ((all_denoised - denoised_mean)**2).mean(dim= 0, keepdims= True)
        denoised_mse = ((all_denoised - clean)**2).mean(dim = 0, keepdims=True)

        bias[sigma] = denoised_bias.cpu()
        var[sigma] = denoised_var.cpu()
        mse[sigma] = denoised_mse.cpu()
        mean[sigma] = denoised_mean.cpu()

    return bias, var , mse, mean


def calc_denoised_bias_var_fc(denoisers,loader, sigma_range, device):
    bias = {}
    var ={}
    mse = {}
    mean = {}
    for sigma in sigma_range:
        all_denoised = []
        for i, batch in enumerate(loader,0):
            clean = batch.to(device)
            noisy , noise = add_noise_torch(clean, sigma, device)
            with torch.no_grad():
                denoised = denoisers[sigma](noisy).detach()
                all_denoised.append(denoised )

        all_denoised = torch.hstack(all_denoised)
        denoised_mean = all_denoised.mean(dim=0 , keepdims= True)
        denoised_bias = clean.mean(dim=0 , keepdims= True) - denoised_mean
        denoised_var = ((all_denoised - denoised_mean)**2).mean(dim= 0, keepdims= True)
        denoised_mse = ((all_denoised - clean)**2).mean(dim = 0, keepdims=True)

        bias[sigma] = denoised_bias.cpu()
        var[sigma] = denoised_var.cpu()
        mse[sigma] = denoised_mse.cpu()
        mean[sigma] = denoised_mean.cpu()

    return bias, var , mse, mean


def normalize_im_set_l1(im_set):
    '''
    im_set: tensor of size N,C,H, W
    this function normalizes intensity variations for images
    '''
    if len(im_set.shape) != 4:
        raise ValueError('Input shape error')

    return im_set/im_set.norm(dim = (2,3),p=1, keepdim=True)

def normalized_distance_np(x, y):
    '''
    Euclidean distance / sqrt(im size). Normalization is applied to make it comparible to sigma.
    Since radius of sphere in concentration of measure theorem is sigma/sqrt(im size)
    '''

    return np.linalg.norm(x - y) /(np.sqrt(np.prod(x.shape)))

def normalized_distance_torch(x, y):
    '''
    Euclidean distance / sqrt(im size). Normalization is applied to make it comparible to sigma.
    Since radius of sphere in concentration of measure theorem is sigma/sqrt(im size)
    '''
    if x.device.type == 'cuda':
        x = x.cpu()
    if y.device.type == 'cuda':
        y = y.cpu()
    return torch.norm(x - y) /(np.sqrt(np.prod(x.shape)))

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


def patch_similarities(sample_images, train_set, patch_size=20, stride=20):

    N, C, H, W = sample_images.shape

    H = int(H/stride) * stride
    W = int(W/stride) * stride


    temp1 = []
    temp2=[]
    for i in range(0,H-patch_size+1, stride):
        for j in range(0,W-patch_size+1,stride):

            patch_corrs = im_set_corr(sample_images[:, :, i:i+patch_size, j:j+patch_size],
                                        train_set[:, :, i:i+patch_size, j:j+patch_size] )
            temp1.append(patch_corrs.argmax(dim = 1))
            temp2.append(patch_corrs.max(dim = 1)[0])

        source_ims = torch.vstack(temp1).T
        all_corrs = torch.vstack(temp2).T
    return source_ims, all_corrs


def classify_patches(ids, all_corrs):
    '''
    returns a two tensors
    First tensor has two columns. First col is row number of the sample image whose all 4 patches come from the same train image.
    Second col is the train image id this sample image is equal to.
    '''
    classified = {}
    ids = ids.to(float)
    ids_same = torch.where(ids.var(dim=1)==0)[0]
    train_ids = ids[ids_same]
    corrs = all_corrs[ids_same]
    classified['same'] = [torch.cat([ids_same.reshape(-1,1), train_ids], dim=1), torch.cat([ids_same.reshape(-1,1), corrs], dim=1)]

    ids_diff = torch.where(ids.var(dim=1)!=0)[0]
    train_ids = ids[ids_diff]
    corrs = all_corrs[ids_diff]
    classified['diff'] = [torch.cat([ids_diff.reshape(-1,1), train_ids], dim=1), torch.cat([ids_diff.reshape(-1,1), corrs], dim=1)]

    return classified



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


def remove_repeats_loop(dataset, threshold=.95): 

    dataset_no_repeats = [dataset[0:1]]
    for i in range(1,len(dataset)): 
        corrs = im_set_corr(remove_im_mean(dataset[i:i+1]), remove_im_mean(torch.vstack(dataset_no_repeats)))
        rep_IDs_1 , rep_IDs_2 = torch.where(abs(corrs )>threshold)
        if len(rep_IDs_2)==0: 
            dataset_no_repeats.append(dataset[i:i+1])
        else: 
            pass
    return torch.vstack(dataset_no_repeats)