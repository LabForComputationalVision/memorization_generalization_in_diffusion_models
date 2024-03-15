import numpy as np
import torch
import pywt


###########################################################################
################################ wavelet coeffs ###########################
###########################################################################
def decompose(images,wavelet='db1', mode='symmetric',  orth_forward=True):
    """ One-level wavelet decomposition, (*,1, L, L) to (*, 4, L/2, L/2), using numpy (CPU, no autodiff). """
    low, high = pywt.dwt2(images, wavelet=wavelet, mode=mode)
    # low is a (*, 1,L/2, L/2) array, high is a tuple of (*,1, L/2, L/2) arrays.
    coeffs = np.concatenate((low,) + high, axis=-3)  # (*, 4, L/2, L/2)
    if orth_forward:
        coeffs = coeffs/(2**(j+1))

    return torch.from_numpy(coeffs)


def reconstruct( coeffs,device, wavelet='db1', mode='symmetric', orth_forward=True):
    """ One-level wavelet reconstruction, (*, 4, L/2, L/2) to (*,1, L, L), using numpy (CPU, no autodiff). """
    if device.type == 'cuda':
        coeffs = coeffs.cpu()

    channels = tuple(coeffs[..., c, :, :] for c in range(4))
    images = pywt.idwt2((channels[0], channels[1:]), wavelet=wavelet, mode=mode)  # (*, L, L)
    images =  torch.from_numpy(images).unsqueeze(1).to(device) 
    ## multiply by two to compensate for division by 2 of the coeffs in trainig 
    if orth_forward: 
        images = images * 2 
    return images 



def multi_scale_decompose(images,J,device, wavelet='db1', mode='symmetric', orth_forward=True):
    '''
    multi scale wavelet transform
    @J: number of scales
    @images: numpy array of size (*,1, L, L)
    returns a list of tensor coeffs. length of list is equal to J. Number of channels of each coeffs is 4.
    '''
    all_coeffs = []
    if device.type == 'cuda':
        low = images.cpu()
    else: 
        low = images

    for j in range(J):
        low, high = pywt.dwt2(low, wavelet, mode)
        coeffs = np.concatenate((low,) + high, axis=-3)  # (*, 4, L/2, L/2)
        if orth_forward:
            coeffs = coeffs/(2**(j+1))
        all_coeffs.append(torch.from_numpy(coeffs).to(device= device,dtype=torch.float32) )

    return all_coeffs



def multi_scale_denoising(noisy, denoisers, device, wavelet='db1', mode='symmetric', orth_forward=True):
    '''
    @ noisy: N,C,H,W tensor or array 
    @ returns denoised reconstructions in all scales 
    '''
    all_low = []
    
    J = len(denoisers)-1
    
    all_coeffs = multi_scale_decompose(noisy,J, device=device,wavelet=wavelet, mode=mode, orth_forward=orth_forward)

    with torch.no_grad():
        out = denoisers['low'](all_coeffs[-1][:,0:1]) # low freq denoising  
        low = all_coeffs[-1][:,0:1] - out # skip connection 
        all_low.append(low)
        for j in range(J-1, -1, -1): 
            inp = torch.cat((low, all_coeffs[j][:, 1::]), dim = 1)
            out = denoisers[j](inp )
            denoised = all_coeffs[j][:,1::] - out
            low = reconstruct(torch.cat(( low,denoised ), dim = 1), device, orth_forward=orth_forward)

            all_low.append(low)

    return all_low



def multi_scale_identity_denoising(noisy, denoisers, device, wavelet='db1', mode='symmetric', orth_forward=True):
    '''
    @ noisy: N,C,H,W tensor or array 
    @ returns denoised reconstructions in all scales 
    '''
    all_low = []

    J = len(denoisers)-1

    all_coeffs = multi_scale_decompose(noisy,J, device=device,wavelet=wavelet, mode=mode, orth_forward=orth_forward)

    with torch.no_grad():
        out = denoisers['low'](all_coeffs[-1][:,0:1]) # low freq denoising  
        low = all_coeffs[-1][:,0:1] - out # skip connection 
        all_low.append(low)
        for j in range(J-1, -1, -1):

            do_nothing = torch.zeros_like(all_coeffs[j][:,1::] )
    
            low = reconstruct(torch.cat(( low,do_nothing ), dim = 1), device, orth_forward=orth_forward)

            all_low.append(low)

    return all_low