import numpy as np
import torch
import time
from wavelet_func import reconstruct
from dataloader_func import rescale_image_range

### Takes a tensor of size (n_ch, im_d1, im_d2)
### and returns a tensor of size (n_ch, im_d1, im_d2)




def univ_inv_sol(model, x_c ,task,device,sig_0=1, sig_L=.01, h0=.01 , beta=.01 , freq=5,seed = None, init_im = None, init_noise_mean=0,max_T=None, fixed_h=False):
    
    '''
    @x_c:  M^T.x)
    @task: the specific linear inverse problem
    @sig_0: initial sigma (largest)
    @sig_L: final sigma (smallest)
    @h0: 1st step size
    @beta:controls added noise in each iteration (0,1]. if 1, no noise is added. As it decreases more noise added.
    '''

    
    M_T = task.M_T #low rank measurement matrix - in function form
    M = task.M #inverse of M_T
    
    n_ch, im_d1,im_d2 = M(x_c).size()
    N = n_ch* im_d1*im_d2
    intermed_Ys=[]
    sigmas = []
    means = []
    if seed is not None: 
        torch.manual_seed(seed)
        
    # initialize y
    if init_im is None: 
        e = torch.ones_like(M(x_c), requires_grad= False , device=device) * init_noise_mean 
        # if seed is None:
        y = torch.normal((e - M(M_T(e))) + M(x_c), sig_0 ).to(device)
        # else: 
            # torch.manual_seed(seed)        
            # y = torch.normal((e - M(M_T(e))) + M(x_c), sig_0).to(device)

        y = y.unsqueeze(0)
        y.requires_grad = False
        
    else:
        y = init_im.unsqueeze(0)
        y.requires_grad = False
        
        
    if freq > 0:
        intermed_Ys.append(y.squeeze(0))
        
    f_y = model(y)

    
    sigma = torch.norm(f_y)/np.sqrt(N)
    sigmas.append(sigma.item())
    means.append(y.mean().item())
    t=1
    start_time_total = time.time()
    
    # snr = 20*torch.log10((y.std()/sigma)).item()
    # snr_L = 20*torch.log10((torch.tensor([1])/sig_L)).item()
    # while sigma > sig_L*(abs(y).max().item() - abs(y).min().item()):  # update stopping criteria since y range could be smaller than 1
    while sigma > sig_L:  # update stopping criteria since y range could be smaller than 1
    # while snr < snr_L:


        h = h0
        if fixed_h is False:
            h = h0*t/(1+ (h0*(t-1)) )

        with torch.no_grad():
            f_y = model(y)

        d = f_y - M(M_T(f_y[0])) + ( M(M_T(y[0]))  - M(x_c) )


        sigma = torch.norm(d)/np.sqrt(N)
        sigmas.append(sigma.item())

        gamma = sigma*np.sqrt(((1 - (beta*h))**2 - (1-h)**2 ))

        noise = torch.randn(n_ch, im_d1,im_d2, device=device) 


        if freq > 0 and t%freq== 0:
            print('-----------------------------', t)
            print('sigma ' , sigma.item() )
            print('mean ', y.mean().item() )
            intermed_Ys.append(y.squeeze(0))

        y = y -  h*d + gamma*noise
        means.append(y.mean().item())
        # snr = 20*torch.log10((y.std()/sigma)).item()        
        
        t +=1
        
        if max_T is not None and t>max_T: 
            print('max T surpassed')
            break
        if sigma > 2: 
            print('not converging')
            break

    print("-------- final sigma, " , sigma.item() )
    print('-------- final mean ', y.mean().item() )
    # print("-------- final snr, " , 20*torch.log10((y.std()/sigma)).item())
    print("-------- total number of iterations, " , t,"-------- average time per iteration (s), " , np.round((time.time() - start_time_total)/(t)  ,4) )

    f_y = model(y)
    denoised_y = y - f_y    
    intermed_Ys.append(denoised_y.detach().squeeze(0))
    sigma = torch.norm(f_y)/np.sqrt(N)
    sigmas.append(sigma.item())    
    means.append(denoised_y.mean().item())

    return denoised_y.squeeze(0).detach(), intermed_Ys, sigmas, means

