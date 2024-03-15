import numpy as np
import torch
import time
from wavelet_func import reconstruct
from dataloader_func import rescale_image_range

### Takes a tensor of size (n_ch, im_d1, im_d2)
### and returns a tensor of size (n_ch, im_d1, im_d2)

def one_scale_synthesis(model, init_im , sig_0=1, sig_L=.01, h0=.01 , beta=.01 , freq=0,device=None, fixed_h = False,max_T=None, seed=None):
    
    '''
    @init_im: either tuple=(C,H,W) representing size of low res image being synthesied, 
    or input tensor of size (3*C+1, H, W)
    @sig_0: initial sigma (largest)
    @sig_L: final sigma (smallest)
    @h0: 1st step size
    @beta:controls added noise in each iteration (0,1]. if 1, no noise is added. As it decreases more noise added.
    '''
    

        
    intermed_Ys=[]
    sigmas = []
    means = []

    ### initialization 
    if type(init_im) is tuple: ### If generating low resolution image. Unconditional synthesis from global prior
        C, H,W = init_im
        conditional=False
        
    else:  ### If generating details. Conditional  
        C, H,W = init_im.size()
        C = 3*C
        conditional=True
        
    if seed is not None: 
        torch.manual_seed(seed)
    
    e = torch.zeros((C ,H,W), requires_grad= False , device=device)
    N = C*H*W #correct?
    y = torch.normal(e, sig_0)      
    y = y.unsqueeze(0)
    y.requires_grad = False
        
        
    if freq > 0:
        intermed_Ys.append(y.squeeze(0))

    t=1
    sigma = torch.tensor(sig_0)
    start_time_total = time.time()
    snr = 20*torch.log10((y.std()/sigma)).item()
    snr_L = 20*torch.log10((torch.tensor([1])/sig_L)).item()    
    # while sigma > sig_L*(abs(y).max().item() - abs(y).min().item()):  # update stopping criteria since y range could be smaller than 1
    # while sigma > sig_L:  # update stopping criteria since y range could be smaller than 1
    while snr < snr_L: 
        h = h0
        
        if fixed_h is False:
            h = h0*t/(1+ (h0*(t-1)) )

            
        with torch.no_grad():
            if conditional:
                f_y = model(torch.cat((init_im.unsqueeze(0), y), dim=1)) 
            else: 
                f_y = model(y)

        sigma = torch.norm(f_y)/np.sqrt(N)
        sigmas.append(sigma)
        gamma = sigma*np.sqrt(((1 - (beta*h))**2 - (1-h)**2 ))
        noise = torch.randn(C, H, W, device=device) 
        
        if freq > 0 and t%freq== 0:
            print('-----------------------------', t)
            print('sigma ' , sigma.item() )
            print('mean ', y.mean().item() )
            intermed_Ys.append(y.squeeze(0))

        y = y -  h*f_y + gamma*noise 
        means.append(y.mean().item())
        snr = 20*torch.log10((y.std()/sigma)).item()        
        
        
        t +=1
        if max_T is not None and t>max_T:
            print('max T surpassed')
            break
        if sigma > 2:
            print('not converging')
            break
    print('-------- total number of iterations: ', t)
    print("-------- final sigma, " , sigma.item() )
    print('-------- final mean ', y.mean().item() )
    print("-------- final snr, " , 20*torch.log10((y.std()/sigma)).item() )

    if conditional:
        denoised_y = y - model(torch.cat((init_im.unsqueeze(0), y), dim=1)) 
    else:             
        denoised_y = y - model(y)

    return denoised_y.squeeze(0), intermed_Ys, sigmas, means




def multi_scale_synthesis(models, init_im  , sig_0, sig_L, h0 , beta , freq,device = None, orth_forward=True, seeds =None, fixed_h = False):
    '''
    @model: denoiser to be used in the algorithm
    @init_im: either tuple=(C,H,W) representing size of low res image being synthesied, 
    or input tensor of size (3*C+1, H, W) if the goal is to generate the low pass as well
    @sig_0: initial sigma (largest)
    @sig_L: final sigma (smallest)
    @h0: 1st step size
    @beta:controls added noise in each iteration (0,1]. if 1, no noise is added. As it decreases more noise added.
    @freq: frequency at which intermediate steps will be logged 
    @seeds: if not none, it should be set to an integer to set the seeds manually 
    @orth_forward: True if wavelet coefficents are normalized. False if they have not (default in pytorch)
    @fixed_h: if False, it uses an ascending h schedule to speed up convergence in later stages
    '''

    J = len(models)-1
    if seeds is not None and len(models)!=len(seeds): 
        raise ValueError('len(seed) and number of models do not match!')
        

    all_out=[]
    all_im = []
    all_inter =[]
    if type(init_im) is tuple: ### If generating low resolution image. Unconditional synthesis from global prior
        print('-------------------- generating low pass image')
        if seeds['low'] is not None: 
            torch.manual_seed(seeds['low'])  
            
        low, inter,_,_ = one_scale_synthesis(model=models['low'], init_im=init_im , sig_0=sig_0['low'], 
                                         sig_L=sig_L['low'], h0=h0['low'], beta=beta['low'] , freq=freq['low'] 
                                         ,device=device, fixed_h = fixed_h['low'])
        all_inter.append(inter)   
        print('-------- im range: ', low.detach().min().item(), low.detach().max().item())
        im = rescale_image_range(low.detach(),.8 ) # rescale low pass to [0,1]
        im_max = im.max()
    else: 
        im = init_im 
        im_max = im.max()
        all_inter.append([])   
        
    all_out.append(im)
    all_im.append(im)
    
    
    print('--------', im.shape)
         
    for j in range(J-1,-1,-1): 
        print('-------------------- scale: ', j)
        if seeds[j] is not None: 
            torch.manual_seed(seeds[j])  
        else: 
            torch.random.seed()
        
        coeffs, inter,_,_ = one_scale_synthesis(models[j], im , sig_0=sig_0[j], sig_L=sig_L[j], h0=h0[j] ,
                                     beta=beta[j] , freq=freq[j] ,device=device, fixed_h = fixed_h[j] )
        all_out.append(coeffs.squeeze(0).detach())

        im = reconstruct( torch.cat([im,coeffs.detach()] ,dim = 0).unsqueeze(0),
                         device, 
                         orth_forward = orth_forward)
        
        # im = rescale_image_range(im.detach(),im_max ) 
        
        im = im.squeeze(0)
        all_im.append(im)
        all_inter.append(inter)
        print('--------',im.shape)
        print('-------- im range: ', im.detach().min().item(), im.detach().max().item())
    

        
    return all_im, all_inter, all_out






def batch_synthesis(model, x_size ,device,sig_0=1, sig_L=.01, h0=.01 , beta=.01 , freq=5,seeds = None, init_im = None, init_noise_mean=0,max_T=None):

    '''
    @x_size: tuple of (B, C, H, W)
    @sig_0: initial sigma (largest)
    @sig_L: final sigma (smallest)
    @h0: 1st step size
    @beta:controls added noise in each iteration (0,1]. if 1, no noise is added. As it decreases more noise added.
    @ init_im: tensor of size (B, C, H, W)
    returns a tensor of size (B, C, H, W)
    '''

    B, n_ch, im_d1,im_d2 = x_size
    N = n_ch* im_d1*im_d2
    if n_ch != 1:
        raise ValueError('Note: torch.norm doesnt work for 3 dims, so this doesnt work for C=3. Fix later. ')


    intermed_Ys=[]
    sigmas = []
    means = []

    # initialize y
    if init_im is None:
        e = torch.ones( (x_size), device=device,requires_grad= False )*init_noise_mean
        if seeds is None:
            y = torch.normal( mean=e , std=torch.ones( (x_size), device=device )*sig_0)
        else:
            if len(seeds) != B:
                raise ValueError('seed length does not match Batch size ')
            noise_samples = []
            for seed in seeds:
                torch.manual_seed(seed)
                noise_samples.append(torch.normal( mean=e[0] , std=sig_0))
            y = torch.stack(noise_samples)
    else:
        y = init_im



    if freq > 0:
        intermed_Ys.append(y)

    with torch.no_grad():
        f_y = model(y)


    sigma = f_y.norm(dim=(2,3),keepdim=True)/np.sqrt(N)
    sigmas.append(sigma )
    means.append(y.mean(dim=(2,3)) )
    t=1
    start_time_total = time.time()
    min_range = 1

    while sigma.max() > sig_L*min_range:  # update stopping criteria since y range could be smaller than 1
    # while sigma > sig_L:

#         h = h0*t/(1+ (h0*(t-1)) )
        h = h0
        with torch.no_grad():
            f_y = model(y)


        sigma = f_y.norm(dim=(2,3), keepdim=True)/np.sqrt(N)
        sigmas.append(sigma )

        gamma = sigma*np.sqrt(((1 - (beta*h))**2 - (1-h)**2 ))

        noise = torch.randn(B, n_ch, im_d1,im_d2 , device=device)




        if freq > 0 and t%freq== 0:
            # print('-----------------------------', t)
            # print('sigma ' , sigma.item() )
            # print('mean ', y.mean().item() )
            intermed_Ys.append(y )


        y = y -  h*f_y  + gamma*noise
        means.append(y.mean(dim=(2,3)) )


        t +=1

        if max_T is not None and t>max_T:
            print('max T surpassed')
            break
        if sigma.max() > 1.5:
            print('not converging')
            break


        min_range = abs(y[sigma.argmax()]).max() - abs(y[sigma.argmax()]).min()

    print("-------- total number of iterations, " , t,"-------- average time per iteration (s), " , np.round((time.time() - start_time_total)/(t)  ,4),'--------final max sigma ' , sigma.max().item() )
    # print("-------- average time per iteration (s), " , np.round((time.time() - start_time_total)/(t)  ,4) )
    # print('--------final sigma ' , sigma.item() )

    denoised_y = y - model(y)


    return denoised_y.detach(), intermed_Ys, sigmas, means




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


def iterative_denoising(model, y ,sig_L=.01, h0=.01 , beta=1 , 
                        freq=0,device=None, fixed_h = True, max_T=None):
    
    '''
    @y: noisy image to denoise
    @sig_L: final sigma (smallest)
    @h0: 1st step size
    @beta:controls added noise in each iteration (0,1]. if 1, no noise is added. As it decreases more noise added.
    '''
    

    C, H,W = y.size()
    N = C * H * W
    t=1
    sigma = 1
    y = y.unsqueeze(0)
    
    # snr = 0
    # snr_L = 20*torch.log10((torch.tensor([1])/sig_L)).item()    
    # while sigma > sig_L*(abs(y).max().item() - abs(y).min().item()):  # update stopping criteria since y range could be smaller than 1
    while sigma > sig_L:  # update stopping criteria since y range could be smaller than 1
    # while snr < snr_L: 
        h = h0
        
        if fixed_h is False:
            h = h0*t/(1+ (h0*(t-1)) )

            
        with torch.no_grad():
            f_y = model(y)

        sigma = torch.norm(f_y)/np.sqrt(N)
        gamma = sigma*np.sqrt(((1 - (beta*h))**2 - (1-h)**2 ))
        noise = torch.randn(C, H, W, device=device) 

        if freq > 0 and t%freq== 0:
            print('-----------------------------', t)
            print('sigma ' , sigma.item() )
            print('mean ', y.mean().item() )

        y = y -  h*f_y + gamma*noise 
        snr = 20*torch.log10((y.std()/sigma)).item()        
        
        
        t +=1
        if max_T is not None and t>max_T:
            print('max T surpassed')
            break
        if sigma > 1.5:
            print('not converging')
            break
    print('-------- total number of iterations: ', t)
    # print("-------- final sigma, " , sigma.item() )
    # print('-------- final mean ', y.mean().item() )
    # print("-------- final snr, " , 20*torch.log10((y.std()/sigma)).item() )

    denoised_y = y - model(y)

    return denoised_y.squeeze(0).detach()




def walk_on_manifold(x, sig0, model, device, n_steps, momentum=None ,
                                    sig_L=.01, h0=.05 , beta=1 , 
                                    freq=0, fixed_h = True, max_T=None):    
    
    x = x.to(device)
    z = torch.randn_like(x, device = device) * sig0

    all_steps = [] 
    all_steps.append(x)
    
    for i in range(n_steps): 

        z = torch.randn_like(x, device = device) *sig0 
        y = x+ z    
        x = iterative_denoising(model, y, device=device, h0=h0, beta=beta , 
                        freq=freq, fixed_h = fixed_h, max_T=max_T)
        if momentum is not None: 
            delta = (x - all_steps[-1] )*momentum
            # delta = delta - delta.mean()
            # sig = (sig0**2 - delta.var())**.5
            all_steps.append(x)
            x = x+ delta
            
        else: 
            all_steps.append(x)
            
    return torch.vstack(all_steps).unsqueeze(1)
