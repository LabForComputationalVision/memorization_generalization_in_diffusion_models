import numpy as np
import torch
def calc_jacobian( inp,model,layer_num=None, channel_num=None):


    ############## prepare the static model
    # model = Net(all_params)
    
    for param in model.parameters():
        param.requires_grad = False

    model.eval()

    ############## find Jacobian
    if inp.requires_grad is False: 
        inp.requires_grad =True

    if layer_num is not None: 
        out = model(inp, layer_num, channel_num)
    else:
        out = model(inp)
    jacob = []
    
    # start_time_total = time.time()
    for i in range(inp.size()[2]):
        for j in range(inp.size()[3]):
            part_der = torch.autograd.grad(out[0,0,i,j], inp, retain_graph=True) # this gives me a 20*20
            jacob.append( part_der[0][0,0].data.view(-1)) # flatten it to 400
    #print("----total time to compute jacobian --- %s seconds ---" % (time.time() - start_time_total))

    return torch.stack(jacob)



def calc_jacobian_rows( inp,model, i,j):
    '''
    @ im: torch image C,H,W
    '''
    ############## prepare the static model
    for param in model.parameters():
        param.requires_grad = False


    model.eval()

    if inp.requires_grad is False: 
        inp.requires_grad =True
    ##############  Jacobian
    out = model(inp)

    part_der = torch.autograd.grad(out[0,0,i,j], inp, retain_graph=True) # this gives me a 20*20
    jacob =  part_der[0][0,0] # flatten it to 400

    
    return jacob

def calc_jacobian_row_MS( inp,models, i,j, device):
    '''
    @ im: torch image C,H,W
    '''
    ############## prepare the static model
    for k,model in  models.items():
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

        
    if inp.requires_grad is False: 
        inp.requires_grad =True
        
    ##############  Jacobian
    out = multi_scale_denoising(inp, models, device)

    part_der = torch.autograd.grad(out[0,0,i,j], inp, retain_graph=True) # this gives me a 20*20
    jacob =  part_der[0][0,0] # flatten it to 400

    
    return jacob


def approx_subspace_proj(U_sub,S_sub,V_sub,x):
    '''returns projection onto the tangent plane'''
    temp = torch.matmul(V_sub, x.flatten())
    temp = torch.matmul(torch.diag(S_sub), temp)
    return torch.matmul(U_sub, temp).reshape(x.shape)



def pca_denoised(im,denoisers, sigmas, B, N):

    Ls={}
    Qs={}
    denoised_ims_centered = {}
    clean_ims = torch.tile(im , dims = (B,1,1,1)).to(device)
    noises = torch.randn_like(clean_ims, device=device)

    for sigma in sigmas:
        noisy_ims = clean_ims +  noises* sigma/255

        with torch.no_grad():
            residuals = denoisers[N](noisy_ims )

        denoised_ims = noisy_ims - residuals
        denoised_ims_centered[sigma] = denoised_ims- denoised_ims.mean(dim= 0, keepdims=True)
        cov = torch.cov(torch.flatten(denoised_ims_centered[sigma].squeeze(), start_dim= 1, end_dim=2).T)
        L, Q = torch.linalg.eig(cov)
        Ls[sigma] = torch.real(L)
        Qs[sigma] = torch.real(Q)

    return Ls, Qs, denoised_ims


def traj_projections(inter_ys, denoiser): 
    with torch.no_grad():
        out = inter_ys - denoiser(inter_ys).detach()
    return out