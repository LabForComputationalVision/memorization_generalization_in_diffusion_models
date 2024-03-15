import numpy as np
import matplotlib
import matplotlib.patches as patches
matplotlib.use('Agg')
import matplotlib.pylab as plt
import torch.nn as nn
import torch
import os
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from quality_metrics_func import batch_psnr_numpy,normalized_distance_np
from dataloader_func import rescale_image_range,rescale_image

######################################################################################
############################################### plots train time #####################
######################################################################################

def plot_loss(loss_list, loss_list_test, file_name):

    fig , axs = plt.subplots(1,2 , figsize= (12,5), sharey = True)
    axs[0].plot( range(len(loss_list)), loss_list, 'b-o', label = 'train')
    axs[0].set_ylabel('MSE')
    axs[0].set_title('min train loss ' + str(round(min(loss_list),3)) + ' from epoch ' + str(loss_list.index(min(loss_list))) + '\n final loss '+str(round(loss_list[-1],3)))
    axs[0].legend()
    axs[1].plot( range(len(loss_list_test)), loss_list_test, 'r-o', label = 'test loss')
    axs[1].set_title('min test loss ' + str(round(min(loss_list_test),3)) + ' from epoch ' + str(loss_list_test.index(min(loss_list_test))) + '\n final test loss '+str(round(loss_list_test[-1],3)))
    axs[1].legend()   
    plt.savefig(file_name)

def plot_psnr(psnr_list,psnr_list_test, file_name):

    fig , axs = plt.subplots(1,2 , figsize= (12,5), sharey = True)
    axs[0].plot( range(len(psnr_list)), psnr_list, 'orange','-o', label = 'train')
    axs[0].set_ylabel('PSNR')
    axs[0].set_title('max train psnr ' + str(round(max(psnr_list),3)) + ' from epoch ' + str(psnr_list.index(max(psnr_list))) + '\n final psnr '+str(round(psnr_list[-1],3)))
    axs[0].legend()
    axs[1].plot( range(len(psnr_list_test)), psnr_list_test, 'green','-o', label = 'test')
    axs[1].set_title('max test psnr ' + str(round(max(psnr_list_test),3)) + ' from epoch ' + str(psnr_list_test.index(max(psnr_list_test))) + '\n final psnr '+str(round(psnr_list_test[-1],3)))
    axs[1].legend()    
    plt.savefig(file_name)


def plot_denoised_image( clean, noisy, denoised ,dir_name):
    clean = clean.detach().cpu().permute(0,2,3,1).squeeze().numpy()
    noisy = noisy.detach().cpu().permute(0,2,3,1).squeeze().numpy()
    denoised = denoised.detach().cpu().permute(0,2,3,1).squeeze().numpy()

    f, axes = plt.subplots(3, 3 )
    ax = axes.ravel()

    for i ,j in zip(range(0, 9, 3), range(3) ):
        fig = ax[i].imshow(clean[j], 'gray')
        ax[i].set_axis_off()
        ax[i].set_title('clean')
        plt.colorbar(fig, ax=ax[i], fraction=.05)

        fig = ax[i+1].imshow(noisy[j], 'gray')
        ax[i+1].set_title( 'PSNR '+ str(round(peak_signal_noise_ratio(clean[j],noisy[j], data_range=1),3) ) + '\n SSIM ' + str(round(structural_similarity(clean[j], noisy[j],multichannel=True),3)), fontsize = 5)
        ax[i+1].set_axis_off()
        plt.colorbar(fig, ax=ax[i+1], fraction=.05)

        fig = ax[i+2].imshow(denoised[j], 'gray')
        ax[i+2].set_title(  'PSNR '+ str(round(peak_signal_noise_ratio(clean[j],denoised[j], data_range=1),3)) + '\n SSIM ' + str(round(structural_similarity(clean[j], denoised[j] ,multichannel=True),3)),fontsize = 5)
        ax[i+2].set_axis_off()
        plt.colorbar(fig, ax=ax[i+2], fraction=.05)

    file_name = dir_name + '/denoised_test_image.png'
    plt.savefig(file_name, dpi = 300)
    plt.close('all')

def plot_denoised_range(clean, noisy, denoised, noise_range, file_name,data_max, writer=None, h=None):
    clean = torch.stack([clean]*noise_range.shape[0])
    clean = clean.detach().cpu().permute(0,2,3,1).squeeze().numpy()
    noisy = noisy.detach().cpu().permute(0,2,3,1).squeeze().numpy()
    denoised = denoised.detach().cpu().permute(0,2,3,1).squeeze().numpy()

    psnr_noisy = batch_psnr_numpy(clean, noisy ,data_max )
    psnr_denoised = batch_psnr_numpy(clean, denoised ,data_max) 
    f, ax = plt.subplots(noise_range.shape[0], 3 , figsize = (3*3, noise_range.shape[0]*3))
    f.tight_layout()
    for i  in  range(noise_range.shape[0]):
        fig = ax[i,0].imshow(clean[i], 'gray')
        ax[i,0].set_axis_off()
        ax[i,0].set_title('clean')
        plt.colorbar(fig, ax=ax[i,0], fraction=.05)

        fig = ax[i,1].imshow(noisy[i], 'gray')
        ax[i,1].set_title( 'PSNR '+ str(round(psnr_noisy[i],3))   , fontsize = 15)
        ax[i,1].set_axis_off()
        plt.colorbar(fig, ax=ax[i,1], fraction=.05)

        fig = ax[i,2].imshow(denoised[i], 'gray')
        ax[i,2].set_title( 'PSNR '+ str(round(psnr_denoised[i],3))   , fontsize = 15)
        ax[i,2].set_axis_off()
        plt.colorbar(fig, ax=ax[i,2], fraction=.05)
   
    plt.savefig(file_name, dpi = 300)
    
    if writer is not None:
        writer.add_figure(file_name.split('/')[-1], f, global_step=h) 

    plt.close('all')

######################################################################################
############################################### plots test time ######################
######################################################################################


def plot_synthesis(intermed_Ys, sample):
    f, axs = plt.subplots(1,len(intermed_Ys), figsize = ( 4*len(intermed_Ys),4))
    axs = axs.ravel()

    #### plot intermediate steps
    for ax in range(len(intermed_Ys)):
        if torch.cuda.is_available():
            intermed_Ys[ax] = intermed_Ys[ax].cpu()

        x = intermed_Ys[ax].permute(1,2,0).detach().numpy()
        if x.shape[2] == 1: # if grayscale
            fig = axs[ax].imshow(x.squeeze(-1), 'gray')
        else: # if color
            fig = axs[ax].imshow(rescale_image(x))
        axs[ax].axis('off')

    #### plot final sample
    if torch.cuda.is_available():
        sample =sample.cpu()

    sample = sample.permute(1,2,0).detach().numpy()
    if sample.shape[2] == 1: # if grayscale
        fig = axs[-1].imshow(sample.squeeze(-1),'gray' )
    else: # if color
        fig = axs[-1].imshow(rescale_image(sample))

    axs[-1].axis('off')
    print('value range', np.round(np.min(sample ),2), np.round(np.max(sample),2) )


def plot_sample(x, corrupted, sample):
    if torch.cuda.is_available():
        x = x.cpu()
        corrupted = corrupted.cpu()
        sample = sample.cpu()

    x = x.permute(1,2,0)
    corrupted = corrupted.permute(1,2,0)
    sample = sample.detach().permute(1,2,0)

    if x.size()!=corrupted.size():
        h_diff = x.size()[0] - corrupted.size()[0]
        w_diff = x.size()[1] - corrupted.size()[1]
        x = x[0:x.size()[0]-h_diff,0:x.size()[1]-w_diff,: ]
        print('NOTE: psnr and ssim calculated using a cropped original image, because the original image is not divisible by the downsampling scale factor.')

    f, axs = plt.subplots(1,3, figsize = (15,5))
    axs = axs.ravel()
    if x.shape[2] == 1: # if gray scale image
        fig = axs[0].imshow( x.squeeze(-1), 'gray', vmin=0, vmax = 1)
        axs[0].set_title('original')

        fig = axs[1].imshow(corrupted.squeeze(-1), 'gray',vmin=0, vmax = 1)
        ssim = np.round(structural_similarity(x.squeeze(-1).numpy(), corrupted.squeeze(-1).numpy()  ) ,3 )
        psnr = np.round(peak_signal_noise_ratio(x.numpy(), corrupted.numpy() , data_range = 1),2)
        axs[1].set_title('partially measured image \n psnr: '+str( psnr) + '\n ssim '+ str(ssim) );

        fig = axs[2].imshow(sample.squeeze(-1),'gray' ,vmin=0, vmax = 1)
        ssim = np.round(structural_similarity(x.squeeze(-1).numpy(), sample.squeeze(-1).numpy()  ) ,3 )
        psnr = np.round(peak_signal_noise_ratio(x.numpy(), sample.numpy(), data_range = 1 ),2)
        axs[2].set_title('reconstructed \n psnr: '+str( psnr) + '\n ssim '+ str(ssim) );


    else: # if color image
        fig = axs[0].imshow( x, vmin=0, vmax = 1)
        axs[0].set_title('original')

        fig = axs[1].imshow( torch.clip(corrupted,0,1), vmin=0, vmax = 1)
        ssim = np.round(structural_similarity(x.numpy(), corrupted.numpy(), multichannel=True  ) ,3 )
        psnr = np.round(peak_signal_noise_ratio(x.numpy(), corrupted.numpy() , data_range = 1), 2)
        axs[1].set_title('partially measured image \n psnr: '+str( psnr) + '\n ssim '+ str(ssim) );

        fig = axs[2].imshow(torch.clip(sample, 0,1),vmin=0, vmax = 1)
        ssim = np.round(structural_similarity(x.numpy(), sample.numpy() , multichannel=True) ,3)
        psnr = np.round(peak_signal_noise_ratio(x.numpy(), sample.numpy() , data_range = 1),2)
        axs[2].set_title('reconstructed \n psnr: '+str( psnr) + '\n ssim '+ str(ssim) );


    for i in range(3):
        axs[i].axis('off')



def plot_all_coeffs(sample, intermed_Ys, orient):
    n_rows = int(np.ceil(len(intermed_Ys)/8))

    f, axs = plt.subplots(n_rows,8, figsize = ( 8*4, n_rows*4))
    f.tight_layout()
    axs = axs.ravel()

    #### plot intermediate steps
    for ax in range(len(intermed_Ys)):
        if torch.cuda.is_available():
            intermed_Ys[ax] = intermed_Ys[ax].cpu()

        x = intermed_Ys[ax].detach().permute(1,2,0).numpy()
        fig = axs[ax].imshow(x[:,:,orient], 'gray')
        plt.colorbar(fig, ax=axs[ax], fraction=.03)

    #### plot final sample
    if torch.cuda.is_available():
        sample =sample.cpu()

    sample = sample.detach().permute(1,2,0).numpy()
    fig = axs[-1].imshow(sample[:,:,orient],'gray' )

    axs[-1].axis('off')
    plt.colorbar(fig, ax=axs[-1], fraction=.05)


    for ax in range(len(intermed_Ys),n_rows*8 ):
        axs[ax].axis('off')

def plot_all_samples(sample, intermed_Ys):
    n_rows = int(np.ceil(len(intermed_Ys)/8))

    f, axs = plt.subplots(n_rows,8, figsize = ( 8*4, n_rows*4))
    f.tight_layout()
    axs = axs.ravel()

    #### plot intermediate steps
    for ax in range(len(intermed_Ys)):
        if torch.cuda.is_available():
            intermed_Ys[ax] = intermed_Ys[ax].cpu()

        x = intermed_Ys[ax].detach().permute(1,2,0).numpy()
        if x.shape[2] == 1:
            fig = axs[ax].imshow(x.squeeze(-1), 'gray')
            plt.colorbar(fig, ax=axs[ax], fraction=.03)

        else:
            fig = axs[ax].imshow(rescale_image(x))
        axs[ax].axis('off')

    #### plot final sample
    if torch.cuda.is_available():
        sample =sample.cpu()

    sample = sample.detach().permute(1,2,0).numpy()
    if sample.shape[2] == 1:
        fig = axs[-1].imshow(sample.squeeze(-1),'gray' )
    else:
        fig = axs[-1].imshow(rescale_image(sample))
    axs[-1].axis('off')
    plt.colorbar(fig, ax=axs[-1], fraction=.05)


    for ax in range(len(intermed_Ys),n_rows*8 ):
        axs[ax].axis('off')


def plot_corrupted_im(x_c):
    try:

        if torch.cuda.is_available():
            plt.imshow(x_c.squeeze(0).cpu(), 'gray', vmin=0, vmax = 1)
        else:
            plt.imshow(x_c.squeeze(0), 'gray', vmin=0, vmax = 1)
    except TypeError:
        if torch.cuda.is_available():
            plt.imshow(x_c.permute(1,2,0).cpu(), vmin=0, vmax = 1)
        else:
            plt.imshow(x_c.permute(1,2,0) , vmin=0, vmax = 1)

    plt.colorbar()


def print_dim(measurment_dim, image_dim):
    print('*** Retained {} / {} ({}%) of dimensions'.format(int(measurment_dim), image_dim
                                                   , np.round(int(measurment_dim)/int(image_dim)*100,
                                                              decimals=3) ))

def visualize_coeffs_tiled(coeffs, figsize):
    '''visulaizes wavelet coefficients in a tiled fashion. Assumes coeffs are from a complete representatoin
    i.e. there are only 3 bands per scale
    @coeffs: a list of tuples of arrays. Example for a wavelet pyramid of height 3: 
    [cA_n, (cH3_n, cV3_n, cD3_n), (cH2_n, cV2_n, cD2_n) , (cH1_n, cV1_n, cD1_n)]'''
    
    
    levels = len(coeffs) - 1
    
    
    
    image_size = 0
    for i in range(len(coeffs)): 
        image_size = coeffs[i][0].shape[0] + image_size
    
    temp = rescale_image(coeffs[0])    
    for i in range(1, levels+1):
        temp1 = np.hstack((temp, rescale_image(coeffs[i][0])))
        temp2 = np.hstack((rescale_image(coeffs[i][1]), rescale_image(coeffs[i][2])))
        temp = np.vstack((temp1, temp2))
    plt.figure(figsize= figsize)
    
    plt.imshow(temp, 'gray')
    plt.xlim(-.5,image_size-.5)
    plt.ylim(image_size-.5,-.5)  
    plt.axis('off')

    # Add lines
    for i in range(levels): 
        plt.plot([image_size/(2)**(i+1)-.5, image_size/2**(i+1)-.5], [0-.5, image_size/ 2**(i) -.5], color='y', linestyle='-', linewidth=1)
        plt.plot([0, image_size/ 2**(i)-.5],[image_size/(2)**(i+1)-.5, image_size/2**(i+1)-.5], color='y', linestyle='-', linewidth=1)

    
    
######################################################################################


def plot_denoising(clean, noisy, denoised, sup_label, device, vmin=0, vmax=1, im_size=3): 
    f, axs = plt.subplots(1,3, figsize = (3 * im_size, im_size+1) )
    f.suptitle(sup_label, fontsize = 20)
    if device.type == 'cuda':
        clean = clean.cpu().squeeze().numpy()
        noisy = noisy.cpu().squeeze().numpy()
        denoised = denoised.cpu().squeeze().numpy()
        
    axs[0].imshow(clean, 'gray', vmin=vmin, vmax = vmax) 
    axs[0].set_title('clean')
    axs[1].imshow(noisy, 'gray',vmin=vmin, vmax = vmax)
    axs[1].set_title(  'PSNR '+ str(round(peak_signal_noise_ratio(clean,noisy, data_range=1),3)) + '\n SSIM ' + str(round(structural_similarity(clean, noisy ,channel_axis=True),3)),fontsize = 12)    
    axs[2].imshow(denoised, 'gray',vmin=vmin, vmax = vmax)
    axs[2].set_title(  'PSNR '+ str(round(peak_signal_noise_ratio(clean,denoised, data_range=1),3)) + '\n SSIM ' + str(round(structural_similarity(clean, denoised ,channel_axis=True),3)),fontsize = 12)    
    
    for i in range(3): 
        axs[i].axis('off')


def plot_many_denoised(x, y, x_hat,  device, suptitle, label, train, vmin=0, vmax=1, im_size=3, n_columns=7):
    '''plot many denoised images from the same clean image (either different denoisers or different noise levels )
    @x: a single clean image
    @y: dictionary of noisy images
    @x_hat: dictionary of denoised images
    @label: to be used for subplot title
    @train: if True, clean comes from train set. If False, it comes from test set
    '''
    ############ plot clean image


    x = x.cpu().permute(0,2,3,1).squeeze().numpy()

    f, axs = plt.subplots(1 ,1, figsize = ( im_size+.5, im_size+.5) )
    f.suptitle(suptitle)
    plt.tight_layout()


    axs.imshow(x, 'gray', vmin=vmin, vmax = vmax)
    if train:
        axs.set_title('clean ' +  'Train image' ,fontsize = im_size*5)
    else:
        axs.set_title('clean ' +  'Test image' ,fontsize = im_size*5)
    axs.axis('off')

    ############ plot noisy images
    n_rows = int(len(y)/n_columns)
    if len(y)%n_columns != 0:
        n_rows = n_rows+1

    im_labels = [key for key,im in y.items() ]

    for i in range(len(y)):
        y[im_labels[i]] = y[im_labels[i]].permute(0,2,3,1).cpu().squeeze().numpy()

    f, axs = plt.subplots(n_rows ,n_columns, figsize = ( im_size*n_columns, n_rows*im_size ) )
    axs = axs.ravel()
    plt.tight_layout()

    for i in range(len(y)):
        axs[i].imshow(y[im_labels[i]], 'gray',vmin=vmin, vmax = vmax)
        axs[i].set_title( 'PSNR '+ str(round(peak_signal_noise_ratio(x,y[im_labels[i]], data_range=1),3))
                         + '\n distance from clean \n or (noise sigma ): ' + str( round(normalized_distance_np(x, y[im_labels[i]]) ,3)),
                         fontsize =im_size*5)

    for i in range(len(axs)):
        axs[i].axis('off')

    ############ plot denoised images
    im_labels = [key for key,im in x_hat.items() ]

    n_rows = int(len(x_hat)/n_columns)
    if len(x_hat)%n_columns != 0:
        n_rows = n_rows+1

    for i in range(len(x_hat)):
        x_hat[im_labels[i]] = x_hat[im_labels[i]].permute(0,2,3,1).cpu().squeeze().numpy()

    f, axs = plt.subplots(n_rows ,n_columns, figsize = ( im_size*n_columns, n_rows*im_size ) )
    axs = axs.ravel()
    plt.tight_layout()

    for i in range(len(x_hat)):
        axs[i].imshow(x_hat[im_labels[i]], 'gray',vmin=vmin, vmax = vmax)
        axs[i].set_title( label + str(im_labels[i])+
                               '\n PSNR '+ str(round(peak_signal_noise_ratio(x,x_hat[im_labels[i]], data_range=1),3)) +
                      '\n distance from clean: ' + str( round(normalized_distance_np(x, x_hat[im_labels[i]]) ,3)),
                                                      fontsize = im_size*5)

    for i in range(len(axs)):
        axs[i].axis('off')


def show_im_set(dataset , N=None , im_size=3,vmin=None, vmax=None, label='None', colorbar=False,n_columns=7, sub_labels=None, colormap='gray',
               norm=None, font_size=30):
    if N is None:
        N = dataset.shape[0]
    device = dataset.device
    if device.type == 'cuda':
        dataset = dataset.cpu()

    if dataset.shape[1] != 1:
        dataset = dataset.permute(0,2,3,1)


    n_rows = int(N/n_columns)
    if N%n_columns != 0:
        n_rows = n_rows+1


    f, axs = plt.subplots(n_rows, n_columns, figsize = (im_size * n_columns , im_size * n_rows))
    axs = axs.ravel()
    f.suptitle( label , fontsize = im_size*5)
    plt.tight_layout()
    for i in range(N ):
        if dataset.shape[1] == 1:
            fig = axs[i].imshow(dataset[i,0], colormap,norm=norm, vmin=vmin, vmax = vmax)
        else:
            fig = axs[i].imshow(rescale_image_range(dataset[i],1), colormap,norm=norm)

        if colorbar:
            plt.colorbar(fig, ax=axs[i], fraction=.05)
        if sub_labels is not None:
            axs[i].set_title( str(sub_labels[i]), fontsize = font_size)

        else:
            axs[i].set_title(str(i))
    for i in range(len(axs)):
        axs[i].axis('off')

    dataset = dataset.to(device)
    
    
def plot_single_im(x, size=(2,2), vmin=None, vmax=None, colorbar = False, label = None):

    if x.device.type == 'cuda':
        x = x.cpu().squeeze()
    else:
        x = x.squeeze()

    if len(x.shape)==3:
        x = x.permute(1,2,0)
        
    plt.figure(figsize = size)
    if len(x.shape)==3:
        plt.imshow(rescale_image_range(x,1) )
    else: 
        plt.imshow(x, 'gray' , vmin=vmin, vmax=vmax )
        
    plt.axis('off')
    if colorbar:
        plt.colorbar()
    if label is not None:
        plt.title(label)
       
def plot_matching_patches(sample , train_set, ids, corrs, im_size=3,vmin=None, vmax=None, colorbar=False):

    mid = int(sample.shape[2]/2)
    loc = [(0-.5,0-.5),(mid-.5,0-.5),(0-.5,mid-.5),(mid-.5,mid-.5)]
    if sample.device.type == 'cuda':
        sample = sample.cpu().squeeze()
    else: 
        sample = sample.squeeze()
        
    if train_set.device.type == 'cuda':
        train_set = train_set.cpu().squeeze(1)
    else: 
        train_set = train_set.squeeze(1)
            
    f, axs = plt.subplots(1, 7, figsize = (im_size * 7 , im_size * 1))
    # f.suptitle( 'train images with highest similarity scores' , fontsize = im_size*5)
    
    plt.tight_layout()
    
    fig = axs[0].imshow(sample, 'gray',vmin=vmin, vmax = vmax)
    axs[0].set_title('sample')

    for i in range(4): 
        fig = axs[i+3].imshow(train_set[ids[i]], 'gray',vmin=vmin, vmax = vmax)
        if colorbar:
            plt.colorbar(fig, ax=axs[i+3], fraction=.05)
        axs[i+3].set_title( 'corr: ' +str(round(corrs[i].item(),4) ))
        pa = patches.Rectangle((loc[i]), width=20, height=20, angle=0.0,  edgecolor = 'red', facecolor='none' )
        axs[i+3].add_patch(pa);         
        
    temp = torch.zeros_like(sample)        
    temp[0:mid,0:mid ] = train_set[ids[0]][0:mid,0:mid]
    temp[0:mid,mid:mid*2 ] = train_set[ids[1]][0:mid,mid:mid*2]
    temp[mid:mid*2,0:mid ] = train_set[ids[2]][mid:mid*2,0:mid]
    temp[mid:mid*2,mid:mid*2 ] = train_set[ids[3]][mid:mid*2,mid:mid*2 ]
        
    fig = axs[1].imshow(temp, 'gray',vmin=vmin, vmax = vmax)
    axs[1].set_title('patches of training examples')

    fig = axs[2].imshow(rescale_image_range(temp,1) - rescale_image_range(sample,1), 'gray',vmin=vmin, vmax = vmax)
    axs[2].set_title('difference bw sample \n and combination training image')
    
    for i in range(len(axs)): 
        axs[i].axis('off')   
