## This module takes a network, a loss function, optimizer and data and trains a DNN denoiser 

import numpy as np
import torch.nn as nn
import os
import time
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import sys
from network import *
from dataloader_func import weights_init_kaiming, add_noise_torch, add_noise_torch_range
from quality_metrics_func import batch_ave_psnr_torch, calc_psnr
from plotting_func import plot_loss,plot_psnr, plot_denoised_range

########################################################### util training functions ###########################################################

def show_denoised_range(model, im,noise_range, args, file_name, writer,h):
    '''
    Takes an image, adds different levels of noise, denoises them and plots the denoised and noisy 
    '''
    model.eval()
    with torch.no_grad():
        noisy , noise = add_noise_torch_range(im, noise_range, device=args.device,coarse=args.coarse)
        output = model(noisy)
        if args.skip:
            if args.coarse:
                denoised = noisy - output
            else:
                denoised = noisy[:,1::] - output
        else:
            denoised = output

    if args.coarse:
        file_name = file_name + '.png'
        plot_denoised_range(im, noisy, denoised, noise_range, args.dir_name+ file_name, 1,writer,h)

    else:
        for o in range(3):
            file_name = file_name +str(o)+ '.png'
            plot_denoised_range(im, noisy[:,o+1:o+2], denoised[:,o:o+1], noise_range, args.dir_name+file_name, 1,writer,h)

########################################################### training ###########################################################
def one_iter(model, batch ,criterion, args):
    ## clean data
    clean = batch.to(args.device)
    if args.rescale:
        clean = clean * torch.rand(size=(batch.size()[0], 1,1,1),device = args.device)
    ## add noise 
    noisy , noise = add_noise_torch(clean, args.noise_level_range, args.device, args.quadratic_noise, args.coarse) 
    
    ## denoise 
    output = model(noisy)
    
    ## handle skip and coarse vs fine
    if args.skip:
        target = noise
        if args.coarse:
            denoised = noisy - output
        else:
            denoised = noisy[:,1::] - output
    else:
        if args.coarse:
            target = clean
        else:
            target = clean[1::] #C=3

        denoised = output
    ## compute loss
    loss = criterion(output, target)/ (clean.size()[0])
    
    ## compute psnr
    if args.coarse:
        psnr = batch_ave_psnr_torch(clean, denoised ,max_I=1.)
    else:
        psnr = batch_ave_psnr_torch(clean[:,1::], denoised ,max_I=1.)

    return model, loss, psnr



def train_epoch(model, trainloader,criterion,optimizer, args):
    loss_sum = 0
    psnr_sum = 0
    model.train()    
    for i, batch in enumerate(trainloader, 0):    
        optimizer.zero_grad()
        model, loss, psnr = one_iter(model, batch,criterion, args)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        psnr_sum += psnr.item()


    return model, loss_sum/(i+1), psnr_sum/(i+1)



def test_epoch(model, testloader,criterion, args):
    loss_sum = 0
    psnr_sum = 0
    model.eval()    
    with torch.no_grad():
        for i, batch in enumerate(testloader, 0):
            model, loss, psnr = one_iter(model, batch,criterion, args)
            loss_sum+= loss.item()
            psnr_sum += psnr.item()

    return loss_sum/(i+1), psnr_sum/(i+1)


def run_training(model, trainloader, testloader,criterion,optimizer, args):
    '''
    trains a denoiser neural network
    '''


    ###
    start_time_total = time.time()
    epoch_loss_list_train = [] #delet if TB works
    epoch_psnr_list_train = []#delet if TB works
    epoch_loss_list_test = []#delet if TB works
    epoch_psnr_list_test = []#delet if TB works
    writer = SummaryWriter(log_dir=args.dir_name)
    im_train = next(iter(trainloader))[0].to(args.device)
    im_test = next(iter(testloader))[0].to(args.device)
    
    ### loop over number of epochs
    for h in range(args.num_epochs):
        print('epoch ', h )
        if h >= args.lr_freq and h%args.lr_freq==0:
            for param_group in optimizer.param_groups:
                args.lr = args.lr/2
                param_group["lr"] = args.lr

        #train
        model, epoch_loss_train, epoch_psnr_train = train_epoch(model, trainloader,criterion,optimizer, args)
        epoch_loss_list_train.append(epoch_loss_train)#delet if TB works
        epoch_psnr_list_train.append(epoch_psnr_train)#delet if TB works
        writer.add_scalar('PSNR/Train', epoch_psnr_train, global_step=h)
        writer.add_scalar('Loss/Train', epoch_loss_train, global_step=h)
        print('train loss = ', epoch_loss_train, 'train psnr = ',epoch_psnr_train )
        
        #eval
        epoch_loss_test, epoch_psnr_test = test_epoch(model, testloader,criterion, args)
        epoch_loss_list_test.append(epoch_loss_test)#delet if TB works
        epoch_psnr_list_test.append(epoch_psnr_test)#delet if TB works
        writer.add_scalar('PSNR/Test', epoch_psnr_test, global_step=h)
        writer.add_scalar('Loss/Test', epoch_loss_test, global_step=h)
        print('test loss = ', epoch_loss_test, 'test psnr = ',epoch_psnr_test )
        
        #plot and save
        plot_loss(epoch_loss_list_train, epoch_loss_list_test, args.dir_name+'/loss_epoch.png')#delet if TB works
        plot_psnr(epoch_psnr_list_train , epoch_psnr_list_test ,args.dir_name+'/psnr_epoch.png' )#delet if TB works
        noise_range = torch.logspace(0,2.5,4, device=args.device).reshape(4,1,1,1)
        show_denoised_range(model,im_train, noise_range, args, '/denoised_train_image',writer,h)
        show_denoised_range(model,im_test, noise_range, args, '/denoised_test_image',writer,h)
        noise_range_psnr = torch.logspace(0,2.5,10, device=args.device).reshape(10,1,1,1)
        # psnr_range , mse_range = psnr_over_range(model, testloader, noise_range_psnr,criterion, args)
        psnr_range = calc_psnr(model,testloader, noise_range_psnr, args.device)
        writer.add_scalars('psnr_range', psnr_range, global_step=h )
        # writer.add_scalars('mse_range', mse_range, global_step=h )
      
        
        #save model after each epoch
        if args.coarse:
            torch.save(model.state_dict(), args.dir_name  + '/model.pt')
        else:
            torch.save(model.state_dict(), args.dir_name  + '/model_scale'+str(args.SLURM_ARRAY_TASK_ID)+'.pt')


    print("--- %s seconds ---" % (time.time() - start_time_total))
    writer.close()
    
    return model