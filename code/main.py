import numpy as np
import torch.nn as nn
import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import argparse
import sys
from network import *
from model_loader_func import initialize_network
from trainer import run_training
from dataloader_func import weights_init_kaiming
import pickle

####################################################################################################################
########################################################## Experiment specific functions #######################
####################################################################################################################
def build_path(args):
    '''
    build the path to save results of trainig. 
    General pattern is: architecture name, data name, noise level, etc 
    This should change depending on specific of path names
    '''
    dir_name = args.dir_name + args.arch_name + '/'+ args.data_name + '/'+str(args.noise_level_range[0])+'to'+ str(args.noise_level_range[1]) 
    if args.RF is not None: 
        dir_name = dir_name + '_RF_'+str(args.RF)+'x'+str(args.RF) 

    if args.set_size is not None: 
        dir_name = dir_name + '_set_size_' + str(args.set_size)
    if args.swap:
        dir_name = dir_name + '_swapped'

    return dir_name


def make_loader(train_set, test_set, args):

    trainloader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False)

    return trainloader, testloader



def prep_data_swap(train_coeffs,args, save):

    # prep swap exp data
    if args.swap is False:
        train_set = train_coeffs[0:args.set_size]
        test_set = train_coeffs[int(-1*args.set_size)::] #for these experiments, take test image from end of train set
    else:
        train_set = train_coeffs[int(-1*args.set_size)::]
        test_set = train_coeffs[0:args.set_size] #for these experiments, take test image from end of train set
    # Don't need the whole test set:
    test_set = test_set[0:args.batch_size]
    if save: 
        torch.save(train_set, args.dir_name + '/train_set.pt')
        torch.save(test_set, args.dir_name + '/test_set.pt')

        
    return train_set, test_set


def repeat_images(train_set,args, N_total): 
    n = int(N_total/args.set_size)
    train_set = torch.tile(train_set,(n,1,1,1)  )
    return train_set





####################################################################################################################
################################################# main #################################################
####################################################################################################################
def main():
    parser = argparse.ArgumentParser(description='training a denoiser')
    
    ### general architecture variables general
    parser.add_argument('--arch_name', type=str, default= 'UNet') ## choose model architecture    
    parser.add_argument('--kernel_size', default= 3)
    parser.add_argument('--padding', default= 1)
    parser.add_argument('--num_kernels', default= 64)
    parser.add_argument('--num_channels', default= 1, help='set 1 for grayscale and 3 for color')
    parser.add_argument('--skip', default= True)
    parser.add_argument('--bias', default=False)

    ### architecture variables BF_CNN and BF_CNN_RF    
    parser.add_argument('--first_layer_linear', default= False, help='For BF_CNN model')    
    parser.add_argument('--num_layers', default= 21)  
    parser.add_argument('--coarse', default = True, help = 'For BF_CNN_RF model. Denoiser for coarse or fine coefficients')
    parser.add_argument('--RF' , default = 90, help='For BF_CNN_RF model or just as a name tag. Receptive field of the network') # only values in this set {5,9,13, 23, 43}
    
    ### architecture variables UNet    
    parser.add_argument('--num_blocks', default= 3)    
    parser.add_argument('--num_enc_conv', default= 2, help='min is 3')  
    parser.add_argument('--num_mid_conv', default= 2, help='min is 3')  
    parser.add_argument('--num_dec_conv', default= 2, help='min is 3') 
    parser.add_argument('--pool_window', default= 2, help='min is 2') 

    ### optimization variables
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--lr_freq', type=int, default=100)


    ### dataset variables
    parser.add_argument('--noise_level_range', default= [0,255])
    parser.add_argument('--quadratic_noise', default = True) #!! make this false and see if it improves performance for medium and large noise levels. Right now we have overfitting in these noise regimes 
    parser.add_argument('--rescale', default=False ,help='rescale intensities. Do not rescale for conditional denoisers.')
    parser.add_argument('--swap', default=False)
    parser.add_argument('--set_size', default=None)
    parser.add_argument('--alpha', default=None)

    ### directory-related variables
    parser.add_argument('--data_name', type=str )    
    parser.add_argument('--data_root_path', default= '../datasets/')    
    parser.add_argument('--dir_name', default= '../denoisers/') #folder where outputs will be saved (modify accordingly)
    ## other variables
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--debug', default=False)
    parser.add_argument('--SLURM_ARRAY_TASK_ID',type=int)

    args = parser.parse_args()
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #################################################################################
    
    args.set_size = 10**(args.SLURM_ARRAY_TASK_ID)
    args.data_path = args.data_root_path + args.data_name
    
    ### load raw data
    train_set = torch.load(args.data_path + 'train_no_repeats_80x80.pt')
    
    print('all data: ', train_set.size() )    
    if int(train_set.shape[0]/2) < args.set_size: 
        args.set_size = int(train_set.shape[0]/2)

    ### build path
    args.dir_name = build_path(args)
    if not os.path.exists(args.dir_name):
        os.makedirs(args.dir_name)
        
        
    ### select a sub-set and swap if needed 
    train_set, test_set = prep_data_swap(train_set, args, save=False)    
    print('train: ', train_set.size(), 'test: ', test_set.size() )
    args.num_channels = train_set.shape[1] #set number of input channels

    ### for debug mode
    if args.debug:
        train_set = train_set[0:args.batch_size]
        test_set = test_set[0:args.batch_size]
        args.num_epochs = 5
    else:
        ### repeat train images 
        train_set  = repeat_images(train_set,args, N_total=250000) # N_total=1000000 for 40x40 images 
        
    print('train: ', train_set.size(), 'test: ', test_set.size() )

    trainloader, testloader = make_loader(train_set, test_set, args)

    ### initialize a model
    model = initialize_network(args.arch_name, args)
    model.apply(weights_init_kaiming)
    if torch.cuda.is_available():
        print('[ Using CUDA ]')
        model = nn.DataParallel(model).cuda()
    print('number of parameters is ' , sum(p.numel() for p in model.parameters() if p.requires_grad))

    ## select criterion and optimizer
    criterion = nn.MSELoss(reduction='sum')
    optimizer = Adam(filter(lambda p: p.requires_grad,model.parameters()), lr = args.lr)

    ## save model args in case 
    with open( args.dir_name +'/exp_arguments.pkl', 'wb') as f:
        pickle.dump(args.__dict__, f)

    ### train 
    model = run_training(model, trainloader, testloader,criterion,optimizer, args)


if __name__ == "__main__" :
    main()


