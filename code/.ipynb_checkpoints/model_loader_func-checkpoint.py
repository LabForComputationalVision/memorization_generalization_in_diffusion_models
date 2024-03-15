import numpy as np
import torch
import torch.nn as nn
import os
import argparse
from network import *

##################################################################################################

def initialize_network(network_name, args):
    '''
    Function to dynamically initialize a neural network by class name
    '''
    if network_name in globals() and issubclass(globals()[network_name], nn.Module):
        return globals()[network_name](args)
    else:
        raise ValueError(f"Network {network_name} not found or not a subclass of nn.Module")



def init_UNet(my_args=None):
    '''
    loads flat BF_CNN with RF flexibility.
    @my_args
    '''

    parser = argparse.ArgumentParser(description='Unet')
    parser.add_argument('--kernel_size', default= 3)
    parser.add_argument('--padding', default= 1)
    parser.add_argument('--num_kernels', default= 64)
    parser.add_argument('--RF',default = 90)
    parser.add_argument('--num_channels', default= 1)
    parser.add_argument('--bias', default=False)    
    parser.add_argument('--num_enc_conv', default= 2, help='min is 3')
    parser.add_argument('--num_mid_conv', default= 2, help='min is 3')
    parser.add_argument('--num_dec_conv', default= 2, help='min is 3')
    parser.add_argument('--pool_window', default= 2, help='min is 2')

    parser.add_argument('--num_blocks', help='This is set according to the desired RF')
    
    args = parser.parse_args('')

    
    if my_args is not None: ## update args with given args
        for key, value in vars(my_args).items():
                vars(args)[key] = value

    
    if RF == 18: 
        my_args.num_blocks = 1        
    elif RF == 45: 
        my_args.num_blocks = 2
    elif RF ==90: 
        my_args.num_blocks = 3        
    elif RF == 180: 
        my_args.num_blocks = 4
    else: 
        raise ValueError
        
        
    model = UNet(args)
    
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def read_trained_params(model, path): 
               
    if torch.cuda.is_available():
        learned_params =torch.load(path)
    else:
        learned_params =torch.load(path, map_location='cpu' )
        
    ## unwrap if in Dataparallel 
    new_state_dict = {}
    for key,value in learned_params.items(): 
        if key.split('.')[0] == 'module': 
            new_key = '.'.join(key.split('.')[1::])
            new_state_dict[new_key] = value

        else: 
            new_state_dict[key] = value
        

    model.load_state_dict(new_state_dict)        
    model.eval();

    return model


def load_UNet(base_path, training_data_name, training_noise,RF=None, set_size=None, swap=False, my_args=None): 
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--RF', default = RF)
    my_args = parser.parse_args('')
    
    
    init = init_UNet( my_args=my_args)
    
    ## build path 
    model_path = os.path.join(base_path,training_data_name,training_noise )    
    if RF is not None: 
        model_path = model_path+'_RF_'+str(RF)+'x'+str(RF)
    if set_size is not None: 
        model_path = model_path +'_set_size_'+str(set_size)
    if swap: 
        model_path = model_path +'_swapped'
    model_path = os.path.join(model_path,'model.pt')   
    ## load model
    model = read_trained_params(init, model_path)
    return model



