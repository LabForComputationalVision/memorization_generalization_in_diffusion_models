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
    loads flat BF_CNN with standard parameters. They can be changed through @my_args argument. 
    '''

    parser = argparse.ArgumentParser(description='Unet')
    parser.add_argument('--kernel_size', default= 3, help='Filter size')
    parser.add_argument('--padding', default= 1)
    parser.add_argument('--num_kernels', default= 64, help='number of channels in intermediate layers')
    parser.add_argument('--RF',default = 90, help='Determines the receptive field size of the UN')
    parser.add_argument('--num_channels', default= 1, help='Number of input channels: 3 for RGB and 1 for grayscale')
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

    RF_blocks = {18:1 , 45:2, 90:3, 180:4}                
    if args.RF in RF_blocks.keys(): 
        args.num_blocks = RF_blocks[args.RF]
    else: 
        raise ValueError('Choose an RF size in ' + str(RF_blocks.keys()))
                
        
    model = UNet(args)
    
    if torch.cuda.is_available():
        model = model.cuda()
    return model




def init_BF_CNN_RF(RF=43, coarse=True,num_channels=1, my_args=None):
    '''
    loads flat BF_CNN with RF flexibility.
    @my_args
    '''

    parser = argparse.ArgumentParser(description='flat BF_CNN')
    parser.add_argument('--kernel_size', default= 3)
    parser.add_argument('--padding', default= 1)
    parser.add_argument('--num_kernels', default= 64)
    parser.add_argument('--num_layers', default= 21)
    parser.add_argument('--RF',default = RF)
    parser.add_argument('--coarse', default= coarse) 
    parser.add_argument('--num_channels', default= num_channels)

    args = parser.parse_args('')
    
    if my_args is not None: ## update args with given args
        for key, value in vars(my_args).items():
                vars(args)[key] = value
                
    model = BF_CNN_RF(args)
    
    if torch.cuda.is_available():
        model = model.cuda()
    return model




def init_BF_CNN(my_args=None, activations=False):
    '''
    loads flat BF_CNN with RF flexibility.
    @ grayscale: if True, number of input and output channels are set to 1. Otherwise 3
    '''

    parser = argparse.ArgumentParser(description='flat BF_CNN')
    parser.add_argument('--kernel_size', default= 3)
    parser.add_argument('--padding', default= 1)
    parser.add_argument('--num_kernels', default= 64)
    parser.add_argument('--num_layers', default= 20)
    parser.add_argument('--num_channels', default= 1)
    parser.add_argument('--coarse', default= True) 
    parser.add_argument('--first_layer_linear', default= True) 
    
    args = parser.parse_args('')

    if my_args is not None: ## update args with given args
        for key, value in vars(my_args).items():
                vars(args)[key] = value
    
    if activations is False: 
        model = BF_CNN(args)
    else: 
        model = BF_CNN_activations(args)
        
    if torch.cuda.is_available():
        model = model.cuda()
            
    return model

##########################################
def load_UNet(base_path, training_data_name, training_noise,RF=None, set_size=None, swap=False, my_args=None): 
    '''Loads a saved trained UNet model
    @base_path: root path to the folder where the related models are saved
    @training_data_name: name of training data to be added to the root path 
    @training_noise: range of the standard deviation of the noise to be added to the path 
    @RF: receptive field of the network, to set the size of the network. Also added to the path. 
    @set_size: size of the training set the network was trained on. 
    @swap: True is the network was trained on the second half of the dataset
    @my_args: paramters of the network, such as RF, if different from the standard initialization 
    '''
    if my_args is None: 
        parser = argparse.ArgumentParser(description='')
        parser.add_argument('--RF', default = RF)
        my_args = parser.parse_args('')
    else: 
        my_args.RF = RF
        
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


def load_BF_CNN(base_path, training_data_name, training_noise,set_size = None, n_classes=None, my_args=None, activations=False):
    
    init = init_BF_CNN(my_args, activations)
    ## build path 
    model_path = os.path.join(base_path,training_data_name,training_noise )
    if set_size is not None: 
        model_path = model_path + '_set_size_'+str(set_size) 
    if n_classes is not None:
        model_path = model_path + '_n_classes_'+str(n_classes) 
         
    model_path = os.path.join(model_path,'model.pt')
    ## load model
    model = read_trained_params(init, model_path)
    return model




def load_BF_CNN_RF(base_path, training_data_name, training_noise, RF=None, set_size=None, swap=False, my_args=None): 

    init = init_BF_CNN_RF(RF=RF, coarse=True, my_args=my_args)
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

##########################################
def read_trained_params(model, path): 
    '''reads parametres of saved models into an initialized network'''
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

