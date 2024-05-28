import numpy as np
import torch.nn as nn
import torch

################################################# network class #################################################
class UNet(nn.Module): 
    def __init__(self, args): 
        super(UNet,self).__init__()
        
        self.pool_window = args.pool_window
        self.num_blocks = args.num_blocks
        ########## Encoder ##########
        self.encoder = nn.ModuleDict([])
        for b in range(self.num_blocks):
            self.encoder[str(b)] = self.init_encoder_block(b,args)
                                
        ########## Mid-layers ##########
        mid_block = nn.ModuleList([])
        for l in range(args.num_mid_conv):
            if l==0:
                mid_block.append(nn.Conv2d(args.num_kernels*(2**b) ,args.num_kernels*(2**(b+1)), args.kernel_size, padding=args.padding , bias=args.bias))
            else: 
                mid_block.append(nn.Conv2d(args.num_kernels*(2**(b+1)) ,args.num_kernels*(2**(b+1)), args.kernel_size, padding=args.padding , bias=args.bias))    
            mid_block.append(BF_batchNorm(args.num_kernels*(2**(b+1)) ))
            mid_block.append(nn.ReLU(inplace=True))
            
        self.mid_block = nn.Sequential(*mid_block)
                                    
        ########## Decoder ##########
        self.decoder = nn.ModuleDict([])
        self.upsample = nn.ModuleDict([])
        for b in range(self.num_blocks-1,-1,-1):
            self.upsample[str(b)], self.decoder[str(b)] = self.init_decoder_block(b,args)
        
        
        
    def forward(self, x):
        pool =  nn.AvgPool2d(kernel_size=self.pool_window, stride=2, padding=int((self.pool_window-1)/2) )  
        ########## Encoder ##########
        unpooled = []
        for b in range(self.num_blocks): 
            x_unpooled = self.encoder[str(b)](x)
            x = pool(x_unpooled)
            unpooled.append(x_unpooled)
            
        ########## Mid-layers ##########
        x = self.mid_block(x)
        
        ########## Decoder ##########
        for b in range(self.num_blocks-1, -1, -1):
            x = self.upsample[str(b)](x)
            x = torch.cat([x, unpooled[b]], dim = 1)
            x = self.decoder[str(b)](x)    

        return x
    
    
    def init_encoder_block(self, b, args):
        enc_layers = nn.ModuleList([])
        if b==0:
            enc_layers.append(nn.Conv2d(args.num_channels ,args.num_kernels, args.kernel_size, padding=args.padding, bias=args.bias))
            enc_layers.append(nn.ReLU(inplace=True))
            for l in range(1,args.num_enc_conv): 
                enc_layers.append(nn.Conv2d(args.num_kernels ,args.num_kernels, args.kernel_size, padding=args.padding, bias=args.bias))
                enc_layers.append(BF_batchNorm(args.num_kernels))
                enc_layers.append(nn.ReLU(inplace=True))
        else: 
            for l in range(args.num_enc_conv): 
                if l==0:
                    enc_layers.append(nn.Conv2d(args.num_kernels*(2**(b-1)) ,args.num_kernels*(2**b), args.kernel_size, padding=args.padding, bias=args.bias))
                else: 
                    enc_layers.append(nn.Conv2d(args.num_kernels*(2**b) ,args.num_kernels*(2**b), args.kernel_size, padding=args.padding, bias=args.bias))
                enc_layers.append(BF_batchNorm(args.num_kernels*(2**b)))
                enc_layers.append(nn.ReLU(inplace=True))
                
                
        return nn.Sequential(*enc_layers)
    
    def init_decoder_block(self, b, args):
        dec_layers = nn.ModuleList([])
        
        #initiate the last block:
        if b==0:
            for l in range(args.num_dec_conv-1): 
                if l==0:                    
                    upsample = nn.ConvTranspose2d(args.num_kernels*2, args.num_kernels, kernel_size=2, stride=2,bias=False)
                    dec_layers.append(nn.Conv2d(args.num_kernels*2, args.num_kernels, kernel_size=args.kernel_size, padding=args.padding,bias=False))                                    
                else: 
                    dec_layers.append(nn.Conv2d(args.num_kernels ,args.num_kernels, args.kernel_size, padding=args.padding, bias=args.bias))                                    
                dec_layers.append(BF_batchNorm(args.num_kernels))
                dec_layers.append(nn.ReLU(inplace=True))
                
            dec_layers.append(nn.Conv2d(args.num_kernels, args.num_channels, kernel_size=args.kernel_size, padding=args.padding,bias=False))
            
        #other blocks
        else: 
            for l in range(args.num_dec_conv): 
                if l==0:
                    upsample= nn.ConvTranspose2d(args.num_kernels*(2**(b+1)), args.num_kernels*(2**b), kernel_size=2, stride=2,bias=False)
                    dec_layers.append(nn.Conv2d(args.num_kernels*(2**(b+1)), args.num_kernels*(2**b), kernel_size=args.kernel_size, padding=args.padding,bias=False))                                    
                else: 
                    dec_layers.append(nn.Conv2d(args.num_kernels*(2**b) ,args.num_kernels*(2**b), args.kernel_size, padding=args.padding, bias=args.bias))

                dec_layers.append(BF_batchNorm(args.num_kernels*(2**b)))
                dec_layers.append(nn.ReLU(inplace=True))
        return upsample, nn.Sequential(*dec_layers)

  
        
class BF_CNN_RF(nn.Module):

    def __init__(self, args):
        super(BF_CNN_RF, self).__init__()
        if args.num_layers != 21:
            raise ValueError('number of layers must be 21 ')

        if args.RF not in [5,8,9,13,23,43]:
            raise ValueError('choose a receptive field in [5,8,9,13,23,43]')

        #this creates RF=9x9, because of the way interspersing 3x3 layers in my code work. Improve code later 
        if args.RF == 9:
            args.RF = 8

        self.num_layers = args.num_layers #21
        self.conv_layers = nn.ModuleList([])
        self.BN_layers = nn.ModuleList([])


        if args.coarse:
            self.conv_layers.append(nn.Conv2d(args.num_channels,args.num_kernels, args.kernel_size, padding=args.padding , bias=False))

        else:
            self.conv_layers.append(nn.Conv2d(args.num_channels*3+1,args.num_kernels, args.kernel_size, padding=args.padding , bias=False))


        for l in range(1,self.num_layers-1):
            if l%((args.num_layers - 1)/ (((args.RF-1)/2)-1)) != 0: ### set some of kernel sizes to 1x1
                kernel_size = 1
                padding = 0
            else:
                kernel_size = args.kernel_size
                padding = args.padding
            self.conv_layers.append(nn.Conv2d(args.num_kernels ,args.num_kernels, kernel_size, padding=padding , bias=False))
            self.BN_layers.append(BF_batchNorm(args.num_kernels ))

        if args.coarse:
            self.conv_layers.append(nn.Conv2d(args.num_kernels,args.num_channels, args.kernel_size, padding=args.padding , bias=False))
        else:
            self.conv_layers.append(nn.Conv2d(args.num_kernels,args.num_channels*3, args.kernel_size, padding=args.padding , bias=False))



    def forward(self, x):
        # activations = []
        relu = nn.ReLU(inplace=True)
        x = self.conv_layers[0](x) #first layer linear

        for l in range(1,self.num_layers-1):
            x = self.conv_layers[l](x)
            x = self.BN_layers[l-1](x)
            # activations.append((x>0))
            x = relu(x)

        x = self.conv_layers[-1](x)
        # return x, activations
        return x


class BF_batchNorm(nn.Module):
    def __init__(self, num_kernels):
        super(BF_batchNorm, self).__init__()
        self.register_buffer("running_sd", torch.ones(1,num_kernels,1,1))
        g = (torch.randn( (1,num_kernels,1,1) )*(2./9./64.)).clamp_(-0.025,0.025)
        self.gammas = nn.Parameter(g, requires_grad=True)

    def forward(self, x):
        training_mode = self.training       
        sd_x = torch.sqrt(x.var(dim=(0,2,3) ,keepdim = True, unbiased=False)+ 1e-05)
        if training_mode:
            x = x / sd_x.expand_as(x)
            with torch.no_grad():
                self.running_sd.copy_((1-.1) * self.running_sd.data + .1 * sd_x)

            x = x * self.gammas.expand_as(x)

        else:
            x = x / self.running_sd.expand_as(x)
            x = x * self.gammas.expand_as(x)

        return x





class BF_CNN(nn.Module):

    def __init__(self, args): 
        super(BF_CNN, self).__init__()


        self.num_layers = args.num_layers
        self.first_layer_linear = args.first_layer_linear
        
        self.conv_layers = nn.ModuleList([])
        self.BN_layers = nn.ModuleList([])


        self.conv_layers.append(nn.Conv2d(args.num_channels,args.num_kernels, args.kernel_size, padding=args.padding , bias=False))

        for l in range(1,self.num_layers-1):
            self.conv_layers.append(nn.Conv2d(args.num_kernels ,args.num_kernels, args.kernel_size, padding=args.padding , bias=False))
            self.BN_layers.append(BF_batchNorm(args.num_kernels ))

        self.conv_layers.append(nn.Conv2d(args.num_kernels,args.num_channels, args.kernel_size, padding=args.padding , bias=False))



    def forward(self, x):
        relu = nn.ReLU(inplace=True)

        x = self.conv_layers[0](x) #first layer linear (different from orginal/old implementation)
        if self.first_layer_linear is False: 
            x = relu(x)

        for l in range(1,self.num_layers-1):
            x = self.conv_layers[l](x)
            x = self.BN_layers[l-1](x)
            x = relu(x)
        x = self.conv_layers[-1](x)

        return x 

    
