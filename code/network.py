import numpy as np
import torch.nn as nn
import torch

################################################# network class #################################################

class UNet_efficient(nn.Module): 
    '''
    UNet inspired by our local multi-scale model
    Number of levels should be such that the coarse level with 20 layers (RF=41x41) receives a 40x40 image. 
    - Encoder: each block does not need to be so deep. The role of encoder is mainly to down sample the image, ot denoise. Exp1: make encoder fully linear
    - We want the denoising to be done but decoders, becuase they receive denoise coarse, so they can condition on that.
    -----------
    - Coarse level: Make the mid layer deep enough so that the final RF at the last layer in the coarse level covers the whole input image (80x80). 
    - This is the difficult part of denoising due to global dependencies (if the UNet manages to pass the low-pass image to this level)
    -----------
    - Decoder: these blocks receive the denoised coarse and conditioned on that they can denoise fine. They should contain more layers than the encoder. 
    - Due to skip connections, effective RF is small, but this should be ok due to conditional Markov property of fine features. 
    '''
    def __init__(self, args): 
        super(UNet_efficient, self).__init__()
        
        
        ########## Encoder ##########
        # for input: 80x80xC  
        self.e11 = nn.Conv2d(args.num_channels, args.num_kernels , kernel_size=args.kernel_size, padding=args.padding, bias=args.bias) 
        self.e12 = nn.Conv2d(args.num_kernels , args.num_kernels , kernel_size=args.kernel_size, padding=args.padding, bias=args.bias) 
        self.bn11 = BF_batchNorm(args.num_kernels )        
        self.pool11 = nn.AvgPool2d(kernel_size=2, stride=2) #probably not good. Choose a better method
        # output: 40x40x64

        # for input: 40x40x64 
        self.e21 = nn.Conv2d(args.num_kernels, args.num_kernels , kernel_size=args.kernel_size, padding=args.padding, bias=args.bias) 
        self.bn21 = BF_batchNorm(args.num_kernels )                
        self.e22 = nn.Conv2d(args.num_kernels , args.num_kernels , kernel_size=args.kernel_size, padding=args.padding, bias=args.bias) 
        self.bn22 = BF_batchNorm(args.num_kernels )        
        self.pool21 = nn.AvgPool2d(kernel_size=2, stride=2) 
        # output: 20x20x64
        
        ########## coarse level ##########
        #input: 20x20x64
        self.num_mid_layers = args.num_mid_layers
        self.conv_mid = nn.ModuleList([])
        self.BN_mid = nn.ModuleList([])
        
        for l in range(self.num_mid_layers):
            self.conv_mid.append(nn.Conv2d(args.num_kernels ,args.num_kernels, args.kernel_size, padding=args.padding , bias=args.bias))
            self.BN_mid.append(BF_batchNorm(args.num_kernels ))
        # output: 20x20x64

        ########## Decoder  ##########
        # input: 20x20x64    
        self.num_dec_layers = args.num_dec_layers
        self.conv_dec2 = nn.ModuleList([])
        self.BN_dec2 = nn.ModuleList([])   
        
        self.upconv2 = nn.ConvTranspose2d(args.num_kernels, args.num_kernels, kernel_size=2, stride=2,bias=args.bias) #upsamples output of mid layers
        for l in range(self.num_dec_layers):
            if l==0:
                self.conv_dec2.append(nn.Conv2d(args.num_kernels*2, args.num_kernels, kernel_size=args.kernel_size, padding=args.padding,bias=args.bias))
            else: 
                self.conv_dec2.append(nn.Conv2d(args.num_kernels, args.num_kernels, kernel_size=args.kernel_size, padding=args.padding,bias=args.bias)) #takes concatinated inp
            self.BN_dec2.append(BF_batchNorm(args.num_kernels ))
        # output: 40x40x64

        # input: 40x40x64    
        self.conv_dec1 = nn.ModuleList([])
        self.BN_dec1 = nn.ModuleList([])   
        
        self.upconv1 = nn.ConvTranspose2d(args.num_kernels, args.num_kernels, kernel_size=2, stride=2,bias=args.bias) #upsamples output of last block of decoder
        for l in range(self.num_dec_layers-1):
            if l==0:
                self.conv_dec1.append(nn.Conv2d(args.num_kernels*2, args.num_kernels, kernel_size=args.kernel_size, padding=args.padding,bias=args.bias))
            else: 
                self.conv_dec1.append(nn.Conv2d(args.num_kernels, args.num_kernels, kernel_size=args.kernel_size, padding=args.padding,bias=args.bias)) #takes concatinated inp
            self.BN_dec1.append(BF_batchNorm(args.num_kernels ))
        self.conv_dec1.append(nn.Conv2d(args.num_kernels, args.num_channels, kernel_size=args.kernel_size, padding=args.padding,bias=args.bias))
        # output: 80x80xC 
        
    def forward(self, x):
        relu = nn.ReLU(inplace=True)
        
        ########## Encoder ##########
        x = relu(self.e11(x)) #80x80
        x1_unpooled = relu(self.bn11(self.e12(x)))#80x80
        x = self.pool11(x1_unpooled) #40x40

        x = relu(self.bn21(self.e21(x),training_mode)) #40x40
        x2_unpooled = relu(self.bn22(self.e22(x))) #40x40
        x = self.pool21(x2_unpooled)#20x20
        
        ########## coarse level ##########
        for l in range(self.num_mid_layers):
            x = relu(self.BN_mid[l](self.conv_mid[l](x) ))#20x20
        
        ########## Decoder  ##########
        x = self.upconv2(x) #upsample output of mid laters to 40x40
        x = torch.cat([x, x2_unpooled], dim=1) #concat with output of decoder before downsampling    
        for l in range(self.num_dec_layers):
            x = relu(self.BN_dec2[l](self.conv_dec2[l](x) ))
            
        x = self.upconv1(x) #upsample to 80x80
        x = torch.cat([x, x1_unpooled], dim=1) #concat with output of decoder before downsampling    
        for l in range(self.num_dec_layers-1):
            x = relu(self.BN_dec1[l](self.conv_dec1[l](x) ))
        x = self.conv_dec1[-1](x)
        
        return x
    
 
    
    

# compute the RF in input layer 
# change convtranspose to upsample 

class UNet_copied(nn.Module): 
    '''
    UNet for segmentation from https://towardsdatascience.com/cook-your-first-u-net-in-pytorch-b3297a844cf3
    '''
    def __init__(self, args): 
        super(UNet_copied, self).__init__()
        
        
        # Encoder
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        
        # First block                
        # for input: 256x256xC       
        self.e11 = nn.Conv2d(args.num_channels, args.num_kernels , kernel_size=args.kernel_size, padding=args.padding, bias=False) 
        self.e12 = nn.Conv2d(args.num_kernels , args.num_kernels , kernel_size=args.kernel_size, padding=args.padding, bias=False) 
        self.bn12 = BF_batchNorm(args.num_kernels )        
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2) 
        # output: 128x128x64
        
        # Second block                
        # input: 128x128x64
        self.e21 = nn.Conv2d( args.num_kernels,  args.num_kernels*2, kernel_size=args.kernel_size, padding=args.padding, bias=False)
        self.bn21 = BF_batchNorm(args.num_kernels*2 )                
        self.e22 = nn.Conv2d( args.num_kernels*2,  args.num_kernels*2, kernel_size=args.kernel_size, padding=args.padding, bias=False)
        self.bn22 = BF_batchNorm(args.num_kernels*2 )                        
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2) 
        # output: 64x64x128
        
        # Third block        
        # input: 64x64x128     
        self.e31 = nn.Conv2d( args.num_kernels*2,  args.num_kernels*4, kernel_size=args.kernel_size, padding=args.padding, bias=False)
        self.bn31 = BF_batchNorm(args.num_kernels*4 )                                
        self.e32 = nn.Conv2d( args.num_kernels*4,  args.num_kernels*4, kernel_size=args.kernel_size, padding=args.padding, bias=False)
        self.bn32 = BF_batchNorm(args.num_kernels*4 )                                        
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2) # output: 68x68x256
        # output: 32x32x256
        
        # fourth block        
        # input: 32x32x256     
        self.e41 = nn.Conv2d( args.num_kernels*4,  args.num_kernels*8, kernel_size=args.kernel_size, padding=args.padding, bias=False)
        self.bn41 = BF_batchNorm(args.num_kernels*8 )                                        
        self.e42 = nn.Conv2d( args.num_kernels*8,  args.num_kernels*8, kernel_size=args.kernel_size, padding=args.padding, bias=False)
        self.bn42 = BF_batchNorm(args.num_kernels*8 )                                                
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2) 
        # output: 16x16x512

        ##### Mid layer         
        # input: 16x16x512
        self.e51 = nn.Conv2d( args.num_kernels*8,  args.num_kernels*16, kernel_size=args.kernel_size, padding=args.padding,bias=False)
        self.bn51 = BF_batchNorm(args.num_kernels*16 )                                                
        self.e52 = nn.Conv2d( args.num_kernels*16,  args.num_kernels*16, kernel_size=args.kernel_size, padding=args.padding,bias=False)
        self.bn52 = BF_batchNorm(args.num_kernels*16 )                                                

        
          
        ##### Decoder  
        # input: 16x16x1024                   
        self.upconv1 = nn.ConvTranspose2d(args.num_kernels*16, args.num_kernels*8, kernel_size=2, stride=2,bias=False)
        self.d11 = nn.Conv2d(args.num_kernels*16, args.num_kernels*8, kernel_size=args.kernel_size, padding=args.padding,bias=False)
        self.bn11_inv = BF_batchNorm(args.num_kernels*8 )                                                        
        self.d12 = nn.Conv2d(args.num_kernels*8, args.num_kernels*8, kernel_size=args.kernel_size, padding=args.padding,bias=False)
        self.bn12_inv = BF_batchNorm(args.num_kernels*8 )                                                                
        # output: 32x32x512
        
        # input: 32x32x512        
        self.upconv2 = nn.ConvTranspose2d(args.num_kernels*8, args.num_kernels*4, kernel_size=2, stride=2,bias=False)
        self.d21 = nn.Conv2d(args.num_kernels*8, args.num_kernels*4, kernel_size=args.kernel_size, padding=args.padding,bias=False)
        self.bn21_inv = BF_batchNorm(args.num_kernels*4 )                                                                
        self.d22 = nn.Conv2d(args.num_kernels*4, args.num_kernels*4, kernel_size=args.kernel_size, padding=args.padding,bias=False)
        self.bn22_inv = BF_batchNorm(args.num_kernels*4 )                                                                        
        # output: 64x64x256   
        
        # input: 64x64x256             
        self.upconv3 = nn.ConvTranspose2d(args.num_kernels*4, args.num_kernels*2, kernel_size=2, stride=2,bias=False)
        self.d31 = nn.Conv2d(args.num_kernels*4, args.num_kernels*2, kernel_size=args.kernel_size, padding=args.padding,bias=False)
        self.bn31_inv = BF_batchNorm(args.num_kernels*2 )                                                                        
        self.d32 = nn.Conv2d(args.num_kernels*2, args.num_kernels*2, kernel_size=args.kernel_size, padding=args.padding,bias=False)
        self.bn32_inv = BF_batchNorm(args.num_kernels*2 )                                                                                
        # output: 128x128x128     
        
        # input: 128x128x128       
        self.upconv4 = nn.ConvTranspose2d(args.num_kernels*2, args.num_kernels, kernel_size=2, stride=2,bias=False)
        self.d41 = nn.Conv2d(args.num_kernels*2, args.num_kernels, kernel_size=args.kernel_size, padding=args.padding,bias=False)
        self.bn41_inv = BF_batchNorm(args.num_kernels*1 )                                                                                
        self.d42 = nn.Conv2d(args.num_kernels, args.num_channels, kernel_size=args.kernel_size, padding=args.padding,bias=False)
        # output: 256x256xC   
    
        
    def forward(self, x):
        relu = nn.ReLU(inplace=True)
        # Encoder
        xe11 = relu(self.e11(x))
        xe12 = relu(self.bn12(self.e12(xe11)))
        xp1 = self.pool1(xe12)

        xe21 = relu(self.bn21(self.e21(xp1)))
        xe22 = relu(self.bn22(self.e22(xe21)))
        xp2 = self.pool2(xe22)

        xe31 = relu(self.bn31(self.e31(xp2)))
        xe32 = relu(self.bn32(self.e32(xe31)))
        xp3 = self.pool3(xe32)

        xe41 = relu(self.bn41(self.e41(xp3)))
        xe42 = relu(self.bn42(self.e42(xe41)))
        xp4 = self.pool4(xe42)
        
        # mid-layers
        xe51 = relu(self.bn51(self.e51(xp4)))
        xe52 = relu(self.bn52(self.e52(xe51)))
        
        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = relu(self.bn11_inv(self.d11(xu11)))
        xd12 = relu(self.bn12_inv(self.d12(xd11)))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = relu(self.bn21_inv(self.d21(xu22)))
        xd22 = relu(self.bn22_inv(self.d22(xd21)))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = relu(self.bn31_inv(self.d31(xu33)))
        xd32 = relu(self.bn32_inv(self.d32(xd31)))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = relu(self.bn41_inv(self.d41(xu44)))
        xd42 = self.d42(xd41)

        return xd42

    
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



class BF_CNN_RF_activations(nn.Module):

    def __init__(self, args):
        super(BF_CNN_RF_activations, self).__init__()
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
        responses = []
        relu = nn.ReLU(inplace=True)
        x = self.conv_layers[0](x) #first layer linear
        responses.append(x)
        for l in range(1,self.num_layers-1):
            x = self.conv_layers[l](x)
            x = self.BN_layers[l-1](x)
            x = relu(x)
            responses.append(x)

        x = self.conv_layers[-1](x)
        responses.append(x)
        return x, responses



class BF_CNN_RF_mid_layer_out(nn.Module):

    def __init__(self, args):
        super(BF_CNN_RF_mid_layer_out, self).__init__()
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



    def forward(self, x, layer_num, channel_num):
        # layer_num starts from zero 
        relu = nn.ReLU(inplace=True)
        if layer_num == 0:
            x = self.conv_layers[0](x) #first layer linear
            return x[:, channel_num:channel_num+1 ]


        elif layer_num > 0 and layer_num < self.num_layers - 1:
            x = self.conv_layers[0](x) #first layer linear
            for l in range(1,layer_num):
                x = self.conv_layers[l](x)
                x = self.BN_layers[l-1](x)
                x = relu(x)
            return x[:, channel_num:channel_num+1 ]
        
        elif layer_num == self.num_layers - 1:
            x = self.conv_layers[0](x) #first layer linear
            for l in range(1,sel.num_layers - 1):
                x = self.conv_layers[l](x)
                x = self.BN_layers[l-1](x)
                x = relu(x)

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

    

class BF_CNN_activations(nn.Module):

    def __init__(self, args): 
        super(BF_CNN_activations, self).__init__()


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
        responses = []
        relu = nn.ReLU(inplace=True)
        x = self.conv_layers[0](x) #first layer linear
        
        if self.first_layer_linear is False: 
            x = relu(x)  
            
        responses.append(x)
            
        for l in range(1,self.num_layers-1):
            x = self.conv_layers[l](x)
            x = self.BN_layers[l-1](x)
            x = relu(x)
            responses.append(x)

        x = self.conv_layers[-1](x)
        responses.append(x)
        return x, responses
    

class BF_CNN_blurred_log_prob(nn.Module):

    def __init__(self, args):
        super(BF_CNN_blurred_log_prob, self).__init__()


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

        return x.mean(dim = (2,3), keepdim = True)



class two_layer_denoiser(nn.Module):

    def __init__(self, args):
        super(two_layer_denoiser, self).__init__()

        self.layer0 = nn.Conv2d(in_channels=args.num_channels,out_channels=args.num_kernels, kernel_size=args.kernel_size,padding=args.padding ,bias=args.bias)
        self.layer1 = nn.Conv2d(in_channels=args.num_kernels,out_channels=args.num_channels, kernel_size=args.kernel_size,padding=args.padding ,bias=False)

    def forward(self, x):
        relu = nn.ReLU(inplace=True)

        x = self.layer0(x)
        x = relu(x)
        x = self.layer1(x)

        return x






class linear_BF_CNN(nn.Module):

    def __init__(self, args):
        super(linear_BF_CNN, self).__init__()


        self.num_layers = args.num_layers

        self.conv_layers = nn.ModuleList([])
        self.BN_layers = nn.ModuleList([])


        self.conv_layers.append(nn.Conv2d(args.num_channels,args.num_kernels, args.kernel_size, padding=args.padding , bias=False))

        for l in range(1,self.num_layers-1):
            self.conv_layers.append(nn.Conv2d(args.num_kernels ,args.num_kernels, args.kernel_size, padding=args.padding , bias=False))
            self.BN_layers.append(BF_batchNorm(args.num_kernels ))

        self.conv_layers.append(nn.Conv2d(args.num_kernels,args.num_channels, args.kernel_size, padding=args.padding , bias=False))



    def forward(self, x):

        x = self.conv_layers[0](x) 

        for l in range(1,self.num_layers-1):
            x = self.conv_layers[l](x)
            x = self.BN_layers[l-1](x)
        x = self.conv_layers[-1](x)

        return x

class one_ReLU_BF_CNN(nn.Module):

    def __init__(self, args):
        super(one_ReLU_BF_CNN, self).__init__()


        self.num_layers = args.num_layers

        self.conv_layers = nn.ModuleList([])
        self.BN_layers = nn.ModuleList([])


        self.conv_layers.append(nn.Conv2d(args.num_channels,args.num_kernels, args.kernel_size, padding=args.padding , bias=False))

        for l in range(1,self.num_layers-1):
            self.conv_layers.append(nn.Conv2d(args.num_kernels ,args.num_kernels, args.kernel_size, padding=args.padding , bias=False))
            self.BN_layers.append(BF_batchNorm(args.num_kernels ))


        self.conv_layers.append(nn.Conv2d(args.num_kernels,args.num_channels, args.kernel_size, padding=args.padding , bias=False))



    def forward(self, x):
        relu = nn.ReLU(inplace=True)

        x = self.conv_layers[0](x)

        for l in range(1,self.num_layers-1):
            x = self.conv_layers[l](x)
            x = self.BN_layers[l-1](x)

        x = relu(x) #non-linearity applied only before the last layer 
            
        x = self.conv_layers[-1](x)

        return x





class sd_estimator(nn.Module):

    def __init__(self, args):
        super(sd_estimator, self).__init__()
        self.conv_layers = nn.ModuleList([])
        self.num_layers = args.num_layers

        self.conv_layers.append(nn.Conv2d(args.num_channels,args.num_kernels, args.kernel_size, padding=args.padding , bias=False))

        for l in range(1,self.num_layers-1):
            self.conv_layers.append(nn.Conv2d(args.num_kernels ,args.num_kernels, args.kernel_size, padding=args.padding, bias=False))

        self.conv_layers.append(nn.Conv2d(args.num_kernels,args.num_channels, args.kernel_size, padding=args.padding , bias=False))

    def forward(self, x):
        relu = nn.ReLU(inplace=True)
        x = self.conv_layers[0](x) #first layer linear

        for l in range(1,self.num_layers-1):
            x = self.conv_layers[l](x)
            x = relu(x)


        x = self.conv_layers[-1](x)

        return x.mean(dim = (2,3), keepdim = True)


