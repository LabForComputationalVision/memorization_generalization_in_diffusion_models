import numpy as np
import torch
import torch.nn as nn
import torch.fft


###################################### Inverse problems Tasks ##################################
#############################################################################################
class synthesis:
    def __init__(self):
        super(synthesis, self).__init__()

    def M_T(self, x):
        return torch.zeros_like(x)

    def M(self, x):
        return torch.zeros_like(x)

class inpainting:
    '''
    makes a blanked area in the center
    @x_size : image size, tuple of (n_ch, im_d1,im_d2)
    @x0,y0: center of the blanked area
    @w: width of the blanked area
    @h: height of the blanked area
    '''
    def __init__(self, x_size,x0,y0,h, w):
        super(inpainting, self).__init__()

        n_ch , im_d1, im_d2 = x_size
        self.mask = torch.ones(x_size)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
        c1, c2 = int(x0), int(y0)
        h , w= int(h/2), int(w/2)
        self.mask[0:n_ch, c1-h : c1+h , c2-w:c2+w] = 0

    def M_T(self, x):
        return x*self.mask

    def M(self, x):
        return x*self.mask


class rand_pixels:
    '''
    @x_size : tuple of (n_ch, im_d1,im_d2)
    @p: fraction of dimensions kept in (0,1)
    '''
    def __init__(self, x_size, p):
        super(rand_pixels, self).__init__()

        self.mask = np.zeros(x_size).flatten()
        self.mask[0:int(p*np.prod(x_size))] = 1
        self.mask = torch.tensor(np.random.choice(self.mask, size = x_size , replace = False).astype('float32').reshape(x_size))
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def M_T(self, x):
        return x*self.mask

    def M(self, x):
        return x*self.mask


class super_resolution:
    '''
    block averaging for super resolution.
    creates a low rank matrix (thin and tall) for down sampling
    @s: downsampling factor, int
    @x_size: tuple of three int  (n_ch, im_d1, im_d2)
    '''

    def __init__(self, x_size, s):
        super(super_resolution, self).__init__()

#         if x_size[1]%2 !=0 or x_size[2]%2 != 0 :
#             raise Exception("image dimensions need to be even")

        self.down_sampling_kernel = torch.ones(x_size[0],1,s,s)
        self.down_sampling_kernel = self.down_sampling_kernel/np.linalg.norm(self.down_sampling_kernel[0,0])
        if torch.cuda.is_available():
            self.down_sampling_kernel = self.down_sampling_kernel.cuda()
        self.x_size = x_size
        self.s = s

    def M_T(self, x):
        down_im = torch.nn.functional.conv2d(x.unsqueeze(0), self.down_sampling_kernel, stride= self.s, groups = self.x_size[0])
        return down_im[0]

    def M(self, x):
        rec_im = torch.nn.functional.conv_transpose2d(x.unsqueeze(0), self.down_sampling_kernel, stride= self.s, groups = self.x_size[0])

        return rec_im[0]



class random_basis:
    '''
    @x_size : tuple of (im_d1,im_d2)
    @p: fraction of dimensions kept in (0,1)
    '''
    def __init__(self, x_size, p):
        super(random_basis, self).__init__()
        n_ch , im_d1, im_d2 = x_size
        self.x_size = x_size
        self.U, _ = torch.qr(torch.randn(int(np.prod(x_size)),int(np.prod(x_size)*p) ))
        if torch.cuda.is_available():
            self.U = self.U.cuda()

    def M_T(self, x):
        # gets 2d or 3d image and returns flatten partial measurement(1d)
        return torch.matmul(self.U.T,x.flatten())

    def M(self, x):
        # gets flatten partial measurement (1d), and returns 2d or 3d reconstruction
        return torch.matmul(self.U,x).reshape(self.x_size[0], self.x_size[1], self.x_size[2])


#### important: when using fftn from torch the reconstruction is more lossy than when fft2 from numpy
#### the difference between reconstruction and clean image in pytorch is of order of e-8, but in numpy is e-16

class spectral_super_resolution:
    '''
    creates a mask for dropping high frequency coefficients
    @im_d: dimension of the input image is (im_d, im_d)
    @p: portion of coefficients to keep
    '''
    def __init__(self, x_size, p):
        super(spectral_super_resolution, self).__init__()

        self.x_size = x_size

        xf = int(round(x_size[1]*np.sqrt(p) )/2)
        yf = int(round(x_size[1]*x_size[2]*p/xf )/4)

        mask = torch.ones((x_size[1],x_size[2]))

        mask[xf:x_size[1]-xf,:]=0
        mask[:, yf:x_size[2]-yf]=0
        self.mask = mask
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def M_T(self, x):
        # returns fft of each of the three color channels independently
        return self.mask*torch.fft.fftn(x, norm= 'ortho', dim = (1,2),s = (self.x_size[1], self.x_size[2]) )

    def M(self, x):
        return torch.real(torch.fft.ifftn(x, norm= 'ortho',  dim = (1,2), s = (self.x_size[1], self.x_size[2]) ))





class retinal_cones:

    def __init__(self, x_size, mask):
        super(retinal_cones, self).__init__()
        if x_size[0] != 1:
            raise TypeError('Only works on grayscale images')
        if x_size[1] != mask.shape[0] or x_size[2] != mask.shape[1]:
            raise TypeError('input image size does not match mask size')

        self.mask = torch.tensor( mask.astype('float32') )
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def M_T(self, x):
        return x*self.mask

    def M(self, x):
        return x*self.mask






