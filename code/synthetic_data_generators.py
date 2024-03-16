import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import torch.nn as nn
import torch
import os
from PIL import Image
import gzip
from matplotlib.patches import Circle, Polygon
import random
from wavelets import OneLevelWaveletTransform
from dataloader_func import rescale_image_range
 

######################################################################################
################################ data generators  ################################
######################################################################################
def make_gaussian_images(args, set_size,noise_level): 
    if type(noise_level) == list:
        std = torch.randint(low = noise_level[0], high=noise_level[1] ,
                            size = (set_size,1,1,1)  )/255
    else:
        std = torch.ones(set_size,1,1,1) * noise_level/255
     
    train_set = torch.randn(set_size,1, args.patch_size[0], args.patch_size[1]) * std
    test_set = torch.randn(set_size,1, args.patch_size[0], args.patch_size[1]) * std
    print('train: ', train_set.size(), 'test: ', test_set.size() )
    return train_set, test_set

def make_pink_noise_dataset(data_shape):
    N, C, W, H = data_shape
    x = torch.fft.fftfreq(W)
    y = torch.fft.fftfreq(H)
    X, Y  = torch.meshgrid(x, y)
    multiplier = 1/(X**2 + Y**2)
    multiplier[0,0] = W*H #replace inf with largest value
    multiplier = np.sqrt(multiplier/(W*H)) 
    
    white_noise = torch.randn(N,C,W,H)*.8 +.5 #change mean and variance to make data range to 0 to 1
    pink_noise = torch.real(torch.fft.ifft2(multiplier*torch.fft.fft2(white_noise))) 
    return pink_noise


def generate_triangles(N):
    images = []
    '''N:number of generated samples'''
    for j in range(N):
        c = np.random.rand(1)[0]
        background_color = (c,c,c)
        fig, ax = plt.subplots(figsize = (4,4), dpi =13,facecolor=background_color)
        # ax.set_xlim(0,43)
        # ax.set_ylim(0,43)
        
        d = np.random.rand(1)[0]
        face_color =(d,d,d)
        polygon = Polygon((np.random.rand(3,2)  ), closed=True, color= face_color )

        ax.add_patch(polygon)
        ax.axis('off')
        plt.savefig('myfig.png', bbox_inches='tight' )
        plt.close()


        im = torch.tensor(plt.imread('myfig.png').mean(axis=2)).unsqueeze(0)
        images.append(im)
    return torch.stack(images)

def generate_circles(N):
    '''N:number of generated samples'''
    
    images = []
    
    for j in range(N):
        c = np.random.rand(1)[0]
        background_color = (c,c,c)
        fig, ax = plt.subplots(figsize = (4,4), dpi =25,facecolor=background_color)
        ax.set_xlim(0,80)
        ax.set_ylim(0,80)
        # fig, ax = plt.subplots(figsize = (4,4), dpi =13,facecolor=background_color)
        # ax.set_xlim(0,43)
        # ax.set_ylim(0,43)
        
        d = np.random.rand(1)[0]
        face_color =(d,d,d)
        center = np.random.randint( low = 10, high = 70, size = (2,) )
        max_r = min(min(center[0], 80-center[0]), min(center[1], 80-center[1]))
        # center = np.random.randint( low = 5, high = 37, size = (2,) )
        # max_r = min(min(center[0], 41-center[0]), min(center[1], 41-center[1]))        
        r = np.random.randint(low = 5, high = max_r)
        disc = Circle(center, radius=r , color= face_color )

        ax.add_patch(disc)
        ax.axis('off')
        plt.savefig('myfig.png', bbox_inches='tight' )
        plt.close()


        im = torch.tensor(plt.imread('myfig.png').mean(axis=2)).unsqueeze(0)
        images.append(im)
    return torch.stack(images)



def generate_polygon_edges(K):
    '''
    K: number of edges (even int)
    '''
    if K >3 and K%2 != 0 : 
        raise ValueError ('K must be even interger number ')
    # Generate two lists, X and Y, of N random integers between 0 and C. Make sure there are no duplicates.
    X = torch.FloatTensor(random.sample(range(40), K)) # random ints without repeat 
    Y = torch.FloatTensor(random.sample(range(40), K)) # random ints without repeat 

    # Sort X and Y and store their maximum and minimum elements.
    X = X.sort()[0]
    X_max , X_min = X[-1::], X[0:1]
    X = X[1:-1]

    Y = Y.sort()[0]
    Y_max , Y_min = Y[-1::], Y[0:1]
    Y = Y[1:-1]

    # Randomly divide the other (not max or min) elements into two groups: X1 and X2, and Y1 and Y2.
    X = X[torch.randperm(X.shape[0])]
    X1 = X[0:int((10 - 2)/2)]
    X2 = X[int((10 - 2)/2)::]

    Y = Y[torch.randperm(Y.shape[0])]
    Y1 = Y[0:int((10 - 2)/2)]
    Y2 = Y[int((10 - 2)/2)::]
    # Re-insert the minimum and maximum elements at the start and end of these lists (minX at the start of X1 and X2, maxX at the end, etc.).

    X1 = torch.concatenate((X_min, X1, X_max))
    X2 = torch.concatenate((X_min, X2, X_max))
    Y1 = torch.concatenate((Y_min, Y1, Y_max))
    Y2 = torch.concatenate((Y_min, Y2, Y_max))


    # Find the consecutive differences (X1[i + 1] - X1[i]), reversing the order for the second group (X2[i] - X2[i + 1]). Store these in lists XVec and YVec.

    XVec = torch.hstack([X1[1::] - X1[0:-1], X2[0:-1] - X2[1::]])
    YVec = torch.hstack([Y1[1::] - Y1[0:-1],  Y2[0:-1] - Y2[1::]])


    # Randomize (shuffle) YVec and treat each pair XVec[i] and YVec[i] as a 2D vector.
    YVec = YVec[torch.randperm(YVec.shape[0])]
    vectors = torch.stack([XVec,YVec]).T

    # Sort these vectors by angle and then lay them end-to-end to form a polygon.

    thetas = torch.arctan(YVec/XVec)*180/torch.pi
    thetas = torch.where( XVec < 0, thetas+180, thetas )  
    thetas = torch.where( (XVec > 0) & (YVec <= 0) , thetas+360, thetas ) 
    vectors = vectors[thetas.sort()[1]]

    return vectors


def draw_polygon(vectors, my_dpi=12):
    c = np.random.rand(1)[0]
    background_color = (c,c,c)
    new_vec = torch.zeros_like(vectors)
    plt.figure(figsize=(5, 5),dpi=my_dpi, facecolor= background_color , edgecolor=background_color  )
    d = np.random.rand(1)[0]
    face_color =(d,d,d)
    # Move the polygon to the original min and max coordinates.
    x0 = 0
    y0 = 0
    x1 = vectors[0][0]
    y1 = vectors[0][1]
    plt.plot([x0,x1],[y0,y1],color=face_color)
    plt.axis('off')

    new_vec[0,0] = x1
    new_vec[0,1] = y1

    for i in range(1, vectors.shape[0]):
        x0 = x1
        y0 = y1
        x1 = vectors[i][0]+x0
        y1 = vectors[i][1]+y0
        plt.plot([x0,x1],[y0,y1], color=face_color)
        new_vec[i,0] = x1
        new_vec[i,1] = y1
    return new_vec

def make_fft_filter_1d(N):
    filt = abs(torch.fft.fftshift(torch.fft.fftfreq(N)))
    filt = 1/filt/N
    i = torch.where(filt == torch.inf)[0]
    filt[i.item()] = filt[i.item()+1]

    return filt



def make_fft_filter_2d(N):
    filt = torch.fft.fftshift(torch.fft.fftfreq(N))
    X, Y = torch.meshgrid(filt, filt)
    filt2 = (X**2 + Y**2).sqrt()
    filt2 = 1/filt2/N
    i, j = torch.where(filt2 == torch.inf)
    filt2[i.item(), j.item()] = filt2[i.item()+1, j.item()]
    return filt2


def make_fft_filter_2d_seprable(N):  
    filt = torch.fft.fftshift(torch.fft.fftfreq(N))
    i = torch.where(filt == 0)[0]
    filt[i.item()] = filt[i.item()+1]
    filt2 = abs(torch.matmul(filt.reshape(-1,1), filt.reshape(1,-1))).sqrt()
    filt2 = 1/filt2/N
    return filt2 

def make_C_alpha_contour(alpha, filt, N = 43): 
    ders = torch.rand(size = (N,), dtype=torch.float32)*2-1
    integrated = torch.fft.ifft(torch.fft.ifftshift( ( torch.fft.fftshift(torch.fft.fft(ders)) *( filt** (alpha))) )).real
    return integrated


def make_C_beta_background(beta, filt2, N = 43): 
    ders2 = torch.rand(size = (N,N), dtype=torch.float32)*2-1
    integrated =torch.fft.ifft2(torch.fft.ifftshift( ( torch.fft.fftshift(torch.fft.fft2(ders2)) * (filt2**(beta))) )).real 
    return integrated
 
    
# def make_C_alpha_images(alpha, beta, separable=False, N=43,num_samples=1, constant_background=False, factor=1 , antialiasing=0, wavelet="db2", mode="reflect"):
#     '''
#     N: image size. 
#     K: number of images
#     for vertical blurring: (factor, 1)
#     '''
#     all_im = []


#     if antialiasing > 0: 
#         N = N * (2**antialiasing) - 2 ** antialiasing
    

#     # filter for 2d
#     if separable:
#         filt2 = make_fft_filter_2d_seprable(N)
#     else: 
#         filt2 = make_fft_filter_2d(N)
        
#     #filter for 1d 
#     filt = make_fft_filter_1d(N* factor)
    
#     #filter for averaging the mask 
#     ave_filt = nn.Conv2d(1,1,factor, stride = factor, padding = 0
#                          , padding_mode='reflect', bias = None)
#     ave_filt.weight = torch.nn.Parameter(torch.ones((1,1,factor,factor))/(factor**2))
    
#     ### genearte images 
#     while len(all_im)<num_samples:
#         ## make the backgrounds
#         if constant_background: 
#             background1 = np.random.rand(1)[0] * torch.ones(size = (N,N))
#             background2 = np.random.rand(1)[0] * torch.ones(size = (N,N))
#         else: 
#             background1 = make_C_beta_background(beta,filt2, N)
#             background2 = make_C_beta_background(beta,filt2, N)        

#             ## rescale background to create an intensity difference between the two parts 
#             # thresh = torch.rand(1).item() # change to this to enfornce a larger gap between the mean of the two backgrounds 
#             if torch.randint(low=0, high=2, size=(1,)).item() == 0:
#                 background1 = rescale_image_range(background1,max_I=1, min_I=torch.rand(1).item())
#                 background2 = rescale_image_range(background2,max_I=torch.rand(1).item(), min_I=0)
#             else:
#                 background2 = rescale_image_range(background2,max_I=1, min_I=torch.rand(1).item())
#                 background1 = rescale_image_range(background1,max_I=torch.rand(1).item(), min_I=0)
                
#         ## make the contours
#         contour = make_C_alpha_contour(alpha ,filt, N*factor )
#         contour = torch.round(contour*N*factor).type(torch.int) + int((N*factor)/2)      
        
#         #### mask and replace   
#         mask = torch.ones((N*factor , N*factor))
#         for i in range(N*factor):
#             mask[0:contour[i],i]=0
            
#         mask_down = ave_filt(mask.unsqueeze(0).unsqueeze(0)).squeeze().detach()
#         im = background1 * mask_down + background2 * (1-mask_down) 

            
#         all_im.append(im)

#     all_im = torch.stack(all_im).unsqueeze(1) 
#     all_im = downsample(all_im.detach(), num_times=antialiasing, wavelet=wavelet, mode=mode)# (N, 1, H/2^j, W/2^j)
#     if antialiasing >0: 
#         all_im = rescale_image_range(all_im, max_I=1, min_I=0)
#     return all_im



def make_C_alpha_images(alpha, beta, separable=False, im_size=43,num_samples=1, constant_background=False, factor=(1,1) , antialiasing=0, wavelet="db2", mode="reflect"):
    '''
    im_size: image size. 
    num_samples: number of images
    for vertical blurring: (factor, 1)
    '''
    all_im = []


    if antialiasing > 0: 
        im_size = im_size * (2**antialiasing) - 2 ** antialiasing

    filt = make_fft_filter_1d(im_size)
    
    if separable:
        filt2 = make_fft_filter_2d_seprable(im_size)
    else: 
        filt2 = make_fft_filter_2d(im_size)

    ave_filt = nn.Conv2d(1,1,(factor[0],factor[1]), stride = (1,1), padding = (int(factor[0]/2),int(factor[1]/2)) , padding_mode='reflect', bias = None)
    ave_filt.weight = torch.nn.Parameter(torch.ones((1,1,factor[0],factor[1]))/(factor[0]* factor[1]))
    while len(all_im)<num_samples:
        contour = make_C_alpha_contour(alpha ,filt, im_size )
        contour = torch.round(contour*im_size).type(torch.int) + int((im_size)/2)    
        
        if constant_background: 
            background1 = np.random.rand(1)[0] * torch.ones(size = (im_size,im_size))
            background2 = np.random.rand(1)[0] * torch.ones(size = (im_size,im_size))

        else: 
            background1 = make_C_beta_background(beta,filt2, im_size)
            background2 = make_C_beta_background(beta,filt2, im_size)        

            ## rescale background to create an intensity difference between the two parts 
            # thresh = torch.rand(1).item() # change to this to enfornce a larger gap between the mean of the two backgrounds 
            if torch.randint(low=0, high=2, size=(1,)).item() == 0:
                background1 = rescale_image_range(background1,max_I=1, min_I=torch.rand(1).item())
                background2 = rescale_image_range(background2,max_I=torch.rand(1).item(), min_I=0)
            else:
                background2 = rescale_image_range(background2,max_I=1, min_I=torch.rand(1).item())
                background1 = rescale_image_range(background1,max_I=torch.rand(1).item(), min_I=0)
        #### mask and replace         
        mask = torch.ones((im_size , im_size))
        for i in range(im_size):
            mask[0:contour[i],i]=0
        if factor[0] >1 or factor[1]>1:
            mask_down = ave_filt(mask.unsqueeze(0).unsqueeze(0)).squeeze().detach()
            if factor[0]%2==0: 
                mask_down = mask_down[0:-1, :]
            if factor[1]%2==0: 
                mask_down = mask_down[:,0:-1]
                
            im = background1 * mask_down + background2 * (1-mask_down) 
        else: 
            im = background1 * mask + background2 * (1-mask) 
            
        all_im.append(im)

    all_im = torch.stack(all_im).unsqueeze(1) 
    all_im = downsample(all_im.detach(), num_times=antialiasing, wavelet=wavelet, mode=mode)# (N, 1, H/2^j, W/2^j)
    if antialiasing >0: 
        all_im = rescale_image_range(all_im, max_I=1, min_I=0)
    return all_im 



def downsample(x, num_times, wavelet="db2", mode="periodization"):
    """ Downsample an (*, H, W) image `num_times` times using the given wavelet filter. """
    transform = OneLevelWaveletTransform(wavelet=wavelet, mode=mode)

    for _ in range(num_times):
        x = transform.decompose(x)[..., 0, :, :]  # (*, H/2^j, W/2^j)

    return x

def circle_dataset(patch_size, num_patches, translate=False, scale=False, foreground=False, background=False,
                   wrap=False, continuous=True, antialiasing=0, wavelet="db2", mode="reflect"):
    """ Returns (N, 1, H, W) images of circles.
    :param patch_size: (H, W) tuple
    :param num_patches: number of samples N
    :param translate: whether to translate the circles
    :param scale: whether to scale the circles
    :param foreground: whether to randomize the foreground color
    :param background: whether to randomize the background color
    :param wrap: whether to generate wrapping circles at the boundaries
    :param continuous: whether consider continuous positions and sizes as opposed to integers
    :param antialiasing: generate larger images and downsample them this number of times
    :param wavelet: wavelet used for downsampling
    :param mode: padding mode for downsampling
    """
    assert patch_size[0] == patch_size[1]  # XXX too lazy to implement rectangular images
    size = patch_size[0] * 2 ** antialiasing  # Generate large images by the aliasing factor

    # Use numpy for randint because pytorch doesn't support low and high arguments as tensors
    rand_np = np.random.uniform if continuous else np.random.randint
    rand = lambda *args, **kwargs: torch.from_numpy(rand_np(*args, **kwargs))

    min_r = 3
    radius = rand(size=(num_patches,), low=min_r, high=(size - min_r) // 2) if scale \
        else torch.full(size=(num_patches,), fill_value=size // 4)  # (N,)

    r = 0 if wrap else radius[:, None]
    center = rand(size=(num_patches, 2), low=r, high=size - r) if translate \
        else torch.full(size=(num_patches, 2), fill_value=size // 2, dtype=torch.int32)  # (N, 2)

    foreground = torch.rand((num_patches,)) if foreground else torch.ones((num_patches,))  # (N,)
    background = torch.rand((num_patches,)) if background else torch.zeros((num_patches,))  # (N,)

    y, x = torch.meshgrid(*(torch.arange(s) for s in (size, size)), indexing="ij")  # (H, W) both
    u = torch.stack((x, y), dim=-1)  # (H, W, 2)

    # Compute modulo in [-(size // 2), (size - 1) // 2]
    diff_mod = lambda diff: (diff + size / 2) % size - size / 2
    squared_dist = torch.sum(diff_mod(u[None] - center[:, None, None]) ** 2, dim=-1)  # (N, H, W)
    circle = squared_dist <= radius[:, None, None].float() ** 2  # (N, H, W)

    data = background[:, None, None] + (foreground - background)[:, None, None] * circle.float()  # (N, H, W)
    data = data[:, None]  # (N, 1, H, W)
    data = downsample(data, num_times=antialiasing, wavelet=wavelet, mode=mode)  # (N, 1, H/2^j, W/2^j)

    return data


def rotation_matrix(angle):
    """ Construct rotation matrices from a given angle. (*,) to (*, 2, 2). """
    return torch.stack((
        torch.stack((torch.cos(angle), -torch.sin(angle)), dim=-1),  # (*, 2)
        torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1),  # (*, 2)
    ), dim=-2)  # (*, 2, 2)


def gaussian_spectrum(patch_size, c=0.05, alpha=2):
    """ Returns a (*L) real covariance spectrum for a stationary circular Gaussian process of spectrum 1/(c + ||omega||^alpha).
    The spectrum is normalized so that pixel marginals have unit variance.
    :param patch_size: desired spatial shape (can be a 1- or 2-tuple)
    :param c: constant in the spectrum definition
    :param alpha: exponent in the spectrum definition
    """
    device = torch.device("cpu")
    omega = torch.stack(torch.meshgrid(*(
        N * torch.fft.fftfreq(N, device=device) for N in patch_size
    ), indexing="ij"), dim=-1)  # (H, W, d)
    omega_norm = torch.sqrt(torch.sum(omega ** 2, dim=-1))  # (H, W)
    spectrum = 1 / (c + omega_norm ** alpha)
    spectrum /= spectrum.mean()
    return spectrum


def generate_gaussian(patch_size, num_patches, c=0.05, alpha=2):
    """ Returns a (N, 1, *L) set of samples from a stationary circular Gaussian distribution with a power-law spectrum.
    :param patch_size: desired spatial shape (can be a 1- or 2-tuple for 1d or 2d process)
    :param num_patches: number N of samples to generate
    :param c: constant in the spectrum definition
    :param alpha: exponent in the spectrum definition
    """
    device = torch.device("cpu")

    # Compute normalized spectrum.
    spectrum = gaussian_spectrum(patch_size, c, alpha)

    # Sample from a stationary Gaussian in Fourier domain.
    shape = (num_patches, 1) + patch_size + (2,)
    noise_fft = torch.view_as_complex(torch.randn(shape, device=device))  # (*, 1, H, W) complex
    # E[|noise_fft|²] = 2 for now, so we rescale to have the right spectrum.
    # Note: we keep this factor of sqrt(2) because we discard the imaginary part after the IFFT.
    noise_fft *= torch.sqrt(spectrum)
    # Go to space domain, discarding the imaginary part.
    noise = torch.real(torch.fft.ifftn(noise_fft, norm="ortho", dim=tuple(range(-len(patch_size), 0))))  # (*, 1, H, W) real

    return noise


def c_alpha_dataset(patch_size, num_patches, alpha, beta, degree=3, antialiasing=0, wavelet="db2",
                    mode="periodization"):
    patch_size = patch_size * 2 ** antialiasing

    # Use numpy for rand because pytorch doesn't support low and high arguments as tensors
    rand = lambda *args, **kwargs: torch.from_numpy(np.random.uniform(*args, **kwargs))

    # Contour generation: we generate 1d curves for each edge.

    # The contour has a 1d spectrum of |omega|^(-2alpha), i.e., a variance decay of k^(-2alpha).
    contours = generate_gaussian(patch_size=(patch_size,), num_patches=num_patches * degree,
                                 alpha=2 * alpha ).reshape((num_patches, degree, patch_size))  # (N, k, H)
    # We center and rescale them so that they are centered around a given value.
    contours -= contours.mean(dim=-1, keepdim=True)
    mean = rand(size=(num_patches, degree), low=0.25, high=0.5)  # (N, k)
    contours = mean[:, :, None] + contours / 3  # (N, k, H)

    # For each edge, generate a random orientation.
    angles = torch.linspace(0, 2 * np.pi, 3 * degree + 1)
    # edge i will have a random orientation in [angles[3*i+1], angles[3*i+2]]
    orientations = rand(size=(num_patches, degree), low=angles[None, 1::3], high=angles[None, 2::3])  # (N, k)
    rotations = rotation_matrix(orientations)  # (N, k, 2, 2)

    # Compute the intersection of the half-spaces defined by the edges.

    # Compute the normal and tangent coordinates for the given orientation.
    u = torch.stack(torch.meshgrid(*(torch.linspace(-1, 1, s) for s in (patch_size, patch_size)),
                                   indexing="ij"), dim=-1)  # (H, W, 2)
    u_rot = torch.sum(rotations[:, :, None, None, :, :] * u[None, None, :, :, None, :], dim=-1)  # (N, k, H, W, 2)
    normal, tangent = u_rot.unbind(dim=-1)  # (N, k, H, W) both

    # Compute the half-space by indexing the contour along the tangent coordinate.
    index = (torch.arange(num_patches, dtype=torch.long)[:, None, None, None],
             torch.arange(degree, dtype=torch.long)[None, :, None, None],
             torch.round(patch_size * tangent / 2).type(torch.long) % patch_size)
    half_spaces = normal + contours[index] > 0  # (N, k, H, W)

    # Compute the intersection between all the half-spaces.
    intersections = torch.all(half_spaces, dim=1)  # (N, H, W)

    # Foreground and background generation.

    def random_mean_pair():
        # Return a random pair of numbers where one is in [0, 0.5] and the other in [0.5, 1.0]
        means = torch.rand((2, num_patches))
        offset = torch.zeros((2, num_patches))
        offset[torch.randint(size=(num_patches,), low=0, high=2), torch.arange(num_patches, dtype=torch.long)] = 1
        return (means + offset) / 2

    def gen_region(mean):
        # They have a 2d spectrum of |omega|^(-2beta), i.e., a variance decay of k^(-beta).
        regions = generate_gaussian(patch_size=(patch_size, patch_size), num_patches=num_patches,
                                    alpha=2 * beta )  # (N, 1, H, W), pixel has unit variance
        # We center and rescale them so that they are centered around a given value.
        regions -= regions.mean(dim=(-1, -2), keepdim=True)
        return mean[:, None, None, None] + regions / 6

    mean_foregrounds, mean_backgrounds = random_mean_pair()  # (N,) both
    foregrounds = gen_region(mean_foregrounds)  # (N, 1, H, W)
    backgrounds = gen_region(mean_backgrounds)  # (N, 1, H, W)

    data = backgrounds + (foregrounds - backgrounds) * intersections[:, None].float()  # (N, 1, H, W)
    data = downsample(data, num_times=antialiasing, wavelet=wavelet, mode=mode)  # (N, 1, H/2^j, W/2^j)
    return data


