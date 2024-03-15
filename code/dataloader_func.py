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

######################################################################################
################################ data loaders ################################
######################################################################################
def single_image_loader(data_set_dire_path, image_number):

    if 'mnist' in data_set_dire_path.split('/'):
        f = gzip.open(data_set_dire_path + '/t10k-images-idx3-ubyte.gz','r')
        f.read(16)
        buf = f.read(28 * 28 *10000)
        data = np.frombuffer(buf, dtype=np.uint8).astype(float)/255
        x = torch.tensor(data.reshape( 10000,28, 28).astype('float32'))[image_number:image_number+1]

    else:
        all_names = os.listdir(data_set_dire_path)
        file_name = all_names[image_number]
        im = plt.imread(data_set_dire_path + file_name)
        if len(im.shape) == 3:
            x = torch.tensor(im).permute(2,0,1)
        elif len(im.shape) == 2:
            x = torch.tensor(im.reshape(1, im.shape[0], im.shape[1]))

    return x


def load_mnist(data_dir_path): 
    f = gzip.open( data_dir_path + '/train-images-idx3-ubyte.gz','r')
    f.read(16)
    buf = f.read(28 * 28 *60000)
    data = np.frombuffer(buf, dtype=np.uint8).astype(float)/255
    train_set = torch.tensor(data.reshape( 60000,28, 28).astype('float32')).unsqueeze(1)
    
    f = gzip.open( data_dir_path + '/t10k-images-idx3-ubyte.gz','r')
    f.read(16)
    buf = f.read(28 * 28 *10000)
    data = np.frombuffer(buf, dtype=np.uint8).astype(float)/255
    test_set = torch.tensor(data.reshape( 10000,28, 28).astype('float32')).unsqueeze(1)
    
    return train_set, test_set

def load_set14( path):
    test_names = os.listdir(path)
    # read and prep test images
    test_images = []
    for file_name in test_names:
        image = plt.imread(path + file_name)
        test_images.append(image );
    return np.array(test_images)


def load_CelebA_dataset( train_folder_path, test_folder_path, s=1,n=-1):
    '''
    return images in a dict of two arrays. all images horizontal. Output arrays are in dims: B, H, W, C
    This function does not change image range.
    '''
    image_dict = {};
    train_names = os.listdir(train_folder_path)[0:n]
    test_names = os.listdir(test_folder_path)[0:n]

    # read and prep train images
    train_images = []
    for file_name in train_names:
        im = plt.imread(train_folder_path + file_name)
        im = im[29:-29, 9:-9] #crop to 160x160
        if s != 1:
            im =  resize_image(im, s)
        train_images.append(im);

    # read and prep test images
    test_images = []
    for file_name in test_names:
        im = plt.imread(test_folder_path + file_name)
        im = im[29:-29, 9:-9]
        if s != 1:
            im =  resize_image(im, s)
        test_images.append(im);


    image_dict['train'] = np.array(train_images)
    image_dict['test'] = np.array(test_images)

    return image_dict



def load_CelebA_HQ_dataset( train_folder_path, test_folder_path, s=1,n=-1):
    '''
    return images in a dict of two arrays. all images horizontal. Output arrays are in dims: B, H, W, C
    This function does not change image range.
    '''
    image_dict = {};
    train_names = os.listdir(train_folder_path)[0:n]
    test_names = os.listdir(test_folder_path)[0:n]

    # read and prep train images
    train_images = []
    for file_name in train_names:
        im = plt.imread(train_folder_path + file_name)
        if s != 1:
            im =  resize_image(im, s)
        train_images.append(im);

    # read and prep test images
    test_images = []
    for file_name in test_names:
        im = plt.imread(test_folder_path + file_name)
        if s != 1:
            im =  resize_image(im, s)
        test_images.append(im);


    image_dict['train'] = np.array(train_images)
    image_dict['test'] = np.array(test_images)

    return image_dict

def load_texture_dataset( train_folder_path, test_folder_path, s=1,n=-1):
    '''
    return images in a dict of two arrays. all images horizontal. Output arrays are in dims: B, H, W, C
    This function does not change image range.
    '''
    image_dict = {};
    train_names = os.listdir(train_folder_path)[0:n]
    test_names = os.listdir(test_folder_path)[0:n]

    # read and prep train images
    train_images = []
    for file_name in train_names:
        im = plt.imread(train_folder_path + file_name)
        if im.shape == (1024,1024,3):
            if s != 1:
                im =  resize_image(im, s)

            train_images.append(im);

    # read and prep test images
    test_images = []
    for file_name in test_names:
        im = plt.imread(test_folder_path + file_name)
        if im.shape == (1024,1024,3):
            if s != 1:
                 im =  resize_image(im, s)
            test_images.append(im);


    image_dict['train'] = np.array(train_images)
    image_dict['test'] = np.array(test_images)

    return image_dict


def load_bedrooms_dataset( train_folder_path, test_folder_path, s=1,n=-1):
    '''
    return images in a dict of two arrays. all images horizontal. Output arrays are in dims: B, H, W, C
    This function does not change image range.
    '''
    image_dict = {};
    train_names = os.listdir(train_folder_path)[0:n]
    test_names = os.listdir(test_folder_path)[0:n]

    # read and prep train images
    train_images = []
    for file_name in train_names:
        im = plt.imread(train_folder_path + file_name)
        im = im[0:256, 0:256]

        if s != 1:
            im =  resize_image(im, s)

        train_images.append(im);

    # read and prep test images
    test_images = []
    for file_name in test_names:
        im = plt.imread(test_folder_path + file_name)
        im = im[0:256, 0:256]
        if s != 1:
            im =  resize_image(im, s)
        test_images.append(im);


    image_dict['train'] = np.array(train_images)
    image_dict['test'] = np.array(test_images)

    return image_dict


def load_Botswana_dataset( folder_path, s=1):
    '''
    Output arrays are in dims: B, H, W, C
    This function does not change image range.
    '''
    folder_names = os.listdir(folder_path)
    train_images = []
    for folder in folder_names:
        if os.path.isdir(folder_path + folder): 
            file_names = os.listdir(folder_path + folder)
            for file in file_names: 
                im = plt.imread(os.path.join(folder_path ,folder,file))
                if s != 1:
                    im =  resize_image(im, s)
                train_images.append(im);
                        
    return np.array(train_images)


def load_Reach_dataset( folder_path, s=1):
    '''
    Images in this dataset have different sizes 
    '''
    folder_names = os.listdir(folder_path)
    print(len(folder_names))
    train_images = []
    for folder in folder_names:
        if os.path.isdir(folder_path + folder): 
            file_names = os.listdir(folder_path + folder)
            for file in file_names: 
                if file.split('.')[-1] == 'jpg': 
                    im = plt.imread(os.path.join(folder_path ,folder,file))
                    if s != 1:
                        im =  resize_image(im , s)
                    train_images.append(im);
    print(len(train_images))
    return train_images

def patch_Reach_dataset(dataset, patch_size, stride): 
    patches = []
    for im in dataset: 
        if (im.shape[0] >= patch_size[0]) & (im.shape[1]>= patch_size[1]):
            try:
                if (len(im.shape)==3):
                    if (im.shape[2] ==3):
                        temp = patch_generator_np(np.expand_dims(im,0).transpose(0,3,1,2), patch_size, stride)
                        patches.append(temp)
                        if len(patches)>101000: 
                            return np.concatenate(patches)
                            break
            except ValueError: 
                pass 
    return np.concatenate(patches)

def prep_Reach_patches(patches, test_set_size): 
    
    patches = (patches/255).astype('float32')
    patches = patches.mean(1, keepdims= True)# convert to grayscale
    patches = torch.FloatTensor(patches) # (B, C, H, W)

    return patches[0:-test_set_size], patches[-test_set_size::]

######################################################################################
################################ image pre-processing ################################
######################################################################################
def prep_celeba(all_images,k=None, mean_zero=False):
    '''
    @k: number of replica of each image with a different intensity
    '''
    all_images_tensor = {}

    train_images = int_to_float(all_images['train'])
    train_images = rgb_to_gray(train_images) # convert to grayscale
    if mean_zero:
        #train_images = remove_mean(train_images)
        train_images = train_images - train_images.mean()
    if k is not None:
        train_images = change_intensity_dataset(train_images,k)
    train_images = torch.FloatTensor(train_images).permute(0,3,1,2).contiguous() # (B, C, H, W)

    test_images = int_to_float(all_images['test'])
    test_images = rgb_to_gray(test_images)
    if mean_zero:
        #test_images = remove_mean(test_images)
        test_images = test_images - test_images.mean()
    if k is not None:
        test_images = change_intensity_dataset(test_images ,k )
    test_images = torch.FloatTensor(test_images).permute(0,3,1,2).contiguous() # (B, C, H, W)

    return train_images, test_images


def prep_texture_images(all_images,gray=True):
    '''
    @k: number of replica of each image with a different intensity
    '''
    all_images_tensor = {}

    train_images = int_to_float(all_images['train'])
    if gray:
        train_images = rgb_to_gray(train_images) # convert to grayscale
    train_images = torch.FloatTensor(train_images).permute(0,3,1,2).contiguous() # (B, C, H, W)
    
    
    test_images = int_to_float(all_images['test'])
    if gray:
        test_images = rgb_to_gray(test_images)
    test_images = torch.FloatTensor(test_images).permute(0,3,1,2).contiguous() # (B, C, H, W)

    return train_images, test_images


def resize_image(im, s):
    image_pil = Image.fromarray(im) # data type needed is uint8
    newsize = (int(image_pil.size[0] * s), int(image_pil.size[1] * s))
    image_pil_resize = image_pil.resize(newsize, resample=Image.BICUBIC)
    image_re = np.array(image_pil_resize)
    return image_re


def float_to_int(X):
    if X.dtype != 'uint8':
        return X
    else:
        return (X*255).astype('uint8')


def int_to_float(X):    
    if X.dtype == 'uint8':
        return (X/255).astype('float32')
    else:
        return X
    
def convert_8bit_to4bit(x): 
    
    for i in range(0,255,16): 
        print(i)
        mask = (x>=i) & (x<i+16)
        x[mask] = i/16
    return x

def rgb_to_gray(data):
    n, h, w, c = data.shape
    return data.mean(3).reshape(n,h,w,1)

def change_intensity(  im , k):
    temp = np.zeros((k,im.shape[0], im.shape[1], im.shape[2])).astype('float32')
    for i in range(k):
        temp[i] = np.random.rand(1).astype('float32') * im
    return temp

def change_intensity_dataset(dataset, k):
    temp = []
    for im in dataset:
        temp.append(change_intensity(im,k))
    return np.concatenate(temp)

def rescale_image_range(im,  max_I, min_I=0):

    temp = (im - im.min())  /((im.max() - im.min()))
    return temp *(max_I-min_I) + min_I



def rescale_image(im):
    if im.device.type == 'cuda': 
        im = im.cpu()

    if type(im) == torch.Tensor:
        im = im.numpy()
    return ((im - im.min()) * (1/(im.max() - im.min()) * 255)).astype('uint8')


def remove_mean(images):
    '''
    remove mean of images in a numpy batch of size N,W,H,1
    '''
    return images - images.mean(axis=(1,2)).reshape(-1,1,1,1)

def patch_generator(all_images, patch_size, stride):
    '''images: a 4D tensor of image: B, C, H, W
    patch_size: a tuple indicating the size of patches
    stride: a tuple indicating the size of the strides
    '''
    im_height = all_images.shape[2]
    im_width = all_images.shape[3]

    h = int(im_height/stride[0]) * stride[0]
    w = int(im_width/stride[1]) * stride[1]

    all_patches = []
    for x in range(0,h- patch_size[0] + 1, stride[0]):
        for y in range(0,w - patch_size[1] + 1, stride[1]):
            patch = all_images[:,:, x:x+patch_size[0] , y:y+patch_size[1]]
            all_patches.append(patch)

    return torch.cat(all_patches)



def patch_generator_np(all_images, patch_size, stride):
    '''images: a 4D numpy of image: B, C, H, W
    patch_size: a tuple indicating the size of patches
    stride: a tuple indicating the size of the strides
    '''
    im_height = all_images.shape[2]
    im_width = all_images.shape[3]

    h = int(im_height/stride[0]) * stride[0]
    w = int(im_width/stride[1]) * stride[1]

    all_patches = []
    for x in range(0,h- patch_size[0] + 1, stride[0]):
        for y in range(0,w - patch_size[1] + 1, stride[1]):
            patch = all_images[:,:, x:x+patch_size[0] , y:y+patch_size[1]]
            all_patches.append(patch)

    return np.concatenate(all_patches)


def patch_generator_with_scale(all_images, patch_size, stride, scales, resample = Image.BICUBIC):
    '''images: a 4D numpy array of image: B, H, W, C
    patch_size: a tuple indicating the size of patches
    stride: a tuple indicating the size of the strides
    scales: a list of float values by which the image is scaled
    '''

    all_images_patches = []
    # loop through all images in the set
    for image in all_images:
        image_patches = [] # holder for all patches of one image from different scales
        # loop through all the scales in the list
        for i in range(len(scales)):
            # resize the image (and blur if needed)
            image_pil = Image.fromarray(image) # data type needed is uint8
            # if blur is True:
                 # image_pil = image_pil.convert('L').filter(ImageFilter.GaussianBlur(1))
                 # image_pil = image_pil.convert('F')
            newsize = (int(image_pil.size[0] * scales[i]), int(image_pil.size[1] * scales[i]))
            image_pil_resize = image_pil.resize(newsize, resample=resample)
            image_re = np.array(image_pil_resize)


            im_height = image_re.shape[0]
            im_width = image_re.shape[1]


            patches = []
            h = int(im_height/stride[0]) * stride[0]
            w = int(im_width/stride[1]) * stride[1]
            # create patches for an image of a certain scale
            for x in range(0,h- patch_size[0] + 1, stride[0]):
                for y in range(0,w - patch_size[1] + 1, stride[1]):
                    # patches[counter] = image_re[ x:x+patch_size[0] , y:y+patch_size[1]]
                    patch = image_re[ x:x+patch_size[0] , y:y+patch_size[1]]
                    # patches.append(patch.reshape(1, patch.shape[0], patch.shape[1])) # add a dimension
                    patches.append(patch) # add a dimension

            patches = np.stack(patches, 0) # all the patches from one image at one scale
            image_patches.append(patches)
        image_patches = np.concatenate(image_patches, axis=0)
        all_images_patches.append(image_patches)
    return np.concatenate(all_images_patches, axis=0)


def data_augmentation(image,mode):
    if mode == 1:
        return image

    if mode == 2: # flipped
        image = np.flipud(image);
        return image

    elif mode == 3: # rotation 90
        image = np.rot90(image,1);
        return image;

    elif mode == 4 :# rotation 90 & flipped
        image = np.rot90(image,1);
        image = np.flipud(image);
        return image;

    elif mode == 5: # rotation 180
        image = np.rot90(image,2);
        return image;

    elif mode == 6: # rotation 180 & flipped
        image = np.rot90(image,2);
        image = np.flipud(image);
        return image;

    elif mode == 7: # rotation 270
        image = np.rot90(image,3);
        return image;

    elif mode == 8: # rotation 270 & flipped
        image = np.rot90(image,3);
        image = np.flipud(image);
        return image;
    else:
        raise ValueError('the requested mode is not defined')


def augment_training_data(train_set):

    augmented_train_set = np.zeros_like(train_set)
    for i in range(train_set.shape[0]):
        mode = np.random.randint(1,9)
        augmented_train_set[i,:,:] =  data_augmentation(train_set[i,:,:], mode)

    train_set = np.concatenate((train_set, augmented_train_set))
    return train_set

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    #elif classname.find('BatchNorm') != -1:
    #    m.weight.data.normal_(mean=0, std=sqrt(2./9./64.)).clamp_(-0.025,0.025)

    #    nn.init.constant(m.bias.data, 0.0)



###########################################################################
################################ wavelet coeffs ###########################
###########################################################################

def prep_celeba_for_wavelet(all_images, k=None):
    '''returns numpy arrays'''
    all_images_tensor = {}

    train_images = int_to_float(all_images['train'])
    train_images = rgb_to_gray(train_images) # convert to grayscale
    if k is not None:
        train_images = change_intensity_dataset(train_images,k)
    
    train_images = train_images.transpose(0,3,1,2)
    test_images = int_to_float(all_images['test'])
    test_images = rgb_to_gray(test_images)
    if k is not None:
        test_images = change_intensity_dataset(test_images ,k )
    test_images = test_images.transpose(0,3,1,2) 
    return train_images, test_images



def load_data_multi_scale(args):
    '''
    returns 2 torch tensors, normalized coefficients
    '''
    train_path = args.data_path + 'train_scale'+str(args.j)+'.pt'
    test_path = args.data_path + 'test_scale'+str(args.j)+'.pt'


    train_coeffs = torch.load(train_path) 
    test_coeffs = torch.load(test_path) 


    # correct for the missing normalizer in the default pywavelet Haar decompose function
    train_coeffs = train_coeffs/(2**(args.j+1))
    test_coeffs = test_coeffs/(2**(args.j+1))

    print('train: ', train_coeffs.size(), 'test: ', test_coeffs.size() )
    print('train low mean: ', train_coeffs[:,0].mean().item(), 'test low mean: ', test_coeffs[:,0].mean().item() )
    print('train high mean: ', train_coeffs[:,1::].mean().item(), 'test high mean: ', test_coeffs[:,1::].mean().item() )

    print('train low std: ', train_coeffs[:,0].std().item(), 'test low std: ', test_coeffs[:,0].std().item() )
    print('train high std: ', train_coeffs[:,1::].std().item(), 'test high std: ', test_coeffs[:,1::].std().item() )

    print('train low max: ', abs(train_coeffs[:,0]).max().item(), 'test low max: ', abs(test_coeffs[:,0]).max().item() )
    print('train high max: ', abs(train_coeffs[:,1::]).max().item(), 'test high max: ', abs(test_coeffs[:,1::]).max().item() )

    if args.coarse:
        train_coeffs = train_coeffs[:,0:1]
        test_coeffs = test_coeffs[:,0:1]

    return train_coeffs, test_coeffs

###########################################################################
################################ add noise ################################
###########################################################################
def add_noise_torch(all_patches, noise_level, device=None, quadratic=False, coarse=True, return_std=False):
    '''
    For images and wave coeffs 
    Gets images in the form of torch  tensors of size (B, C, H, W)
    '''
    N, C, H, W = all_patches.size()

    #for blind denoising
    if type(noise_level) == list:
        # std = torch.randint(low = noise_level[0], high=noise_level[1] , size = (N,1,1,1) ,device = device )/255
        std = torch.rand(size = (N,1,1,1), device = device) 
        
        if quadratic is True:
            std = std *(noise_level[1]**.5- noise_level[0]**.5)  +noise_level[0]**.5 
            std = std ** 2
        else:
            std = std *(noise_level[1]- noise_level[0])  +noise_level[0]

        std = std/255            

    #for specific noise
    else:
        std = torch.ones(N,1,1,1,  device = device) * noise_level/255

    if coarse:
        noise_samples = torch.randn(size = all_patches.size() , device = device) * std
        noisy = noise_samples+ all_patches
    else:
        noise_samples = torch.randn(size = (N,3,H,W) , device = device) * std
        noisy = torch.cat((all_patches[:, 0:1, :,:], all_patches[:, 1::, :,:] + noise_samples), dim=1)

    if return_std: 
        return noisy, noise_samples, std
        
    else: 
        return noisy, noise_samples

def add_noise_torch_range(im, noise_range, device=None, coarse=True):
    '''
    For images and wave coeffs
    Gets image- torch  tensors of size (1, H, W)
    @noise_rangel: sigmas for images in range 0-255 -tensor
    '''
    images = torch.stack([im]*noise_range.shape[0])
    noise_range = noise_range/255
    noise_samples = torch.randn(size = images.size() , device = device) * noise_range

    if coarse:
        noisy = images + noise_samples
    else:
        noise_samples = noise_samples[:,0:3]
        noisy = torch.cat((images[:, 0:1, :,:], images[:, 1::, :,:] + noise_samples), dim=1)

    return noisy , noise_samples


def add_noise_coeffs_torch_range(im, noise_range, device=None):
    '''
    Gets image- torch  tensors of size (1, H, W)
    @noise_rangel: sigmas for images in range 0-255 -tensor
    '''
    images = torch.stack([im]*noise_range.shape[0])
    noise_range = noise_range/255
    noise_samples = torch.randn(size = images.size() , device = device) * noise_range
    noise_samples = noise_samples[:,0:3]
    noisy = torch.cat((images[:, 0:1, :,:], images[:, 1::, :,:] + noise_samples), dim=1)
    return noisy , noise_samples




#def add_noise_coeffs_torch(all_patches, noise_level, mode='B', device=None, quadratic=False):
#    '''
#    Gets images in the form of torch  tensors of size (B, C, H, W)
#    '''
#    N, C, H, W = all_patches.size()
#    #if (mode == 'S' and type(noise_level) != int):
#    #    raise TypeError('noise level needs to be in range 0 to 255')
#    #elif(mode == 'B' and type(noise_level[0]) != int):
#    #    raise TypeError('noise level needs to be in range 0 to 255')
#
#    #if all_patches.dtype == torch.uint8:
#    #    raise TypeError('Data type required is float32')
#
#    #for blind denoising
#    if mode == 'B':
#        std = torch.randint(low = noise_level[0], high=noise_level[1] ,
#                            size = (N,1,1,1) ,device = device )/255
#
#        if quadratic is True:
#            std = std ** 2
#    #for specific noise
#    else:
#        std = torch.ones(N,1,1,1,  device = device) * noise_level/255
#
#    noise_samples = torch.randn(size = (N,3,H,W) , device = device) * std
#    noisy = torch.cat((all_patches[:, 0:1, :,:], all_patches[:, 1::, :,:] + noise_samples), dim=1)
#
#    return noisy, noise_samples




######################################################################################
################################ image loader test time ##############################
######################################################################################
class test_image:
    def __init__(self, grayscale,path, image_num):
        super(test_image, self).__init__()

        self.grayscale = grayscale
        self.path = path
        self.image_num = image_num

        self.im = single_image_loader(self.path,self.image_num)
        if self.im.dtype == torch.uint8:
            self.im = self.im/255
        if self.im.size()[0] == 3 and grayscale==True:
            raise Exception('model is trained for grayscale images. Load a grayscale image')
        elif self.im.size()[0] == 1 and grayscale==False:
            raise Exception('model is trained for color images. Load a color image')
        if torch.cuda.is_available():
            self.im = self.im.cuda()

    def show(self):
        if self.grayscale is True:
            if torch.cuda.is_available():
                plt.imshow(self.im.squeeze(0).cpu(), 'gray', vmin=0, vmax = 1)
            else:
                plt.imshow(self.im.squeeze(0), 'gray', vmin=0, vmax = 1)
        else:
            if torch.cuda.is_available():
                plt.imshow(self.im.permute(1,2,0).cpu(), vmin=0, vmax = 1)
            else:
                plt.imshow(self.im.permute(1,2,0), vmin=0, vmax = 1)

        plt.title('test image')
        plt.colorbar()
#         plt.axis('off');

    def crop(self, x0,y0,h,w):
        self.cropped_im = self.im[:, x0:x0+h, y0:y0+w]
        if self.grayscale is True:
            if torch.cuda.is_available():
                plt.imshow(self.cropped_im.squeeze(0).cpu(), 'gray', vmin=0, vmax = 1)
            else:
                plt.imshow(self.cropped_im.squeeze(0), 'gray', vmin=0, vmax = 1)

        else:
            if torch.cuda.is_available():
                plt.imshow(self.cropped_im.permute(1,2,0).cpu(), vmin=0, vmax = 1)
            else:
                plt.imshow(self.cropped_im.permute(1,2,0), vmin=0, vmax = 1)

        plt.title('cropped test image')
        plt.colorbar()
#         plt.axis('off');
        return self.cropped_im

