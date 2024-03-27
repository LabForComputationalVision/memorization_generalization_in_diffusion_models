import time
import torch
from synthetic_data_generators import make_C_alpha_images
from dataloader_func import rescale_image_range
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--alpha' ) 
    parser.add_argument('--beta')  
    parser.add_argument('--size',type=int , default=80) #image size   
    parser.add_argument('--num_samples',type=int , default=100000)   
    parser.add_argument('--factor', default=(2,2)) # default blur factor

    args = parser.parse_args()
    args.beta = args.alpha     
    path = '~/datasets/C_alpha_data/C_alpha'+str(args.alpha)+'_beta'+str(args.beta)
    

    if os.path.exists(path): #to avoid over-writing the existing dataset 
        pass
    else:
        os.makedirs(path)


        start_time_total = time.time()


        for split in [ 'train', 'test']:
            ims = make_C_alpha_images(alpha = args.alpha, 
                                      beta = args.beta, 
                                      im_size=args.size, 
                                      num_samples=args.num_samples, 
                                      factor=args.factor,
                                      )

            
            print('saving data of size: ' , ims.shape)
            torch.save(ims , f"{path}/{split}.pt")

        print("--- %s seconds ---" % (time.time() - start_time_total))


if __name__ == "__main__":
    main()
