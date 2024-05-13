import numpy as np
import os
import time
import torch
from dataloader_func import load_Reach_dataset, patch_Reach_dataset,prep_Reach_patches
from quality_metrics_func import remove_repeats_loop, remove_repeats


def main():
    start_time_total = time.time()
    
    train_folder_path = dir_path + 'train/img_align_celeba/'
    test_folder_path = dir_path + 'test/'

    all_ims = load_CelebA_dataset( train_folder_path, test_folder_path, s=.125)
    train_set, test_set  = prep_celeba(all_ims)
        
    print('train: ', train_set.shape )
    print('test: ', test_set.shape )

    # remove repeated images 
    data_cleaned = remove_repeats(train_set, threshold=.95)
    print('shape after removed repeats:' , data_cleaned.shape)
    
    torch.save(data_cleaned, dir_path + '/train80x80_no_repeats.pt')
    torch.save(test_set, dir_path + '/test80x80.pt')
        
    print("--- %s seconds ---" % (time.time() - start_time_total))

    

if __name__ == "__main__" :
    main()    
    
    
    
