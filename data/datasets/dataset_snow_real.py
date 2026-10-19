import os, sys
from glob import glob
import random
# PyTorch library
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as transforms
try:
    from .datapipeline import MyDataset_Crop, MyDataset_Unpaired
    from .utils import *
except:
    from datapipeline import MyDataset_Crop, MyDataset_Unpaired
    from utils import *

def main_dataset_snow_real(global_rank = 1,
                         path_train='Datasets/allweather/',
                         path_test = 'Datasets/',
                         batch_size_train=4, 
                         batch_size_test=1, 
                         verbose=False, 
                         cropsize=512, 
                         flips = None,
                         num_workers=1, 
                         crop_type='Random',
                         world_size = 1):
    
    ############################## Creating the dataloader for Training
    PATH_TRAIN_ALLWEATHER = path_train
    
    print(PATH_TRAIN_ALLWEATHER)

    # paths to the blur and sharp sets of images

    paths_blur_jpg = glob(os.path.join(PATH_TRAIN_ALLWEATHER, 'input', '*.jpg'))
    paths_blur_png = glob(os.path.join(PATH_TRAIN_ALLWEATHER, 'input', '*.png'))
    paths_blur = paths_blur_jpg + paths_blur_png

    paths_sharp_jpg = glob(os.path.join(PATH_TRAIN_ALLWEATHER, 'gt', '*.jpg'))
    paths_sharp_png = glob(os.path.join(PATH_TRAIN_ALLWEATHER, 'gt', '*.png'))
    paths_sharp = paths_sharp_jpg + paths_sharp_png

    # check if all the image routes are correct
    check_paths([paths_blur, paths_sharp])

    if verbose:
        print('Images in the subsets of AllWeather: \n')
        print("    -Images in the PATH_LOW_TRAINING folder: ", len(paths_blur))
        print("    -Images in the PATH_HIGH_TRAINING folder: ", len(paths_sharp))


    tensor_transform = transforms.ToTensor()
    if flips:
        flip_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # flips horizontal with p=0.5
            transforms.RandomVerticalFlip()  # flips vertical with p = 0.5
        ])
    else:
        flip_transform = None

    # Load the datasets
    train_dataset = MyDataset_Crop(paths_blur, paths_sharp, cropsize=cropsize,
                                   tensor_transform=tensor_transform, flips=flip_transform, test=False, crop_type=crop_type)


    # Now we need to apply the Distributed sampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, shuffle= True, rank=global_rank)

    samplers = []
    # samplers = {'train': train_sampler, 'test': [test_sampler_gopro, test_sampler_lolblur]}
    samplers.append(train_sampler)
    


    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_train, shuffle=False,
                            num_workers=num_workers, pin_memory=True, drop_last=True, sampler=train_sampler)
    


    ################## Creating the random seed
    random.seed(42)
    
    ############################## Creating the dataloader for Testing

    ################## Snow Real
    paths_input_outdoor = glob(os.path.join(path_test, '*.jpg'))

    check_paths([paths_input_outdoor])

    if verbose:
        print('Images in the subsets of OutdoorRain: \n')
        print("    -Images in the PATH_LOW_VALID folder: ", len(paths_input_outdoor))

    #the test sets will be especific to lolv2 real and synth
    test_dataset_outdoor = MyDataset_Unpaired(paths_input_outdoor, cropsize=None,
                                  tensor_transform=tensor_transform, test=True, crop_type=crop_type)


    # Now we need to apply the Distributed sampler
   
    test_sampler_outdoor = DistributedSampler(test_dataset_outdoor, num_replicas=world_size, shuffle= True, rank=global_rank)
    
    samplers.append(test_sampler_outdoor)
    


    
    test_loader_outdoor = DataLoader(dataset=test_dataset_outdoor, batch_size=batch_size_test, shuffle=False,
                            num_workers=num_workers, pin_memory=True, drop_last=False, sampler=test_sampler_outdoor)
   
     

    test_loader = {'snow':test_loader_outdoor}

    
    return train_loader, test_loader, samplers

if __name__ == '__main__':
    train_loader, test_loader, samplers = main_dataset_snow_real(global_rank=0,path_train='/media/Data/allweather',
                                                                  path_test='/media/Data/unpaired_allweather',
                                                               verbose=True)
    #print(test_loader.keys())
    # print(samplers.keys())