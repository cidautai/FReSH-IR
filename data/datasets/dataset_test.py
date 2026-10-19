import os, sys
from glob import glob
import random
# PyTorch library
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as transforms
try:
    from .datapipeline import MyDataset_Crop
    from .utils import *
except:
    from datapipeline import MyDataset_Crop
    from utils import *

def test_dataset_allweather(global_rank = 1,
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
    ################## Snow100K Dataset

    PATH_TEST_SNOW100K = os.path.join(path_test, 'Snow100K/test/Snow100K-L/')
    paths_input_snow100k = glob(os.path.join(PATH_TEST_SNOW100K, 'synthetic', '*.jpg'))
    paths_gt_snow100k = glob(os.path.join(PATH_TEST_SNOW100K, 'gt', '*.jpg'))


    check_paths([paths_input_snow100k, paths_gt_snow100k])
    if verbose:
        print('Images in the subsets of Snow100K: \n')
        print("    -Images in the PATH_LOW_VALID folder: ", len(paths_input_snow100k))
        print("    -Images in the PATH_HIGH_VALID folder: ", len(paths_gt_snow100k))

    ################## Raindrop A Dataset 
    PATH_TEST_RAINDROP = os.path.join(path_test, 'RaindropData/test/')
    paths_input_raindrop_png = glob(os.path.join(PATH_TEST_RAINDROP, 'data', '*.png'))
    paths_gt_raindrop_png = glob(os.path.join(PATH_TEST_RAINDROP, 'gt', '*.png'))

    check_paths([paths_input_raindrop_png, paths_gt_raindrop_png])
    if verbose:
        print('Images in the subsets of Raindrop: \n')
        print("    -Images in the PATH_LOW_VALID folder: ", len(paths_input_raindrop_png))
        print("    -Images in the PATH_HIGH_VALID folder: ", len(paths_gt_raindrop_png))

    ################## Raindrop Dataset 

    paths_input_raindrop_jpg = glob(os.path.join(PATH_TEST_RAINDROP, 'data', '*.jpg'))
    paths_gt_raindrop_jpg = glob(os.path.join(PATH_TEST_RAINDROP, 'gt', '*.jpg'))
    
    paths_input_raindrop = paths_input_raindrop_png + paths_input_raindrop_jpg
    paths_gt_raindrop = paths_gt_raindrop_png + paths_gt_raindrop_jpg

    check_paths([paths_input_raindrop, paths_gt_raindrop])
    if verbose:
        print('Images in the subsets of Raindrop: \n')
        print("    -Images in the PATH_LOW_VALID folder: ", len(paths_input_raindrop))
        print("    -Images in the PATH_HIGH_VALID folder: ", len(paths_gt_raindrop))


    ################## OutdoorRain Dataset
    PATH_TEST_OUTDOOR = os.path.join(path_test, 'OutdoorRain/Test1/')
    paths_input_outdoor = glob(os.path.join(PATH_TEST_OUTDOOR, 'input', '*.png'))
    paths_gt_outdoor = glob(os.path.join(PATH_TEST_OUTDOOR, 'gt', '*.png'))

    check_paths([paths_input_outdoor, paths_gt_outdoor])

    if verbose:
        print('Images in the subsets of OutdoorRain: \n')
        print("    -Images in the PATH_LOW_VALID folder: ", len(paths_input_outdoor))
        print("    -Images in the PATH_HIGH_VALID folder: ", len(paths_gt_outdoor))

    #the test sets will be especific to lolv2 real and synth
    test_dataset_outdoor = MyDataset_Crop(paths_input_outdoor, paths_gt_outdoor, cropsize=None,
                                  tensor_transform=tensor_transform, test=True, crop_type=crop_type)

    test_dataset_snow100k = MyDataset_Crop(paths_input_snow100k, paths_gt_snow100k, cropsize=None,
                                  tensor_transform=tensor_transform, test=True, crop_type=crop_type)
    
    test_dataset_raindropa = MyDataset_Crop(paths_input_raindrop_png, paths_gt_raindrop_png, cropsize=None,
                                  tensor_transform=tensor_transform, test=True, crop_type=crop_type)

    test_dataset_raindrop = MyDataset_Crop(paths_input_raindrop, paths_gt_raindrop, cropsize=None,
                                  tensor_transform=tensor_transform, test=True, crop_type=crop_type)

    # Now we need to apply the Distributed sampler
   
    test_sampler_outdoor = DistributedSampler(test_dataset_outdoor, num_replicas=world_size, shuffle= True, rank=global_rank)
    test_sampler_snow100k = DistributedSampler(test_dataset_snow100k, num_replicas=world_size, shuffle= True, rank=global_rank)
    test_sampler_raindrop = DistributedSampler(test_dataset_raindrop, num_replicas=world_size, shuffle= True, rank=global_rank)
    test_sampler_raindropa = DistributedSampler(test_dataset_raindropa, num_replicas=world_size, shuffle= True, rank=global_rank)

    samplers.append(test_sampler_outdoor)
    samplers.append(test_sampler_snow100k)
    samplers.append(test_sampler_raindrop)
    samplers.append(test_sampler_raindropa)


    
    test_loader_outdoor = DataLoader(dataset=test_dataset_outdoor, batch_size=batch_size_test, shuffle=False,
                            num_workers=num_workers, pin_memory=True, drop_last=False, sampler=test_sampler_outdoor)
    test_loader_snow100k = DataLoader(dataset=test_dataset_snow100k, batch_size=batch_size_test, shuffle=False,
                            num_workers=num_workers, pin_memory=True, drop_last=False, sampler=test_sampler_snow100k)
    test_loader_raindrop = DataLoader(dataset=test_dataset_raindrop, batch_size=batch_size_test, shuffle=False,
                            num_workers=num_workers, pin_memory=True, drop_last=False, sampler=test_sampler_raindrop)
    test_loader_raindropa = DataLoader(dataset=test_dataset_raindropa, batch_size=batch_size_test, shuffle=False,
                            num_workers=num_workers, pin_memory=True, drop_last=False, sampler=test_sampler_raindropa)
     

    test_loader = {'outdoor':test_loader_outdoor, 'snow100k':test_loader_snow100k, 'raindropA':test_loader_raindropa, 'raindrop':test_loader_raindrop}

    
    return train_loader, test_loader, samplers

if __name__ == '__main__':
    train_loader, test_loader, samplers = test_dataset_allweather(global_rank=0,path_train='/media/Data/allweather',
                                                                  path_test='/media/Data/',
                                                               verbose=True)
    # print(test_loader.keys())
    # print(samplers.keys())