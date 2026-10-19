import numpy as np 
import os

from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torch.nn as nn
import cv2 as cv

def crop_center(img, cropx=224, cropy=256):
    """    
    Crops the center region of an image to the specified dimensions.

    This function takes an input image and returns a cropped version of the image,
    centered around the middle of the original image. The dimensions of the cropped
    region are specified by the `cropx` and `cropy` parameters.

    Parameters:
        img (numpy.ndarray): The input image as a NumPy array with shape (height, width, channels).
        cropx (int, optional): The width of the cropped region. Default is 224.
        cropy (int, optional): The height of the cropped region. Default is 256.

    Returns:
        numpy.ndarray: The center-cropped image with dimensions (cropy, cropx, channels).

    Raises:
        AttributeError: If the input image does not have the expected shape attribute.
        ValueError: If the crop dimensions are larger than the input image dimensions.

    Example:
        >>> import numpy as np
        >>> img = np.random.rand(500, 600, 3)  # Example image
        >>> cropped_img = crop_center(img, cropx=224, cropy=256)
        >>> print(cropped_img.shape)
        (256, 224, 3)
    """

    y,x,_  = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

class CropTo4(nn.Module):
    """
    A PyTorch module that crops two input images into four equal quadrants each. 
    If the dimensions of the input images are not divisible by 2, the images are padded with zeros 
    to ensure proper cropping.
    Methods:
        __init__():
            Initializes the CropTo4 module.
        forward(img1, img2):
            Crops the input images into four equal quadrants each. If the dimensions of the images 
            are not divisible by 2, they are padded with zeros before cropping.
            Args:
                img1 (torch.Tensor): The first input image tensor of shape (N, C, H, W).
                img2 (torch.Tensor): The second input image tensor of shape (N, C, H, W).
            Returns:
                Tuple[List[torch.Tensor], List[torch.Tensor]]: Two lists of cropped image tensors 
                (each containing 4 tensors corresponding to the quadrants of the input images).
        pad(img):
            Pads the input image with zeros if its dimensions are not divisible by 2.
            Args:
                img (torch.Tensor): The input image tensor of shape (N, C, H, W).
            Returns:
                torch.Tensor: The padded image tensor with dimensions adjusted to be divisible by 2.
    """
    def __init__(self):
        super(CropTo4, self).__init__()
        

    def forward(self, img1, img2):
        
        #pad the images with zeros if their size is lower than the cropsize
        img1 = self.pad(img1)
        img2 = self.pad(img2)
        _, _, h, w = img1.shape
        crops1 = [TF.crop(img1, 0, 0, h//2, w//2), TF.crop(img1, 0, w//2, h//2, w//2),
                  TF.crop(img1, h//2, 0, h//2, w//2),TF.crop(img1, h//2, w//2, h//2, w//2)]
        crops2 = [TF.crop(img2, 0, 0, h//2, w//2), TF.crop(img2, 0, w//2, h//2, w//2),
                  TF.crop(img2, h//2, 0, h//2, w//2),TF.crop(img2, h//2, w//2, h//2, w//2)]
            
        return crops1, crops2

    def pad(self, img):
        _, _, h, w = img.shape
        mod_pad_h = (h - h//2) % 2
        mod_pad_w = (w - w//2) % 2
        img = F.pad(img, (0, mod_pad_w, 0, mod_pad_h), mode = 'constant', value = 0)
        
        return img

class RandomCropSame:
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img1, img2):
        
        #pad the images with zeros if their size is lower than the cropsize
        if img1.shape[1] <= self.size[0] or img1.shape[2] <= self.size[1]:
            img1 = self.pad(img1)
            img2 = self.pad(img2)
        
        i, j, th, tw = self.get_params(img1, self.size)
            
        return TF.crop(img1, i, j, th, tw), TF.crop(img2, i, j, th, tw)  # Use th and tw here

    def get_params(self, img, output_size):
        h, w = img.shape[1], img.shape[2]
        th, tw = output_size
        
        if w <= tw or h <= th:
            return 0, 0, h, w
        
        # Calculate the starting top-left corner (i, j) such that the entire crop is within the image.
        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()
        
        return i, j, th, tw

    def pad(self, img):
        _, h, w = img.shape
        mod_pad_h = self.size[0] - h
        mod_pad_w = self.size[1] - w
        img = F.pad(img, (0, mod_pad_w, 0, mod_pad_h))
        
        return img

class MyDataset_Crop(Dataset):
    """
    A Dataset of the low and high light images with data values in each channel in the range 0-1 (normalized).
    """
    
    def __init__(self, images_low, images_high, cropsize = None, tensor_transform = None, flips=None, test=False, crop_type = 'Random'):
        """
        -images_high: list of RGB images of normal-light used for training or testing the model
        - images_low: list of RGB images of low-light used for training or testing the model
        - test: indicates if the dataset is for training (False) or testing (True)
        -image_size: contains the dimension of the final image (H, W, C). This is important
                     to do the propper crop of the image.
        """
        self.imgs_low   = sorted(images_low)
        self.imgs_high  = sorted(images_high)
        self.test       = test
        self.cropsize   = cropsize
        self.to_tensor  = tensor_transform
        self.flips      = flips
        
        if self.cropsize:
            if crop_type   == 'Random':
                
                self.random_crop = RandomCropSame(self.cropsize)
                self.center_crop = None
            elif crop_type == 'Center':
                
                self.center_crop = transforms.CenterCrop(cropsize)  
                self.random_crop = None

    def __len__(self):
        return len(self.imgs_low)

    def __getitem__(self, idx):
        """
        Given a (random) index. The dataloader selects the corresponding image path, and loads the image.
        Then it returns the image, after applying any required transformation.
        """
        img_low  = self.imgs_low[idx]
        img_high = self.imgs_high[idx]
        if 'RaindropData' in img_low and 'RaindropData' in img_high:
            low_extension = img_low.split('/')[-1].split('.')[1]
            high_extension = img_high.split('/')[-1].split('.')[1]
            low_name = img_low.split('/')[-1].split('.')[0].split('_')[0]
            high_name = img_high.split('/')[-1].split('.')[0].split('_')[0]
            if (low_name== high_name) and (low_extension != high_extension):
                img_high = img_high.replace(high_extension, low_extension)


        # Load the image and convert to numpy array
        rgb_low  = Image.open(img_low).convert('RGB')
        rgb_high = Image.open(img_high).convert('RGB')

        # if self.test:
        #     if rgb_high.size != rgb_low.size:
        #         new_size = rgb_low.size
        #         rgb_high = rgb_high.resize(new_size, Image.BICUBIC)


        if self.to_tensor: #transform the image to have the adequate properties
            rgb_low  = self.to_tensor(rgb_low)
            rgb_high = self.to_tensor(rgb_high)

        # print(rgb_high.shape, rgb_low.shape)
        if self.cropsize: # do random crops of the image
            if self.random_crop:   
                rgb_high, rgb_low = self.random_crop(rgb_high, rgb_low)
            elif self.center_crop:
                rgb_high, rgb_low = self.center_crop(rgb_high), self.center_crop(rgb_low)

        # stack high and low to do the exact same flip on the two images
        high_and_low = torch.stack((rgb_high, rgb_low))
        if self.flips:
            high_and_low      = self.flips(high_and_low)
            rgb_high, rgb_low = high_and_low #separate again the images
        

        # print(rgb_high.shape, rgb_low.shape)

        return {'gt': rgb_high, 'input': rgb_low}

class MyDataset_Unpaired(Dataset):
    """
    A Dataset of the low and high light images with data values in each channel in the range 0-1 (normalized).
    """
    
    def __init__(self, images_low,  cropsize = None, tensor_transform = None, flips=None, test=False, crop_type = 'Random'):
        """
        - images_low: list of RGB images of low-light used for training or testing the model
        - test: indicates if the dataset is for training (False) or testing (True)
        -image_size: contains the dimension of the final image (H, W, C). This is important
                     to do the propper crop of the image.
        """
        self.imgs_low   = sorted(images_low)
        self.test       = test
        self.cropsize   = cropsize
        self.to_tensor  = tensor_transform
        self.flips      = flips
        
        if self.cropsize:
            if crop_type   == 'Random':
                
                self.random_crop = RandomCropSame(self.cropsize)
                self.center_crop = None
            elif crop_type == 'Center':
                
                self.center_crop = transforms.CenterCrop(cropsize)  
                self.random_crop = None

    def __len__(self):
        return len(self.imgs_low)

    def __getitem__(self, idx):
        """
        Given a (random) index. The dataloader selects the corresponding image path, and loads the image.
        Then it returns the image, after applying any required transformation.
        """
        img_low  = self.imgs_low[idx]

        # Load the image and convert to numpy array
        rgb_low  = Image.open(img_low).convert('RGB')

        # if self.test:
        #     if rgb_high.size != rgb_low.size:
        #         new_size = rgb_low.size
        #         rgb_high = rgb_high.resize(new_size, Image.BICUBIC)



        if self.to_tensor: #transform the image to have the adequate properties
            rgb_low  = self.to_tensor(rgb_low)

        if self.flips:
            rgb_low = self.flips(rgb_low) #separate again the images
        
        # print(rgb_high.shape, rgb_low.shape)
        if self.cropsize: # do random crops of the image
            if self.random_crop:   
                rgb_low = self.random_crop(rgb_low)
            elif self.center_crop:
                rgb_low = self.center_crop(rgb_low)

        # print(rgb_high.shape, rgb_low.shape)

        return {'input': rgb_low}
    
    
def tensor_to_pil(tensor):
    img = (tensor.permute(1,2,0).numpy()*255).astype(np.uint8)
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    img = Image.fromarray(img)
    return img

def labeler_train(path):

    if path.endswith('.jpg'):
        label = 0
    elif path.split('/')[-1].split('_')[-1].startswith('rain'):
        label = 1
    elif path.split('/')[-1].startswith('im'):
        label = 2
    else:
        label = 3
    return torch.tensor(label)

def labeler_test(path):

    if path.endswith('.jpg'):
        label = 0
    elif path.split('/')[-1].split('_')[-1].startswith('rain'):
        label = 1
    elif path.split('/')[-1].startswith('im'):
        label = 2
    else:
        label = 3

    return torch.tensor(label)

class MyDataset_Crop_Labels(Dataset):
    """
    A Dataset of the low and high light images with data values in each channel in the range 0-1 (normalized).
    """
    
    def __init__(self, images_low, images_high, cropsize = None, tensor_transform = None, flips=None, test=False, crop_type = 'Random'):
        """
        -images_high: list of RGB images of normal-light used for training or testing the model
        - images_low: list of RGB images of low-light used for training or testing the model
        - test: indicates if the dataset is for training (False) or testing (True)
        -image_size: contains the dimension of the final image (H, W, C). This is important
                     to do the propper crop of the image.
        """
        self.imgs_low   = sorted(images_low)
        self.imgs_high  = sorted(images_high)
        self.test       = test
        self.cropsize   = cropsize
        self.to_tensor  = tensor_transform
        self.flips      = flips
        
        if self.cropsize:
            if crop_type   == 'Random':
                
                self.random_crop = RandomCropSame(self.cropsize)
                self.center_crop = None
            elif crop_type == 'Center':
                
                self.center_crop = transforms.CenterCrop(cropsize)  
                self.random_crop = None

    def __len__(self):
        return len(self.imgs_low)

    def __getitem__(self, idx):
        """
        Given a (random) index. The dataloader selects the corresponding image path, and loads the image.
        Then it returns the image, after applying any required transformation.
        """
        
        img_low  = self.imgs_low[idx]
        img_high = self.imgs_high[idx]
        
        if self.test:
            label = labeler_test(img_low)
        else:
            label = labeler_train(img_low)
        
        # Load the image and convert to numpy array
        rgb_low  = Image.open(img_low).convert('RGB')
        rgb_high = Image.open(img_high).convert('RGB')

        if self.to_tensor: #transform the image to have the adequate properties
            rgb_low  = self.to_tensor(rgb_low)
            rgb_high = self.to_tensor(rgb_high)

        # stack high and low to do the exact same flip on the two images
        high_and_low = torch.stack((rgb_high, rgb_low))
        if self.flips:
            high_and_low      = self.flips(high_and_low)
            rgb_high, rgb_low = high_and_low #separate again the images
        
        # print(rgb_high.shape, rgb_low.shape)
        if self.cropsize: # do random crops of the image
            if self.random_crop:   
                rgb_high, rgb_low = self.random_crop(rgb_high, rgb_low)
            elif self.center_crop:
                rgb_high, rgb_low = self.center_crop(rgb_high), self.center_crop(rgb_low)
    
        return rgb_high, rgb_low, label


if __name__== '__main__':

    pass
