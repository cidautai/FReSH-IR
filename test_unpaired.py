import os
import argparse
from options.options import parse

parser = argparse.ArgumentParser(description="Script for testing")
parser.add_argument('-p', '--config', type=str, default='options/test/Unpaired.yml', help = 'Config file of testing unpaired')
parser.add_argument('-c', '--checkpoints_path', type=str, default='./models/bests/FReSHIR.pt',
                help="Checkpoints path")
args = parser.parse_args()

opt = parse(args.config)

import torch
from archs import create_model
from torchvision.transforms import Resize
from options.options import parse
from utils.utils import *
from tqdm import tqdm
from data import create_data
from losses import create_unpaired_losses

import torch.nn.functional as F

def pad_tensor(tensor, multiple = 16):
    """
    Pads a 4D tensor to ensure its height and width are multiples of a specified number.
    This function calculates the necessary padding for the height and width of the input tensor
    to make them divisible by the given `multiple`. It then applies zero-padding to the tensor
    using PyTorch's `F.pad`.
    Parameters:
        tensor (torch.Tensor): A 4D tensor with shape (N, C, H, W), where N is the batch size,
                               C is the number of channels, H is the height, and W is the width.
        multiple (int, optional): The number to which the height and width should be padded to
                                  be multiples of. Default is 16.
    Returns:
        torch.Tensor: The padded tensor with height and width adjusted to be multiples of the
                      specified `multiple`.
    """
    multiple = multiple
    _, _, H, W = tensor.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h), value = 0)
    
    return tensor

def load_model(local_rank, model, path_weights, global_rank=1):
    """
    Loads a model's weights from a specified checkpoint file and maps them to the appropriate device.
    Args:
        local_rank (int): The local rank of the current process, used to determine the target device for loading the model.
        model (torch.nn.Module): The PyTorch model instance to load the weights into.
        path_weights (str): The file path to the checkpoint containing the model's weights and additional metadata.
        global_rank: the rank of the process in the distributed setup.
    Returns:
        torch.nn.Module: The model with the loaded weights.
    Raises:
        FileNotFoundError: If the specified checkpoint file does not exist.
        KeyError: If the checkpoint does not contain the expected keys, such as 'model_state_dict' or 'epoch'.
    Notes:
        - The function prints the epoch value stored in the checkpoint for reference.
        - Assumes the checkpoint file contains a dictionary with at least the keys 'model_state_dict' and 'epoch'.
        - The `map_location` argument ensures that the weights are loaded onto the correct device based on the rank.
    """    
    map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
    checkpoints = torch.load(path_weights, map_location=map_location, weights_only=False)
    if 'module.' in list(checkpoints['model_state_dict'].keys())[0]:
        weights = {k.replace('module.', ''): v for k, v in checkpoints['model_state_dict'].items()}
    else:
        weights = checkpoints['model_state_dict']
    if global_rank == 0:
        print('Values on epoch:', checkpoints['epoch'])
    model.load_state_dict(weights)
    if global_rank == 0:
        print('Loaded weights correctly')
    return model

resize = opt['Resize']
scale = opt['scale']

def resize_for_clip(x: torch.Tensor, target_size: int = 512) -> torch.Tensor:
    """
    Redimensiona un batch de imágenes (N,C,H,W) a target_size x target_size
    usando interpolación bicúbica para evaluación CLIP-IQA+.
    """
    return F.interpolate(x, size=(target_size, target_size), mode='bicubic', align_corners=False)

def eval_one_loader_unpaired(model, test_loader, losses, metrics, local_rank, global_rank=1):
    """
    Evaluates a model on a test data loader for unpaired data.
    This function processes a test data loader to evaluate a model's performance
    using specified loss functions and metrics. It supports optional resizing and 
    padding of input images to ensure compatibility with the model. The results 
    are aggregated and printed for each metric.
    Args:
        model (torch.nn.Module): The model to be evaluated.
        test_loader (DataLoader): DataLoader providing the test dataset.
        losses (dict): A dictionary of loss functions to compute metrics.
                       Keys are metric names, and values are callable loss functions.
        metrics (dict): A dictionary to store aggregated metric results.
                        Keys are metric names, and values are initialized to zero.
        local_rank (int): The rank of the current process in a distributed setup.
        world_size (int): The total number of processes in the distributed setup.
        global_rank (int): The rank of the current process in the global context.
    Returns:
        None: The function prints the final results for each metric.
    Notes:
        - If the input image dimensions are large (height or width >= 1500), the 
          function downsamples the image by a specified scale factor.
        - Padding is applied only when resizing is performed to ensure the model 
          operates correctly.
        - The function uses `torch.no_grad()` to disable gradient computation 
          during evaluation.
        - Results are clamped to the range [0, 1] before computing metrics.
        - Progress is displayed using a progress bar for rank 0 in a distributed setup.
    """

    if global_rank==0:
        pbar = tqdm(total = len(test_loader))

    model.eval()
    # denoiser.eval()
    # sr_module.eval()
    for batch_dict in test_loader: # we just need to take the degraded image for these metrics

        ind_metric = {name: None for name in metrics.keys()}
        element = batch_dict['input'].to(local_rank)

        _, _, H, W = element.shape
        downsample = Resize([int(dim//scale) for dim in (H, W)]) if resize and (H >=1500 or W>=1500) else torch.nn.Identity()
        element = downsample(element)    

        if resize: # only add padd if resize is needed, to ensure that the model works properly
            element = pad_tensor(element)

        with torch.no_grad():
            # denoised_dict = denoiser(element)
            # denoised = denoised_dict['output']
            output_dict = model(element)
            # output = output_dict['output']
            # sr_dict = sr_module(output)
            # print(denoised.shape, output.shape, sr_dict['output'].shape)


        result = output_dict['output']
        result = torch.clamp(result, 0., 1.)
        
        upsample = Resize((H, W)) if resize else torch.nn.Identity()
        result = upsample(result)
        result = result[:, :, :H, :W]
        
        # result = resize_for_clip(result, target_size=512)
        
        for name, loss in losses.items(): # calculate the metrics for each image
            ind_metric[name] = loss[name](result)

        for metric in metrics.keys(): # add the metrics for each image to the total
            metrics[metric] = metrics[metric] + ind_metric[metric]

        pbar.update(1)

    print('Final results:')
    for name, value in metrics.items(): # calculate the mean metrics 
        print(f'In metric {name} we get: {value / len(test_loader)}')   

    if global_rank == 0:
        pbar.close()    

def eval_unpaired():
    if torch.cuda.device_count() > 1:
        # Initialize distributed environment
        dist.init_process_group(backend='nccl')
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        # Valores por defecto para single-GPU
        global_rank = 0
        world_size = 1
        local_rank = 0

    # Set device
    torch.cuda.set_device(local_rank)

    # load the dataloaders
    _, test_loader, _ = create_data(global_rank, world_size=world_size, opt=opt['datasets'])

    # define network and print complexity
    model, _, _ = create_model(opt['network'], local_rank, global_rank=global_rank)
    # denoiser = SCUNetWrapper(weights_path='./SCUNet_model/model_zoo/scunet_color_real_psnr.pth', device=f"cuda:{local_rank}")
    # sr_module = SRVGGWrapper()
    # load checkpoints
    model = load_model(local_rank, model, path_weights=args.checkpoints_path, global_rank=global_rank)
    if global_rank == 0:
        print('Using weights in:', args.checkpoints_path)

    # load losses
    names = opt['unsupervised_losses']
    losses = create_unpaired_losses(names, local_rank)
    metrics = {name: 0. for name in names}

    for key, loader in test_loader.items():
        if global_rank == 0:
            print(f' ---- Evaluating the model in {key} dataset')
        eval_one_loader_unpaired(model, loader, losses, metrics, local_rank, global_rank=global_rank)
        metrics = {name: 0. for name in names}

    if torch.cuda.device_count() > 1:
        dist.destroy_process_group()

if __name__ == '__main__':
    eval_unpaired()