import torch
import torch.distributed as dist
import sys
sys.path.append('../losses')
sys.path.append('../data/datasets/datapipeline')
from losses import SSIM
import numpy as np
from lpips import LPIPS
from .utils import calculate_psnr

calc_SSIM = SSIM(data_range=1.)

def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    if world_size < 2:
        return rt
    else:
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= world_size
        return rt

def eval_one_loader(model, test_loader, metrics, local_rank=0, side_loss = False,multi_side = False, world_size = 1):
    """
    Evaluates a model on a given test data loader and computes metrics.
    Args:
        model (torch.nn.Module): The model to be evaluated.
        test_loader (torch.utils.data.DataLoader): DataLoader containing the test dataset.
        metrics (dict): Dictionary to store or update evaluation metrics.
        local_rank (int, optional): Device rank for distributed or GPU computation. Defaults to 0.
        side_loss (bool, optional): Flag to enable or disable side loss computation. Defaults to False.
    Returns:
        tuple: 
            - metrics (dict): Updated metrics after evaluation.
            - imgs_dict (dict): Dictionary containing output images and optionally side loss-related images.
                Keys include:
                - 'output': The enhanced output image.
                - 'side_align_1', 'side_align_4', 'side_align_7' (if side_loss=True): Edge alignment differences.
                - 'side_darkir' (if side_loss=True): Side dark image batch.
    """

    calc_LPIPS = LPIPS(net='vgg', verbose = False).to(local_rank)
    model.eval()
    mean_metrics = {'valid_psnr':[], 'valid_psnrY':[], 'valid_ssim':[], 'valid_lpips':[]}
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()

    with torch.no_grad():
        # Now we need to go over the test_loader and evaluate the results of the epoch
        for batch_dict in test_loader:

            low_batch = batch_dict['input'].to(local_rank)    
            high_batch = batch_dict['gt'].to(local_rank)  
            if side_loss:
                output_dict = model(low_batch, side_loss = side_loss)
            elif multi_side:
                 output_dict = model(low_batch, multi_side = multi_side)

            else:
                output_dict = model(low_batch)

        # end=time.time()
            # calculate the metrics
            valid_psnr_batch = calculate_psnr(output_dict['output'], high_batch, test_y_channel=False)
            valid_psnrY_batch = calculate_psnr(output_dict['output'], high_batch, test_y_channel=True)
            valid_ssim_batch = calc_SSIM(output_dict['output'], high_batch)
            valid_lpips_batch = calc_LPIPS(output_dict['output'], high_batch)
            
            mean_metrics['valid_psnr'].append(np.mean(valid_psnr_batch))
            mean_metrics['valid_ssim'].append(valid_ssim_batch.item())
            mean_metrics['valid_lpips'].append(torch.mean(valid_lpips_batch).item())
            mean_metrics['valid_psnrY'].append(np.mean(valid_psnrY_batch))

    end.record()
    torch.cuda.synchronize()
    total_time = start.elapsed_time(end)
    mean_time = total_time / len(test_loader)

    # now calculate the mean of each metrics in all the batches and gpus
    valid_psnr_tensor = reduce_tensor(torch.tensor(np.mean(mean_metrics['valid_psnr'])).to(local_rank), world_size=world_size)
    valid_ssim_tensor = reduce_tensor(torch.tensor(np.mean(mean_metrics['valid_ssim'])).to(local_rank),world_size=world_size)
    valid_lpips_tensor = reduce_tensor(torch.tensor(np.mean(mean_metrics['valid_lpips'])).to(local_rank), world_size=world_size)
    valid_psnrY_tensor = reduce_tensor(torch.tensor(np.mean(mean_metrics['valid_psnrY'])).to(local_rank), world_size=world_size)

    metrics['valid_psnr'] = valid_psnr_tensor.item()
    metrics['valid_ssim'] = valid_ssim_tensor.item()
    metrics['valid_lpips'] = valid_lpips_tensor.item()
    metrics['valid_psnrY'] = valid_psnrY_tensor.item()

    if side_loss:
        imgs_dict = {'input': low_batch[0],'output':output_dict['output'][0], 'gt': high_batch[0]}
    elif multi_side:
        imgs_dict = {'input': low_batch[0],'output':output_dict['output'][0], 'gt': high_batch[0]}
    else:
        imgs_dict = {'input': low_batch[0],'output':output_dict['output'][0], 'gt': high_batch[0]}

    return metrics, imgs_dict, mean_time

def eval_model(model, test_loader, metrics, local_rank=None, side_loss = False, multi_side = False, world_size = 1):
    """
    Evaluates a given model on one or multiple test data loaders and computes the specified metrics.
    This function processes the test data loaders, which can either be a single data loader or a dictionary 
    of multiple data loaders. It evaluates the model on each data loader, computes the metrics, and optionally 
    handles side loss computation. The results are returned as metrics and optionally a dictionary of images.
    Args:
        model (torch.nn.Module): The model to be evaluated.
        test_loader (Union[torch.utils.data.DataLoader, dict]): A single test data loader or a dictionary 
            of multiple data loaders. If a single data loader is provided, it is wrapped in a dictionary.
        metrics (dict): A dictionary to store the computed metrics for evaluation.
        local_rank (int, optional): The rank of the current process in distributed training. Defaults to None.
        side_loss (bool, optional): Whether to compute side loss during evaluation. Defaults to False.
    Returns:
        tuple: 
            - If `test_loader` contains multiple data loaders, returns a tuple of:
                - all_metrics (dict): A dictionary containing metrics for each data loader.
                - all_imgs_dict (dict): A dictionary containing image data for each data loader.
            - If `test_loader` contains a single data loader, returns a tuple of:
                - metrics (dict): The computed metrics for the single data loader.
                - imgs_dict (dict): The image data for the single data loader.
    """

    #first you need to assert that test_loader is a dictionary
    if type(test_loader) != dict:
        test_loader = {'data': test_loader}

    all_metrics = {}
    all_imgs_dict = {}
    all_mean_time = {}
    total_batches = 0
    weighted_time = 0

    for key, loader in test_loader.items():

        all_metrics[f'{key}'] = {}
        metrics, imgs_dict, mean_time = eval_one_loader(model, loader, all_metrics[f'{key}'], local_rank=local_rank, side_loss=side_loss, multi_side = multi_side, world_size=world_size)
        all_metrics[f'{key}'] = metrics
        all_imgs_dict[f'{key}'] = imgs_dict
        
        num_batches = len(loader)
        total_batches += num_batches
        weighted_time += mean_time * num_batches
        
    all_mean_time = weighted_time/total_batches

    return all_metrics, all_imgs_dict, all_mean_time
    