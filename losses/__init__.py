from .loss import PixelLoss, SSIM, VGGLoss, EdgeLoss, EnhanceLoss, MultiguidanceLoss,FourierEdgeLoss, ClassificationLoss
import pyiqa

def adjust_list_length(lst, target_length):
    """
    Adjusts the length of a list to match the target length by either appending
    the last element or removing elements from the end.

    Parameters:
        lst (list): The list to adjust.
        target_length (int): The desired length of the list.

    Returns:
        list: The adjusted list with the specified length.
    """
    assert isinstance(lst, list), "First argument must be a list."
    assert isinstance(target_length, int), "Second argument must be an integer."
    assert target_length >= 0, "Target length must be non-negative."

    while len(lst) < target_length:
        lst.append(lst[-1] if lst else None)
    while len(lst) > target_length:
        lst.pop()
    return lst

def create_unpaired_losses(list_of_losses = ['musiq', 'niqe', 'nrqm'], local_rank=0):
    """
    Creates a dictionary of unpaired loss metrics.
    This function initializes a set of loss metrics specified in the input list
    and maps them to the given device rank. Each loss metric is created using
    the `pyiqa.create_metric` function and is stored in a nested dictionary
    structure.
    Args:
        list_of_losses (list of str, optional): A list of loss metric names to be
            created. Defaults to ['musiq', 'niqe', 'nrqm'].
        local_rank (int, optional): The device rank (e.g., GPU index) to which the
            loss metrics should be moved. Defaults to 0.
    Returns:
        dict: A dictionary where each key is a loss metric name, and the value
        is another dictionary containing the metric object associated with the
        corresponding name.
    """
    losses = {}
    for name in list_of_losses:
        losses[name] = {name: pyiqa.create_metric(name).to(local_rank)}
    
    return losses

def create_loss(opt, local_rank, step = 0, global_rank = 1):
    """
    Creates and returns a dictionary of loss functions and their corresponding weights 
    for evaluating a model based on the provided options.
    This function dynamically initializes various loss functions based on the configuration 
    specified in the `opt` dictionary for a given training step. It supports multiple types 
    of losses such as pixel loss, perceptual loss, edge loss, edge alignment loss, frequency 
    loss, enhance loss, YUV loss, and color consistency loss (or every loss that you defined 
    previously in loss.py and import to this script). Each loss is initialized with 
    its respective parameters and moved to the specified device local_rank.
    Args:
        opt (dict): A dictionary containing configuration options for the losses. 
                    It includes keys for each loss type (e.g., 'pixel', 'perceptual', etc.) 
                    and their respective parameters such as weights, criteria, and reductions.
        local_rank (int): The device rank (e.g., GPU ID) where the losses should be moved.
        step (int, optional): The current training step, used to index into step-specific 
                                configurations in the `opt` dictionary. Defaults to 0.
        global_rank (int, optional): The global rank of the process in distributed training.
    Returns:
        dict: A dictionary containing the initialized loss functions, where the keys are 
                the names of the losses (e.g., 'pixel_loss', 'perceptual_loss') and the values 
                are the corresponding loss objects.
    Note: 
        The flags to each loss are processed by the adjust_list_length function, which ensures that
        the length of the list is equal to the number of steps. This prevents errors in the YAML file
        if the user does not define the same number of weights for each step. The function will
        automatically adjust the length of the list to be equal to the number of steps.
    """
    
    losses = dict()
    weight_losses = dict()
    
    # these are the keywords that need to be checked if have the proper length
    processing_keys = ['flag', 'weight']

    # process all the keys in the optimization dictionary and adjust their length if needed
    for key, value in opt.items():
        if key in processing_keys:
            opt[key] = adjust_list_length(value, opt['optimization']['steps'])

    if opt['pixel_flag'][step]: 
        pixel_loss = PixelLoss(loss_weight = opt['pixel_weight'][step],
                                criterion = opt['pixel_criterion'],
                                reduction = opt['pixel_reduction']).to(local_rank)
        losses['pixel_loss'] = pixel_loss
        weight_losses['pixel_weight'] = opt['pixel_weight'][step]
        if global_rank == 0: print(f"Using pixel loss {opt['pixel_criterion']} with weight {opt['pixel_weight'][step]}")
    
    # the edge loss
    if opt['edge_flag'][step]: 
        edge_loss = EdgeLoss(loss_weight = opt['edge_weight'][step],
                                criterion = opt['edge_criterion'],
                                reduction = opt['edge_reduction'],
                                rank = local_rank).to(local_rank)
        losses['edge_loss'] = edge_loss
        weight_losses['edge_weight'] = opt['edge_weight'][step]
        if global_rank==0: print(f"Using edge loss {opt['edge_criterion']} with weight {opt['edge_weight'][step]}")
    
    if opt['fourier_flag'][step]:
        frequency_loss = FourierEdgeLoss(loss_weight = opt['fourier_weight'][step],
                                reduction = opt['fourier_reduction'],
                                criterion = opt['fourier_criterion'],
                                radius_ratio = opt['cutoff'][step]).to(local_rank)
        losses['fourier_loss'] = frequency_loss
        weight_losses['fourier_weight'] = opt['fourier_weight'][step]
        if global_rank==0: print(f"Using fourier loss {opt['fourier_criterion']} with weight {opt['fourier_weight'][step]}")
    
    if opt['fourier_out_flag'][step]:
        frequency_loss = FourierEdgeLoss(crop_size=384,loss_weight = opt['fourier_out_weight'][step],
                                reduction = opt['fourier_out_reduction'],
                                criterion = opt['fourier_out_criterion'],
                                radius_ratio = opt['out_cutoff'][step]).to(local_rank)
        losses['fourier_out_loss'] = frequency_loss
        weight_losses['fourier_out_weight'] = opt['fourier_out_weight'][step]
        if global_rank==0: print(f"Using fourier out loss {opt['fourier_out_criterion']} with weight {opt['fourier_out_weight'][step]}")

    # the enhance loss
    if opt['enhance_flag'][step]:
        enhance_loss = EnhanceLoss(loss_weight= opt['enhance_weight'][step],
                                reduction = opt['enhance_reduction'],
                                criterion = opt['enhance_criterion']).to(local_rank)
        losses['enhance_loss'] = enhance_loss
        weight_losses['enhance_weight'] = opt['enhance_weight'][step]
        if global_rank==0: print(f"Using enhance loss {opt['enhance_criterion']} with weight {opt['enhance_weight'][step]}")
 

    if opt['enhance_out_flag'][step]:
        enhance_loss = EnhanceLoss(loss_weight= opt['enhance_out_weight'][step],
                                reduction = opt['enhance_out_reduction'],
                                criterion = opt['enhance_out_criterion']).to(local_rank)
        losses['enhance_out_loss'] = enhance_loss
        weight_losses['enhance_out_weight'] = opt['enhance_out_weight'][step]
        if global_rank==0: print(f"Using enhance out loss {opt['enhance_out_criterion']} with weight {opt['enhance_out_weight'][step]}")
 
    if opt['classify_flag'][step]:
        classification_loss = ClassificationLoss(loss_weight = opt['classify_weight'][step]).to(local_rank)
        losses['classification_loss'] = classification_loss
        weight_losses['classification_weight'] = opt['classify_weight'][step]
        if global_rank == 0: print(f"Using classification loss with weight {opt['classify_weight'][step]}")
    
    if opt['multiguidance_flag'][step]:
        multiguidance_loss = MultiguidanceLoss(loss_weight = opt['multiguidance_weight'][step], 
                                               criterion = opt['multiguidance_criterion'],
                                               reduction = opt['multiguidance_reduction'],
                                               loss_type = opt['multiguidance_type'], 
                                               cutoff = opt['multiguidance_cutoff'][step], 
                                               crop_size = opt['multiguidance_crop'],rank = local_rank).to(local_rank)
        losses['multiguidance_loss'] = multiguidance_loss
        weight_losses['multiguidance_weight'] = opt['multiguidance_weight'][step]
        if global_rank == 0: print(f"Using multiguidance loss {opt['multiguidance_type']} type {opt['multiguidance_type']} with weight {opt['multiguidance_weight'][step]}")

    # if rank0 print the losses used
    if global_rank==0: 
        print('Used losses: ',losses.keys())
        print(f'With weights {weight_losses.values()}','***************************')
    return losses

def calculate_loss(all_losses,
                   output_dict,
                   high_batch,
                   gt_label = None,
                   get_individual_losses = False, 
                   scale_factor=8):
    """
    Calculates and returns the total optimization loss and optionally individual losses.

    This function computes various loss components based on the provided loss functions 
    and input data. It aggregates these losses into a total optimization loss, which 
    can be used for model training. Optionally, it can also return a dictionary of 
    individual loss values.

    Args:
        all_losses (dict): A dictionary containing loss functions. The keys are the 
            names of the losses (e.g., 'pixel_loss', 'perceptual_loss'), and the values 
            are the corresponding callable loss functions.
        output_dict (dict): A dictionary containing the model's output data. It may 
            include keys such as 'output', 'side_align', 'out_side', and 'side_darkir', 
            depending on the losses being calculated.
        high_batch (torch.Tensor): The ground truth high-resolution batch used for 
            calculating the losses.
        get_individual_losses (bool, optional): If True, the function returns a 
            dictionary of individual loss values in addition to the total optimization 
            loss. Defaults to False.
        scale_factor (int, optional): A scaling factor used for certain loss functions 
            (e.g., 'enhance_loss' and 'yuv_loss'). Defaults to 8.

    Returns:
        float: The total optimization loss, which is the sum of all calculated losses.
        dict (optional): A dictionary of individual loss values, where the keys are 
            the loss names and the values are the corresponding loss values. This is 
            returned only if `get_individual_losses` is True.
    """


    losses = dict()
    optim_loss = 0
    if 'pixel_loss' in all_losses:
        l_pixel, l_total = all_losses['pixel_loss'](output_dict['output'], high_batch)
        losses['pixel_loss'] = l_pixel
        optim_loss += l_total

    if 'edge_loss' in all_losses and 'out_side' in output_dict: 
        l_edge, l_total = all_losses['edge_loss'](output_dict['out_side'], high_batch, scale_factor = scale_factor)
        losses['edge_loss'] = l_edge
        optim_loss += l_total
    
    if 'fourier_loss' in all_losses and 'out_side' in output_dict:
        l_frequency, l_total = all_losses['fourier_loss'](output_dict['out_side'], high_batch, scale_factor = scale_factor)
        losses['fourier_loss'] = l_frequency
        optim_loss += l_total
    
    if 'fourier_out_loss' in all_losses:
        l_frequency, l_total = all_losses['fourier_out_loss'](output_dict['output'], high_batch, scale_factor = 1)
        losses['fourier_out_loss'] = l_frequency
        optim_loss += l_total

    if 'enhance_loss' in all_losses and 'out_side' in output_dict:
        l_enhance, l_total = all_losses['enhance_loss'](output_dict['out_side'], high_batch, scale_factor = scale_factor) 
        losses['enhance_loss'] = l_enhance
        optim_loss += l_total
    
    if 'enhance_out_loss' in all_losses:
        l_enhance, l_total = all_losses['enhance_out_loss'](output_dict['output'], high_batch, scale_factor = 1) 
        losses['enhance_out_loss'] = l_enhance
        optim_loss += l_total

    if 'classify_loss' in all_losses and 'pred_labels' in output_dict:
        l_class, l_total = all_losses['classify_loss'](gt_label,output_dict['pred_labels'])
        losses['classify_loss'] = l_class
        optim_loss += l_total

    if 'multiguidance_loss' in all_losses and 'out_side' in output_dict:
        l_multiguidance, l_total = all_losses['multiguidance_loss'](output_dict['out_side'], high_batch)
        losses['multiguidance_loss'] = l_multiguidance
        optim_loss += l_total

    if get_individual_losses:
        return optim_loss, losses
    else:
        return optim_loss

__all__ = ['create_loss', 'calculate_loss', 'SSIM', 'VGGLoss', 'create_unpaired_losses']
