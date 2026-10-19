import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.nn as nn


from .FReSHIR import FReSHIR


def freeze_parameters(model,
                      substring: str):
    """
    Freezes the parameters of a given model that do not contain a specified substring in their names.

    This function iterates through all the named parameters of the provided model and sets the
    `requires_grad` attribute to `False` for those parameters whose names do not include the
    specified substring. This is useful for fine-tuning or transfer learning scenarios where
    only specific parts of the model need to be updated during training.

    Args:
        model: The model whose parameters are to be selectively frozen. It should have a
               `named_parameters` method, such as PyTorch models.
        substring (str): The substring to search for in parameter names. Parameters containing
                         this substring in their names will remain trainable.

    Returns:
        The model with updated `requires_grad` attributes for its parameters.
    """

    for name, param in model.named_parameters():
        if substring not in name:
            param.requires_grad = False
    return model

def create_model(opt, local_rank, global_rank = 1):
    """
    Creates and initializes a deep learning model based on the provided configuration.
    This function selects a specific model architecture based on the `name` key in the
    `opt` dictionary and initializes it with the corresponding parameters. It also
    handles device placement, distributed data parallelism (DDP), and optionally computes
    the model's computational complexity and parameter count for the rank 0 process.
    Args:
        opt (dict): The network key from the config file. Needed parameters to instantiate the model
            can be defined in the config file when needed. They need to be passes to the model
            as arguments.
        local_rank (int): The rank of the current process in distributed training. Used for device placement
            and logging.
        global_rank (int, optional): The global rank of the process in distributed training.
    Returns:
        tuple: A tuple containing:
            - model (torch.nn.Module): The initialized model wrapped in DistributedDataParallel (DDP).
            - macs (str or None): The computational complexity of the model in MACs (if global_rank is 0).
            - params (str or None): The number of parameters in the model (if global_rank is 0).
        MACs are calculated using an input tensor of size [9,256,256] for all models.
    Raises:
        NotImplementedError: If the specified model name in `opt['name']` is not supported.
    """
    # print(rank)
    name = opt['name']

    if name == 'FReSHIR':
        model = FReSHIR(img_channel=opt['img_channels'],
                        width=opt['width'],
                        middle_blk_num=opt['middle_blk_num_enc'],
                        enc_blk_nums=opt['enc_blk_nums'],
                        dec_blk_nums=opt['dec_blk_nums'],
                        cutoff = opt['cutoff'],
                        dilations = opt['dilations'])

    else:
        raise NotImplementedError('This network is not implemented')
    if global_rank ==0:
        from ptflops import get_model_complexity_info
        print(f'Using {name} network')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        input_size = (3, 256, 256) # the image channels must be the same as your network input!
        macs, params = get_model_complexity_info(model, input_size, print_per_layer_stat = False)
        print(f'Computational complexity at {input_size}: {macs}')
        print('Number of parameters: ', params)
        dummy_input = torch.randn(1, *input_size).to(device)
        model.eval()
        # Warm-up
        for _ in range(10):
            _ = model(dummy_input)

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        n_runs = 100
        timings = []

        with torch.no_grad():
            for _ in range(n_runs):
                starter.record()
                _ = model(dummy_input)
                ender.record()

                torch.cuda.synchronize()

                curr_time = starter.elapsed_time(ender)  # milliseconds
                timings.append(curr_time)

        avg_runtime = sum(timings) / len(timings)
        print(f"Average runtime per inference: {avg_runtime:.3f} ms")
    else:
        macs, params = None, None

    # print(rank)
    model.to(local_rank)

    if 'find_unused_params' in opt: find_unused_params = opt['find_unused_params']
    else: find_unused_params = False
    world_size = torch.cuda.device_count()
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=find_unused_params)

    return model, macs, params

def create_optim_scheduler(opt, model):
    """
    Creates and returns an optimizer and its corresponding learning rate scheduler.
    This function initializes an AdamW optimizer with the parameters of the given model
    that require gradients. It also sets up a learning rate scheduler based on the
    specified learning rate scheme in the configuration.
    Args:
        opt (dict): A dictionary containing training configuration parameters (keys from train).
        model (torch.nn.Module): The PyTorch model whose parameters will be optimized.
    Returns:
        tuple: A tuple containing:
            - optim (torch.optim.Optimizer): The initialized AdamW optimizer.
            - scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler
              corresponding to the specified scheme.
    Raises:
        NotImplementedError: If the specified learning rate scheme is not implemented.
    """
    '''
    Returns the optim and its scheduler.
    opt: a dictionary of the yaml config file with the train key
    '''
    optim = torch.optim.AdamW( filter(lambda p: p.requires_grad, model.parameters()) ,
                            lr = opt['lr_initial'],
                            weight_decay = opt['weight_decay'],
                            betas = opt['betas'])
    T_max = int(sum(opt['epochs'])) # total number of epochs in the multi-step list of epochs

    if opt['lr_scheme'] == 'CosineAnnealing':
        scheduler = CosineAnnealingLR(optim, T_max=T_max, eta_min=opt['eta_min'])
    else:
        raise NotImplementedError('scheduler not implemented')

    return optim, scheduler

def load_weights(model, model_weights, global_rank = 1):
    """
    Loads the weights of a pretrained model into a new model, ensuring that only
    the weights with matching keys between the old and new models are updated.
    This function is useful when transferring weights between models with
    potentially different architectures or when fine-tuning a model. It will print
    how many keys were loaded and how many keys were not loaded.
    Args:
        model (torch.nn.Module): The new model instance where the weights will be loaded.
        old_weights (dict): A dictionary containing the pretrained weights, typically
                            obtained from `state_dict()` of another model.
    Returns:
        torch.nn.Module: The new model with the updated weights.
    """
    '''
    Loads the weights of a pretrained model, picking only the weights that are
    in the new model.
    '''
    new_weights = model.state_dict()

    # Calculate and print key statistics
    total_checkpoint_keys = len(model_weights)
    total_model_keys = len(new_weights)
    matching_keys = len(set(model_weights.keys()) & set(new_weights.keys()))

    if global_rank == 0:
        print(f"Keys in checkpoint: {total_checkpoint_keys}")
        print(f"Keys in model state dict: {total_model_keys}")
        print(f"Matching keys: {matching_keys}")

    new_weights.update({k: v for k, v in model_weights.items() if k in new_weights})

    model.load_state_dict(new_weights)

    return model

def load_optim(optim, optim_weights):
    """
    Loads the state of an optimizer by updating its weights with the provided
    optimizer weights, while ensuring only the matching keys between the current
    optimizer state and the provided weights are updated.

    This function is useful when loading a pre-trained optimizer state into a
    new model, ensuring compatibility by only updating the weights that exist
    in both the new optimizer and the provided state.

    Args:
        optim (torch.optim.Optimizer): The optimizer whose state is to be updated.
        optim_weights (dict): A dictionary containing the state of the optimizer
            to be loaded, typically obtained from a checkpoint.

    Returns:
        torch.optim.Optimizer: The optimizer with its state partially updated
        based on the provided weights.
    """
    '''
    Loads the values of the optimizer picking only the weights that are in the new model.
    '''
    optim_new_weights = optim.state_dict()
    optim_new_weights.update({k:v for k, v in optim_weights.items() if k in optim_new_weights})
    return optim

def resume_model(model,
                 optim,
                 scheduler,
                 path_resume,
                 local_rank, global_rank = 1,
                 resume:str=None):
    """
    Resumes the training process by loading model, optimizer, and scheduler states from a checkpoint.
    This function is designed to handle the resumption of training, particularly in distributed training setups.
    It loads the model and optimizer weights from a specified checkpoint file if the `resume` flag is set to True.
    If not resuming, it initializes the training process from scratch.
    Args:
        model (torch.nn.Module): The model to load weights into.
        optim (torch.optim.Optimizer): The optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        path_model (str): Path to the checkpoint file containing saved states.
        local_rank (int): The rank of the current process in distributed training.
        resume (str, optional): A flag indicating whether to resume training from a checkpoint. Defaults to None.
        global_rank (int, optional): The global rank of the process in distributed training. Defaults to 1.
    Returns:
        tuple: A tuple containing:
            - model (torch.nn.Module): The model with loaded weights (if resuming).
            - optim (torch.optim.Optimizer): The optimizer with loaded state (if resuming).
            - scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler (unchanged).
            - start_epochs (int): The starting epoch for training (0 if not resuming).
    """
    '''
    Returns the loaded weights of model and optimizer if resume flag is True
    '''
    map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank} # location where the weights need to be mapped (needed for Distributed training)
    if resume:
        checkpoints = torch.load(path_resume, map_location=map_location, weights_only=False) # load the checkpoints of resumed training
        weights = checkpoints['model_state_dict'] # pick the weights of the model
        model = load_weights(model, model_weights=weights, global_rank=global_rank) # load the weights into the model
        optim = load_optim(optim, optim_weights = checkpoints['optimizer_state_dict'])
        if global_rank == 0: print('Loaded weights')
    else:
        if global_rank==0: print('Starting from zero the training')

    return model, optim, scheduler

def save_checkpoint(model, optim, scheduler, metrics_train, paths, global_rank = None):
    """
    Saves the model checkpoint after each epoch and updates the best model based on PSNR (Peak Signal-to-Noise Ratio).
    This function saves the current state of the model, optimizer, and scheduler to a specified file path.
    Additionally, it compares the current training PSNR with the best PSNR recorded so far. If the current PSNR
    is higher, it updates the best model checkpoint and the best PSNR value.
    Args:
        model (torch.nn.Module): The model to save.
        optim (torch.optim.Optimizer): The optimizer associated with the model.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        metrics_train (dict): A dictionary containing training metrics, including:
            - 'epoch' (int): The current epoch number.
            - 'train_loss' (float): The training loss for the current epoch.
            - 'train_psnr' (float): The PSNR for the current epoch.
            - 'best_psnr' (float): The best PSNR recorded so far.
        paths (dict): A dictionary containing file paths for saving the model:
            - 'new' (str): Path to save the current model checkpoint.
            - 'best' (str): Path to save the best model checkpoint.
        global_rank (int, optional): The rank of the process in distributed training. If not rank 0, the function
                                returns without saving. Defaults to None.
    Returns:
        float: The updated best PSNR value.
    Notes:
        - The function only saves the model if `rank` is 0 or not specified.
        - If an error occurs during saving, the function prints an error message.
    """

    best_psnr = metrics_train['best_psnr']
    if global_rank!=0:
        return best_psnr

    weights = model.state_dict()

    model_to_save = {
        'epoch': metrics_train['epoch'],
        'model_state_dict': weights,
        'optimizer_state_dict': optim.state_dict(),
        'loss': metrics_train['train_loss'],
        'scheduler_state_dict': scheduler.state_dict()
    }

    try:
        torch.save(model_to_save, paths['new'])
        # Save best model if new valid_psnr is higher than the best one
        if metrics_train['train_psnr'] >= metrics_train['best_psnr']:
            torch.save(model_to_save, paths['best'])
            metrics_train['best_psnr'] = metrics_train['train_psnr']  # update best psnr
    except Exception as e:
        print(f"Error saving model: {e}")
    return metrics_train['best_psnr']

def save_checkpoint_class(model, optim, scheduler, metrics_eval, metrics_train, paths, global_rank = 1):

    '''
    Save the .pt of the model after each epoch.

    '''
    last_final_score = metrics_train['final_score']
    final_score = metrics_eval['final_score']
    if global_rank != 0:
        return None

    if type(next(iter(metrics_eval.values()))) != dict:
        metrics_eval = {'metrics': metrics_eval}

    weights = model.state_dict()

    # Save the model after every epoch
    model_to_save = {
        'epoch': metrics_train['epoch'],
        'model_state_dict': weights,
        'optimizer_state_dict': optim.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }

    try:
        torch.save(model_to_save, paths['new'])

        # Save best model if new valid_psnr is higher than the best one
        if final_score >= last_final_score:
            torch.save(model_to_save, paths['best'])
            last_final_score = final_score  # update best psnr
    except Exception as e:
        print(f"Error saving model: {e}")
    return last_final_score

__all__ = ['create_model', 'resume_model', 'create_optim_scheduler', 'save_checkpoint',
           'load_optim', 'load_weights', 'save_checkpoint', 'save_checkpoint_class']


