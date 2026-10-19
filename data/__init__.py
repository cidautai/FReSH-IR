from .datasets.utils import shuffle_sampler
from .datasets.dataset_allweather import main_dataset_allweather
from .datasets.dataset_test import test_dataset_allweather
from .datasets.dataset_snow_real import main_dataset_snow_real


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

def check_if_defined(opt, key, value, step=0):
    '''
    Check if a key is defined in the provided options dictionary and return its corresponding value.

    Parameters:
        opt (dict): The dictionary containing options.
        key (str): The key to check in the dictionary.
        value (Any): The default value to return if the key is not found in the dictionary.
        step (int, optional): The index to use if the value associated with the key is a list. Defaults to 0.

    Returns:
        Any: The value associated with the key in the dictionary if it exists and is not a list,
             or the value at the specified index if it is a list. If the key is not found,
             the provided default value is returned.
    '''
    if key not in opt:
        return value
    elif key in opt and type(opt[key]) != list:
        return opt[key]
    else:
        return adjust_list_length(opt[key], step+1)[step]   # we adjust the length of the list to be the same as the number of steps (prevent errors in the yaml file)

def create_data(global_rank, world_size, opt, step=0):
    """
    Creates and initializes data loaders for training and testing datasets based on the provided configuration.
    Parameters:
        global_rank (int): The rank of the current process in distributed training.
        world_size (int): The total number of processes in distributed training.
        opt (dict): A dictionary containing dataset configuration options, typically parsed from a YAML file.
            - name (str): The name of the dataset to use.
            - train (dict): Configuration for the training dataset.
        step (int, optional): The current training step, used for dynamic configuration. Defaults to `0`.
    Returns:
        tuple: A tuple containing:
            - train_loader (DataLoader): Data loader for the training dataset.
            - test_loader (DataLoader): Data loader for the validation dataset.
            - samplers (optional): Samplers used for distributed data loading, if applicable.
    Raises:
        NotImplementedError: If the specified dataset name in `opt['name']` is not implemented.
    """
    '''
    opt: a dictionary from the yaml config key datasets
    '''
    name = opt['name']
    train_path=opt['train']['train_path']
    test_path = opt['val']['test_path']
    verbose = opt['train']['verbose']
    batch_size_test = opt['val']['batch_size_test']

    # the following parameters are only defined in the train set
    batch_size_train = check_if_defined(opt['train'], 'batch_size_train', world_size, step=step) // world_size
    flips = check_if_defined(opt['train'], 'flips', False,step=0)
    cropsize = check_if_defined(opt['train'], 'cropsize', None, step=step)
    num_workers = check_if_defined(opt['train'], 'num_workers', 1, step=0)
    crop_type = check_if_defined(opt['train'], 'crop_type', 'Random', step=0)

    if global_rank != 0:
        verbose = False
    samplers = None

    if name == 'AllWeather':
        train_loader, test_loaders, samplers = main_dataset_allweather(path_train=train_path,
                                                                   path_test=test_path,
                                                                   batch_size_train=batch_size_train,
                                                                   batch_size_test=batch_size_test,
                                                                   verbose=verbose,
                                                                   cropsize=cropsize,
                                                                   num_workers=num_workers,
                                                                   crop_type=crop_type,
                                                                   world_size=world_size,
                                                                   global_rank = global_rank,
                                                                   flips=flips)

    elif name == 'Test':
        train_loader, test_loaders, samplers = test_dataset_allweather(path_train=train_path,
                                                                   path_test=test_path,
                                                                   batch_size_train=batch_size_train,
                                                                   batch_size_test=batch_size_test,
                                                                   verbose=verbose,
                                                                   cropsize=cropsize,
                                                                   num_workers=num_workers,
                                                                   crop_type=crop_type,
                                                                   world_size=world_size,
                                                                   global_rank = global_rank,
                                                                   flips=flips)

    elif name == 'Snow Real':
        train_loader, test_loaders, samplers = main_dataset_snow_real(path_train=train_path,
                                                                   path_test=test_path,
                                                                   batch_size_train=batch_size_train,
                                                                   batch_size_test=batch_size_test,
                                                                   verbose=verbose,
                                                                   cropsize=cropsize,
                                                                   num_workers=num_workers,
                                                                   crop_type=crop_type,
                                                                   world_size=world_size,
                                                                   global_rank = global_rank,
                                                                   flips=flips)
    else:
        raise NotImplementedError(f'{name} is not implemented')

    if global_rank ==0: print(f'Using {name} Dataset')

    return train_loader, test_loaders, samplers


__all__ = ['create_data', 'shuffle_sampler']
