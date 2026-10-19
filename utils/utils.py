import os
import torch.distributed as dist
import numpy as np
import torch
import random


def setup(rank, world_size, port = '12355', master_addr = 'localhost'):
    """
    Sets up the distributed training environment.
    This function initializes the process group for distributed training using the NCCL backend.
    It sets the necessary environment variables for the master address and port, and ensures
    synchronization across processes by calling a barrier.
    Args:
        rank (int): The rank of the current process within the distributed group.
        world_size (int): The total number of processes participating in the distributed training.
        port (str, optional): The port number used for communication between processes. Defaults to '12355'.
        master_addr (str, optional): The address of the master node. Defaults to 'localhost'.
    Raises:
        RuntimeError: If the process group fails to initialize or if there are issues with synchronization.
    """
    os.environ['MASTER_PORT'] = port
    os.environ['MASTER_ADDR'] = master_addr

    # print('before starting the process')
    dist.init_process_group("nccl", init_method="env://",rank=rank, world_size=world_size)

def cleanup():
    """
    Cleans up the distributed processing environment by destroying the process group.

    This function is typically used in distributed computing setups to release resources
    and properly shut down the distributed process group created by the `torch.distributed` module.
    """
    dist.destroy_process_group()

def create_path_models(opt):
    """
    Generates and returns three paths related to model storage based on the provided options.
    Parameters:
        opt (dict): A dictionary containing the following keys:
            - 'path_resume' (str): The base path for the model.
            - 'path_save' (str, optional): A new path for the model.
            - 'best' (str): The directory where the best model path will be created.
    Returns:
        tuple: A tuple containing:
            - PATH_MODEL (str): The base path for the model.
            - NEW_PATH_MODEL (str): The new path for the model, or the base path if 'path_save' is not provided.
            - BEST_PATH_MODEL (str): The path for the best model, constructed by combining 'best' and the basename of NEW_PATH_MODEL.
    Note:
        - When resuming the training, you will need to have different paths for PATH_MODEL and NEW_PATH_MODEL. Otherwise, the previous
          weights will be overwritten.
        - When training from scratch, PATH_MODEL and NEW_PATH_MODEL can be the same.
    """
    PATH_MODEL     = opt['path_resume']
    NEW_PATH_MODEL = opt['path_save']
    BEST_PATH_MODEL = opt['best']

    return PATH_MODEL, NEW_PATH_MODEL, BEST_PATH_MODEL



def set_random_seed(seed):
    """
    Sets the random seed for reproducibility across various libraries.

    This function ensures that the random number generators in Python's `random` module,
    NumPy, and PyTorch produce deterministic results by setting their seeds to the
    specified value. It also configures PyTorch to use the same seed for all CUDA devices.

    Args:
        seed (int): The seed value to set for all random number generators.

    Raises:
        ValueError: If the provided seed is not an integer.
    """
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':

    pass
