import os
from tqdm import tqdm
from options.options import parse
import argparse
import re

parser = argparse.ArgumentParser(description="Script for test")
parser.add_argument('-p', '--config', type=str, default='./options/test/Baseline.yml', help='Config file of prediction')
parser.add_argument('-c', '--checkpoints_path', type=str, default='./models/bests/FReSHIR.pt',
                help="Checkpoints path")
args = parser.parse_args()

opt = parse(args.config)

# PyTorch library
import torch
import torch.distributed as dist

from data.datasets.datapipeline import *
from archs import *
from losses import *
from data import *
from scripts.test_schedule import eval_model

def load_model(local_rank, model, path_weights, global_rank=1):
    """
    Loads a model's weights from a specified checkpoint file and maps them to the appropriate device.
    """
    map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
    checkpoints = torch.load(path_weights, map_location=map_location, weights_only=False)
    weights = checkpoints['model_state_dict']
    # new_weights = {re.sub(r'(encoders\.\d+)\.(\d+\.)', r'\1.modules_list.\2', k): v for k, v in weights.items()}
    # print(weights.keys())
    new_weights = {k.replace('module.', ''): v for k, v in weights.items()}
    # print(new_weights.keys())
    if global_rank == 0: print('Values on epoch:', checkpoints['epoch'])
    model.load_state_dict(new_weights)
    if global_rank == 0: print('Loaded weights correctly')
    return model

def run_evaluation():

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
    model, _, _= create_model(opt['network'], local_rank=local_rank, global_rank=global_rank)

    # load checkpoints of the model
    model = load_model(local_rank, model, args.checkpoints_path, global_rank=global_rank)
    metrics_eval = {}

    # eval phase
    model.eval()
    metrics_eval, _, time_eval = eval_model(model, test_loader, metrics_eval, local_rank=local_rank, 
                               side_loss=opt['network']['guiding_branches'],
                               world_size=world_size)
    
    # print some results
    if global_rank == 0:
        if type(next(iter(metrics_eval.values()))) == dict:
            for key, metric_eval in metrics_eval.items():
                print(f" \t {key} --- PSNR: {metric_eval['valid_psnr']}, PSNR-Y: {metric_eval['valid_psnrY']}, SSIM: {metric_eval['valid_ssim']}, LPIPS: {metric_eval['valid_lpips']}")
        else:
            print(f" \t {opt['datasets']['name']} --- PSNR: {metrics_eval['valid_psnr']}, PSNR-Y: {metrics_eval['valid_psnrY']}, SSIM: {metrics_eval['valid_ssim']}, LPIPS: {metrics_eval['valid_lpips']}")
        print(f" \t Time for evaluation: {time_eval} ms")
    
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == '__main__':
    run_evaluation()