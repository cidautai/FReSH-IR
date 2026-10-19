import os, sys
import numpy as np
from PIL import Image
from options.options import parse
import argparse

parser = argparse.ArgumentParser(description="Script for inference on a folder of images")
parser.add_argument('-p', '--config', type=str, default='./options/predict/Baseline.yml', help='Config file')
parser.add_argument('-i', '--inp_path', type=str, required=True, help="Input folder path")
parser.add_argument('-o', '--out_path', type=str, default='./results', help="Output folder path")
parser.add_argument('-c', '--checkpoints_path', type=str, default='./models/bests/FReSHIR.pt', help="Checkpoint path")
args = parser.parse_args()
path_options = args.config
opt = parse(path_options)

import torch
from tqdm import tqdm
from torchvision.transforms import Resize

import torch.nn.functional as F
import torchvision.transforms as transforms
from archs import create_model

pil_to_tensor = transforms.ToTensor()

def load_image(path):
    img = Image.open(path).convert('RGB')
    img_tensor = pil_to_tensor(img)
    return img_tensor

def save_tensor(tensor, path):
    tensor = tensor.squeeze(0)
    array = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(array)
    img.save(path)

def pad_tensor(tensor, multiple=8):
    _, _, H, W = tensor.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h), value=0)
    return tensor

def load_model(rank, model, path_weights, global_rank=1):
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    checkpoints = torch.load(path_weights, map_location=map_location, weights_only=False)
    if 'module.' in list(checkpoints['model_state_dict'].keys())[0]:
        weights = {k.replace('module.', ''): v for k, v in checkpoints['model_state_dict'].items()}
    else:
        weights = checkpoints['model_state_dict']
    model.load_state_dict(weights, strict=True)
    if global_rank == 0:
        print('Loaded weights correctly')
    return model

resize = opt['Resize']
scale  = opt['scale']

def predict_folder():

    global_rank = 0
    local_rank = 0

    model, _, _ = create_model(opt['network'], local_rank=local_rank, global_rank=global_rank)
    model = load_model(local_rank, model, path_weights=args.checkpoints_path, global_rank=global_rank)

    PATH_IMAGES = args.inp_path
    PATH_RESULTS = args.out_path

    os.makedirs(PATH_RESULTS, exist_ok=True)

    test_set = [os.path.join(PATH_IMAGES, path) for path in os.listdir(PATH_IMAGES)
                if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]

    pbar = tqdm(total=len(test_set))

    model.eval()

    for img_name in test_set:

        img_tensor = load_image(img_name)
        tensor = img_tensor.to(local_rank).unsqueeze(0)
        _, _, H, W = tensor.shape

        downsample = Resize([int(dim / scale) for dim in (H, W)]) if resize else torch.nn.Identity()
        tensor = downsample(tensor)
        tensor = pad_tensor(tensor)

        with torch.no_grad():
            output_dict = model(tensor)

        upsample = Resize((H, W)) if resize else torch.nn.Identity()
        output = upsample(output_dict['output'])
        output = torch.clamp(output, 0., 1.)
        output = output[:, :, :H, :W]

        save_tensor(output, os.path.join(PATH_RESULTS, os.path.basename(img_name)))
        pbar.update(1)

        del output
        torch.cuda.empty_cache()

    pbar.close()
    print('Finished inference!')

if __name__ == '__main__':
    predict_folder()
