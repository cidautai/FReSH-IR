import os
import numpy as np
from options.options import parse
import argparse

parser = argparse.ArgumentParser(description="Script for inference on a video file")
parser.add_argument('-p', '--config', type=str, default='./options/predict/Baseline.yml', help='Config file')
parser.add_argument('-i', '--inp_path', type=str, required=True, help="Input video path")
parser.add_argument('-o', '--out_path', type=str, default='./results/output.mp4', help="Output video path")
parser.add_argument('-c', '--checkpoints_path', type=str, default='./models/bests/FReSHIR.pt', help="Checkpoint path")
args = parser.parse_args()
path_options = args.config
opt = parse(path_options)

import torch
import cv2
from tqdm import tqdm

import torch.nn.functional as F
import torchvision.transforms as transforms
from archs import create_model

pil_to_tensor = transforms.ToTensor()

def frame_to_tensor(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = pil_to_tensor(frame_rgb)
    return tensor

def tensor_to_frame(tensor):
    array = (tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    frame_bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    return frame_bgr

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

def predict_video():

    global_rank = 0
    local_rank = 0

    model, _, _ = create_model(opt['network'], local_rank=local_rank, global_rank=global_rank)
    model = load_model(local_rank, model, path_weights=args.checkpoints_path, global_rank=global_rank)
    model.eval()

    cap = cv2.VideoCapture(args.inp_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {args.inp_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.out_path, fourcc, fps, (width, height))

    pbar = tqdm(total=total_frames)

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            H, W = frame.shape[:2]
            tensor = frame_to_tensor(frame).to(local_rank).unsqueeze(0)

            if resize:
                new_H, new_W = int(H / scale), int(W / scale)
                tensor = torch.nn.functional.interpolate(tensor, size=(new_H, new_W), mode='bilinear', align_corners=False)

            tensor = pad_tensor(tensor)

            output_dict = model(tensor)
            output = output_dict['output']
            output = torch.clamp(output, 0., 1.)

            if resize:
                output = torch.nn.functional.interpolate(output, size=(H, W), mode='bilinear', align_corners=False)

            output = output[:, :, :H, :W]
            result_frame = tensor_to_frame(output)
            out.write(result_frame)

            del output
            torch.cuda.empty_cache()
            pbar.update(1)

    cap.release()
    out.release()
    pbar.close()
    print(f'Finished! Output saved to {args.out_path}')

if __name__ == '__main__':
    predict_video()
