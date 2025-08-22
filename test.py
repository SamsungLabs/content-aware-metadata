import os
import cv2
import copy
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import UNet
from dataset_raw import DatasetRAWTest
from util import calc_psnr, get_kmap_from_prob
from pytorch_msssim import ssim


parser = argparse.ArgumentParser(description='Full fixed samples')
parser.add_argument(
    '--data-dir', default='/datadir', type=str, help='folder of training and validation images')
parser.add_argument(
    '--file-type', default='png', type=str, help='image file type (png or tif)')
parser.add_argument(
    '--checkpoint-dir', default='/checkpointdir', type=str, help='folder of checkpoint')
parser.add_argument(
    '--num-iters', type=int, default=10, help='number of iterations')
parser.add_argument(
    '--patch-size', type=int, default=128, help='patch size')
parser.add_argument(
    '--init-features', type=int, default=32, help='init_features of UNet')
parser.add_argument(
    '--k', type=float, default=1.5625, help='percentage of samples to pick')
parser.add_argument(
    '--batch-size', type=int, default=1, help='batch size (DO NOT CHANGE)')
parser.add_argument(
    '--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument(
    '--extra', type=str, default='', help='extra identifier for save folder name')

args = parser.parse_args()

grid_size = args.patch_size ** 2 * args.k / 100
if np.sqrt(grid_size) != int(np.sqrt(grid_size)):
    print('Warning: superpixel grid seeds may not match the percentage of samples.')
grid_size = args.patch_size // int(np.sqrt(grid_size))

savefoldername=('k' + str(args.k)
    + '_lr' + str(args.lr)
    + '_i' + str(args.num_iters)
    + '_b' + str(args.batch_size)
    + '_ft' + str(args.init_features)
    + args.extra
)

root = os.path.join('./outputs/', savefoldername)
if not os.path.exists(root):
    os.makedirs(root)

image_datasets = {x: DatasetRAWTest(os.path.join(args.data_dir,x), ftype=args.file_type)
                  for x in ['test']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                             shuffle=False, num_workers=0)
              for x in ['test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sampler = UNet(in_channels=6, out_channels=9, init_features=args.init_features, sigmoid=False)
sampler.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'best_sampler.pt')))
sampler = sampler.to(device)
sampler.eval()

reconstructor = UNet(7, out_channels=3, init_features=args.init_features)
reconstructor.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'best_reconstructor.pt')))
reconstructor = reconstructor.to(device)
reconstructor.eval()

avg_psnr = 0
avg_ssim = 0
for i, (inputs, targets) in enumerate(dataloaders['test']):

    # forward: sampler
    outputs = sampler(torch.cat((inputs, targets), 1))
    prob = F.softmax(outputs, 1)

    # sampling process
    kmap = get_kmap_from_prob(prob, grid_size)

    inputs2 = torch.cat([inputs, kmap * targets, kmap], dim=1)

    online_sampler = copy.deepcopy(reconstructor)
    online_sampler.eval()
    optimizer = optim.Adam(online_sampler.parameters(), lr=args.lr)

    for _ in range(args.num_iters):
        optimizer.zero_grad()
        outputs = online_sampler(inputs2.detach())
        loss = (kmap.detach() * (outputs - targets).abs()).mean()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        outputs = online_sampler(inputs2)

    # evaluation metrics
    psnrout = calc_psnr(outputs, targets)
    ssimout = ssim((outputs * 65535).floor(), (targets * 65535).floor(), data_range=65535, size_average=True)
    avg_psnr += psnrout.item()
    avg_ssim += ssimout.item()

    # save images
    # this is only for visualization
    inputs = inputs.squeeze().permute(1, 2, 0).cpu().detach().numpy()
    inputs = cv2.cvtColor(inputs, cv2.COLOR_RGB2BGR)
    targets = targets.squeeze().permute(1, 2, 0).cpu().detach().numpy()
    targets = cv2.cvtColor(targets, cv2.COLOR_RGB2BGR)
    outputs = outputs.squeeze().permute(1, 2, 0).cpu().detach().numpy()
    outputs = cv2.cvtColor(outputs, cv2.COLOR_RGB2BGR)
    kmap = kmap.squeeze().cpu().detach().numpy()
    cv2.imwrite(os.path.join(root, '{:07d}_in.png'.format(i)), (inputs * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(root, '{:07d}_gt.png'.format(i)), (targets ** (1 / 2.2) * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(root, '{:07d}_out.png'.format(i)), (outputs ** (1 / 2.2) * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(root, '{:07d}_kmap.png'.format(i)), ((1 - kmap) * 255).astype(np.uint8))

avg_psnr /= len(image_datasets['test'])
avg_ssim /= len(image_datasets['test'])
print('PSNR: {:4f}'.format(avg_psnr))
print('SSIM: {:4f}'.format(avg_ssim))
