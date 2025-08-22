import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from model import UNet
from dataset_raw import DatasetRAW
from util import poolfeat, upfeat, calc_psnr, get_kmap_from_prob

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Learn samples: full float')
parser.add_argument(
    '--data-dir', default='/datadir', type=str, help='folder of training and validation images')
parser.add_argument(
    '--file-type', default='png', type=str, help='image file type (png or tif)')
parser.add_argument(
    '--num-epochs', type=int, default=120, help='number of epochs')
parser.add_argument(
    '--tboard-freq', type=int, default=200, help='frequency of writing to tensorboard')
parser.add_argument(
    '--k', type=float, default=1.5625, help='percentage of samples to pick')
parser.add_argument(
    '--patch-size', type=int, default=128, help='patch size')
parser.add_argument(
    '--init-features', type=int, default=32, help='init_features of UNet')
parser.add_argument(
    '--stride', type=int, default=52, help='stride when cropping patches')
parser.add_argument(
    '--batch-size', type=int, default=128, help='batch size')
parser.add_argument(
    '--lr', type=float, default=0.001, help='learning rate')

parser.add_argument(
    '--lambda-slic', type=float, default=0.0001, help='weight of SLIC loss')
parser.add_argument(
    '--slic-alpha', type=float, default=0.2, help='alpha in SLIC loss')
parser.add_argument(
    '--slic-m', type=float, default=10.0, help='weight m in SLIC loss')

parser.add_argument(
    '--lambda-meta', type=float, default=0.01, help='weight of meta loss')
parser.add_argument(
    '--inner-lr', type=float, default=0.001, help='learning rate of inner loop')
parser.add_argument(
    '--inner-steps', type=int, default=5, help='number of update steps in inner loop')

parser.add_argument(
    '--extra', type=str, default='', help='extra identifier for save folder name')

args = parser.parse_args()

grid_size = args.patch_size ** 2 * args.k / 100
if np.sqrt(grid_size) != int(np.sqrt(grid_size)):
    print('Warning: superpixel grid seeds may not match the percentage of samples.')
grid_size = args.patch_size // int(np.sqrt(grid_size))


savefoldername = ('k' + str(args.k)
    + '_lr' + str(args.lr)
    + '_e' + str(args.num_epochs)
    + '_b' + str(args.batch_size)
    + '_p' + str(args.patch_size)
    + '_s' + str(args.stride)
    + '_ft' + str(args.init_features)
    + '_ls' + str(args.lambda_slic)
    + '_sa' + str(args.slic_alpha)
    + '_sm' + str(args.slic_m)
    + '_lm' + str(args.lambda_meta)
    + '_il' + str(args.inner_lr)
    + '_is' + str(args.inner_steps)
    + '_' + args.extra
)

writer = SummaryWriter(os.path.join('./logs', savefoldername))
mysavepath = os.path.join('./models', savefoldername)

if not(os.path.exists(mysavepath) and os.path.isdir(mysavepath)):
    os.makedirs(mysavepath)

image_datasets = {x: DatasetRAW(os.path.join(args.data_dir, x), args.batch_size, args.patch_size,args.stride, to_gpu=False, ftype=args.file_type)
                  for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                              shuffle=True, num_workers=0)
              for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

epoch_loss = {x: 0.0 for x in ['train', 'val']}
epoch_recon = {x: 0.0 for x in ['train', 'val']}
epoch_slic = {x: 0.0 for x in ['train', 'val']}
epoch_recon_meta = {x: 0.0 for x in ['train', 'val']}
epoch_psnr = {x: 0.0 for x in ['train', 'val']}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sampler = UNet(in_channels=6, out_channels=9, init_features=args.init_features, sigmoid=False)
sampler = sampler.to(device)
sampler = nn.DataParallel(sampler)

reconstructor = UNet(in_channels=7, out_channels=3, init_features=args.init_features)
reconstructor = reconstructor.to(device)
reconstructor = nn.DataParallel(reconstructor)

params = list(sampler.parameters()) + list(reconstructor.parameters())
optimizer = optim.Adam(params, lr=args.lr)

# function to generate random mask
def genmask_random(kp, img):
    B, C, H, W = img.size()
    k = int(H * W * kp / 100)
    kmapo = torch.zeros_like(img)
    kmapo = kmapo.view(B, C, -1)
    for i in range(B):
        samples = np.random.choice(H * W, k, replace=False)
        kmapo[i, :, samples] = 1.
    kmapo = kmapo.view(B, C, H, W)
    return kmapo[:, [0], :, :]

# training loop starts here
since = time.time()

best_loss = 10 ** 6
best_psnr = 0.0

for epoch in range(args.num_epochs):
    running_loss_tboard = 0.0
    running_recon_tboard = 0.0
    running_slic_tboard = 0.0
    running_recon_meta_tboard = 0.0
    running_psnr_tboard = 0.0

    print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            sampler.train()
            reconstructor.train()
        else:
            sampler.eval()
            reconstructor.eval()

        running_loss = 0.0
        running_recon = 0.0
        running_slic = 0.0
        running_recon_meta = 0.0
        running_psnr = 0.0

        # counter for tboard
        if phase == 'train':
            i = 0
                
        # Iterate over data.            
        for inputs, targets in dataloaders[phase]:

            inputs, targets = inputs.to(device), targets.to(device)

            coords = torch.stack(torch.meshgrid(torch.arange(args.patch_size, device=device), torch.arange(args.patch_size, device=device)), 0)
            coords = coords[None].repeat(inputs.shape[0], 1, 1, 1).float()
            
            if phase == 'train':
                i += 1         
            
            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                # forward: sampler network
                outputs = sampler(torch.cat((inputs, targets), 1))
                prob = F.softmax(outputs, 1)

                # superpixel loss
                inputs_targets_rgbxy = torch.cat([255 * inputs, 255 * targets, coords], 1)
                pooled_labxy = poolfeat(inputs_targets_rgbxy, prob, grid_size, grid_size)
                reconstr_feat = upfeat(pooled_labxy, prob, grid_size, grid_size)
                slic = args.slic_alpha * F.mse_loss(reconstr_feat[:, :3, :, :], inputs_targets_rgbxy[:, :3, :, :]) + \
                    (1 - args.slic_alpha) * F.mse_loss(reconstr_feat[:, 3:6, :, :], inputs_targets_rgbxy[:, 3:6, :, :]) + \
                    args.slic_m ** 2 / grid_size ** 2 * F.mse_loss(reconstr_feat[:, 6:, :, :], inputs_targets_rgbxy[:, 6:, :, :])

                # sampling process
                kmap = get_kmap_from_prob(prob, grid_size)

                # forward: reconstruction network
                inputs2 = torch.cat([inputs, kmap * targets, kmap], dim=1)
                outputs2 = reconstructor(inputs2)
                recon = F.l1_loss(outputs2, targets)

                psnrout = calc_psnr(torch.clip(outputs2 * (1 - kmap), 0, 1), targets * (1 - kmap))

                # backward + optimize only if in training phase
                if phase == 'train':
                    # online optimization
                    # we use random mask instead of learned mask
                    kmap_meta = genmask_random(args.k, inputs)
                    inputs2_meta = torch.cat([inputs, kmap_meta * targets, kmap_meta], dim=1)

                    updated_params = list(reconstructor.module.parameters())
                    # inner loop
                    for _ in range(args.inner_steps):
                        outputs2_inner = reconstructor.module.forward_meta(inputs2_meta, updated_params)
                        loss_inner = (kmap_meta * (outputs2_inner - targets).abs()).mean()
                        grad = torch.autograd.grad(loss_inner, updated_params)
                        updated_params = list(map(lambda p: p[1] - args.inner_lr * p[0], zip(grad, updated_params)))

                    # outer loop
                    outputs2_outer = reconstructor.module.forward_meta(inputs2_meta, updated_params)
                    recon_meta = F.l1_loss(outputs2_outer, targets)
                    grad_meta = torch.autograd.grad(recon_meta, updated_params)

                    # training objective
                    loss = recon + args.lambda_slic * slic
                    # recon_meta = torch.zeros_like(recon)

                    loss.backward()

                    # manually add gradient updates for recon_meta
                    for param, g in zip(reconstructor.module.parameters(), grad_meta):
                        param.grad += args.lambda_meta * g

                    optimizer.step()

                    running_loss_tboard += loss.item()
                    running_recon_tboard += recon.item()
                    running_slic_tboard += slic.item()
                    running_recon_meta_tboard += recon_meta.item()
                    running_psnr_tboard += psnrout.item()
                    if i % args.tboard_freq == args.tboard_freq - 1:
                        
                        # ...log the running loss
                        writer.add_scalar('loss',
                                        running_loss_tboard / args.tboard_freq,
                                        epoch * len(dataloaders[phase]) + i)

                        writer.add_scalar('recon',
                                        running_recon_tboard / args.tboard_freq,
                                        epoch * len(dataloaders[phase]) + i)

                        writer.add_scalar('slic',
                                        running_slic_tboard / args.tboard_freq,
                                        epoch * len(dataloaders[phase]) + i)

                        writer.add_scalar('recon_meta',
                                        running_recon_meta_tboard / args.tboard_freq,
                                        epoch * len(dataloaders[phase]) + i)
                        
                        writer.add_scalar('psnr',
                                        running_psnr_tboard / args.tboard_freq,
                                        epoch * len(dataloaders[phase]) + i)
                        
                        running_loss_tboard = 0.0
                        running_recon_tboard = 0.0
                        running_slic_tboard = 0.0
                        running_recon_meta_tboard = 0.0
                        running_psnr_tboard = 0.0
                else:
                    # at validation time, we do not compute recon_meta to reduce training time
                    recon_meta = torch.zeros_like(recon)
                    loss = recon + args.lambda_slic * slic

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_recon += recon.item() * inputs.size(0)
            running_slic += slic.item() * inputs.size(0)
            running_recon_meta += recon_meta.item() * inputs.size(0)
            running_psnr += psnrout.item() * inputs.size(0)

        epoch_loss[phase] = running_loss / dataset_sizes[phase]
        epoch_recon[phase] = running_recon / dataset_sizes[phase]
        epoch_slic[phase] = running_slic / dataset_sizes[phase]
        epoch_recon_meta[phase] = running_recon_meta / dataset_sizes[phase]
        epoch_psnr[phase] = running_psnr / dataset_sizes[phase]
        
        
        if phase == 'val':
            # ...log the running loss
            writer.add_scalars('loss',
                              {'train': epoch_loss['train'],'val': epoch_loss['val']},
                              (epoch+1) * len(dataloaders['train']))

            writer.add_scalars('recon',
                              {'train': epoch_recon['train'],'val': epoch_recon['val']},
                              (epoch+1) * len(dataloaders['train']))

            writer.add_scalars('slic',
                              {'train': epoch_slic['train'],'val': epoch_slic['val']},
                              (epoch+1) * len(dataloaders['train']))

            writer.add_scalars('recon_meta',
                              {'train': epoch_recon_meta['train'],'val': epoch_recon_meta['val']},
                              (epoch+1) * len(dataloaders['train']))
            
            writer.add_scalars('psnr',
                              {'train': epoch_psnr['train'],'val': epoch_psnr['val']},
                              (epoch+1) * len(dataloaders['train']))

            # log images
            img_grid = torchvision.utils.make_grid(
                torch.cat((inputs, kmap.repeat(1, 3, 1, 1), torch.clip(outputs2 * (1 - kmap) + targets * kmap, 0, 1), targets), 2),
                normalize=True,
                range=(0, 1)
            )
            writer.add_image('val_epoch_' + str(epoch), img_grid)
                        

        print('{} Loss: {:.6f}, Recon: {:.6f}, SLIC: {:.6f}, Recon_meta: {:.6f}, PSNR: {:.4f}'.format(
            phase, epoch_loss[phase], epoch_recon[phase], epoch_slic[phase], epoch_recon_meta[phase], epoch_psnr[phase]))

        # deep copy the sampler
        if phase == 'val' and epoch_loss[phase] < best_loss:
            best_loss = epoch_loss[phase]
            best_psnr = epoch_psnr[phase]
            torch.save(sampler.module.state_dict(), os.path.join(mysavepath, 'best_sampler.pt'))
            torch.save(reconstructor.module.state_dict(), os.path.join(mysavepath, 'best_reconstructor.pt'))

    print()

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
print('Best val loss: {:4f}'.format(best_loss))
print('Best val psnr: {:4f}'.format(best_psnr))
