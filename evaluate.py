import argparse
import os
from collections import OrderedDict
from glob import glob
from pathlib import Path

import pandas as pd
import torch
from torch._C import device
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from albumentations.augmentations import transforms
from albumentations.augmentations import geometric
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
from yaml import loader

import archs
import losses
from dataset import Dataset
from metrics import dice_coef, iou_score
from utils import AverageMeter, str2bool


def evaluate(net, config, val_loader):
    avg_meters = {'dice': AverageMeter()}

    # switch to evaluate mode
    net.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                outputs = net(input)
                dice = dice_coef(outputs[-1], target)
            else:
                output = net(input)
                dice = dice_coef(output, target)

            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([('dice', avg_meters['dice'].avg)])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('dice', avg_meters['dice'].avg)])
    
    
if __name__ == '__main__':
    
    dir_img = Path('./data_zebra_crossing/images/')
    dir_mask = Path('./data_zebra_crossing/masks/')
    
    cudnn.benchmark = True #加快卷积效率
    
    #读取网络配置文件 config
    with open('models/data_zebra_crossing_NestedUNet_woDS/config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    #根据config创建网络
    net = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])
    #读取网络参数 .pth
    net.load_state_dict(torch.load('models/%s/model.pth' % config['name']))
    net.to(torch.device('cuda'))
    net.eval()
    
    #创建dataloader
    val_img_ids = glob(os.path.join('data_zebra_crossing', 'images', '*' + config['img_ext']))
    val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids]
    
    val_transform = Compose([
        geometric.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(config['dataset'], 'images'),
        mask_dir=os.path.join(config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)
    
    #评价
    dice = evaluate(net, config, val_loader)
    print(dice)