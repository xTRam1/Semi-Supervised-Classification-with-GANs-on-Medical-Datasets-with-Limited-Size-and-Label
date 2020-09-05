# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cfg
from resNets import Generator, Discriminator
from utils.utils import set_log_dir, create_logger
from torch.utils.data import DataLoader, TensorDataset

import torch
import os
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import torch.nn as nn
from torchvision.utils import make_grid, save_image
from torch.autograd import Variable
from imageio import imsave
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def main():
    args = cfg.parse_args()
    torch.cuda.manual_seed(args.random_seed)
    assert args.exp_name
    assert os.path.exists(args.load_path)
    args.path_helper = set_log_dir('logs_eval', args.exp_name)
    logger = create_logger(args.path_helper['log_path'], phase='test')

    gen_net = Generator(args=args).cuda()

    # set writer
    print(f'=> resuming from {args.load_path}')
    assert os.path.exists(args.load_path)
    checkpoint_file = os.path.join(args.load_path, 'Model', 'checkpoint_best1.pth')
    assert os.path.exists(checkpoint_file)
    checkpoint = torch.load(checkpoint_file)
    gen_net.load_state_dict(checkpoint['gen_state_dict'])
    
    logger.info(args)
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'valid_global_steps': 0,
    }

    z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))
    gen_imgs = gen_net(z).mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
    for img_idx, img in enumerate(gen_imgs):
        file_name = os.path.join(fid_buffer_dir, f'iter{iter_idx}_b{img_idx}.png')
        imsave(file_name, img)
    
    print('Images saved at: ' + fid_buffer_dir)

if __name__ == '__main__':
    main()








