# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time 

import cfg
from functional import train, validate, LinearLrDecay, load_params, copy_params, get_val_acc
from utils.utils import set_log_dir, save_checkpoint, create_logger

import torch
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from copy import deepcopy
from resNets import Generator, Discriminator
from Datasets import get_label_unlabel_dataset, get_train_validation_test_data

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main():
    args = cfg.parse_args()
    torch.cuda.manual_seed(args.random_seed)

    # import network
    gen_net = Generator(args=args).cuda()
    dis_net = Discriminator(args=args, policy='color,translation,cutout').cuda()
    print(dis_net.policy)

    # set optimizer
    gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.parameters()),
                                     args.g_lr, (args.beta1, args.beta2))
    dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dis_net.parameters()),
                                     args.d_lr, (args.beta1, args.beta2))
                                     
    # Schedulers
    gen_scheduler = LinearLrDecay(gen_optimizer, args.g_lr, 0.0, args.lr_decay_start_iter, args.max_iter * args.n_critic)
    dis_scheduler = LinearLrDecay(dis_optimizer, args.d_lr, 0.0, args.lr_decay_start_iter, args.max_iter * args.n_critic)

    # set up dataloader for training
    labeled, unlabeled = get_label_unlabel_dataset(args.train_csv_path, args.train_img_path, args.ratio)
    unlabel_loader1 = DataLoader(unlabeled, batch_size=args.gen_batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    unlabel_loader2 = DataLoader(unlabeled, batch_size=args.gen_batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    label_loader = DataLoader(labeled, batch_size=args.gen_batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)

    # set up dataloader for validation and test
    train_data, val_data, test_data = get_train_validation_test_data(args.train_csv_path, 
                args.train_img_path, args.val_csv_path, args.val_img_path, args.test_csv_path, args.test_img_path)
    val_loader = DataLoader(val_data, batch_size=args.gen_batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_data, batch_size=args.gen_batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    train_loader = DataLoader(train_data, batch_size=args.gen_batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)

    # epoch number for dis_net
    args.max_epoch = args.max_epoch * args.n_critic
    if args.max_iter:
        args.max_epoch = np.ceil(args.max_iter * args.n_critic / len(unlabel_loader1))

    # initial
    fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (25, args.latent_dim)))
    start_epoch = 0
    best_acc1 = 0
    best_acc2 = 0
    best_acc3 = 0

    # set writer
    if args.load_path:
        print(f'=> resuming from {args.load_path}')
        assert os.path.exists(args.load_path)
        checkpoint = torch.load(args.load_path)
        start_epoch = checkpoint['epoch']
        gen_net.load_state_dict(checkpoint['gen_state_dict'])
        dis_net.load_state_dict(checkpoint['dis_state_dict'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])

        args.path_helper = checkpoint['path_helper']
        logger = create_logger(args.path_helper['log_path'])
        logger.info(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')
    else:
        # create new log dir
        assert args.exp_name
        args.path_helper = set_log_dir('logs', args.exp_name)
        logger = create_logger(args.path_helper['log_path'])

    logger.info(args)
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'train_global_steps': start_epoch * len(unlabel_loader1),
        'valid_global_steps': start_epoch // args.val_freq,
    }
    
    # train loop
    val_no = 0
    lr_schedulers = (gen_scheduler, dis_scheduler) if args.lr_decay else None
    start = time.time()
    for epoch in tqdm(range(int(start_epoch), int(args.max_epoch)), desc='total progress'):
        best_curr_acc1, best_curr_acc2, best_curr_acc3 = train(args, gen_net, dis_net, gen_optimizer, dis_optimizer, 
            unlabel_loader1, unlabel_loader2, label_loader, train_loader, val_loader, epoch, writer_dict, best_acc1, best_acc2, best_acc3, lr_schedulers)
        
        best_acc1, best_acc2, best_acc3 = best_curr_acc1, best_curr_acc2, best_curr_acc3 
        
        if epoch and epoch % args.val_freq == 0 or epoch == int(args.max_epoch)-1:
            val_acc = validate(args, fixed_z, gen_net, dis_net, writer_dict, val_loader, val_no)
            logger.info(f'Validation Accuracy {val_acc} || @ epoch {epoch}.')
        
        save_checkpoint({
        'epoch': epoch + 1,
        'gen_state_dict': gen_net.state_dict(),
        'dis_state_dict': dis_net.state_dict(),
        'gen_optimizer': gen_optimizer.state_dict(),
        'dis_optimizer': dis_optimizer.state_dict(),
        'path_helper': args.path_helper
        }, False, False, False, args.path_helper['ckpt_path'], filename='checkpoint_last.pth')
        val_no += 1
    
    end =  time.time()

    final_val_acc = get_val_acc(val_loader, dis_net)
    final_test_acc = get_val_acc(test_loader, dis_net)
    time_elapsed = end - start

    print('\n Final Validation Accuracy:', final_val_acc.data, 
    '\n Final Test Accuracy:', final_test_acc.data, 
    '\n Time Elapsed:', time_elapsed, 'seconds.')

if __name__ == '__main__':
    main()
