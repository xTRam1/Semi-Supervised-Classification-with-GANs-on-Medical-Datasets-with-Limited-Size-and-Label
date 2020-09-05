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
from functional import get_val_acc
from torch.utils.data import DataLoader, TensorDataset

import torch
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from Datasets import get_label_unlabel_dataset, get_train_validation_test_data

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def main():
    args = cfg.parse_args()
    torch.cuda.manual_seed(args.random_seed)
    assert args.exp_name
    assert os.path.exists(args.load_path)
    args.path_helper = set_log_dir('logs_eval', args.exp_name)
    logger = create_logger(args.path_helper['log_path'], phase='test')

    dis_net1 = Discriminator(args=args).cuda()
    dis_net2 = Discriminator(args=args).cuda()
    dis_net3 = Discriminator(args=args).cuda()
    dis_net4 = Discriminator(args=args).cuda()

    # set writer
    print(f'=> resuming from {args.load_path}')
    assert os.path.exists(args.load_path)
    checkpoint_file1 = os.path.join(args.load_path, 'Model', 'checkpoint_best1.pth')
    checkpoint_file2 = os.path.join(args.load_path, 'Model', 'checkpoint_best2.pth')
    checkpoint_file3 = os.path.join(args.load_path, 'Model', 'checkpoint_best3.pth')
    checkpoint_file4 = os.path.join(args.load_path, 'Model', 'checkpoint_last.pth')
    assert os.path.exists(checkpoint_file1)
    assert os.path.exists(checkpoint_file2)
    assert os.path.exists(checkpoint_file3)
    assert os.path.exists(checkpoint_file4)
    checkpoint1 = torch.load(checkpoint_file1)
    checkpoint2 = torch.load(checkpoint_file2)
    checkpoint3 = torch.load(checkpoint_file3)
    checkpoint4 = torch.load(checkpoint_file4)
    
    print(checkpoint1['epoch'] + 1)
    print(checkpoint2['epoch'] + 1)
    print(checkpoint3['epoch'] + 1)
    print(checkpoint4['epoch'] + 1)
    
    dis_net1.load_state_dict(checkpoint1['dis_state_dict'])
    logger.info(f'=> loaded checkpoint {checkpoint_file1}')
    dis_net2.load_state_dict(checkpoint2['dis_state_dict'])
    logger.info(f'=> loaded checkpoint {checkpoint_file2}')
    dis_net3.load_state_dict(checkpoint3['dis_state_dict'])
    logger.info(f'=> loaded checkpoint {checkpoint_file3}')
    dis_net4.load_state_dict(checkpoint3['dis_state_dict'])
    logger.info(f'=> loaded checkpoint {checkpoint_file4}')
    
    logger.info(args)
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'valid_global_steps': 0,
    }

    train_data, val_data, test_data = get_train_validation_test_data(args.train_csv_path, 
                args.train_img_path, args.val_csv_path, args.val_img_path, args.test_csv_path, args.test_img_path)
    val_loader = DataLoader(val_data, batch_size=args.gen_batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_data, batch_size=args.gen_batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)

    final_val_acc1 = get_val_acc(val_loader, dis_net1)
    final_test_acc1 = get_val_acc(test_loader, dis_net1)
    final_val_acc2 = get_val_acc(val_loader, dis_net2)
    final_test_acc2 = get_val_acc(test_loader, dis_net2)
    final_val_acc3 = get_val_acc(val_loader, dis_net3)
    final_test_acc3 = get_val_acc(test_loader, dis_net3)
    final_val_acc4 = get_val_acc(val_loader, dis_net4)
    final_test_acc4 = get_val_acc(test_loader, dis_net4)
    final_test_acc = (final_test_acc1.data + final_test_acc2.data + final_test_acc3.data) / 3

    print('\n ----- \n 1. Model \n Final Validation Accuracy: {0}, \n Final Test Accuracy: {1}'.format(final_val_acc1.data, final_test_acc1.data))
    print('\n ----- \n 2. Model \n Final Validation Accuracy: {0}, \n Final Test Accuracy: {1}'.format(final_val_acc2.data, final_test_acc2.data))
    print('\n ----- \n 3. Model \n Final Validation Accuracy: {0}, \n Final Test Accuracy: {1}'.format(final_val_acc3.data, final_test_acc3.data))
    print('\n ----- \n Last Model \n Final Validation Accuracy: {0}, \n Final Test Accuracy: {1}'.format(final_val_acc4.data, final_test_acc4.data))
    print('\n ----- \n Final Test Acc:', final_test_acc)

if __name__ == '__main__':
    main()









