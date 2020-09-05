from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils import set_log_dir, create_logger
from functions import get_val_acc
from torch.utils.data import DataLoader, TensorDataset
import torchvision

import torch
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from dataset import get_label_unlabel_dataset, get_train_validation_test_data
import cfg

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def main():
    args = cfg.parse_args()
    torch.cuda.manual_seed(args.random_seed)
    assert args.exp_name
    assert os.path.exists(args.load_path)
    args.path_helper = set_log_dir('logs_eval', args.exp_name)
    logger = create_logger(args.path_helper['log_path'], phase='test')

    resnet1 = torchvision.models.resnet50(pretrained=False, progress=False)
    resnet2 = torchvision.models.resnet50(pretrained=False, progress=False)
    resnet3 = torchvision.models.resnet50(pretrained=False, progress=False)
    resnet4 = torchvision.models.resnet50(pretrained=False, progress=False)
    
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
    resnet1.load_state_dict(checkpoint1['resnet_state_dict'])
    resnet2.load_state_dict(checkpoint2['resnet_state_dict'])
    resnet3.load_state_dict(checkpoint3['resnet_state_dict'])
    resnet4.load_state_dict(checkpoint4['resnet_state_dict'])
    logger.info(f'=> loaded checkpoint {checkpoint_file}')

    logger.info(args)
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'valid_global_steps': 0,
    }

    _, val_data, test_data = get_train_validation_test_data(args.val_csv_path, args.val_img_path, args.test_csv_path, args.test_img_path)
    val_loader = DataLoader(val_data, batch_size=args.gen_batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_data, batch_size=args.gen_batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)

    final_val_acc1 = get_val_acc(val_loader, resnet1)
    final_test_acc1 = get_val_acc(test_loader, resnet1)
    final_val_acc2 = get_val_acc(val_loader, resnet2)
    final_test_acc2 = get_val_acc(test_loader, resnet2)
    final_val_acc3 = get_val_acc(val_loader, resnet3)
    final_test_acc3 = get_val_acc(test_loader, resnet3)
    final_val_acc4 = get_val_acc(val_loader, resnet4)
    final_test_acc4 = get_val_acc(test_loader, resnet4)
    
    final_test_acc = (final_test_acc1.data + final_test_acc2.data + final_test_acc3.data) / 3

    print('\n ----- \n 1. Model \n Final Validation Accuracy: {0}, \n Final Test Accuracy: {1}'.format(final_val_acc1.data, final_test_acc1.data))
    print('\n ----- \n 2. Model \n Final Validation Accuracy: {0}, \n Final Test Accuracy: {1}'.format(final_val_acc2.data, final_test_acc2.data))
    print('\n ----- \n 3. Model \n Final Validation Accuracy: {0}, \n Final Test Accuracy: {1}'.format(final_val_acc3.data, final_test_acc3.data))
    print('\n ----- \n Last Model \n Final Validation Accuracy: {0}, \n Final Test Accuracy: {1}'.format(final_val_acc4.data, final_test_acc4.data))
    print('\n ----- \n Final Test Acc:', final_test_acc)
    

if __name__ == '__main__':
    main()












