# Set up
import copy
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from dataset import get_train_validation_test_data, get_label_unlabel_dataset
from utils import set_log_dir, save_checkpoint, create_logger
import cfg
from functions import train, get_val_acc

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main():
    args = cfg.parse_args()
    torch.cuda.manual_seed(args.random_seed)

    # Setting up the resnet model
    if args.pretrained:
        resnet = torchvision.models.resnet50(pretrained=True, progress=True)
    else:
        resnet = torchvision.models.resnet50(pretrained=False, progress=True)
    num_features = resnet.fc.in_features
    resnet.fc = nn.Linear(num_features, args.num_classes)
    resnet = resnet.cuda()

    # Setting up the optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(resnet.parameters(), lr=args.lr, weight_decay=1e-4)
    elif args.optimizer == 'sgd_momentum':
        optimizer = optim.SGD(resnet.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, resnet.parameters()),
                                     args.g_lr, (args.beta1, args.beta2))
    else:
        optimizer = None
    assert optimizer != None

    criterion = nn.CrossEntropyLoss()

    if args.percentage == 1.0:
        train_data, val_data, test_data = get_train_validation_test_data(args.train_csv_path, 
                args.train_img_path, args.val_csv_path, args.val_img_path, args.test_csv_path, args.test_img_path)
    else:
        train_data = get_label_unlabel_dataset(args.train_csv_path, args.train_img_path, args.percentage)
        _, val_data, test_data = get_train_validation_test_data(args.train_csv_path, 
                args.train_img_path, args.val_csv_path, args.val_img_path, args.test_csv_path, args.test_img_path)

    train_loader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_data, batch_size=args.eval_batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_data, batch_size=args.eval_batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    print('Training Datasize:', len(train_data))

    start_epoch = 0
    best_acc1 = 0
    best_acc2 = 0
    best_acc3 = 0

    # set writer
    if args.load_path:
        print(f'=> resuming from {args.load_path}')
        assert os.path.exists(args.load_path)
        checkpoint_file = os.path.join(args.load_path, 'Model', 'checkpoint_last.pth')
        assert os.path.exists(checkpoint_file)
        checkpoint = torch.load(checkpoint_file)
        start_epoch = checkpoint['epoch']
        resnet.load_state_dict(checkpoint['resnet_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

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
        'train_global_steps': start_epoch * len(train_loader),
        'valid_global_steps': start_epoch // args.val_freq,
    }

    start = time.time()
    for epoch in tqdm(range(int(start_epoch), int(args.max_epoch)), desc='total progress'):
        best_curr_acc1, best_curr_acc2, best_curr_acc3 = train(args, resnet, optimizer, criterion, train_loader, val_loader, epoch, writer_dict, best_acc1, best_acc2, best_acc3)

        best_acc1, best_acc2, best_acc3 = best_curr_acc1, best_curr_acc2, best_curr_acc3
        
        if epoch and epoch % args.val_freq == 0 or epoch == int(args.max_epoch)-1:
            val_acc = get_val_acc(val_loader, resnet)
            logger.info(f'Validation Accuracy {val_acc} || @ epoch {epoch}.')

        save_checkpoint({
            'epoch': epoch + 1,
            'resnet_state_dict': resnet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'path_helper': args.path_helper
        }, False, False, False, args.path_helper['ckpt_path'], filename='checkpoint_last.pth')

    end =  time.time()
    final_val_acc = get_val_acc(val_loader, resnet)
    final_test_acc = get_val_acc(test_loader, resnet)
    time_elapsed = end - start

    print('\n Final Validation Accuracy:', final_val_acc.data, 
    '\n Final Test Accuracy:', final_test_acc.data, 
    '\n Time Elapsed:', time_elapsed, 'seconds.')
    

if __name__ == "__main__":
    main()