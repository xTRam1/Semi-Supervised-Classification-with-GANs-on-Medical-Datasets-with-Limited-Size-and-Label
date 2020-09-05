# Set up
import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.datasets as datasets
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
import pdb
from utils import save_checkpoint


def train(args, resnet : nn.Module, optimizer, criterion, train_loader, val_loader, epoch, writer_dict, best_acc1, best_acc2, best_acc3):
    writer = writer_dict['writer']
    step = 0

    # train mode
    resnet = resnet.train()
    loss = accuracy = 0.
    best_curr_acc1, best_curr_acc2, best_curr_acc3 = 0, 0, 0
    for iter_idx, (x, y) in enumerate(tqdm(train_loader)):
        global_steps = writer_dict['train_global_steps']

        x, y = x.cuda(), y.cuda()
        
        optimizer.zero_grad()
        scores = resnet(x)
        l = criterion(scores, y)
        loss += l
        acc = torch.mean((scores.max(1)[1] == y).float())

        l.backward()
        optimizer.step()

        # verbose
        if iter_idx % args.print_freq == 0:
            loss /= args.print_freq
            accuracy = get_val_acc(train_loader, resnet)
            val_accuracy = get_val_acc(val_loader, resnet)
            print('Hey!')
            tqdm.write(
                "[Epoch %d/%d] [Batch %d/%d] [Loss: %.4f] [Train Acc: %.4f] [Val Acc: %.4f]" %
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), loss, accuracy, val_accuracy))

            if val_accuracy > best_curr_acc1:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'resnet_state_dict': resnet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'path_helper': args.path_helper
                }, True, False, False, args.path_helper['ckpt_path'])
                best_curr_acc1 = val_accuracy
            elif val_accuracy < best_curr_acc1 and val_accuracy > best_curr_acc2:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'resnet_state_dict': resnet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'path_helper': args.path_helper
                }, False, True, False, args.path_helper['ckpt_path'])
                best_curr_acc2 = val_accuracy
            elif val_accuracy < best_curr_acc1 and val_accuracy < best_curr_acc2 and val_accuracy > best_curr_acc3:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'resnet_state_dict': resnet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'path_helper': args.path_helper
                }, False, False, True, args.path_helper['ckpt_path'])
                best_curr_acc3 = val_accuracy

            # TensorBoard scalar writing
            with torch.no_grad():
                writer.add_scalars('Loss/training', loss, global_steps)
                writer.add_scalars('Accuracy', {'Traning Accuracy': accuracy, 'Validation Accuracy': val_accuracy}, global_steps)

        step += 1
        writer_dict['train_global_steps'] = global_steps + 1
        
    return best_curr_acc1, best_curr_acc2, best_curr_acc3 


def get_val_acc(val_loader, resnet):
    val_accuracy = 0
    val_step = 0
    
    for iter_idx, (imgs, labels) in enumerate(val_loader):
        imgs, labels = imgs.cuda(), labels.cuda()
        output_labels = resnet(imgs)
        
        val_acc = torch.mean((output_labels.max(1)[1] == labels).float())
        val_accuracy += val_acc
        val_step += 1

    return val_accuracy / val_step
