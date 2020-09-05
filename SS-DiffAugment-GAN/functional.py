import math
import pdb
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from torch.autograd import Variable
from imageio import imsave
from tqdm import tqdm
from copy import deepcopy
import logging
import matplotlib.pyplot as plt

from utils.utils import save_checkpoint

def log_sum_exp(x, axis = 1):
    m = torch.max(x, dim = 1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(1)), dim = axis))

logger = logging.getLogger(__name__)
def train(args, gen_net: nn.Module, dis_net: nn.Module, gen_optimizer, dis_optimizer, 
            unlabel_loader1, unlabel_loader2, label_loader, train_loader, val_loader, epoch, 
            writer_dict, best_acc1, best_acc2, best_acc3, schedulers):
    
    writer = writer_dict['writer']
    gen_step = 0
    
    # iterators
    label_iterator = label_loader.__iter__()
    unlabel2_iterator = unlabel_loader2.__iter__()

    # train mode
    gen_net = gen_net.train()
    dis_net = dis_net.train()

    loss_supervised = loss_unsupervised = loss_gen = accuracy = 0.
    best_curr_acc1, best_curr_acc2, best_curr_acc3 = best_acc1, best_acc2, best_acc3
    for iter_idx, unlabel1 in enumerate(tqdm(unlabel_loader1)):
        global_steps = writer_dict['train_global_steps']
    
        # Getting the unlabeled and labeled images 
        try:
            unlabel2 = unlabel2_iterator.next()
        except StopIteration:
            unlabel2_iterator = unlabel_loader2.__iter__()
            unlabel2 = unlabel2_iterator.next()
        
        try:
            x, y = label_iterator.next()
        except StopIteration:
            label_iterator = label_loader.__iter__()
            x, y = label_iterator.next()
            
        x, y, unlabel1, unlabel2 = x.cuda(), y.cuda(), unlabel1.cuda(), unlabel2.cuda()
        
        # Training Discriminator
        ll, lu, acc = trainD(args, x, y, unlabel1, dis_net, gen_net, dis_optimizer)
        loss_supervised += ll
        loss_unsupervised += lu
        accuracy += acc

        # Training Generator
        lg = trainG(args, unlabel2, gen_net, dis_net, gen_optimizer, dis_optimizer)
        #if epoch > 1 and lg > 1:
        #    lg = trainG(args, unlabel2, gen_net, dis_net, gen_optimizer, dis_optimizer)
        loss_gen += lg
        
        if schedulers:
            gen_scheduler, dis_scheduler = schedulers
            g_lr = gen_scheduler.step(global_steps)
            d_lr = dis_scheduler.step(global_steps)
            writer.add_scalar('LR/g_lr', g_lr, global_steps)
            writer.add_scalar('LR/d_lr', d_lr, global_steps)

        # verbose
        if gen_step and iter_idx % args.print_freq == 0:
            loss_supervised /= args.print_freq
            loss_unsupervised /= args.print_freq
            loss_gen /= args.print_freq
            accuracy = get_val_acc(label_loader, dis_net)
            all_train_acc = get_val_acc(train_loader, dis_net)
            val_accuracy = get_val_acc(val_loader, dis_net)
            
            tqdm.write(
                "[Epoch %d/%d] [Batch %d/%d] [loss_supervised: %.4f] [loss_unsupervised: %.4f] [loss_gen: %.4f] [train acc: %.4f] [all train acc: %.4f] [val acc: %.4f]" %
                (epoch, args.max_epoch, iter_idx % len(unlabel_loader1), len(unlabel_loader1), loss_supervised, loss_unsupervised, loss_gen, accuracy, all_train_acc, val_accuracy))

            if val_accuracy > best_curr_acc1:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'gen_state_dict': gen_net.state_dict(),
                    'dis_state_dict': dis_net.state_dict(),
                    'gen_optimizer': gen_optimizer.state_dict(),
                    'dis_optimizer': dis_optimizer.state_dict(),
                    'path_helper': args.path_helper
                }, True, False, False, args.path_helper['ckpt_path'])
                best_curr_acc1 = val_accuracy
            elif val_accuracy < best_curr_acc1 and val_accuracy > best_curr_acc2:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'gen_state_dict': gen_net.state_dict(),
                    'dis_state_dict': dis_net.state_dict(),
                    'gen_optimizer': gen_optimizer.state_dict(),
                    'dis_optimizer': dis_optimizer.state_dict(),
                    'path_helper': args.path_helper
                }, False, True, False, args.path_helper['ckpt_path'])
                best_curr_acc2 = val_accuracy
            elif val_accuracy < best_curr_acc1 and val_accuracy < best_curr_acc2 and val_accuracy > best_curr_acc3:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'gen_state_dict': gen_net.state_dict(),
                    'dis_state_dict': dis_net.state_dict(),
                    'gen_optimizer': gen_optimizer.state_dict(),
                    'dis_optimizer': dis_optimizer.state_dict(),
                    'path_helper': args.path_helper
                }, False, False, True, args.path_helper['ckpt_path'])
                best_curr_acc3 = val_accuracy

            # TensorBoard scalar writing
            with torch.no_grad():
                writer.add_scalars('loss', {'loss_supervised':loss_supervised, 'loss_unsupervised':loss_unsupervised, 'loss_gen':loss_gen}, global_steps)
                writer.add_scalars('Accuracy', {'Traning Accuracy': accuracy, 'Validation Accuracy': val_accuracy, 'All Training Accuracy': all_train_acc}, global_steps)

        gen_step += 1
        writer_dict['train_global_steps'] = global_steps + 1

    return best_curr_acc1, best_curr_acc2, best_curr_acc3 

def get_val_acc(val_loader, dis_net):
    val_accuracy = 0
    val_step = 0
    
    for iter_idx, (imgs, labels) in enumerate(val_loader):
        imgs, labels = imgs.cuda(), labels.cuda()
        output_labels = dis_net(imgs, test=True)
        
        val_acc = torch.mean((output_labels.max(1)[1] == labels).float())
        val_accuracy += val_acc
        val_step += 1

    return val_accuracy / val_step


def trainD(args, x_label, y, x_unlabel, dis_net, gen_net, dis_optimizer):
    x_label, x_unlabel, y = Variable(x_label), Variable(x_unlabel), Variable(y, requires_grad=False)
    x_label, x_unlabel, y = x_label.cuda(), x_unlabel.cuda(), y.cuda()
    gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))
            
    # Values are input to Discriminator
    output_label = dis_net(x_label)
    output_unlabel = dis_net(x_unlabel)
    output_fake = dis_net(gen_net(gen_z).detach())
    
    # Loss
    logz_label, logz_unlabel, logz_fake = log_sum_exp(output_label), log_sum_exp(output_unlabel), log_sum_exp(output_fake) # log Ã¢Ë†â€˜e^x_i
    prob_label = torch.gather(output_label, 1, y.unsqueeze(1)) # log e^x_label = x_label 
    loss_supervised = -torch.mean(prob_label) + torch.mean(logz_label)
    loss_unsupervised = 0.5 * (-torch.mean(logz_unlabel) + torch.mean(F.softplus(logz_unlabel))  + # real_data: log Z/(1+Z)
                        torch.mean(F.softplus(logz_fake)) ) # fake_data: log 1/(1+Z)
    loss = loss_supervised + args.unlabel_weight * loss_unsupervised
    
    # Accuracy
    acc = torch.mean((output_label.max(1)[1] == y).float())
    
    # Doing the backpropagation for the Discriminator
    dis_optimizer.zero_grad()
    loss.backward()
    dis_optimizer.step()
    return loss_supervised.data.cpu().numpy(), loss_unsupervised.data.cpu().numpy(), acc
    
    
def trainG(args, x_unlabel, gen_net, dis_net, gen_optimizer, dis_optimizer):
    gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))
    fake = gen_net(gen_z)
            
    # Values are input to Discriminator
    mom_gen, output_fake = dis_net(fake, feature=True)
    mom_unlabel, _ = dis_net(Variable(x_unlabel), feature=True)
    
    # Loss
    mom_gen = torch.mean(mom_gen, dim = 0)
    mom_unlabel = torch.mean(mom_unlabel, dim = 0)
    loss_fm = torch.mean((mom_gen - mom_unlabel) ** 2)
    loss = loss_fm 
    
    # Doing the backpropagation for the Generator
    gen_optimizer.zero_grad()
    dis_optimizer.zero_grad()
    loss.backward()
    gen_optimizer.step()

    return loss.data.cpu().numpy()


def validate(args, fixed_z, gen_net: nn.Module, dis_net, writer_dict, val_loader, val_no):
    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']

    # eval mode
    gen_net = gen_net.eval()

    # generate images
    sample_imgs = gen_net(fixed_z)
    img_grid = make_grid(sample_imgs, nrow=5, normalize=True, scale_each=True)
    show(img_grid)

    fid_buffer_dir = os.path.join(args.path_helper['sample_path'])

    eval_iter = args.num_eval_imgs // args.eval_batch_size
    img_list = list()
    for iter_idx in tqdm(range(eval_iter), desc='sample images'):
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))

        # Generate a batch of images
        gen_imgs = gen_net(z).mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
        for img_idx, img in enumerate(gen_imgs):
            file_name = os.path.join(fid_buffer_dir, f'iter{iter_idx}_b{img_idx}.png')
            imsave(file_name, img)
        img_list.extend(list(gen_imgs))

    writer.add_image('sampled_images', img_grid, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    save_image(img_grid, os.path.join(args.path_helper['fixed_z'], str(val_no) + '.png'))

    val_acc = get_val_acc(val_loader, dis_net) 

    return val_acc


class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten

def show(img):
    npimg = img.detach().cpu().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.show()
