import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--percentage',
        type=float,
        default=1.0,
        help='What is the percentage?')
    parser.add_argument(
        '--pretrained',
        type=str2bool,
        default=True,
        help='Is resnet pretrained?')
    parser.add_argument(
        '--num_classes',
        type=int,
        default=2)
    parser.add_argument(
        '--max_epoch',
        type=int,
        default=200,
        help='number of epochs of training')
    parser.add_argument(
        '--max_iter',
        type=int,
        default=5000,
        help='number of epochs of training')
    parser.add_argument(
        '--n_critic',
        type=int,
        default=1,
        help='number of epochs of training')
    parser.add_argument(
        '--optimizer', 
        type=str,
        default='sgd',
        help='type of optimizer')
    parser.add_argument(
        '-t_b_s',
        '--train_batch_size',
        type=int,
        default=64,
        help='size of the batches')
    parser.add_argument(
        '-e_b_s',
        '--eval_batch_size',
        type=int,
        default=64,
        help='size of the batches')
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='learning rate')  
    parser.add_argument(
        '--lr_decay',
        type=bool,
        default=True,
        help='learning rate decay or not')
    parser.add_argument(
        '--beta1',
        type=float,
        default=0.0,
        help='adam: decay of first order momentum of gradient')
    parser.add_argument(
        '--beta2',
        type=float,
        default=0.9,
        help='adam: decay of first order momentum of gradient')
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='number of cpu threads to use during batch generation')
    parser.add_argument(
        '--img_size',
        type=int,
        default=96,
        help='size of each image dimension')
    parser.add_argument(
        '--val_freq',
        type=int,
        default=20,
        help='interval between each validation (epoch)')
    parser.add_argument(
        '--print_freq',
        type=int,
        default=50,
        help='interval between each verbose')
    parser.add_argument(
        '--load_path',
        type=str,
        help='The reload model path (checkpoint)')
    parser.add_argument(
        '--exp_name',
        type=str,
        help='The name of exp (checkpoint)')
    parser.add_argument(
        '--train_csv_path',
        type=str,
        default='./data',
        help='The path of data set')
    parser.add_argument(
        '--train_img_path',
        type=str,
        default='./data',
        help='The path of data set')
    parser.add_argument(
        '--val_csv_path',
        type=str,
        default='./data',
        help='The path of data set')
    parser.add_argument(
        '--val_img_path',
        type=str,
        default='./data',
        help='The path of data set')
    parser.add_argument(
        '--test_csv_path',
        type=str,
        default='./data',
        help='The path of data set')
    parser.add_argument(
        '--test_img_path',
        type=str,
        default='./data',
        help='The path of data set')
    parser.add_argument('--random_seed', type=int, default=12345)

    args = parser.parse_args()
    return args