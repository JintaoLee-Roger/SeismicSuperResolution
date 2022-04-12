import argparse


parser = argparse.ArgumentParser(description='UNet')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=6,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')

# Model specifications
parser.add_argument('--model', default='unet',
                    help='model name')

# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--decay', type=str, default='200',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')

# Log specifications
parser.add_argument('--load', type=str, default='',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')

# dir configure
parser.add_argument('--apply_field_data', type=bool, default=False)
parser.add_argument('--dir_data_root', type=str, default='../data/',
                    help='root directory of dataset')
parser.add_argument('--dir_lr', type=str, default='nx2',
                    help='if synthetic data, like this: dir_data_root/dir_lr.   '+
                    'if field data, dir_lr is the absolute path of field data.')
parser.add_argument('--dir_hr', type=str, default='sx',
                    help='like this: dir_data_root/dir_hr.')
parser.add_argument('--save_dir_suffix', type=str, default='syn',
                    help='save test result as: result-(save_dir_suffix)/, such as result-syn/')

parser.add_argument('--data_range', type=str, default='1-1200/1301-1450',
                    help='train/test data range')
parser.add_argument('--scale', type=int, default=2,
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=96,
                    help='input patch size')
parser.add_argument('--save', type=str, default='alpha6',
                    help='dir name to save, i.e. generate this dir: ../experiment/(save)/')
parser.add_argument('--save_models', type=bool, default=True,
                    help='save all intermediate models')

parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', type=bool, default=True,
                    help='save output results (sr), maybe used in test (not validation)')
parser.add_argument('--loss', type=float, default=0.6,
                    help='loss function configuration, loss*MSSSIM+(1-loss)*L1')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=150,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--test_only', type=bool, default=False,
                    help='set this option to test the model')
parser.add_argument('--n_GPUs', type=int, default=2,
                    help='number of GPUs')

args = parser.parse_args()

args.seed = 1 # random seed
args.act = 'relu' # activation function
args.momentum = 0.9 # SGD momentum
args.betas = (0.9, 0.999) # ADAM beta
args.epsilon = 1e-8 # ADAM epsilon for numerical stability
args.weight_decay = 0 # weight decay
args.gclip = 0 # gradient clipping threshold (0 = no clipping)
args.feature_scale = 1 # choices=(1, 2, 4, 8, 16)

args.pre_train = ''

if args.loss == 0:
    args.loss = '1*L1'
elif args.loss == 1:
    args.loss = '1*MSSSIM'
else:
    args.loss = str(round(1-args.loss, 2))+'*L1+'+str(round(args.loss, 2))+'*MSSSIM'


for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
