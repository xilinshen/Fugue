import argparse
def moco_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--file', type = str,
                        help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='densenet21',
                        help='model architecture (default: densenet21)')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 1)')
    parser.add_argument('--in_features', type=int, help="number of features")
    parser.add_argument('--num_batches', type=int, help='the total number of batches')
    parser.add_argument('--outdir', help='output folder')
    parser.add_argument('--save_frequency', type=int, default=5, help='save frequency of model checkpoints')
    parser.add_argument('--load-split-file', action = "store_true", default=False,
                                    help='if the integrated file is large, we recommend split into mini-batch and load the fragments into memory for eeach iteration.')
    parser.add_argument('--split-now', action = "store_true", default=False, 
                                    help='')
    parser.add_argument('--split-savedir', type=str, default=None, 
                        help='if split file now, an empty path should be provided to save the splitted mini-batch files')

    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--shuffle-ratio', type=float, default=0.2, help='positional shuffle rate')
    parser.add_argument('--randomzero-ratio', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--moco-k', type=int,
                        help='queue size; number of negative keys. This value needs to be divisible by the batch size.'
                        'The recommended value is about 5% of the samples size')
    parser.add_argument('--moco-m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')


    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum of SGD solver')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--schedule', default=[60, 120], nargs='*', type=int,
                        help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    parser.add_argument('--moco-dim', default=128, type=int,
                        help='feature dimension (default: 128)')
    parser.add_argument('--moco-t', default=0.2, type=float,
                        help='softmax temperature (default: 0.2)')
    parser.add_argument('--mlp', action='store_true',
                        help='use mlp head')
    parser.add_argument('--cos', action='store_true',
                        help='use cosine lr schedule')

    return parser
