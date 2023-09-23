import os
import os.path as osp
import argparse
from torch.nn import DataParallel
import torch
from torch import nn
from tensorboardX import SummaryWriter
from datasets.dataloader import get_data_loader
from misc.utils import init_model, init_random_seed, mkdirs
from misc.saver import Saver
import random
from models import *
from models import DenoiseResnet
from OpensetMethods.OpenMax import openmax
from pdb import set_trace as st
from datasets.odd_data_loader import get_loader_odd


def test(args):
    sub_path = '{}-{}-{}-{}-{}'.format(args.datasetname, args.adv, args.split,
                                       str(args.adv_iter), args.training_type)
    args.seed = init_random_seed(args.manual_seed)
    #####################load datasets#####################

    kdataloader_trn, _, knownclass = get_data_loader(name=args.datasetname, train=True, split=args.split,
                                                     batch_size=args.batchsize, image_size=args.imgsize)
    #
    kdataloader_tst, ukdataloader_tst, knownclass = get_data_loader(name=args.datasetname, train=False,
                                                                    split=args.split,
                                                                    batch_size=args.batchsize, image_size=args.imgsize)

    # mnist_loader, mnist_val_loader, mnist_test_loader = get_loader_odd(split=args.split)
    # knownclass = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    nclass = len(knownclass)
    #####################Network Init#####################
    load_path = osp.join(args.results_path, sub_path, 'model')

    Encoderrestore = osp.join(load_path, f'Encoder-{args.defensesnapshot}.pt')

    Encoder = init_model(
        net=DenoiseResnet.ResnetEncoder(latent_size=args.latent_size, denoise=args.denoise, robust=args.robust),
        init_type=args.init_type, restore=Encoderrestore, idx=args.idx)

    NorClsfierrestore = osp.join(load_path, f'NorClsfier-{args.defensesnapshot}.pt')
    NorClsfier = init_model(net=DenoiseResnet.NorClassifier(latent_size=args.latent_size, num_classes=nclass),
                            init_type=args.init_type, restore=NorClsfierrestore, parallel_reload=args.parallel_train,idx=args.idx)
    Encoder = DataParallel(Encoder)
    NorClsfier = DataParallel(NorClsfier)

    openmax(args, kdataloader_trn, kdataloader_tst, ukdataloader_tst, knownclass, Encoder, NorClsfier, testPy=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AdvOpenset")

    parser.add_argument('--training_type', type=str, default='RG1')
    parser.add_argument('--parallel_train', type=str, default=True)
    parser.add_argument('--datasetname', type=str, default='imagenet')  # cifar10 tinyimagenet svhn
    parser.add_argument('--split', type=str, default='0')
    parser.add_argument('--imgsize', type=int, default=224)  # cifar svhn 32 tinyimagenet 64 imagenet 224

    parser.add_argument('--adv', type=str, default='FGSM')  # clean PGD FGSM
    parser.add_argument('--adv_iter', type=int, default=5)

    parser.add_argument('--defense', type=str, default='Ours_FD')
    parser.add_argument('--denoisemean', type=str, default='gaussian')
    parser.add_argument('--init_type', type=str, default='normal')

    parser.add_argument('--defensesnapshot', type=str, default='final')
    parser.add_argument('--denoise', type=str, default=[True, True, True, True, True])

    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--latent_size', type=int, default=512)

    parser.add_argument('--results_path', type=str, default='./results_R_G/')
    parser.add_argument('--manual_seed', type=int, default=None)
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--robust', type=bool, default=True) # whether to use a robust block
    parser.add_argument('--idx', type=int, default=0) # GPU index

    print(parser.parse_args())
    test(parser.parse_args())
