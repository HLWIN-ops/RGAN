# -*- coding: utf-8 -*
from __future__ import division

import os
import random
from PIL import Image
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.nn import init
import scipy.io as scio
import math
from torch.nn import DataParallel
from torch import nn
from copy import deepcopy
from advertorch.context import ctx_noparamgrad_and_eval

from pdb import set_trace as st


def make_variable(tensor, volatile=False):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().detach().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    return image_numpy.astype(imtype)


def denormalize(x, std, mean):
    """Invert normalization, and then convert array into image."""
    out = x * std + mean
    return out.clamp(0, 1)

def weights_init_const(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.constant_(m.weight.data, 1)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        if m.weight is not None:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)



def weights_init_normal(m):

    classname = m.__class__.__name__
    if (classname.find('Conv2d') != -1):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if (m.bias is not None) and (m.bias.data is not None):
            m.bias.data.zero_()
    elif (classname.find('BatchNorm') != -1):
        if (m.weight is not None) and (m.weight.data is not None):
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
    elif (classname.find('Linear') != -1):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        if m.weight is not None:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

        

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_orthogonal_rnn(m):
    classname = m.__class__.__name__
    if classname.find('LSTM') != -1:
        init.orthogonal_(m.all_weights[0][0], gain=1)
        init.orthogonal_(m.all_weights[0][1], gain=1)
        init.constant_(m.all_weights[0][2], 1)
        init.constant_(m.all_weights[0][3], 1)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    # print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    elif init_type == 'orthogonal_rnn':
        net.apply(weights_init_orthogonal_rnn)
    elif init_type == 'const':
        net.apply(weights_init_const)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)                


def init_random_seed(manual_seed):
    """Init random seed."""
    seed = None
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    # print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed




def init_model(net, restore, init_type, init= True, parallel_reload=True, idx=0):
    """Init models with cuda and weights."""
    # init weights of model
    if init:
        init_weights(net, init_type)
    else:
        print("inside normalization")
    
    # restore model weights
    if restore is not None and os.path.exists(restore):

        # original saved file with DataParallel
        state_dict = torch.load(restore)
        # # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                name = k[7:] # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        # load params
        net.load_state_dict(new_state_dict)
        
        net.restored = True
        print("Restore model from: {}".format(os.path.abspath(restore)))
    else:
        print("******无预训练加载*******")

    if torch.cuda.is_available():
        cudnn.benchmark = True
        device = torch.device(f"cuda:{idx}" if torch.cuda.is_available() else "cpu")
        # net.cuda()
        net.to(device)

    return net


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)




def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil = image_pil.resize((128, 128), resample=Image.BICUBIC)
    image_pil.save(image_path)


def lab_conv(knownclass, label):
    knownclass = sorted(knownclass)
    
    label_convert = torch.zeros(len(label))

    for j in range(len(label)):
        for i in range(len(knownclass)):
            if label[j] == knownclass[i]:
                label_convert[j] = int(knownclass.index(knownclass[i]))

    return label_convert


def noise_fn(image):
    noise = torch.from_numpy(np.random.normal(0, 0.1, image.shape))
    inputs_scale = ((image ** 2).sum()) ** 0.5
    noise_scale = ((noise ** 2).sum()) ** 0.5
    # noise = noise * 0.04 * inputs_scale / noise_scale
    noise = noise * 0.08 * inputs_scale / noise_scale
    image_noisy = torch.clamp(image + noise, -1, 1)
    return image_noisy.float()


class ROA(object):
    '''
    Make sticker
    '''

    def __init__(self, model1, model2, size):
        self.model1 = model1
        self.model2 = model2
        self.img_size = size
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param size: the image size 
        """

    def exhaustive_search(self, X, y, alpha, num_iter, width, height, xskip, yskip, random=False):
        """
        :param X: images from the pytorch dataloaders
        :param y: labels from the pytorch dataloaders
        :param alpha: the learning rate of inside PGD attacks
        :param num_iter: the number of iterations of inside PGD attacks
        :param width: the width of ROA
        :param height: the height of ROA
        :param xskip: the skip (stride) when searching in x axis
        :param yskip: the skip (stride) when searching in y axis
        :param random: the initialization the ROA before inside PGD attacks,
                       True is random initialization, False is 0.5 initialization
        """

        with torch.set_grad_enabled(False):

            model1 = self.model1
            model2 = self.model2
            size = self.img_size

            model1.eval()
            model2.eval()
            # device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
            # X = X.to(device)
            # y = y.to(device)

            max_loss = torch.zeros(y.shape[0]).to(y.device) - 100
            all_loss = torch.zeros(y.shape[0]).to(y.device)


            xtimes = (size - width) // xskip
            ytimes = (size - height) // yskip

            output_j = torch.zeros(y.shape[0]).to(y.device)
            output_i = torch.zeros(y.shape[0]).to(y.device)


            count = torch.zeros(y.shape[0])
            ones = torch.ones(y.shape[0])

            for i in range(xtimes):
                for j in range(ytimes):
                    sticker = X.clone()
                    sticker[:, :, yskip * j:(yskip * j + height), xskip * i:(xskip * i + width)] = 1 / 2
                    all_loss = nn.CrossEntropyLoss(reduction='none')(model2(model1(sticker)), y)
                    padding_j = (torch.zeros(y.shape[0]) + j).to(y.device)
                    padding_i = (torch.zeros(y.shape[0]) + i).to(y.device)
                    output_j[all_loss > max_loss] = padding_j[all_loss > max_loss]
                    output_i[all_loss > max_loss] = padding_i[all_loss > max_loss]
                    count += (all_loss == max_loss).type(torch.FloatTensor)
                    max_loss = torch.max(max_loss, all_loss)

            same_loss = np.transpose(np.argwhere(count >= xtimes * ytimes * 0.9))
            for ind in same_loss:
                output_j[ind] = torch.randint(ytimes, (1,)).type(output_j.dtype)
                output_i[ind] = torch.randint(xtimes, (1,)).type(output_i.dtype)

            zero_loss = np.transpose(np.argwhere(max_loss.cpu() == 0))
            for ind in zero_loss:
                output_j[ind] = torch.randint(ytimes, (1,)).type(output_j.dtype)
                output_i[ind] = torch.randint(xtimes, (1,)).type(output_i.dtype)

        with torch.set_grad_enabled(True):
            return self.inside_pgd(X, y, width, height, alpha, num_iter, xskip, yskip, output_j, output_i)

    def gradient_based_search(self, X, y, alpha, num_iter, width, height, xskip, yskip, potential_nums, random=False):
        """
        :param X: images from the pytorch dataloaders
        :param y: labels from the pytorch dataloaders
        :param alpha: the learning rate of inside PGD attacks
        :param num_iter: the number of iterations of inside PGD attacks
        :param width: the width of ROA
        :param height: the height of ROA
        :param xskip: the skip (stride) when searching in x axis
        :param yskip: the skip (stride) when searching in y axis
        :param potential_nums: the number of keeping potential candidate position
        :param random: the initialization the ROA before inside PGD attacks,
                       True is random initialization, False is 0.5 initialization
        """

        model1 = self.model1
        model2 = self.model2
        size = self.img_size

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        gradient = torch.zeros_like(X, requires_grad=True).to(device)
        X1 = torch.zeros_like(X, requires_grad=True)
        X = X.to(device)
        y = y.to(device)
        X1.data = X.detach().to(device)

        loss = nn.CrossEntropyLoss()(model2(model1(X1)), y)
        loss.backward()

        gradient.data = X1.grad.detach()
        max_val, indice = torch.max(torch.abs(gradient.view(gradient.shape[0], -1)), 1)
        gradient = gradient / max_val[:, None, None, None]
        X1.grad.zero_()

        xtimes = (size - width) // xskip
        ytimes = (size - height) // yskip
        # print(xtimes,ytimes)

        nums = potential_nums
        output_j1 = torch.zeros(y.shape[0]).repeat(nums).view(y.shape[0], nums)
        output_i1 = torch.zeros(y.shape[0]).repeat(nums).view(y.shape[0], nums)
        matrix = torch.zeros([ytimes * xtimes]).repeat(1, y.shape[0]).view(y.shape[0], ytimes * xtimes)
        max_loss = torch.zeros(y.shape[0]).to(y.device)
        all_loss = torch.zeros(y.shape[0]).to(y.device)

        for i in range(xtimes):
            for j in range(ytimes):
                num = gradient[:, :, yskip * j:(yskip * j + height), xskip * i:(xskip * i + width)]
                loss = torch.sum(torch.sum(torch.sum(torch.mul(num, num), 1), 1), 1)
                matrix[:, j * xtimes + i] = loss
        topk_values, topk_indices = torch.topk(matrix, nums)
        output_j1 = topk_indices // xtimes
        output_i1 = topk_indices % xtimes

        output_j = torch.zeros(y.shape[0]) + output_j1[:, 0].float()
        output_i = torch.zeros(y.shape[0]) + output_i1[:, 0].float()

        with torch.set_grad_enabled(False):
            for l in range(output_j1.size(1)):
                sticker = X.clone()
                for m in range(output_j1.size(0)):
                    sticker[m, :, yskip * output_j1[m, l]:(yskip * output_j1[m, l] + height),
                    xskip * output_i1[m, l]:(xskip * output_i1[m, l] + width)] = 1 / 2
                sticker1 = sticker.detach()
                all_loss = nn.CrossEntropyLoss(reduction='none')(model2(model1(sticker1)), y)
                padding_j = torch.zeros(y.shape[0]) + output_j1[:, l].float()
                padding_i = torch.zeros(y.shape[0]) + output_i1[:, l].float()
                output_j[all_loss > max_loss] = padding_j[all_loss > max_loss]
                output_i[all_loss > max_loss] = padding_i[all_loss > max_loss]
                max_loss = torch.max(max_loss, all_loss)

        return self.inside_pgd(X, y, width, height, alpha, num_iter, xskip, yskip, output_j, output_i)

    def inside_pgd(self, X, y, width, height, alpha, num_iter, xskip, yskip, out_j, out_i, random=False):
        model1 = self.model1
        model2 = self.model2
        model2.eval()
        model1.eval()
        sticker = torch.zeros(X.shape, requires_grad=False)
        for num, ii in enumerate(out_i):
            j = int(out_j[num].item())
            i = int(ii.item())
            sticker[num, :, yskip * j:(yskip * j + height), xskip * i:(xskip * i + width)] = 1
        sticker = sticker.to(y.device)

        if random == False:
            delta = torch.zeros_like(X, requires_grad=True) + 1 / 2
        else:
            delta = torch.rand_like(X, requires_grad=True).to(y.device)
            delta.data = delta.data * 255

        X1 = torch.rand_like(X, requires_grad=True).to(y.device)
        X1.data = X.detach() * (1 - sticker) + ((delta.detach()) * sticker)

        for t in range(num_iter):
            loss = nn.CrossEntropyLoss()(model2(model1(X1)), y)
            loss.backward()
            X1.data = (X1.detach() + alpha * X1.grad.detach().sign() * sticker)
            X1.data = (X1.detach()).clamp(0, 1)
            X1.grad.zero_()
        return (X1).detach()

