import os
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from models.DenoiseResnet import print_networks


class resnet18(nn.Module):

    def __init__(self, opt, num_classes=10):
        super(resnet18, self).__init__()
        self.opt = opt
        self.isTrain = self.opt.train
        self.target_model = torchvision.models.resnet18(pretrained=False)
        # self.target_model.load_state_dict(torch.load('./pretrained/resnet18.pth'))
        self.num_fits = self.target_model.fc.in_features
        # if num_classes==10:
        #     self.target_model.conv1.kernel_size = (3, 3)
        #     self.target_model.conv1.stride = (1, 1)
        #     self.target_model.conv1.padding = (1, 1)
        #     self.target_model.maxpool.stride = 1
        self.target_model.fc = nn.Linear(self.num_fits, num_classes)

        assert (torch.cuda.is_available())
        # self.target_model.cuda()
        self.netresnet18 = self.target_model
        self.softmax = nn.Softmax(dim=-1)
        print_networks(self,'peer_resnet18')

    def forward(self, x):
        self.logits = self.netresnet18(x)
        return self.logits


    def backward(self):
        self.loss_loss = F.cross_entropy(self.logits, self.label).to(self.device)
        self.loss_loss.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizers.zero_grad()
        self.backward()
        self.optimizers.step()
