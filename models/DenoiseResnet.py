import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from .cbam import CBAM
from .activation import avaliable_activations
from .normalization import avaliable_normalizations
import math

def conv(in_planes, out_planes, stride=1, kernel_size=3, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,
                     groups=groups, bias=False)

class SqueezeExcitationLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SqueezeExcitationLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class RobustResBlock(nn.Module):
    # expansion = 4
    expansion = 1

    # BN + Skip Connection
    def __init__(self, in_planes, planes, stride=1, kernel_size=3, scales=4, base_width=160, cardinality=1,
                 activation='ReLU', normalization='BatchNorm', se_reduction=16, **kwargs):
        super(RobustResBlock, self).__init__()
        width = int(math.floor(planes * (base_width / 160))) * cardinality
        self.act = avaliable_activations[activation](inplace=True)
        self.bn1 = avaliable_normalizations[normalization](in_planes)
        self.conv1 = nn.Conv2d(in_planes, width * scales, kernel_size=1, bias=False)
        if stride > 1:
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        self.bn2 = avaliable_normalizations[normalization](width * scales)
        if scales == 1:
            self.conv2 = conv(width, width, stride=stride, kernel_size=kernel_size, groups=cardinality)
            self.bn3 = avaliable_normalizations[normalization](width)
        else:
            self.conv2 = nn.ModuleList(
                [conv(width, width, stride=stride, kernel_size=kernel_size, groups=cardinality) for _ in
                 range(scales - 1)])
            self.bn3 = nn.ModuleList([avaliable_normalizations[normalization](width) for _ in range(scales - 1)])
        self.conv3 = nn.Conv2d(width * scales, planes * self.expansion, kernel_size=1, bias=False)
        self.se_bn = avaliable_normalizations[normalization](planes * self.expansion)
        self.se = SqueezeExcitationLayer(planes * self.expansion, reduction=se_reduction)

        if stride > 1 or (in_planes != planes * self.expansion):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )
        self.stride = stride
        self.scales = scales

    def forward(self, x):
        out = self.act(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.act(self.bn2(self.conv1(x)))

        if self.scales == 1:
            out = self.act(self.bn3(self.conv2(out)))
        else:
            xs = torch.chunk(out, self.scales, 1)
            ys = []
            for s in range(self.scales - 1):
                if s == 0 or self.stride > 1:  # if stride > 1, acts like normal bottleneck, without adding
                    input = xs[s]
                else:
                    input = xs[s] + ys[-1]
                ys.append(self.act(self.bn3[s](self.conv2[s](input))))
            ys.append(xs[s + 1] if self.stride == 1 else self.pool(xs[s + 1]))
            out = torch.cat(ys, 1)
        out = self.conv3(out)
        return out + shortcut + self.se(self.se_bn(out))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, latent_size=512, denoise=[True, True, True, True, True]):
        super(ResNet, self).__init__()
        self.denoise = denoise
        self.in_planes = 64

        # print('Latent_size of Encoder: {:.1f}'.format(latent_size))
        # print('denoisemean is : {}'.format(denoisemean))

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if self.denoise[0]:
            self.layer0denoise = CBAM(64, 16)
        if self.denoise[1]:
            self.layer1denoise = CBAM(64, 16)
        if self.denoise[2]:
            self.layer2denoise = CBAM(128, 16)
        if self.denoise[3]:
            self.layer3denoise = CBAM(256, 16)
        if self.denoise[4]:
            self.layer4denoise = CBAM(512, 16)

        self.fc1 = nn.Sequential(nn.Linear(512 * block.expansion, latent_size),
                                 nn.BatchNorm1d(latent_size),
                                 nn.ReLU(True)
                                 )
        print_networks(self, 'encoder')

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.denoise[0]:
            out = self.layer0denoise(out)
        out = self.layer1(out)
        if self.denoise[1]:
            out = self.layer1denoise(out)
        out = self.layer2(out)
        if self.denoise[2]:
            out = self.layer2denoise(out)
        out = self.layer3(out)
        if self.denoise[3]:
            out = self.layer3denoise(out)
        out = self.layer4(out)
        if self.denoise[4]:
            out = self.layer4denoise(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        embed_feat = self.fc1(out)
        return embed_feat


def ResnetEncoder(latent_size, denoise, robust=False):
    block = BasicBlock
    if robust:
        block = RobustResBlock
    return ResNet(block=block, num_blocks=[2, 2, 2, 2], latent_size=latent_size, denoise=denoise)
    # return ResNet(block=RobustResBlock, num_blocks=[2, 2, 2, 2],latent_size=latent_size, denoise=denoise)


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


class ResnetDecoder(nn.Module):
    def __init__(self, latent_size=512, datasetname=None, **kwargs):
        super(self.__class__, self).__init__()
        self.latent_size = latent_size
        self.name = datasetname
        if datasetname == 'imagenet':
            self.fc1 = nn.Linear(latent_size, 512 * 14 * 14, bias=False)
            up = [
                nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, True),

                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, True),

                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, True),

                nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
                nn.Tanh()
            ]
            self.Deconv = nn.Sequential(*up)
            print_networks(self, 'Decoder')

        else:
            self.fc1 = nn.Linear(latent_size, 512 * 2 * 2, bias=False)
            up = [
                nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, True),

                nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, True),

                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, True)
            ]
            if datasetname == 'tinyimagenet':
                up += [
                    nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU(0.2, True)
                ]
            up += [
                nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1),
                nn.Tanh()
            ]
            self.Deconv = nn.Sequential(*up)
            print_networks(self, 'Decoder')

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.fc1(x)
        if self.name == 'imagenet':
            x = x.resize(batch_size, 512, 14, 14)
        else:
            x = x.resize(batch_size, 512, 2, 2)
        x = self.Deconv(x)

        return x


class NorClassifier(nn.Module):
    def __init__(self, latent_size=512, num_classes=10):
        super(self.__class__, self).__init__()

        self.norcls = nn.Linear(latent_size, num_classes)
        print_networks(self, 'NorClassifier')

    def forward(self, x):
        out = self.norcls(x)
        return out


class SSDClassifier(nn.Module):
    def __init__(self, latent_size=512):
        super(self.__class__, self).__init__()

        self.rotcls = nn.Linear(latent_size, 4)
        print_networks(self, 'SSDClassifier')

    def forward(self, x):
        out = self.rotcls(x)
        return out


def print_networks(net, name):
    """Print the total number of parameters in the network and (if verbose) network architecture

    Parameters:
        verbose (bool) -- if verbose: print the network architecture
    """
    print('---------- Networks initialized -------------')
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
        # print(net)
        # a = str(net)
    print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
    print('-----------------------------------------------')
