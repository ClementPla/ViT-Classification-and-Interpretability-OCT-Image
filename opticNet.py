import torch
import torch.nn as nn
import torch.nn.functional as F
from nntools.nnet import AbstractNet


class ResConv(nn.Module):
    def __init__(self, input_chan, features, stride, base_name=''):
        super(ResConv, self).__init__()

        self.branch_1 = nn.Sequential(nn.BatchNorm2d(input_chan),
                                      nn.ReLU(),
                                      nn.Conv2d(input_chan, features[0], stride=1, kernel_size=1),
                                      nn.BatchNorm2d(features[0]),
                                      nn.ReLU(),
                                      nn.ReflectionPad2d((1, 0, 1, 0)),
                                      nn.Conv2d(features[0], features[1], stride=stride, kernel_size=(2, 2), padding=0),
                                      nn.BatchNorm2d(features[1]),
                                      nn.ReLU(),
                                      nn.Conv2d(features[1], features[2], kernel_size=(1, 1), stride=(1, 1))
                                      )
        self.branch_2 = nn.Sequential(nn.BatchNorm2d(input_chan), nn.ReLU(), nn.Conv2d(input_chan, features[2],
                                                                                       kernel_size=(1, 1),
                                                                                       stride=stride))
        self.base_name = base_name

    def forward(self, x):
        return self.branch_1(x) + self.branch_2(x)

    def param_order(self):

        params = ['1/bn_1', '1/conv_1', '1/bn_2', '1/conv_2', '1/bn_3', '1/conv_3', '2/bn_1', '2/conv_1']
        return [self.base_name + name for name in params]


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, bias=True, padding=0):
        super(SeparableConv2d, self).__init__()

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding=padding,
                                   dilation=dilation,
                                   groups=in_channels,
                                   bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class ResIdentity(nn.Module):
    def __init__(self, input_chan, features):
        super(ResIdentity, self).__init__()
        self.layer_1 = nn.Sequential(nn.BatchNorm2d(input_chan), nn.ReLU(), nn.Conv2d(input_chan,
                                                                                      features[0],
                                                                                      kernel_size=(1, 1)))

        self.branch_1 = nn.Sequential(nn.BatchNorm2d(features[0]), nn.ReLU(), nn.Conv2d(features[0], features[1],
                                                                                        dilation=(2, 2), padding=1,
                                                                                        kernel_size=(2, 2)))

        self.branch_2 = nn.Sequential(nn.BatchNorm2d(features[0]), nn.ReLU(), SeparableConv2d(features[0], features[1],
                                                                                              dilation=(2, 2),
                                                                                              padding=1,
                                                                                              kernel_size=(2, 2)))
        self.out = nn.Sequential(nn.BatchNorm2d(features[1]), nn.ReLU(), nn.Conv2d(features[1],
                                                                                   features[2],
                                                                                   kernel_size=(1, 1)))

    def forward(self, x):
        x_shortcut = x
        x = self.layer_1(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x = x1 + x2
        return x_shortcut + self.out(x)


class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape
        x = F.adaptive_max_pool2d(x, (h // 2, w // 2))
        x = F.interpolate(x, size=(h, w), mode='bilinear')
        return torch.sigmoid(x)


class RDBI(nn.Module):
    def __init__(self, input_chan, features, iterations):
        super(RDBI, self).__init__()
        layers = [ResIdentity(input_chan, features)]
        for i in range(iterations - 1):
            layers.append(ResIdentity(features[-1], features))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MidBlock(nn.Module):
    def __init__(self, input_chan, rdbi_features, rdbi_iteration, res_conv_features=None, stride=1):
        super(MidBlock, self).__init__()
        self.encoder_decoder = EncoderDecoder()
        self.rdbi = RDBI(input_chan, rdbi_features, rdbi_iteration)
        if res_conv_features is not None:
            self.res_conv = ResConv(input_chan, res_conv_features, stride)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x):
        x1 = self.encoder_decoder(x)
        x2 = self.rdbi(x)
        x = x1 * x2
        x = x + x1 + x2
        return self.res_conv(x)


class OpticNet(AbstractNet):
    def __init__(self, n_channels=3, n_classes=4):
        super(OpticNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.first_layer = nn.Sequential(nn.Conv2d(self.n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=3),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(),
                                         ResConv(64, [64, 64, 256], 1))
        self.mid1 = MidBlock(256, [32, 32, 256], 4, [128, 128, 512], 2)
        self.mid2 = MidBlock(512, [64, 64, 512], 4, [256, 256, 1024], 2)
        self.mid3 = MidBlock(1024, [128, 128, 1024], 3, [512, 512, 2048], 2)
        self.mid4 = MidBlock(2048, [256, 256, 2048], 3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(2048, 256), nn.Linear(256, self.n_classes))

    def forward(self, x):
        x = self.first_layer(x)
        x = self.mid1(x)

        x = self.mid2(x)

        x = self.mid3(x)

        x = self.mid4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return self.fc(x)


if __name__ == '__main__':
    model = OpticNet(3, 5)

    foo = torch.zeros((1, 3, 224, 224))
    model(foo)
