import torch
from torch import nn
import torch.nn.init as init
from lc_help_funcs import concat_input_labels, create_random_labels, to_one_hot
from constants import *


class MyConvo2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init = True,  stride = 1, bias = True):
        super(MyConvo2d, self).__init__()
        self.he_init = he_init
        self.padding = int((kernel_size - 1 ) /2)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=self.padding, bias = bias)

    def forward(self, input):
        output = self.conv(input)
        return output


class ConvMeanPool(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init = True):
        super(ConvMeanPool, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size, he_init = self.he_init)

    def forward(self, input):
        output = self.conv(input)
        output = (output[: ,: ,::2 ,::2] + output[: ,: ,1::2 ,::2] + output[: ,: ,::2 ,1::2] + output[: ,: ,1::2
                                                                                               ,1::2]) / 4
        return output


class MeanPoolConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init = True):
        super(MeanPoolConv, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size, he_init = self.he_init)

    def forward(self, input):
        output = input
        output = ((output[:, :, ::2, ::2] + output[:, :, 1::2, ::2] + output[:, :, ::2, 1::2] + output[:, :, 1::2,
                                                                                                1::2]) / 4)
        output = self.conv(output)
        return output


class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, input_height, input_width, input_depth) = output.size()
        output_depth = int(input_depth / self.block_size_sq)
        output_width = int(input_width * self.block_size)
        output_height = int(input_height * self.block_size)
        t_1 = output.reshape(batch_size, input_height, input_width, self.block_size_sq, output_depth)
        spl = t_1.split(self.block_size, 3)
        stacks = [t_t.reshape(batch_size ,input_height ,output_width ,output_depth) for t_t in spl]
        output = torch.stack(stacks ,0).transpose(0 ,1).permute(0 ,2 ,1 ,3 ,4).reshape(batch_size ,output_height
                                                                                       ,output_width ,output_depth)
        output = output.permute(0, 3, 1, 2)
        return output


class UpSampleConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init = True, bias=True):
        super(UpSampleConv, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size, he_init = self.he_init, bias=bias)
        self.depth_to_space = DepthToSpace(2)

    def forward(self, input):
        output = input
        output = torch.cat((output, output, output, output), 1)
        output = self.depth_to_space(output)
        output = self.conv(output)
        return output


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, resample=None, hw=DIM):
        super(ResidualBlock, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.resample = resample
        self.bn1 = None
        self.bn2 = None
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        if resample == 'down':
            self.bn1 = nn.LayerNorm([input_dim, hw, hw])
            self.bn2 = nn.LayerNorm([input_dim, hw, hw])
        elif resample == 'up':
            self.bn1 = nn.BatchNorm2d(input_dim)
            self.bn2 = nn.BatchNorm2d(output_dim)
        elif resample is None:
            self.bn1 = nn.BatchNorm2d(output_dim)
            self.bn2 = nn.LayerNorm([input_dim, hw, hw])
        else:
            raise Exception('invalid resample value')

        if resample == 'down':
            self.conv_shortcut = MeanPoolConv(input_dim, output_dim, kernel_size = 1, he_init = False)
            self.conv_1 = MyConvo2d(input_dim, input_dim, kernel_size = kernel_size, bias = False)
            self.conv_2 = ConvMeanPool(input_dim, output_dim, kernel_size = kernel_size)
        elif resample == 'up':
            self.conv_shortcut = UpSampleConv(input_dim, output_dim, kernel_size = 1, he_init = False)
            self.conv_1 = UpSampleConv(input_dim, output_dim, kernel_size = kernel_size, bias = False)
            self.conv_2 = MyConvo2d(output_dim, output_dim, kernel_size = kernel_size)
        elif resample is None:
            self.conv_shortcut = MyConvo2d(input_dim, output_dim, kernel_size = 1, he_init = False)
            self.conv_1 = MyConvo2d(input_dim, input_dim, kernel_size = kernel_size, bias = False)
            self.conv_2 = MyConvo2d(input_dim, output_dim, kernel_size = kernel_size)
        else:
            raise Exception('invalid resample value')

    def forward(self, input):
        if self.input_dim == self.output_dim and self.resample is None:
            shortcut = input
        else:
            shortcut = self.conv_shortcut(input)

        output = input
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.conv_1(output)
        output = self.bn2(output)
        output = self.relu2(output)
        output = self.conv_2(output)

        return shortcut + output


class ResidualBlock_lc(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, resample=None, hw=DIM, num_of_classes=0): # cohen we added num of classes
        super(ResidualBlock_lc, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.resample = resample
        self.bn1 = None
        self.bn2 = None
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.num_of_classes = num_of_classes
        if resample == 'down':
            self.bn1 = nn.LayerNorm([input_dim, hw, hw])
            self.bn2 = nn.LayerNorm([input_dim, hw, hw])
        elif resample == 'up':
            self.bn1 = nn.BatchNorm2d(input_dim)
            self.bn2 = nn.BatchNorm2d(output_dim)
        elif resample is None:
            self.bn1 = nn.BatchNorm2d(output_dim)
            self.bn2 = nn.LayerNorm([input_dim, hw, hw])
        else:
            raise Exception('invalid resample value')

        if resample == 'down':
            self.conv_shortcut = MeanPoolConv(input_dim, output_dim, kernel_size = 1, he_init = False) # cohen - our article does that only in resnet 32, in other networks uses ConvMeanPool
            self.conv_1 = MyConvo2d(input_dim + self.num_of_classes, input_dim, kernel_size = kernel_size) # cohen we deleted - bias = False
            self.conv_2 = ConvMeanPool(input_dim + self.num_of_classes, output_dim, kernel_size = kernel_size)
        elif resample == 'up':
            self.conv_shortcut = UpSampleConv(input_dim, output_dim, kernel_size = 1, he_init = False)
            self.conv_1 = UpSampleConv(input_dim +self.num_of_classes, output_dim, kernel_size = kernel_size) # cohen we deleted - bias = False
            self.conv_2 = MyConvo2d(output_dim +self.num_of_classes, output_dim, kernel_size = kernel_size)
        elif resample is None:
            self.conv_shortcut = MyConvo2d(input_dim, output_dim, kernel_size = 1, he_init = False)
            self.conv_1 = MyConvo2d(input_dim + self.num_of_classes, input_dim, kernel_size = kernel_size) # cohen we deleted - bias = False
            self.conv_2 = MyConvo2d(input_dim +self.num_of_classes, output_dim, kernel_size = kernel_size)
        else:
            raise Exception('invalid resample value')

    def forward(self, input, labels):
        if self.input_dim == self.output_dim and self.resample is None:
            shortcut = input
        else:
            shortcut = self.conv_shortcut(input)

        output = self.bn1(input)
        output = self.relu1(output)
        output = concat_input_labels(output, labels, self.num_of_classes) # cohen we added line and func
        output = self.conv_1(output)
        output = self.bn2(output)
        output = self.relu2(output)
        output = concat_input_labels(output, labels, self.num_of_classes) # cohen we added line and func
        output = self.conv_2(output)
        return shortcut + output


class ReLULayer(nn.Module):
    def __init__(self, n_in, n_out):
        super(ReLULayer, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.linear(input)
        output = self.relu(output)
        return output


class FCGenerator(nn.Module):
    def __init__(self, fc_dim=512):
        super(FCGenerator, self).__init__()
        self.relulayer1 = ReLULayer(128, fc_dim)
        self.relulayer2 = ReLULayer(fc_dim, fc_dim)
        self.relulayer3 = ReLULayer(fc_dim, fc_dim)
        self.relulayer4 = ReLULayer(fc_dim, fc_dim)
        self.linear = nn.Linear(fc_dim, OUTPUT_DIM)
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.relulayer1(input)
        output = self.relulayer2(output)
        output = self.relulayer3(output)
        output = self.relulayer4(output)
        output = self.linear(output)
        output = self.tanh(output)
        return output


class Generator(nn.Module):
    def __init__(self, dim=G_DIM , output_dim=OUTPUT_DIM, num_of_classes=0):
        super(Generator, self).__init__()

        self.dim = dim
        self.num_of_classes = num_of_classes

        self.ln1 = nn.Linear(128 +num_of_classes, 4* 4 * 8 * self.dim)
        self.rb1 = ResidualBlock_lc(8 * self.dim, 8 * self.dim, 3, resample='up',
                                    num_of_classes=num_of_classes)  # cohen added num classes
        self.rb2 = ResidualBlock_lc(8 * self.dim, 4 * self.dim, 3, resample='up',
                                    num_of_classes=num_of_classes)  # cohen added num classes
        self.rb3 = ResidualBlock_lc(4 * self.dim, 2 * self.dim, 3, resample='up',
                                    num_of_classes=num_of_classes)  # cohen added num classes
        self.rb4 = ResidualBlock_lc(2 * self.dim, 1 * self.dim, 3, resample='up',
                                    num_of_classes=num_of_classes)  # cohen added num classes
        self.bn = nn.BatchNorm2d(self.dim)

        self.conv1 = MyConvo2d(1 * self.dim + num_of_classes, 3, 3)  # cohen added num classes
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, input, labels):
        # concat input and lables
        input = torch.cat((input, labels), 1)
        output = self.ln1(input.contiguous())
        output = output.view(-1, 8 * self.dim, 4, 4)
        output = self.rb1(output, labels)
        output = self.rb2(output, labels)
        output = self.rb3(output, labels)
        output = self.rb4(output, labels)

        output = self.bn(output)
        output = self.relu(output)
        output = concat_input_labels(output, labels, self.num_of_classes)
        output = self.conv1(output)
        output = self.tanh(output)
        output = output.view(-1, OUTPUT_DIM)
        return output


class Discriminator(nn.Module):
    def __init__(self, dim=D_DIM, num_of_classes=0):
        super(Discriminator, self).__init__()

        self.dim = dim
        self.num_of_classes = num_of_classes  # cohen we added line

        self.conv1 = MyConvo2d(3 + num_of_classes, self.dim, 3, he_init=False)
        self.rb1 = ResidualBlock_lc(self.dim, 2 * self.dim, 3, resample='down', hw=DIM, num_of_classes=num_of_classes)
        self.rb2 = ResidualBlock_lc(2 * self.dim, 4 * self.dim, 3, resample='down', hw=int(DIM / 2),
                                    num_of_classes=num_of_classes)
        self.rb3 = ResidualBlock_lc(4 * self.dim, 8 * self.dim, 3, resample='down', hw=int(DIM / 4),
                                    num_of_classes=num_of_classes)
        self.rb4 = ResidualBlock_lc(8 * self.dim, 8 * self.dim, 3, resample='down', hw=int(DIM / 8),
                                    num_of_classes=num_of_classes)
        self.ln1 = nn.Linear(4 * 4 * 8 * self.dim + num_of_classes, num_of_classes)

    def forward(self, input, labels):

        input = input.view(-1, 3, 64, 64)
        output = concat_input_labels(input, labels, self.num_of_classes)
        output = self.conv1(output)
        output = self.rb1(output, labels)
        output = self.rb2(output, labels)
        output = self.rb3(output, labels)
        output = self.rb4(output, labels)
        output = output.view(-1, 4 * 4 * 8 * self.dim)
        output = torch.cat((output, labels), 1)
        output = self.ln1(output)
        output = output.view(-1)
        return output


def weights_init(m):
    """
    :param m: the layer which we want to init
    """
    if isinstance(m, MyConvo2d):
        if m.conv.weight is not None:
            if m.he_init:
                init.kaiming_uniform_(m.conv.weight)
            else:
                init.xavier_uniform_(m.conv.weight)
        if m.conv.bias is not None:
            init.constant_(m.conv.bias, 0.0)
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)
