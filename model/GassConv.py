import math

import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils import weight_norm
import torch.nn.functional as F


class Chomp2d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp2d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()

class PeriodicityModule(nn.Module):
    def __init__(self, all_a, all_p, all_f, all_mean, device):
        super(PeriodicityModule, self).__init__()
        all_a = torch.tensor(all_a, dtype=torch.float32).to(device)
        all_p = torch.tensor(all_p, dtype=torch.float32).to(device)
        all_f = torch.tensor(all_f, dtype=torch.float32).to(device)
        all_mean = torch.tensor(all_mean, dtype=torch.float32).to(device)
        self.num_series, self.K = all_a.shape #num_series * K
        # inner cos
        self.layers1_weight = torch.nn.Parameter(all_f.unsqueeze(1)) # num_series * 1 * K
        self.layers1_bias = torch.nn.Parameter(all_p.unsqueeze(1)) # num_series * 1 * K
        # outside cos
        self.layers2_weight = torch.nn.Parameter(all_a.unsqueeze(-1)) # num_series * K * 1
        self.layers2_bias = torch.nn.Parameter(all_mean.unsqueeze(-1).unsqueeze(-1)) # num_series * 1 * 1

    def forward(self, x, series_id, inputKmask=None):
        # x: barch*backcast_length, series_id:batch
        #output: batch* backcast_length
        batch_layer1_weight = self.layers1_weight.index_select(0, series_id.view(-1))
        batch_layer1_bias = self.layers1_bias.index_select(0, series_id.view(-1))
        batch_layer2_weight = self.layers2_weight.index_select(0, series_id.view(-1))  # batch * K * 1
        batch_layer2_bias = self.layers2_bias.index_select(0, series_id.view(-1)) # batch * 1 * 1
        # forward function
        # batch * backcast * k
        x = 2 * np.pi * torch.matmul(x.unsqueeze(-1), batch_layer1_weight) + batch_layer1_bias
        if inputKmask is None:
            x = torch.cos(x)
        else:
            inputKmask = inputKmask.unsqueeze(1).repeat(1, x.shape[2], 1)
            x = torch.mul(torch.cos(x), inputKmask)
        x = torch.matmul(x, batch_layer2_weight) + batch_layer2_bias
        return x.squeeze(-1)


class SpatioConvBlack(nn.Module):
    def __init__(self, ks, c):
        super(SpatioConvBlack, self).__init__()
        self.theta = nn.Parameter(torch.FloatTensor(c, c, ks))
        self.b = nn.Parameter(torch.FloatTensor(1, c, 1, 1))
        self.reset_parameters()
    def reset_parameters(self):
        init.kaiming_uniform(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1/math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x, Lk):
        residual = x
        x_c = torch.einsum("knm, bimt -> binkt", Lk, x)
        x_gc = torch.einsum("iok, binkt->bont", self.theta, x_c) + self.b
        x_gc = torch.relu(x_gc+residual)

        return x_gc




class GaussBlack(nn.Module):
    def __init__(self, n_input, n_output, g_kernel_data, kernel_size, stride, dilation, padding, dropout=0.2):
        super(GaussBlack, self).__init__()
        self.g_data = g_kernel_data
        self.g_data = nn.Parameter(data=torch.FloatTensor(self.g_data).expand(n_input, n_input, self.g_data.shape[0], self.g_data.shape[1]),
            requires_grad=False
        )
        # 更新权重，进行学习
        self.conv1 = weight_norm(nn.Conv2d(
            n_input, n_output, (1, kernel_size), stride=stride, padding=padding, dilation=dilation))
        # f.conv2d() weight卷积核(Cout, Cin, H, W)
        # (1, x)对x进行操作
        self.chomp1 = Chomp2d(padding[1])
        self.relu1 = nn.Tanh()
        self.dropout1 = nn.Dropout(dropout)

        # 更新权重，进行学习
        self.conv2 = weight_norm(nn.Conv2d(
            n_output, n_output, (1, kernel_size), stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp2d(padding[1])
        self.relu2 = nn.Tanh()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                               self.conv2, self.chomp2, self.relu2, self.dropout2)
        # self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1)

        # gass
        # self.downsample = nn.Conv2d(n_input, n_output, (1, kernel_size))
        self.downsample = nn.Conv2d(n_input, n_output, (1, 1))
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        # g_data (out_channel, in_channel, H, W)
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # x shape (batch, channel, H, W), H:nodes, W:input_timesteps
        # Gauss Conv
        # out = nn.functional.conv2d(x, self.g_data, stride=1)
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)

        return self.relu(out + res)

class GaussConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, g_kernel, kernel_size=2, dropout=0.2):
        super(GaussConvNet, self).__init__()
        layer = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            # i = 0 第一层等于，输入；后面等于隐层的数量
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layer += [GaussBlack(in_channels, out_channels, g_kernel, kernel_size, stride=1, dilation=dilation_size,
                                 padding=(0, (kernel_size-1)*dilation_size), dropout=dropout)]
        self.network = nn.Sequential(*layer)

    def forward(self, x):
        return self.network(x)

class GTCN(nn.Module):
    def __init__(self, FFT_data, Ks, input_size, input_timesteps, output_size, num_channels, g_kernel, kernel_size, dropout, device):
        super(GTCN, self).__init__()
        self.all_a = FFT_data['all_a']
        self.all_p = FFT_data['all_p']
        self.all_f = FFT_data['all_f']
        self.all_mean = FFT_data['all_mean']
        self.PM = PeriodicityModule(self.all_a, self.all_p, self.all_f, self.all_mean, device)
        self.GTCN = GaussConvNet(input_size, num_channels, g_kernel=g_kernel, kernel_size=kernel_size, dropout=dropout)
        self.Sconv = SpatioConvBlack(Ks, num_channels[-1])
        # gass
        # self.linear = nn.Linear(num_channels[-1]*(input_timesteps-len(num_channels)*kernel_size+len(num_channels)), output_size)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(num_channels[-1] * input_timesteps, output_size)

    def forward(self, inputs, Lk, input_timestamp, ids, mask):
        p_f = self.PM(input_timestamp, ids.long(), mask)
        p_f = p_f.unsqueeze(1)
        inputs = inputs.permute(0, 3, 1, 2)
        y1 = self.GTCN(inputs)
        y2 = self.Sconv(y1, Lk)
        y3 = p_f * y2
        y3 = y3.reshape(y1.shape[0], y1.shape[2], y1.shape[1]*y1.shape[3])
        out = self.linear(y3)
        return out



