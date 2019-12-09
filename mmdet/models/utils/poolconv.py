import torch
import copy
import torch.nn as nn
import torch.nn.functional as F

class pool_conv(nn.Module):
    def __init__(self,
                 pooling_size,
                 in_channel,
                 stride=2,
                 padding=0,
                 # group=1,
                 ):
        # super(pool_conv, self).__init__()
        super(pool_conv,self).__init__()
        self.pooling_size = pooling_size
        self.in_channel = in_channel
        self.stride = stride
        self.padding  = padding
        # self.group = group
        self.zero_kernel = torch.zeros((pooling_size, pooling_size))
        self.kernels = []
        for i in range(pooling_size):
            for j in range(pooling_size):
                pool_conv_kernel = torch.zeros_like(self.zero_kernel)
                pool_conv_kernel[i][j] = 1
                self.kernels.append(pool_conv_kernel.repeat(in_channel, 1, 1, 1))

    def forward(self, inputs):
        if not torch.is_tensor(inputs):
            raise TypeError("Input type for pool conv should be tensor")
        B,C,H,W = inputs.size()
        if not C == self.kernels[0].size(0):
            raise ValueError("Input tensor has different channel size with pool conv")
        results = [F.conv2d(inputs,
                            kernel.to(inputs.device),
                            stride = self.stride,
                            padding = self.padding,
                            groups= self.in_channel)
                   for kernel in self.kernels]
        results = torch.cat(results, dim=1)
        return results







