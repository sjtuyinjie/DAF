import torch
from torch import nn
import math
import numpy as np
from model.modules import fit_predict_torch

class basic_project2(nn.Module):
    def __init__(self, input_ch, output_ch):
        super(basic_project2, self).__init__()
        self.proj = nn.Linear(input_ch, output_ch, bias=True)
    def forward(self, x):
        return self.proj(x)

class kernel_linear_act(nn.Module):
    def __init__(self, input_ch, output_ch):
        super(kernel_linear_act, self).__init__()
        self.block = nn.Sequential(nn.LeakyReLU(negative_slope=0.1), basic_project2(input_ch, output_ch))
    def forward(self, input_x):
        return self.block(input_x)

class kernel_residual_fc_embeds(nn.Module):
    def __init__(self, input_ch, intermediate_ch=512, grid_ch = 64, num_block=8, output_ch=1, grid_gap=0.25, grid_bandwidth=0.25, bandwidth_min=0.1, bandwidth_max=0.5, float_amt=0.1, min_xy=None, max_xy=None, probe=False, num_objs=140):
        super(kernel_residual_fc_embeds, self).__init__()
        # input_ch (int): number of ch going into the network
        # intermediate_ch (int): number of intermediate neurons
        # min_xy, max_xy are the bounding box of the room in real (not normalized) coordinates
        # probe = True returns the features of the last layer

        for k in range(num_block - 1):
            self.register_parameter("left_right_{}".format(k),nn.Parameter(torch.randn(1, 1, 2, intermediate_ch)/math.sqrt(intermediate_ch),requires_grad=True))

        self.proj = basic_project2(input_ch, intermediate_ch)
        self.residual_1 = nn.Sequential(basic_project2(input_ch, intermediate_ch), nn.LeakyReLU(negative_slope=0.1), basic_project2(intermediate_ch, intermediate_ch))
        self.layers = torch.nn.ModuleList()
        for k in range(num_block - 2):
            self.layers.append(kernel_linear_act(intermediate_ch, intermediate_ch))

        self.out_layer = nn.Linear(intermediate_ch, output_ch)
        self.blocks = len(self.layers)
        self.probe = probe

        # for k in range(len(self.layers)):
        # for k in range(1):
        #     setattr(self, "embedding_{}".format(k), nn.Embedding(num_objs, intermediate_ch))
        # ### Make the grid

    def forward(self, input_stuff, obj_index):
        # SAMPLES = input_stuff.shape[1]
        # print(getattr(self, "embedding_{}".format(0))(obj_index)[:,None, None].shape, getattr(self, "left_right_0").shape, "SHAPE")
        # assert False
        my_input = input_stuff

        # out is torch.Size([5, 2000, 2, 512]) MYINPUT
        out = self.proj(my_input).unsqueeze(2).repeat(1, 1, 2, 1) + getattr(self, "left_right_0")

        for k in range(len(self.layers)):
            # print(self.layers[k](out).shape, getattr(self, "left_right_{}".format(k + 1)).shape, getattr(self, "embedding_{}".format(k))(obj_index).shape)
            # exit()
            out = self.layers[k](out) + getattr(self, "left_right_{}".format(k + 1))
            if k == (self.blocks // 2 - 1):
                out = out + self.residual_1(my_input).unsqueeze(2).repeat(1, 1, 2, 1)
        if self.probe:
            return out
        return self.out_layer(out)