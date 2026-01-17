import torch
from torch import nn
import math

class residual_block(nn.Module):
    def __init__(self, input_ch_):
        super(residual_block, self).__init__()
        self.block = nn.Sequential(nn.Linear(input_ch_, input_ch_), nn.LeakyReLU(0.1), nn.Linear(input_ch_, input_ch_))
    def forward(self, input_x):
        return self.block(input_x) + input_x

class FC_layers(nn.Module):
    def __init__(self, input_ch, intermediate_ch, output_ch, num_blocks):
        super(FC_layers, self).__init__()
        layer_list = [nn.Linear(input_ch, intermediate_ch)]
        for block_idx in range(num_blocks):
            layer_list.append(residual_block(intermediate_ch))
        layer_list.append(nn.Linear(intermediate_ch, output_ch))
        self.layers = nn.Sequential(*layer_list)
    def forward(self, input_x):
        return self.layers(input_x)


class residual_block_2(nn.Module):
    def __init__(self, input_ch_):
        super(residual_block_2, self).__init__()
        self.block = nn.Sequential(nn.Linear(input_ch_, input_ch_), nn.LeakyReLU(0.1), nn.Linear(input_ch_, input_ch_), nn.LeakyReLU(0.1))
    def forward(self, input_x):
        return self.block(input_x) + input_x

class FC_layers_2(nn.Module):
    def __init__(self, input_ch, intermediate_ch, output_ch, num_blocks):
        super(FC_layers_2, self).__init__()
        layer_list = [nn.Linear(input_ch, intermediate_ch)]
        for block_idx in range(num_blocks):
            layer_list.append(residual_block_2(intermediate_ch))
        layer_list.append(nn.Linear(intermediate_ch, output_ch))
        self.layers = nn.Sequential(*layer_list)
    def forward(self, input_x):
        return self.layers(input_x)

class FC_layers_2_mod(nn.Module):
    def __init__(self, input_ch, intermediate_ch, output_ch, num_blocks):
        super(FC_layers_2_mod, self).__init__()
        layer_list = [nn.Linear(input_ch, intermediate_ch)]
        for block_idx in range(num_blocks):
            layer_list.append(residual_block_2(intermediate_ch))
        layer_list.append(nn.Linear(intermediate_ch, intermediate_ch))
        self.layers = nn.Sequential(*layer_list)
        self.final = nn.Linear(intermediate_ch, output_ch)
    def forward(self, input_x):
        latent = self.layers(input_x)
        return latent, self.final(latent)


class FC_layers_2_mod_16(nn.Module):
    def __init__(self, input_ch, intermediate_ch, output_ch, num_blocks):
        super(FC_layers_2_mod_16, self).__init__()
        layer_list = [nn.Linear(input_ch, intermediate_ch)]
        for block_idx in range(num_blocks):
            layer_list.append(residual_block_2(intermediate_ch))
        layer_list.append(nn.Linear(intermediate_ch, intermediate_ch))
        self.layers = nn.Sequential(*layer_list)
        self.down_proj = nn.Linear(intermediate_ch, 16)
        self.final = nn.Linear(16, output_ch)
    def forward(self, input_x):
        latent = self.down_proj(self.layers(input_x))
        return latent, self.final(latent)

class FC_layers_3(nn.Module):
    def __init__(self, input_ch, intermediate_ch, num_blocks, max_material_num, max_type_num, max_name_num, residual_latent_ch):
        super(FC_layers_3, self).__init__()
        layer_list = [nn.Linear(input_ch, intermediate_ch)]
        for block_idx in range(num_blocks):
            layer_list.append(residual_block_2(intermediate_ch))
        layer_list.append(nn.Linear(intermediate_ch, intermediate_ch))
        layer_list.append(nn.LeakyReLU(0.1))
        self.layers = nn.Sequential(*layer_list)
        # Double check, is the max material num 0 indexed?

        self.material_prediction = nn.Linear(intermediate_ch, max_material_num)
        self.type_prediction = nn.Linear(intermediate_ch, max_type_num)
        self.residual_latent_mean = nn.Linear(intermediate_ch, residual_latent_ch)
        self.residual_latent_logvar = nn.Linear(intermediate_ch, residual_latent_ch)
        self.position_latent = nn.Linear(intermediate_ch, 2) # We have a 2 channel positional representation
    def forward(self, input_x):
        latent = self.layers(input_x)
        return self.material_prediction(latent), self.type_prediction(latent), self.residual_latent_mean(latent), self.residual_latent_logvar(latent), self.position_latent(latent)