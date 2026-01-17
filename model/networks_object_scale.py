import torch
from torch import nn
import math
import timm


def convert_relu_to_softplus(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.LeakyReLU(0.1))
        else:
            convert_relu_to_softplus(child)


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
        self.block = nn.Sequential(nn.Linear(input_ch_, input_ch_), nn.LeakyReLU(0.1), nn.Linear(input_ch_, input_ch_),
                                   nn.LeakyReLU(0.1))

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

class total2stft(nn.Module):
    def __init__(self,input_ch):
        super(total2stft, self).__init__()
        self.fc1 = nn.Linear(input_ch, 256)
        self.fc2 = nn.Linear(256, 600)
        self.fc3 = nn.Linear(600, 256)
        self.fc4 = nn.Linear(256, 2 * 256 * 600)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = x.view(-1, 2, 256, 600)
        return x



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
    def __init__(self, input_ch, intermediate_ch, num_blocks, max_material_num, max_type_num, max_name_num,
                 residual_latent_ch):
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
        self.position_latent = nn.Linear(intermediate_ch, 2)  # We have a 2 channel positional representation

    def forward(self, input_x):
        latent = self.layers(input_x)
        return self.material_prediction(latent), self.type_prediction(latent), self.residual_latent_mean(
            latent), self.residual_latent_logvar(latent), self.position_latent(latent)


class myresnet(nn.Module):
    def __init__(self, max_material_num,  residual_latent_ch):
        super(myresnet, self).__init__()
        # layer_list = [nn.Linear(input_ch, intermediate_ch)]
        # for block_idx in range(num_blocks):
        #     layer_list.append(residual_block_2(intermediate_ch))
        # layer_list.append(nn.Linear(intermediate_ch, intermediate_ch))
        # layer_list.append(nn.LeakyReLU(0.1))
        # self.layers = nn.Sequential(*layer_list)

        net = timm.create_model('resnet18', pretrained=False, num_classes=0)
        #net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # out 2048
        net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # in_channel = 2048
        # out_channel = intermediate_ch
        # net.fc = nn.Linear(in_channel, out_channel)
        # net = net.to(output_device)



        #convert_relu_to_softplus(net)
        #print(net)
        self.layers = net

        self.material_prediction = nn.Linear(512, max_material_num)
        self.scale_prediction = nn.Linear(512, 1)

        self.residual_latent_mean = nn.Linear(512, residual_latent_ch)
        self.residual_latent_logvar = nn.Linear(512, residual_latent_ch)
        #print("here2")


    def forward(self, input_x):
        latent = self.layers(input_x)
        #latent=0.5*nn.Sigmoid(latent)
        #print(self.material_prediction(self.layers()))
        #exit(-1)
        #print("here1")
        return self.material_prediction(latent),0.5*torch.sigmoid(self.scale_prediction(latent)), self.residual_latent_mean(
            latent), self.residual_latent_logvar(latent)

