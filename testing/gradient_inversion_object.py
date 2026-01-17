import matplotlib.pyplot as plt
import numpy as np
import torch

torch.backends.cudnn.benchmark = True
from inspect import getsourcefile
import os.path as path, sys

current_dir = path.dirname(path.abspath(getsourcefile(lambda: 0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])

import torch

torch.backends.cudnn.benchmark = True

from data_loading.sound_loader_object import soundsamples
import torch.multiprocessing as mp
import os
import socket
from contextlib import closing
import torch.distributed as dist
from model.networks_object import FC_layers, FC_layers_2, FC_layers_3, FC_layers_2_mod, FC_layers_2_mod_16, myresnet
from model.modules import embedding_module_log
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import math
from time import time
from options import Options
import functools
import pickle
from itertools import chain

from torch import nn
import timm
import torch

import timm


def to_torch(input_arr):
    return input_arr[None]


def test_net(rank, other_args):
    pi = math.pi
    output_device = rank
    print("creating dataset")
    dataset = soundsamples(other_args)
    reconstruction_criterion = torch.nn.MSELoss()

    map_location = 'cuda:%d' % rank
    # weights = torch.load("/home/aluo/PycharmProjects/Neural_Acoustic_Fields_regress/results/frl_apartment_4/00088.chkpt", map_location=map_location)

    # weights = torch.load("/home/aluo/PycharmProjects/Neural_Acoustic_Fields_regress/results/apartment_1/00155.chkpt", map_location=map_location)
    # weights = torch.load("//home/zdp21n5/projects/aresnettest/results/apartment_1_vae_resnet_620//00620.chkpt", map_location=map_location)
    weights = torch.load("//data/vision/torralba/scratch/chuang/Projects/nap_vision/results/object_ours/00200.chkpt",
                         map_location=map_location)
    print("Checkpoint loaded")

    # encoder = FC_layers_2_mod_16(input_ch=129+129, intermediate_ch=512, output_ch=40, num_blocks=4).to(output_device)
    # decoder = FC_layers_2(input_ch=16+2, intermediate_ch=512, output_ch=129+129, num_blocks=4).to(output_device)
    # new
    LATENT_CH = 8
    # encoder = myresnet(max_material_num=dataset.max_material,
    #                    max_type_num=dataset.max_type, max_name_num=dataset.max_name, residual_latent_ch=LATENT_CH).to(
    #     output_device)
    print("mat:", dataset.max_material)  # 10

    # encoder = FC_layers_3(input_ch=129 + 129, intermediate_ch=512, num_blocks=3, max_material_num=dataset.max_material,
    #                       max_type_num=dataset.max_type, max_name_num=dataset.max_name, residual_latent_ch=LATENT_CH).to(output_device)
    decoder = FC_layers_2(input_ch=dataset.max_material + LATENT_CH + 1, intermediate_ch=512,
                          output_ch=129 , num_blocks=3).to(output_device)
    ##42  2

   # encoder.load_state_dict(weights["network"][0])
    decoder.load_state_dict(weights["network"][1])
    # exit()

    #encoder.eval()
    decoder.eval()


    # new
    # encoder_DDP = DDP(encoder, find_unused_parameters=False, device_ids=[rank])
    # decoder_DDP = DDP(decoder, find_unused_parameters=False, device_ids=[rank])
    #with torch.no_grad():
    IDX = 0
    all_loss = []
    all_pos = []
    all_gt = []
    all_forward = []
    sum = 0
    sum2 = 0
    sum3 = 0

    for par in decoder.parameters():
        par.requires_grad = False
    type_ok=0

    for IDX in range(1000):  # 1016
        #print(IDX)
        data_stuff = dataset.getitem_testing2(IDX)  # getitem_whole_v3 getitem_testing
        psd = data_stuff["psd_data"].to(output_device, non_blocking=True)[None]
        # gt = data_stuff["training_data"].to(output_device, non_blocking=True)
        #gt = (data_stuff["training_data"]).to(output_device, non_blocking=True)[None]  # spectrom here
        material_num_gt = (data_stuff["material_num"]).to(output_device, non_blocking=True)  # [:, 0]

        scale_gt = (data_stuff["scale_num"]).to(output_device, non_blocking=True)
        #material = torch.randn(1, 10)

        scale = torch.randn(1, 1).to(output_device, non_blocking=True)
        material_logit = torch.rand(1, 7).to(output_device, non_blocking=True)
        latent = torch.randn(1, 8).to(output_device, non_blocking=True)
        scale.requires_grad = True
        material_logit.requires_grad = True
        latent.requires_grad = True
        optim = torch.optim.Adam([material_logit,latent, scale], lr=5e-4)
        #print(type_logit)
        for i in range(50):
            material_final = torch.nn.functional.softmax(material_logit)#.to(output_device, non_blocking=True)
            #print(type_final)
            optim.zero_grad()
            total_latent = torch.cat((material_final,scale, latent), axis=-1)
            psd_predict = decoder(total_latent)
            loss_value = torch.nn.functional.mse_loss(psd_predict, psd)
            #loss_value.requires_grad = True
            loss_value.backward()
            optim.step()
        # print(position)
        # print(position_gt)
        #print(np.linalg.norm(position[0].detach().cpu().numpy()-position_gt[0].detach().cpu().numpy()))
        #print(torch.argmax(type_logit) == type_num_gt)
        # print(scale)
        # print(scale_gt)

        sum = sum + abs(scale - scale_gt)

        if int(torch.argmax(material_logit).item())== int(material_num_gt.item()):
            type_ok += 1
        print("mat:", type_ok / (IDX + 1))
        print("scale:", sum[[0]]/(IDX+1))


        # sum+=np.linalg.norm(position[0].detach().cpu().numpy()-position_gt[0].detach().cpu().numpy())
        # second_index = int(torch.argsort(type_logit)[0][-2].item())
        # third_index = int(torch.argsort(type_logit)[0][-3].item())
        # if int(torch.argmax(type_logit).item())== int(type_num_gt.item())or second_index== int(type_num_gt.item())or third_index== int(type_num_gt.item()):
        #     sum2+=1
        # #sum2+=torch.argmax(type_logit) == type_num_gt
        # print(sum/(IDX+1))
        # print(sum2/(IDX+1))






if __name__ == '__main__':
    cur_args = Options().parse()
    exp_name = cur_args.exp_name

    exp_name_filled = exp_name.format(cur_args.apt)
    cur_args.exp_name = exp_name_filled

    exp_dir = os.path.join(cur_args.save_loc, exp_name_filled)
    cur_args.exp_dir = exp_dir

    result_output_dir = os.path.join(cur_args.save_loc, cur_args.inference_loc)
    cur_args.result_output_dir = result_output_dir
    if not os.path.isdir(result_output_dir):
        os.mkdir(result_output_dir)

    if not os.path.isdir(cur_args.save_loc):
        print("Save directory {} does not exist, need checkpoint folder...".format(cur_args.save_loc))
        exit()
    if not os.path.isdir(cur_args.exp_dir):
        print("Experiment {} does not exist, need experiment folder...".format(cur_args.exp_name))
        exit()
    print("Experiment directory is {}".format(exp_dir))
    world_size = cur_args.gpus
    test_ = test_net(0, cur_args)
