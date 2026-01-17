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

from data_loading.sound_loader_object8 import soundsamples
import torch.multiprocessing as mp
import os
import socket
from contextlib import closing
import torch.distributed as dist
from model.networks_object_scale import FC_layers, FC_layers_2, FC_layers_3, FC_layers_2_mod, FC_layers_2_mod_16, myresnet
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

os.environ['TORCH_HOME']='/data/vision/torralba/scratch/chuang/Projects/find-fallen/pretrained/'
def to_torch(input_arr):
    return input_arr[None]


def test_net(rank, other_args):
    pi = math.pi
    output_device = rank
    print("creating dataset")
    dataset = soundsamples(other_args)
    reconstruction_criterion = torch.nn.MSELoss()


    # weights = torch.load("/home/aluo/PycharmProjects/Neural_Acoustic_Fields_regress/results/frl_apartment_4/00088.chkpt", map_location=map_location)

    # weights = torch.load("/home/aluo/PycharmProjects/Neural_Acoustic_Fields_regress/results/apartment_1/00155.chkpt", map_location=map_location)
    # weights = torch.load("//home/zdp21n5/projects/aresnettest/results/apartment_1_vae_resnet_620//00620.chkpt", map_location=map_location)
    # weights = torch.load("//data/vision/torralba/scratch/chuang/Projects/nap_vision/results/new/00800.chkpt",
    #                      map_location=map_location)#1480
    map_location = 'cuda:%d' % rank
    # weights = torch.load("//data/vision/torralba/scratch/chuang/Projects/nap_vision/results/object_both/01670.chkpt",
    #                      map_location=map_location)
    weights = torch.load("//data/vision/torralba/scratch/chuang/Projects/nap_vision/results/object_scale8/00100.chkpt",
                         map_location=map_location)
    print("Checkpoint loaded")
    latent_dim = 2
    # encoder = FC_layers_2_mod_16(input_ch=129+129, intermediate_ch=512, output_ch=40, num_blocks=4).to(output_device)
    # decoder = FC_layers_2(input_ch=16+2, intermediate_ch=512, output_ch=129+129, num_blocks=4).to(output_device)
    # new
    LATENT_CH = 8
    encoder = myresnet(max_material_num=dataset.max_material, residual_latent_ch=LATENT_CH).to(
        output_device)
    print("mat:", dataset.max_material)  # 10

    # encoder = FC_layers_3(input_ch=129 + 129, intermediate_ch=512, num_blocks=3, max_material_num=dataset.max_material,
    #                       max_type_num=dataset.max_type, max_name_num=dataset.max_name, residual_latent_ch=LATENT_CH).to(output_device)
    decoder = FC_layers_2(input_ch=dataset.max_material + LATENT_CH+1, intermediate_ch=512,
                          output_ch=129 , num_blocks=3).to(output_device)

    encoder.load_state_dict(weights["network"][0])
    decoder.load_state_dict(weights["network"][1])
    # exit()

    encoder.eval()
    decoder.eval()


    # new
    # encoder_DDP = DDP(encoder, find_unused_parameters=False, device_ids=[rank])
    # decoder_DDP = DDP(decoder, find_unused_parameters=False, device_ids=[rank])
    with torch.no_grad():
        IDX = 0
        all_loss = []
        all_pos = []
        all_gt = []
        all_forward = []
        sum = 0
        sum2 = 0
        sum3 = 0
        type_ok = 0
        type_ok2=0
        type_ok3=0

        cnt=0
        for IDX in range(10160):  # 1016
            #print(IDX)
            losses = []
            pos = []

            data_stuff = dataset.getitem_testing2(IDX)  # getitem_whole_v3 getitem_testing

            psd = data_stuff["psd_data"].to(output_device, non_blocking=True)[None]
            # print(psd.shape)
            # gt = data_stuff["training_data"].to(output_device, non_blocking=True)
            gt = (data_stuff["training_data"]).to(output_device, non_blocking=True)[None]  # spectrom here
            #print(gt)
            #print("gt shape:",gt.shape)
            #exit(-1)
            # print("gt shape:",gt.shape)# 1 2 256 600
            material_num_gt = (data_stuff["material_num"]).to(output_device, non_blocking=True)  # [:, 0]
            scale_num_gt = (data_stuff["scale_num"]).to(output_device, non_blocking=True).cpu().numpy()

            index_num=(data_stuff["index_num"]).to(output_device, non_blocking=True).cpu().numpy()
            #print("index:",index_num[0])


            if str(index_num[0]) in ["23", "60", "59", "22", "56", "55", "25", "57"]:
                print(index_num)
                cnt+=1
                #exit(-1)




                material_pred, scale_pred, residual_pred, residual_pred_logvar = encoder(gt)

                #bestcate = np.argmax(material_pred.cpu().numpy())

                sum = sum + abs(scale_pred.cpu().numpy() - scale_num_gt)


                #print("now:", type_ok / (IDX+1))
                print("scale:",sum[0]/cnt)




            all_loss.append(losses)
            all_pos.append(pos)


        #print("SR:", type_ok / 1016)
        with open("stft_v2.pkl", "wb") as f:
            pickle.dump([all_loss, all_pos, all_gt, all_forward], f)
        exit()


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
