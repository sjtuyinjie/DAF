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

from data_loading.sound_loader_stft_v2 import soundsamples
import torch.multiprocessing as mp
import os
import socket
from contextlib import closing
import torch.distributed as dist
from model.networks_stft import FC_layers, FC_layers_2, FC_layers_3, FC_layers_2_mod, FC_layers_2_mod_16, myresnet
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

    map_location = 'cuda:%d' % rank
    # weights = torch.load("/home/aluo/PycharmProjects/Neural_Acoustic_Fields_regress/results/frl_apartment_4/00088.chkpt", map_location=map_location)


    # weights = torch.load("/data/vision/torralba/scratch/chuang/Projects/find-fallen/baseline/planner/stft/03630.chkpt",
    #                      map_location=map_location)
    weights = torch.load("/data/vision/torralba/scratch/chuang/Projects/nap_vision/results/new/00630.chkpt",
                         map_location=map_location)
    print("Checkpoint loaded")
    latent_dim = 2
    # encoder = FC_layers_2_mod_16(input_ch=129+129, intermediate_ch=512, output_ch=40, num_blocks=4).to(output_device)
    # decoder = FC_layers_2(input_ch=16+2, intermediate_ch=512, output_ch=129+129, num_blocks=4).to(output_device)
    # new
    LATENT_CH = 8
    encoder = myresnet(max_material_num=dataset.max_material,
                       max_type_num=dataset.max_type, max_name_num=dataset.max_name, residual_latent_ch=LATENT_CH).to(
        output_device)
    print("mat:", dataset.max_material)  # 10
    print("type:", dataset.max_type)  # 30
    print("name:", dataset.max_name)  # 53
    # encoder = FC_layers_3(input_ch=129 + 129, intermediate_ch=512, num_blocks=3, max_material_num=dataset.max_material,
    #                       max_type_num=dataset.max_type, max_name_num=dataset.max_name, residual_latent_ch=LATENT_CH).to(output_device)
    decoder = FC_layers_2(input_ch=dataset.max_material + LATENT_CH + 2, intermediate_ch=512,
                          output_ch=129 + 129, num_blocks=3).to(output_device)
    ##42  2

    encoder.load_state_dict(weights["network"][0])
    decoder.load_state_dict(weights["network"][1])
    # exit()

    encoder.eval()
    decoder.eval()
    container = dict()
    save_name = os.path.join(other_args.result_output_dir, other_args.apt + "_NAF.pkl")
    # container["mean_std"] = (dataset.std, dataset.mean)
    criterion = torch.nn.MSELoss()

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
        type_ok=0
        for IDX in range(1016):  # 1016
            print(IDX)
            losses = []
            pos = []
            data_stuff = dataset.getitem_testing2(IDX)  # getitem_whole_v3 getitem_testing
            # print(data_stuff)

            # org
            # gt = to_torch(data_stuff[0]).to(output_device, non_blocking=True)
            # obj_type_num = torch.Tensor([data_stuff[2]]).to(output_device, non_blocking=True).long()
            # one_hot = torch.nn.functional.one_hot(obj_type_num, 40)
            # non_norm_position = to_torch(data_stuff[3]).to(output_device, non_blocking=True)
            # orig_non_norm = non_norm_position.cpu().numpy()+0.0

            # new
            # print(type(data_stuff["training_data"]), type(data_stuff["position"]))
            # exit() ctrl + shift + R
            psd = data_stuff["psd_data"].to(output_device, non_blocking=True)[None]
            # print(psd.shape)
            # gt = data_stuff["training_data"].to(output_device, non_blocking=True)
            gt = (data_stuff["training_data"]).to(output_device, non_blocking=True)[None]  # spectrom here
            # print("gt shape:",gt.shape)# 1 2 256 600
            material_num_gt = (data_stuff["material_num"]).to(output_device, non_blocking=True)  # [:, 0]
            type_num_gt = (data_stuff["type_num"]).to(output_device, non_blocking=True)  # [:, 0]
            name_num_gt = (data_stuff["name_num"]).to(output_device, non_blocking=True)  # [:, 0]
            position_gt = (data_stuff["position"]).to(output_device, non_blocking=True)




            material_pred, type_pred, residual_pred, residual_pred_logvar, position_pred = encoder(gt)
            all_forward.append(position_pred + 0.0)
            bestcate=np.argmax(type_pred.cpu().numpy())
            print(bestcate)
            print(type_num_gt)
            if bestcate==type_num_gt:
                type_ok+=1



            positional_embed_module = embedding_module_log(num_freqs=10, ch_dim=-1, max_freq=5)  # org1
            position_embedded = positional_embed_module(position_pred + 0.0)
            original_pred = position_pred + 0.0

            total_latent = torch.cat((material_pred, residual_pred, position_embedded), axis=-1)  # org 1


            recovered_output = decoder(total_latent)


            # new
            position_pred0 = position_pred.cpu().numpy()
            position_gt0 = position_gt.cpu().numpy()
            sum = sum + np.sqrt(np.square(position_pred0[0][0] - position_gt0[0][0]) + np.square(
                position_pred0[0][1] - position_gt0[0][1]))
            if (np.sqrt(np.square(position_pred0[0][0] - position_gt0[0][0]) + np.square(
                    position_pred0[0][1] - position_gt0[0][1])) < 2):
                sum2 += 1
            if (np.sqrt(np.square(position_pred0[0][0] - position_gt0[0][0]) + np.square(
                    position_pred0[0][1] - position_gt0[0][1])) < 5):
                sum3 += 1
            if (IDX > 0):
                print("error:", sum / IDX)

            for x_offset in np.arange(-5, 5, 0.2):
                for y_offset in np.arange(-5, 5, 0.2):
                    position_pred[:, 0] = x_offset
                    position_pred[:, 1] = y_offset
                    position_embedded = positional_embed_module(position_pred + 0.0)

                    pos.append(position_pred.cpu().numpy())

                    total_latent = torch.cat((material_pred, residual_pred, position_pred), axis=-1)
                    reconstruct = decoder(total_latent)  # org

                    losses.append(reconstruction_criterion(reconstruct, psd).item())

            print("sum2", sum2)  # 100:45  200:75 300:73 400:76  500:76 630:76
            print("sum3", sum3)  # 100:97  200:96 300:97 400:98  500:97 600:98

            all_loss.append(losses)
            all_pos.append(pos)
            all_gt.append(position_gt)
        print("SR:",type_ok/1016)



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
