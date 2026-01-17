import matplotlib.pyplot as plt
import numpy as np
import torch
torch.backends.cudnn.benchmark = True
from inspect import getsourcefile
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])



import torch
torch.backends.cudnn.benchmark = True

from data_loading.sound_loader_stft import soundsamples
import torch.multiprocessing as mp
import os
import socket
from contextlib import closing
import torch.distributed as dist
#from model.networks_cnn import FC_layers, FC_layers_2,FC_layers_3, FC_layers_2_mod,FC_layers_2_mod_16,myresnet
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


basedir="vae_log.txt"
f_log=open(basedir, 'w+')
f_log.write("test\n")
f_log.flush()

def to_torch(input_arr):
    return input_arr[None]



def test_net(rank, other_args):
    sum = 0
    pi = math.pi
    output_device = rank
    print("creating dataset")
    dataset = soundsamples(other_args)
    reconstruction_criterion = torch.nn.MSELoss()

    map_location = 'cuda:%d' % rank
    # weights = torch.load("/home/aluo/PycharmProjects/Neural_Acoustic_Fields_regress/results/frl_apartment_4/00088.chkpt", map_location=map_location)

    # weights = torch.load("/home/aluo/PycharmProjects/Neural_Acoustic_Fields_regress/results/apartment_1/00155.chkpt", map_location=map_location)
    #weights = torch.load("//home/zdp21n5/projects/aresnettest/results/apartment_1_vae_resnet_620//00620.chkpt", map_location=map_location)
    weights = torch.load("/home/zdp21n5/projects/aresnettest/results/apartment_1_resnet_report////00100.chkpt",
                         map_location=map_location)
    print("Checkpoint loaded")
    latent_dim = 2
    # encoder = FC_layers_2_mod_16(input_ch=129+129, intermediate_ch=512, output_ch=40, num_blocks=4).to(output_device)
    # decoder = FC_layers_2(input_ch=16+2, intermediate_ch=512, output_ch=129+129, num_blocks=4).to(output_device)
    #new
    LATENT_CH = 8
    # encoder = myresnet(max_material_num=dataset.max_material,
    #                    max_type_num=dataset.max_type, max_name_num=dataset.max_name, residual_latent_ch=LATENT_CH).to(output_device)


    net = timm.create_model('resnet50', pretrained=True, num_classes=0)
    net.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
    in_channel = 2048
    out_channel = 2
    net.fc = nn.Linear(in_channel, out_channel)
    net = net.to(output_device)
    encoder = net


    #encoder.load_state_dict(weights["network"][0])
    #decoder.load_state_dict(weights["network"][1])
    # exit()

    encoder.load_state_dict(weights)

    encoder.eval()
    #decoder.eval()
    container = dict()
    save_name = os.path.join(other_args.result_output_dir, other_args.apt+"_NAF.pkl")
    # container["mean_std"] = (dataset.std, dataset.mean)
    criterion = torch.nn.MSELoss()


    #new
    # encoder_DDP = DDP(encoder, find_unused_parameters=False, device_ids=[rank])
    # decoder_DDP = DDP(decoder, find_unused_parameters=False, device_ids=[rank])
    with torch.no_grad():
        IDX = 0
        all_loss = []
        all_pos = []
        all_gt = []
        for IDX in range(1016):#1016
            print(IDX)
            losses = []
            pos = []
            data_stuff = dataset.getitem_testing2(IDX)#getitem_whole_v3 getitem_testing
            #print(data_stuff)

            #org
            # gt = to_torch(data_stuff[0]).to(output_device, non_blocking=True)
            # obj_type_num = torch.Tensor([data_stuff[2]]).to(output_device, non_blocking=True).long()
            # one_hot = torch.nn.functional.one_hot(obj_type_num, 40)
            # non_norm_position = to_torch(data_stuff[3]).to(output_device, non_blocking=True)
            # orig_non_norm = non_norm_position.cpu().numpy()+0.0

            #new
            # print(type(data_stuff["training_data"]), type(data_stuff["position"]))
            # exit() ctrl + shift + R
            psd = data_stuff["psd_data"].to(output_device, non_blocking=True)[None]
            #print(psd.shape)
            #gt = data_stuff["training_data"].to(output_device, non_blocking=True)
            gt = (data_stuff["training_data"]).to(output_device, non_blocking=True)[None]#spectrom here
            #print(gt.shape)
            material_num_gt =(data_stuff["material_num"]).to(output_device, non_blocking=True)#[:, 0]
            type_num_gt = (data_stuff["type_num"]).to(output_device, non_blocking=True)#[:, 0]
            name_num_gt = (data_stuff["name_num"]).to(output_device, non_blocking=True)#[:, 0]
            position_gt = (data_stuff["position"]).to(output_device, non_blocking=True)

            position_pred = net(gt)
            print("pred:",position_pred)
            print("gt",position_gt)
            position_pred=position_pred.cpu().numpy()
            position_gt=position_gt.cpu().numpy()


            sum=sum+np.sqrt(np.square(position_pred[0][0]-position_gt[0][0])+np.square(position_pred[0][1]-position_gt[0][1]))
            #sum+=np.sqrt(np.square(all_pos[i][best][0][0]-gt_pos[:][0][0]))
            # f_log = open(basedir, 'a')
            # f_log.write("{}: pred {}, gt {}\n".format(
            #         position_pred,position_pred))
            # f_log.flush()

        print(sum/1015)






       


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
