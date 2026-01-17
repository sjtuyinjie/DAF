import torch
from torchsummary import summary
torch.backends.cudnn.benchmark = True

from data_loading.sound_loader_object import soundsamples
import torch.multiprocessing as mp
import os
import socket
from contextlib import closing
import torch.distributed as dist
from model.networks_object_scale import FC_layers_3, FC_layers_2,FC_layers_2_mod_16,myresnet
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

import os, sys,re


from scipy.signal import welch
from scipy.io import wavfile
import librosa
import torchaudio
import tempfile
import os

# STRUCTURE_LOSS_LAMBDA = 5e-2 # should be small
# L2_LAMBDA = 5e-5 # must be small
# POS_LAMBDA = 10 # important
# KLD_LAMBDA = 1e-3
os.environ['TORCH_HOME']='/data/vision/torralba/scratch/chuang/Projects/find-fallen/pretrained/'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:280"
TYPE_LOSS_LAMBDA= 5e-1
STRUCTURE_LOSS_LAMBDA = 5e-1 # should be small
L2_LAMBDA = 5e-5 # must be small
SCALE_LAMBDA = 1 # important
KLD_LAMBDA = 1e-2

#basedir="/home/zdp21n5/projects/aresnettest/results//log.txt"
basedir="result_kitchen.txt"
f_log=open(basedir, 'w+')
f_log.write("test\n")
f_log.flush()



def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('localhost', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def worker_init_fn(worker_id, myrank_info):
    # print(worker_id + myrank_info*100, "SEED")
    np.random.seed(worker_id + myrank_info * 100)


def kld(mu_val, logvar_val):
    return -0.5 * torch.sum(1 + logvar_val - mu_val.pow(2) - logvar_val.exp()) / mu_val.size(0)


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def train_net(rank, world_size, freeport, other_args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = freeport
    print(rank)
    output_device = rank
    # print(rank, world_size, "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAaa")
    # exit()
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    pi = math.pi
    dataset = soundsamples(other_args)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    ranked_worker_init = functools.partial(worker_init_fn, myrank_info=rank)
    sound_loader = torch.utils.data.DataLoader(dataset, batch_size=other_args.batch_size // world_size, shuffle=False,
                                               num_workers=3, worker_init_fn=ranked_worker_init,
                                               persistent_workers=True, sampler=train_sampler, drop_last=False)

    #encoder = FC_layers_2_mod_16(input_ch=129+129, intermediate_ch=512, output_ch=40, num_blocks=4).to(output_device)
    LATENT_CH = 8
    print(dataset.max_material)
    encoder = myresnet( max_material_num=dataset.max_material, residual_latent_ch=LATENT_CH).to(output_device)
    # print(encoder)
    # #summary(encoder, (15, 2, 256, 600))
    # exit(-1)
    # encoder = FC_layers_3(input_ch=129 + 129, intermediate_ch=512, num_blocks=3, max_material_num=dataset.max_material,
    #                        max_type_num=dataset.max_type, max_name_num=dataset.max_name, residual_latent_ch=LATENT_CH).to(output_device)
    decoder = FC_layers_2(input_ch=dataset.max_material + LATENT_CH+1, intermediate_ch=512,
                          output_ch=129, num_blocks=3).to(output_device)
    #print(decoder)
    #exit(-1)



    if rank == 0:
        print("Dataloader requires {} batches".format(len(sound_loader)))#452
    #     print(dataset.max_material + dataset.max_name + LATENT_CH + 2, "AAAAAAAAAAAAAAAAAAAAAAAAAA")
    # exit()

    start_epoch = 1




    # We have conditional forward, must set find_unused_parameters to true
    encoder_DDP = DDP(encoder, find_unused_parameters=True, device_ids=[rank])
    decoder_DDP = DDP(decoder, find_unused_parameters=False, device_ids=[rank])
    reconstruction_criterion = torch.nn.MSELoss()
    scale_criterion = torch.nn.MSELoss()
    structured_criterion = torch.nn.CrossEntropyLoss()

    optimizer_1 = torch.optim.Adam(chain(encoder_DDP.parameters(), decoder_DDP.parameters()), lr=2e-4)
    # optimizer_2 = torch.optim.Adam(chain(decoder_DDP.parameters()), lr=2e-4)

    if rank == 0:
        old_time = time()



    for epoch in range(start_epoch, other_args.epochs):


        material_loss_buffer = 0.0
        scale_loss_buffer = 0.0
        regularization_buffer = 0.0
        reconstruction_buffer = 0.0
        total_loss_buffer = 0.0
        kld_loss_buffer = 0.0

        cur_iter = 0
        # if epoch <= 50:
        for data_stuff in sound_loader:
            psd = data_stuff["psd_data"].to(output_device, non_blocking=True)
            #print(psd.shape)
            gt = data_stuff["training_data"].to(output_device, non_blocking=True)#spec
            #print(gt.shape)

            # torch.Size([15, 258])
            # torch.Size([15, 2, 256, 600])

            # torch.Size([15, 258])
            # torch.Size([15, 2, 256, 600])
            # torch.Size([15])
            # torch.Size([15])
            # torch.Size([15, 1, 2])

            #exit()
            material_num_gt = data_stuff["material_num"].to(output_device, non_blocking=True)[:, 0]
            scale_num_gt = data_stuff["scale_num"].to(output_device, non_blocking=True)#[:, 0]



            #print(material_num_gt.shape)

            # print("psd shape:",psd.shape)
            # print("gt shape:",gt.shape)
            # print("material shape:", material_num_gt.shape)
            # print("position gt shape:",position_gt.shape)
            # psd shape: torch.Size([40, 258])
            # material shape: torch.Size([40])
            # gt shape: torch.Size([40, 2, 256, 600])
            # position gt shape: torch.Size([40, 1, 2])


            optimizer_1.zero_grad(set_to_none=True)
            #print("gt shape;",gt.shape)#torch.Size([15, 1, 257, 1036])
            #gt shape; torch.Size([16, 1, 257, 1036])
            #exit(-1)
            material_pred, scale_pred, residual_pred_mu, residual_pred_logvar = encoder_DDP(gt)#input spectrogram for encoder

            material_final = torch.nn.functional.softmax(material_pred)


            residual_pred = reparameterize(residual_pred_mu, residual_pred_logvar)

            loss_material = structured_criterion(material_pred, material_num_gt) * STRUCTURE_LOSS_LAMBDA

            loss_scale = scale_criterion(scale_pred, scale_num_gt.float()) * SCALE_LAMBDA





            total_loss =loss_material +0*loss_scale #+ loss_kld + loss_regularization + loss_reconstruction
            # print(loss_scale)
            # print(loss_material)
            #total_loss=total_loss.float()
            #print("scale:",loss_scale)

            if rank == 0:
                material_loss_buffer += loss_material.item()
                scale_loss_buffer += loss_scale.item()

                total_loss_buffer += total_loss.item()

                cur_iter += 1

            total_loss.backward()
            optimizer_1.step()
        if rank == 0:
            avg_material = material_loss_buffer / cur_iter
            avg_scale = scale_loss_buffer / cur_iter

            avg_loss = total_loss_buffer / cur_iter


            print(
                "{}: Ending epoch {}, time {},  Total loss: {:.6}, material loss: {:.6},scale loss: {:.6}".format(
                    other_args.exp_name, epoch, time() - old_time, avg_loss, avg_material,avg_scale))
            old_time = time()
            f_log = open(basedir, 'a')
            f_log.write("{}: Ending epoch {}, time {},  Total loss: {:.6},  material loss: {:.6},scale loss: {:.6}\n".format(
                    other_args.exp_name, epoch, time() - old_time, avg_loss, avg_material,avg_scale))
            f_log.flush()
        if rank == 0 and (epoch % 10 == 0 or epoch == 1 or epoch > (other_args.epochs - 3)):
            save_name = str(epoch).zfill(5) + ".chkpt"
            save_dict = {}
            print("saved")
            save_dict["network"] = [encoder_DDP.module.state_dict(), decoder_DDP.module.state_dict()]
            torch.save(save_dict, os.path.join(other_args.exp_dir, save_name))
    print("Wrapping up training {}".format(other_args.exp_name))
    dist.barrier()
    dist.destroy_process_group()
    return 1


if __name__ == '__main__':
    cur_args = Options().parse()
    exp_name = cur_args.exp_name
    exp_name_filled = exp_name.format(cur_args.apt)
    cur_args.exp_name = exp_name_filled
    if not os.path.isdir(cur_args.save_loc):
        print("Save directory {} does not exist, creating...".format(cur_args.save_loc))
        os.mkdir(cur_args.save_loc)
    exp_dir = os.path.join(cur_args.save_loc, exp_name_filled)
    cur_args.exp_dir = exp_dir
    print("Experiment directory is {}".format(exp_dir))
    if not os.path.isdir(exp_dir):
        os.mkdir(exp_dir)
    world_size = cur_args.gpus
    myport = str(find_free_port())
    mp.spawn(train_net, args=(world_size, myport, cur_args), nprocs=world_size, join=True)
