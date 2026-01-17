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

from data_loading.sound_loader_v4 import soundsamples
import torch.multiprocessing as mp
import os
import socket
from contextlib import closing
import torch.distributed as dist
from model.networks_fc import FC_layers, FC_layers_2,FC_layers_3, FC_layers_2_mod,FC_layers_2_mod_16
from model.modules import embedding_module_log
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import math
from time import time
from options import Options
import functools
import pickle
from itertools import chain



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
    weights = torch.load("/media/car/YJ06/amitprojects/projects/neural_audio_physics-master/results/apartment_1_back/00090.chkpt", map_location=map_location)
    print("Checkpoint loaded")
    latent_dim = 2
    # encoder = FC_layers_2_mod_16(input_ch=129+129, intermediate_ch=512, output_ch=40, num_blocks=4).to(output_device)
    # decoder = FC_layers_2(input_ch=16+2, intermediate_ch=512, output_ch=129+129, num_blocks=4).to(output_device)
    #new
    LATENT_CH = 8
    encoder = FC_layers_3(input_ch=129 + 129, intermediate_ch=512, num_blocks=3, max_material_num=dataset.max_material,
                          max_type_num=dataset.max_type, max_name_num=dataset.max_name, residual_latent_ch=LATENT_CH).to(output_device)
    decoder = FC_layers_2(input_ch=dataset.max_material + dataset.max_type + LATENT_CH + 42, intermediate_ch=512,
                          output_ch=129 + 129, num_blocks=3).to(output_device)


    encoder.load_state_dict(weights["network"][0])
    decoder.load_state_dict(weights["network"][1])
    # exit()

    encoder.eval()
    decoder.eval()
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
        for IDX in range(50):
            print(IDX)
            losses = []
            pos = []
            data_stuff = dataset.getitem_whole_v3(IDX)#getitem_whole_v3 getitem_testing

            #org
            # gt = to_torch(data_stuff[0]).to(output_device, non_blocking=True)
            # obj_type_num = torch.Tensor([data_stuff[2]]).to(output_device, non_blocking=True).long()
            # one_hot = torch.nn.functional.one_hot(obj_type_num, 40)
            # non_norm_position = to_torch(data_stuff[3]).to(output_device, non_blocking=True)
            # orig_non_norm = non_norm_position.cpu().numpy()+0.0

            #new

            gt = data_stuff[0].to(output_device, non_blocking=True)
            material_num_gt = data_stuff[1].to(output_device, non_blocking=True)
            type_num_gt = data_stuff[2].to(output_device, non_blocking=True)
            name_num_gt = data_stuff[3].to(output_device, non_blocking=True)
            position_gt = data_stuff[4].to(output_device, non_blocking=True)


            material_pred, type_pred, residual_pred, position_pred = encoder(gt)
            # print('mat:')
            # print(material_pred.shape)
            # print('type:')
            # print(type_pred.shape)
            # print('resi:')
            # print(residual_pred.shape)
            # print('posi:')
            # print(position_pred.shape)
           
            positional_embed_module = embedding_module_log(num_freqs=10, ch_dim=-1)#org1
            position_embedded = positional_embed_module(position_pred)
            #print(position_embedded.shape)   #
            #position_embedded=position_pred
            total_latent = torch.cat((material_pred, type_pred, residual_pred, position_embedded), axis=-1)#org 1
            #print(total_latent.shape)

            recovered_output = decoder(total_latent)


       
            for x_offset in np.arange(-5,5, 0.25):
                for y_offset in np.arange(-5, 5, 0.25):
                    position_embedded[..., 0] = x_offset
                    position_embedded[..., 1] = y_offset

         
                    pos.append(position_embedded.cpu().numpy())

                    total_latent = torch.cat((material_pred, type_pred, residual_pred, position_embedded), axis=-1)
                    recovered_output = decoder(total_latent)
      
                    #reconstruct = decoder(torch.cat((latent, non_norm_position[:, 0]), dim=-1))  #org


                    #new above


                    losses.append(reconstruction_criterion(recovered_output, gt).item())
            all_loss.append(losses)
            all_pos.append(pos)
            all_gt.append(position_gt)

        with open("dump_training_physics_smooth_16.pkl", "wb") as f:
            pickle.dump([all_loss, all_pos, all_gt], f)
        exit()
        # print("GOT SO FAR")
        # exit()

        renorm_3 = np.concatenate((renorm, renorm_2), axis=1)
        # print(dataset.mean.shape, dataset.std.shape)
        # exit()

        # plt.imshow(renorm_3[0])
        # plt.show()
        print(total_out_flat.shape)
        exit()
        
    with open(save_name, "wb") as saver_file_obj:
        pickle.dump(container, saver_file_obj)
        print("Results saved to {}".format(save_name))
    return 1


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
    ## Uncomment to run all rooms
    # for apt in ['apartment_1', 'apartment_2', 'frl_apartment_2', 'frl_apartment_4', 'office_4', 'room_2']:
    #     cur_args.apt = apt
    #     exp_name = cur_args.exp_name
    #     exp_name_filled = exp_name.format(cur_args.apt)
    #     cur_args.exp_name = exp_name_filled
    #
    #     exp_dir = os.path.join(cur_args.save_loc, exp_name_filled)
    #     cur_args.exp_dir = exp_dir
    #
    #     result_output_dir = os.path.join(cur_args.save_loc, cur_args.inference_loc)
    #     cur_args.result_output_dir = result_output_dir
    #     if not os.path.isdir(result_output_dir):
    #         os.mkdir(result_output_dir)
    #
    #     if not os.path.isdir(cur_args.save_loc):
    #         print("Save directory {} does not exist, need checkpoint folder...".format(cur_args.save_loc))
    #         exit()
    #     if not os.path.isdir(cur_args.exp_dir):
    #         print("Experiment {} does not exist, need experiment folder...".format(cur_args.exp_name))
    #         exit()
    #     print("Experiment directory is {}".format(exp_dir))
    #     world_size = cur_args.gpus
    #     test_ = test_net(0, cur_args)
