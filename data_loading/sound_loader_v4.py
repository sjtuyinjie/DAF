import numpy.random
import torch
import os
import pickle
import numpy as np
import random
import h5py
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def listdir(path):
    return [os.path.join(path, x) for x in os.listdir(path)]

def join(*paths):
    return os.path.join(*paths)


class soundsamples(torch.utils.data.Dataset):
    def __init__(self, arg_stuff):
        with open("/home/zdp21n5/projects/aresnettest//all_container.pkl", "rb") as fff:
            metadata = pickle.load(fff)
        # with open("/home/aluo/PycharmProjects/Neural_Acoustic_Fields_regress/complete_metadata_v3_height.pkl", "rb") as fff:
        #     heightmetadata = pickle.load(fff)
        self.metadata = metadata
        # self.heightmetadata = heightmetadata
        self.sound_data = []
        self.full_path = "/home/zdp21n5/datasets/multimodal_challenge_combined/dataset//spectral_data_psd.h5"
        self.sound_data = h5py.File(self.full_path, 'r')
        print("collecting h5py keys")
        self.sound_keys = list(self.sound_data.keys())
        print("finish collecting h5py keys")
        outs = []
        for k in self.sound_keys:
            spec_data = self.sound_data[k]
            outs.append(spec_data)
        outs = np.array(outs)#2 256 871
        print(outs.shape)
        #print(outs)
        #exit()
        self.scaling = np.percentile(outs, 99)
 
        del outs
        self.sound_keys_orig = self.sound_keys.copy()
        # self.sound_keys = [_idx for _idx in self.sound_keys if "mm_craftroom_1a" in _idx]

        max_type = []
        max_name = []
        max_material = []

        for k in list(self.metadata.keys()):
            max_type.append(self.metadata[k]["type_num"])
            max_name.append(self.metadata[k]["name_num"])
            max_material.append(self.metadata[k]["material_num"])
        self.max_type = max(max_type)+1
        self.max_name = max(max_name)+1
        self.max_material = max(max_material)+1


        self.sound_data.close()
        self.sound_data = None
        all_keys = sorted(self.sound_keys)
        random.seed(10)
        random.shuffle(all_keys)
        valid_len = int(len(all_keys) / 10.0)
        self.training = sorted(all_keys[valid_len:])
        training_buffer = []
        for k in self.training:
            # if self.metadata[k]["name_num"] == 1:
            if "mm_craftroom" in k:
                training_buffer.append(k)
        self.training = training_buffer
        # print(len(self.training), "TRAINING")
        # exit()

        self.testing = sorted(all_keys[:valid_len])

        testing_buffer = []
        for k in self.testing:
            # if self.metadata[k]["name_num"] == 1:
            if "mm_craftroom" in k:
                testing_buffer.append(k)
        self.testing = testing_buffer

        with open("training_test.pkl", "wb") as f:
            pickle.dump([all_keys[valid_len:], all_keys[:valid_len]], f)
        # with open("/home/aluo/PycharmProjects/Neural_Acoustic_Fields_regress/mean_std_psd.pkl", "rb") as f:
        #     mean_std = pickle.load(f)
        # self.mean = mean_std[0]
        # self.std = 3.0 * mean_std[1]
        # print(self.mean.shape, self.std.shape)
        # exit()
        # num_samples = arg_stuff.pixel_count
        # self.num_samples = num_samples
        self.pos_reg_amt = arg_stuff.reg_eps
        self.query_str = None
        self.first_run = True
        assert len(metadata) == len(self.sound_keys_orig)

    def __len__(self):
        # return number of samples for a SINGLE orientation
        return len(self.training)

    def __getitem__(self, idx):
        loaded = False
        # while not loaded:
        #     try:
        if self.sound_data is None:
            self.sound_data = h5py.File(self.full_path, 'r')
        myidx = idx
        query_str = self.training[myidx]
        spec_data = self.sound_data[query_str][:][:,None]
        return_dictionary = {}
        cur_metadata = self.metadata[query_str]
        cur_material_str = cur_metadata["material_str"]
        cur_material_num = cur_metadata["material_num"]
        cur_obj_str = cur_metadata["name_str"]
        cur_obj_num = cur_metadata["name_num"]
        cur_type_str = cur_metadata["type_str"]
        cur_type_num = cur_metadata["type_num"]
        # cur_height = self.heightmetadata[query_str]
        # cur_position = cur_metadata["obj_pos_relative"]
        cur_position = [cur_metadata["perception_pos"][0], cur_metadata["perception_pos"][2]]
        #print(spec_data/self.scaling + 1e-2)
        #print(np.log(1e-2)*0.5)
        spec_data = np.log(spec_data/self.scaling + 1e-2) - np.log(1e-2)*0.5
        sound_size = spec_data.shape
        non_norm_start = (np.array(cur_position) + np.random.normal(0,1,2)*0.05)
        total_non_norm_position = torch.from_numpy(non_norm_start)[None].float()
        selected_total = spec_data.reshape(-1)
        selected_total_smooth = gaussian_filter(selected_total, sigma=2.0)
        selected_total = torch.from_numpy(selected_total_smooth).float()
        loaded = True

        return_dictionary = {"training_data":selected_total,
                             "material_num":torch.from_numpy(np.array([cur_material_num])).long(),
                             "type_num":torch.from_numpy(np.array([cur_type_num])).long(),
                             "name_num":torch.from_numpy(np.array([cur_obj_num])).long(),
                             "position":total_non_norm_position
                             }
        return return_dictionary

        # return selected_total,torch.from_numpy(np.array([cur_material_num])).long(), torch.from_numpy(np.array([cur_type_num])).long(), total_non_norm_position, torch.from_numpy(selected_freq).float(), cur_height


    def getitem_testing(self, idx):
        loaded = False
        # while not loaded:
        #     try:
        if self.sound_data is None:
            self.sound_data = h5py.File(self.full_path, 'r')
        myidx = idx
        query_str = self.testing[myidx]
        # idx = 0

        spec_data = self.sound_data[query_str][:][:, None]
        cur_metadata = self.metadata[query_str]
        cur_material_num = cur_metadata["material_num"]
        cur_obj_num = cur_metadata["name_num"]
        cur_type_num = cur_metadata["type_num"]
        #cur_height = self.heightmetadata[query_str]
        cur_position = [cur_metadata["perception_pos"][0], cur_metadata["perception_pos"][2]]
        spec_data = np.log(spec_data/self.scaling + 1e-2) - np.log(1e-2)*0.5
        non_norm_start = np.array(cur_position)
        # Do not inject noise for regularization if we are using this for testing!

        total_non_norm_position = torch.from_numpy(non_norm_start)[None].float()

        selected_total = spec_data.reshape(-1)
        selected_total_smooth = gaussian_filter(selected_total, sigma=2.0)
        selected_total = torch.from_numpy(selected_total_smooth).float()
        loaded = True

        return_dictionary = {"training_data": selected_total,
                             "material_num": torch.from_numpy(np.array([cur_material_num])).long(),
                             "type_num": torch.from_numpy(np.array([cur_type_num])).long(),
                             "name_num": torch.from_numpy(np.array([cur_obj_num])).long(),
                             "position": total_non_norm_position
                             }
        return return_dictionary

        # return selected_total, torch.from_numpy(np.array([cur_material_num])).long(), torch.from_numpy(
        #     np.array([cur_type_num])).long(), total_non_norm_position, torch.from_numpy(
        #     selected_freq).float(), 2.0 * torch.from_numpy(selected_time).float() / float(129.0) - 1.0, cur_height