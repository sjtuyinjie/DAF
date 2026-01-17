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
        with open("/data/vision/torralba/scratch/chuang/Projects/ObjectFolder/all_container_both.pkl", "rb") as fff:
            metadata = pickle.load(fff)

        self.metadata = metadata
        # self.heightmetadata = heightmetadata
        self.sound_data = []
        self.full_path = "/data/vision/torralba/scratch/chuang/Projects/ObjectFolder/object_stft.h5"
        self.sound_data = h5py.File(self.full_path, 'r')
        #new stft
        self.psd_path="/data/vision/torralba/scratch/chuang/Projects/ObjectFolder/object_psd.h5"
        self.psd_file = h5py.File(self.psd_path, 'r')

        print("collecting h5py keys")
        self.sound_keys = list(self.sound_data.keys())
        print("finish collecting h5py keys")

        print("collecting psd keys")
        self.psd_keys = list(self.psd_file.keys())
        print("finish collecting psd keys")
        # outs = []
        # num=0
        # for k in self.sound_keys:
        #     #print(k)
        #     num=num+1
        #     spec_data = self.sound_data[k]
        #     #print(spec_data.shape)
        #     outs.append(spec_data)
        #print(num)
#         outs0 = np.array(outs[0])
#         outs1=np.array(outs[1])
#         self.scaling = np.percentile(outs0, 99)
#         sclaling2=np.percentile(outs1, 99)
#         -43388.50730468758
# -43045.72863281244
        # print(self.scaling)
        # print(sclaling2)
        #del outs
        #self.scaling=-43388
        outs = []

        for k in self.psd_keys:
            mmm = self.psd_file[k]
            outs.append(mmm)
        outs = np.array(outs)#2 256 871

        self.scaling = np.percentile(outs, 99)
        print(self.scaling)
        #exit(-1)
        self.sound_keys_orig = self.sound_keys.copy()
        self.psd_keys_orig = self.psd_keys.copy()
        # self.sound_keys = [_idx for _idx in self.sound_keys if "mm_craftroom_1a" in _idx]

        max_type = []
        max_name = []
        max_material = []
        max_scale = []

        for k in list(self.metadata.keys()):
            print(self.metadata[k]["material_num"])

            max_material.append(self.metadata[k]["material_num"])
            max_scale.append(self.metadata[k]["scale_num"])

        self.max_material = max(max_material)+1

        print("max material:", self.max_material)

        # max type: 30
        # max name: 53
        # max material: 10

        self.sound_data.close()
        self.sound_data = None
        self.psd_file.close()
        self.psd_file = None
        all_keys = sorted(self.sound_keys)
        random.seed(10)
        random.shuffle(all_keys)
        valid_len = int(len(all_keys) / 10.0)
        self.training = sorted(all_keys[valid_len:])
        training_buffer = []

        #exit()
        print("org len:",len(self.training))#3354
        for k in self.training:
            if str(self.metadata[k]["index_num"]) not in ["23", "60", "59", "22", "56", "55", "25", "57"]:
                training_buffer.append(k)
                print(self.metadata[k]["index_num"])
        #exit(-1)

        #craftroom 8594
        #kitchen 8597
        self.training = training_buffer
        print(len(self.training)) #9032  to 8594
        print(len(training_buffer))
        #f=open("./traininglist.txt",'w')
        #f.write(str(self.training))

        #exit()
        # print(len(self.training), "TRAINING")


        self.testing = sorted(all_keys[:])

        testing_buffer = []
        for k in self.testing:
            # if self.metadata[k]["name_num"] == 1:
            # if "mm_craftroom" in k:
            #if self.metadata[k]["index_num"] in ["23", "60", "59", "22", "56", "55", "25", "57"]:
            testing_buffer.append(k)

        self.testing = testing_buffer
        print("testing:",len(self.testing))
        #exit(-1)

        with open("training_test_object.pkl", "wb") as f:
            pickle.dump([all_keys[valid_len:], all_keys[:]], f)
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
        if self.psd_file is None:
            self.psd_file = h5py.File(self.psd_path, 'r')

        myidx = idx
        query_str = self.training[myidx]


        #new stft

        psd_data = self.psd_file[query_str][:][:, None]

        psd_data = np.log(psd_data / self.scaling + 1e-2) - np.log(1e-2) * 0.5
        psd_total = psd_data.reshape(-1)
        psd_total_smooth = gaussian_filter(psd_total, sigma=2.0)
        psd_total = torch.from_numpy(psd_total_smooth).float()
#####


        spec_data = self.sound_data[query_str][:][:]
        return_dictionary = {}
        cur_metadata = self.metadata[query_str]

        cur_material_num = cur_metadata["material_num"]
        cur_scale_num = cur_metadata["scale_num"]
        # print("cur:",cur_scale_num)




        selected_total = torch.from_numpy(spec_data).float()
        loaded = True
        #print(selected_total.shape)#torch.Size([2, 256, 600])

        return_dictionary = {"psd_data":psd_total,
                             "training_data":selected_total,
                             "material_num":torch.from_numpy(np.array([cur_material_num])).long(),
                             "scale_num": torch.from_numpy(np.array([cur_scale_num])).long(),

                             }
        return return_dictionary

        # return selected_total,torch.from_numpy(np.array([cur_material_num])).long(), torch.from_numpy(np.array([cur_type_num])).long(), total_non_norm_position, torch.from_numpy(selected_freq).float(), cur_height



    def getitem_testing2(self, idx):
        loaded = False
        # while not loaded:
        #     try:
        if self.sound_data is None:
            self.sound_data = h5py.File(self.full_path, 'r')
        if self.psd_file is None:
            self.psd_file = h5py.File(self.psd_path, 'r')

        myidx = idx
        #print(len(self.testing))#1016
        query_str = self.testing[myidx]


        #new stft

        psd_data = self.psd_file[query_str][:][:, None]

        psd_data = np.log(psd_data / self.scaling + 1e-2) - np.log(1e-2) * 0.5
        psd_total = psd_data.reshape(-1)
        psd_total_smooth = gaussian_filter(psd_total, sigma=2.0)
        psd_total = torch.from_numpy(psd_total_smooth).float()
#####


        spec_data = self.sound_data[query_str][:][:]

        cur_metadata = self.metadata[query_str]

        cur_material_num = cur_metadata["material_num"]
        cur_scale_num = cur_metadata["scale_num"]
        cur_index_num = cur_metadata["index_num"]
        # print("cur:",cur_scale_num)


        #selected_total = spec_data.reshape(-1)  #not used for spectro
        #selected_total=spec_data
        #selected_total_smooth = gaussian_filter(selected_total, sigma=2.0)
        selected_total = torch.from_numpy(spec_data).float()
        loaded = True
        #print(selected_total.shape)#torch.Size([2, 256, 600])

        return_dictionary = {"psd_data":psd_total,
                             "training_data":selected_total,
                             "material_num":torch.from_numpy(np.array([cur_material_num])).long(),
                             "scale_num": torch.from_numpy(np.array([cur_scale_num])),
                             "index_num":torch.from_numpy(np.array([cur_index_num])),
                             }
        return return_dictionary

