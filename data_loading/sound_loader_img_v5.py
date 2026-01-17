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

def listdir(path):
    return [os.path.join(path, x) for x in os.listdir(path)]

def join(*paths):
    return os.path.join(*paths)


class soundsamples(torch.utils.data.Dataset):
    def __init__(self, arg_stuff):
        with open("/home/aluo/PycharmProjects/Neural_Acoustic_Fields_regress/complete_metadata_v3.pkl", "rb") as fff:
            metadata = pickle.load(fff)
        self.metadata = metadata

        self.sound_data = []
        self.full_path = "/media/aluo/big2/multimodal_challenge_combined/dataset/multimodal_mag_v3.h5"
        self.full_path_spec = "/media/aluo/big2/multimodal_challenge_combined/dataset/multimodal_mag_v4.h5"
        self.sound_data = h5py.File(self.full_path, 'r')
        self.sound_keys = list(self.sound_data.keys())
        outs = []
        for k in self.sound_keys:
            spec_data = self.sound_data[k]
            outs.append(spec_data)
        outs = np.array(outs)
        self.scaling = np.percentile(outs, 99)
        del outs
        self.sound_keys_orig = self.sound_keys.copy()
        # self.sound_keys = [_idx for _idx in self.sound_keys if "mm_craftroom_1a" in _idx]
        self.sound_data.close()
        self.sound_data = None
        self.sound_data_spec = None
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
        with open("/home/aluo/PycharmProjects/Neural_Acoustic_Fields_regress/mean_std_psd.pkl", "rb") as f:
            mean_std = pickle.load(f)
        self.mean = mean_std[0]
        self.std = 3.0 * mean_std[1]
        # print(self.mean.shape, self.std.shape)
        # exit()
        num_samples = arg_stuff.pixel_count
        self.num_samples = num_samples
        self.pos_reg_amt = arg_stuff.reg_eps
        self.query_str = None
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
        if self.sound_data_spec is None:
            self.sound_data_spec = h5py.File(self.full_path_spec, 'r')

        # if idx % 2==0:
        #     idx = 1
        # else:
        #     idx = 0
        # myidx = idx % 100
        myidx = idx
        query_str = self.training[myidx]
        # idx = 0

        spec_data = torch.from_numpy(self.sound_data[query_str][:]).float()[:,None]
        spec_data_img = torch.from_numpy(self.sound_data_spec[query_str][:]).float()
        # print(spec_data.shape)
        # exit()
        cur_metadata = self.metadata[query_str]
        cur_material_str = cur_metadata["material_str"]
        cur_material_num = cur_metadata["material_num"]
        cur_obj_str = cur_metadata["name_str"]
        cur_obj_num = cur_metadata["name_num"]
        cur_type_str = cur_metadata["type_str"]
        cur_type_num = cur_metadata["type_num"]
        cur_position = cur_metadata["obj_pos_relative"]
        # cur_position = [cur_metadata["perception_pos"][0], cur_metadata["perception_pos"][2]]
        cur_material_num = cur_type_num
        # if random.random()<1.0:
        #     # np.log(1e-3) = -6.90775527898213
        spec_data_img = torch.nn.functional.pad(spec_data_img, pad=[0, 1000-min(spec_data_img.shape[2], 1000), 0, 0, 0, 0], value=-6.90775527898213)[:,:128,:512]+6.9



        # actual_spec_len = spec_data.shape[2]
        spec_data_np = spec_data.numpy()
        # probs_flat = np.clip((np.exp(spec_data_np[0]).flatten()+np.exp(spec_data_np[1]).flatten())*0.5, 0.000, 1000.0)
        # probs_flat = probs_flat/np.sum(probs_flat)
        # indices = np.arange(0, len(probs_flat), dtype=np.int32)
        # sample = np.random.choice(indices, size=1500, p=probs_flat, replace=False)
        # unraveled_sample = np.unravel_index(sample, spec_data[0].shape)
        # spec_data = (spec_data - self.mean) / (self.std)
        spec_data = torch.log(spec_data/self.scaling + 1e-2)
        # spec_data = (spec_data - torch.mean(spec_data))
        # spec_data = spec_data/(torch.max(torch.abs(spec_data))+1e-8)
        # 2, freq, time
        sound_size = spec_data.shape
        # selected_time = np.concatenate([np.random.randint(0, sound_size[2], self.num_samples-1500), unraveled_sample[1]])
        # selected_freq = np.concatenate([np.random.randint(0, sound_size[1], self.num_samples-1500), unraveled_sample[0]])

        # selected_time = np.arange(0, actual_spec_len)
        # selected_freq = np.arange(0, 256)
        # selected_time, selected_freq = np.meshgrid(selected_time, selected_freq)
        # selected_time = selected_time.reshape(-1)
        # selected_freq = selected_freq.reshape(-1)
        selected_time = np.arange(40)
        selected_freq = np.array([0]*40)

        non_norm_start = (np.array(cur_position) + np.random.normal(0,1,2)*0.05)

        total_non_norm_position = torch.from_numpy(non_norm_start)[None].float()

        selected_total = spec_data.reshape(-1)
        loaded = True

        return selected_total, spec_data_img, total_non_norm_position

    def getitem2(self, idx):
        loaded = False
        # while not loaded:
        #     try:
        if self.sound_data is None:
            self.sound_data = h5py.File(self.full_path, 'r')

        query_str = self.testing[idx]

        spec_data = torch.from_numpy(self.sound_data[query_str][:]).float()
        # print(spec_data.shape)
        # plt.imshow(self.sound_data[query_str][:][1])
        # plt.show()
        # assert False
        cur_metadata = self.metadata[query_str]
        cur_material_str = cur_metadata["material_str"]
        cur_material_num = cur_metadata["material_num"]
        cur_obj_str = cur_metadata["name_str"]
        cur_obj_num = cur_metadata["name_num"]
        cur_position = cur_metadata["obj_pos_relative"]
        if random.random()<0.05:
            # np.log(1e-3) = -6.90775527898213
            spec_data = torch.nn.functional.pad(spec_data, pad=[0, 3572-spec_data.shape[2], 0, 0, 0, 0], value=-6.90775527898213)



        actual_spec_len = spec_data.shape[2]
        spec_data = (spec_data - self.mean[:,:,:actual_spec_len])/self.std[:,:,:actual_spec_len]
        # 2, freq, time
        sound_size = spec_data.shape
        selected_time = np.random.randint(0, sound_size[2], self.num_samples)
        selected_freq = np.random.randint(0, sound_size[1], self.num_samples)

        non_norm_start = (np.array(cur_position))

        total_non_norm_position = torch.from_numpy(non_norm_start)[None].float()

        selected_total = spec_data[:,selected_freq,selected_time]
        loaded = True
            #
            # except Exception as e:
            #     print(query_str)
            #     print(e)
            #     print("Failed to load sound sample")

        return selected_total,cur_material_num, cur_obj_num, total_non_norm_position, 2.0*torch.from_numpy(selected_freq).float()/255.0 - 1.0, 2.0*torch.from_numpy(selected_time).float()/float(4065-1)-1.0

    def getitem_whole_v3(self, idx):
        loaded = False
        # while not loaded:
        #     try:
        if self.sound_data is None:
            self.sound_data = h5py.File(self.full_path, 'r')
        if self.sound_data_spec is None:
            self.sound_data_spec = h5py.File(self.full_path_spec, 'r')

        # if idx % 2==0:
        #     idx = 1
        # else:
        #     idx = 0
        # myidx = idx % 100
        myidx = idx
        query_str = self.testing[myidx]
        # idx = 0

        spec_data = torch.from_numpy(self.sound_data[query_str][:]).float()[:, None]
        spec_data_img = torch.from_numpy(self.sound_data_spec[query_str][:]).float()
        # print(spec_data.shape)
        # exit()
        cur_metadata = self.metadata[query_str]
        cur_material_str = cur_metadata["material_str"]
        cur_material_num = cur_metadata["material_num"]
        cur_obj_str = cur_metadata["name_str"]
        cur_obj_num = cur_metadata["name_num"]
        cur_type_str = cur_metadata["type_str"]
        cur_type_num = cur_metadata["type_num"]
        cur_position = cur_metadata["obj_pos_relative"]
        # cur_position = [cur_metadata["perception_pos"][0], cur_metadata["perception_pos"][2]]
        cur_material_num = cur_type_num
        # if random.random()<1.0:
        #     # np.log(1e-3) = -6.90775527898213
        spec_data_img = torch.nn.functional.pad(spec_data_img,
                                                pad=[0, 1000 - min(spec_data_img.shape[2], 1000), 0, 0, 0, 0],
                                                value=-6.90775527898213)[:, :128, :512] + 6.9

        # actual_spec_len = spec_data.shape[2]
        spec_data_np = spec_data.numpy()
        # probs_flat = np.clip((np.exp(spec_data_np[0]).flatten()+np.exp(spec_data_np[1]).flatten())*0.5, 0.000, 1000.0)
        # probs_flat = probs_flat/np.sum(probs_flat)
        # indices = np.arange(0, len(probs_flat), dtype=np.int32)
        # sample = np.random.choice(indices, size=1500, p=probs_flat, replace=False)
        # unraveled_sample = np.unravel_index(sample, spec_data[0].shape)
        # spec_data = (spec_data - self.mean) / (self.std)
        spec_data = torch.log(spec_data / self.scaling + 1e-2)
        # spec_data = (spec_data - torch.mean(spec_data))
        # spec_data = spec_data/(torch.max(torch.abs(spec_data))+1e-8)
        # 2, freq, time
        sound_size = spec_data.shape
        # selected_time = np.concatenate([np.random.randint(0, sound_size[2], self.num_samples-1500), unraveled_sample[1]])
        # selected_freq = np.concatenate([np.random.randint(0, sound_size[1], self.num_samples-1500), unraveled_sample[0]])

        # selected_time = np.arange(0, actual_spec_len)
        # selected_freq = np.arange(0, 256)
        # selected_time, selected_freq = np.meshgrid(selected_time, selected_freq)
        # selected_time = selected_time.reshape(-1)
        # selected_freq = selected_freq.reshape(-1)
        selected_time = np.arange(40)
        selected_freq = np.array([0] * 40)

        non_norm_start = (np.array(cur_position) + np.random.normal(0, 1, 2) * 0.00)

        total_non_norm_position = torch.from_numpy(non_norm_start)[None].float()

        selected_total = spec_data.reshape(-1)
        loaded = True

        return selected_total, spec_data_img, total_non_norm_position


    def getitem_whole_v2(self, idx):
            loaded = False
            # while not loaded:
            #     try:
            if self.sound_data is None:
                self.sound_data = h5py.File(self.full_path, 'r')

            query_str = self.training[idx]
            # idx = 0

            spec_data = torch.from_numpy(self.sound_data[query_str][:]).float()[...,:1000]
            cur_metadata = self.metadata[query_str]
            cur_material_str = cur_metadata["material_str"]
            cur_material_num = cur_metadata["material_num"]
            cur_obj_str = cur_metadata["name_str"]
            cur_obj_num = cur_metadata["name_num"]
            cur_type_str = cur_metadata["type_str"]
            cur_type_num = cur_metadata["type_num"]
            cur_position = cur_metadata["obj_pos_relative"]
            cur_material_num = cur_type_num
            if random.random()<1.0:
                # np.log(1e-3) = -6.90775527898213
                spec_data = torch.nn.functional.pad(spec_data, pad=[0, 1000-min(spec_data.shape[2], 1000), 0, 0, 0, 0], value=-6.90775527898213)



            actual_spec_len = spec_data.shape[2]
            spec_data_np = spec_data.numpy()
            probs_flat = np.clip((np.exp(spec_data_np[0]).flatten()+np.exp(spec_data_np[1]).flatten())*0.5, 0.000, 1000.0)
            probs_flat = probs_flat/np.sum(probs_flat)
            indices = np.arange(0, len(probs_flat), dtype=np.int32)
            sample = np.random.choice(indices, size=1500, p=probs_flat, replace=False)
            unraveled_sample = np.unravel_index(sample, spec_data[0].shape)

            spec_data = (spec_data - self.mean[:,:,:actual_spec_len])/self.std[:,:,:actual_spec_len]
            # 2, freq, time
            sound_size = spec_data.shape
            # selected_time = np.concatenate([np.random.randint(0, sound_size[2], self.num_samples-1500), unraveled_sample[1]])
            # selected_freq = np.concatenate([np.random.randint(0, sound_size[1], self.num_samples-1500), unraveled_sample[0]])
            #
            # selected_time = np.arange(0, actual_spec_len)
            # selected_freq = np.arange(0, 256)
            # selected_time, selected_freq = np.meshgrid(selected_time, selected_freq)
            # selected_time = selected_time.reshape(-1)
            # selected_freq = selected_freq.reshape(-1)
            selected_time = np.arange(0, actual_spec_len)
            selected_freq = np.arange(0, 256)
            selected_time, selected_freq = np.meshgrid(selected_time, selected_freq)
            selected_time = selected_time.reshape(-1)
            selected_freq = selected_freq.reshape(-1)

            non_norm_start = np.array(cur_position)

            total_non_norm_position = torch.from_numpy(non_norm_start)[None].float()

            selected_total = spec_data[:,selected_freq,selected_time]
            loaded = True
                #
                # except Exception as e:
                #     print(query_str)
                #     print(e)
                #     print("Failed to load sound sample")

            return selected_total,cur_material_num, cur_type_num, total_non_norm_position, 2.0*torch.from_numpy(selected_freq).float()/255.0 - 1.0, 2.0*torch.from_numpy(selected_time).float()/float(1000)-1.0

    def getitem_whole(self, idx):
        loaded = False
        # while not loaded:
        #     try:
        if self.sound_data is None:
            self.sound_data = h5py.File(self.full_path, 'r')

        # query_str = self.testing[idx]
        query_str = self.training[idx]
        self.spec_data = torch.from_numpy(self.sound_data[query_str][:]).float()[...,:1000]
        spec_data = self.spec_data+0.0
        # if query_str != self.query_str:
        #     spec_data = torch.from_numpy(self.sound_data[query_str][:]).float()[...,:1000]
        #     self.query_str = query_str
        #     self.spec_data = spec_data
        # else:
        #     spec_data = self.spec_data


        cur_metadata = self.metadata[query_str]
        cur_material_str = cur_metadata["material_str"]
        cur_material_num = cur_metadata["material_num"]
        cur_obj_str = cur_metadata["name_str"]
        cur_obj_num = cur_metadata["name_num"]
        cur_position = cur_metadata["obj_pos_relative"]
        cur_type_num = cur_metadata["type_num"]
        self.shapes = spec_data.shape
        # if random.random()<0.00:
        #     # np.log(1e-3) = -6.90775527898213
        #     spec_data = torch.nn.functional.pad(spec_data, pad=[0, 3572-spec_data.shape[2], 0, 0, 0, 0], value=-6.90775527898213)



        # assert False
        actual_spec_len = spec_data.shape[2]
        spec_data = (spec_data - self.mean[:,:,:actual_spec_len])/self.std[:,:,:actual_spec_len]
        # 2, freq, time
        sound_size = spec_data.shape
        # selected_time = np.random.randint(0, sound_size[2], self.num_samples)
        # selected_freq = np.random.randint(0, sound_size[1], self.num_samples)
        selected_time = np.arange(0, actual_spec_len)
        selected_freq = np.arange(0, 256)
        selected_time, selected_freq = np.meshgrid(selected_time, selected_freq)
        selected_time = selected_time.reshape(-1)
        selected_freq = selected_freq.reshape(-1)
        # print(spec_data.shape, "orig shape", actual_spec_len)
        # exit()

        non_norm_start = (np.array(cur_position))

        total_non_norm_position = torch.from_numpy(non_norm_start)[None].float()

        selected_total = spec_data[:,selected_freq,selected_time]
        selected_mean = self.mean[:,selected_freq,selected_time]
        selected_std = self.std[:,selected_freq,selected_time]
        # print(selected_total.shape)
        # exit()
        loaded = True
            #
            # except Exception as e:
            #     print(query_str)
            #     print(e)
            #     print("Failed to load sound sample")

        return selected_total,cur_material_num, cur_type_num, total_non_norm_position, 2.0*torch.from_numpy(selected_freq).float()/255.0 - 1.0, 2.0*torch.from_numpy(selected_time).float()/float(1000)-1.0, selected_mean, selected_std