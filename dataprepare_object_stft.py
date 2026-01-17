import numpy as np
import scipy
import matplotlib.pyplot as plt

import math
import json
from pprint import pprint
import os
import pandas as pd
import pickle
from scipy.signal import welch
import scipy
import pandas as pd


def json_loader(input_filename):
    with open(input_filename, "rb") as f_:
        data_json_ = json.load(f_)
    return data_json_


def return_paths(input_path):
    files_ = os.listdir(input_path)
    return [os.path.join(input_path, _) for _ in files_]


def filter_obj(list_in, require):
    return [__ for __ in list_in if require in __]



import torchaudio
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa
# from skimage.transform import rescale, resize
# from scipy.interpolate import interp1d
import os


def load_audio(path_name, use_torch=True, resample=True, resample_rate=22050):
    # returns in shape (ch, num_sample), as float32 (on Linux at least)
    # by default torchaudio is wav_arr, sample_rate
    # by default wavfile is sample_rfaxate, wav_arr
    if use_torch:
        loaded = torchaudio.load(path_name)
        wave_data_loaded = loaded[0].numpy()
        sr_loaded = loaded[1]
    else:
        loaded = wavfile.read(path_name)

        #print(str(loaded[1]))
        #print(str(loaded[1].dtype))
        #exit()
        #assert str(loaded[1].dtype) == "int16"#float32
        #wave_data_loaded = np.clip(loaded[1] / np.iinfo(loaded[1].dtype).max, -10.0, 10.0).T
        float32_data= loaded[1]#/ 1.414*32767
        # print(min(float32_data))
        # print(max(float32_data))

        # int16_data = float32_data.astype(np.int16)
        #int16_data = float32_data.astype(np.int16)
        # print(min(int16_data))
        # print(max(int16_data))
        #exit(-1)


        #wave_data_loaded = np.clip(int16_data, -10.0, 10.0).T
        #wave_data_loaded = np.clip(float32_data, -10.0, 10.0).T
        wave_data_loaded=float32_data.T

        sr_loaded = loaded[0]
        #print(sr_loaded)
        #print(int16_data)

    resampled_wave = wave_data_loaded
    assert sr_loaded == 44100
    #return np.clip(resampled_wave, -10.0, 10.0)
    return resampled_wave


class get_spec():
    def __init__(self, use_torch=False, power_mod=2, fft_size=512):
        self.n_fft = fft_size
        self.hop = self.n_fft // 4
        if use_torch:
            self.use_torch = True
            #             self.spec_transform = Spectrogram(power=None)
            self.spec_transform = Spectrogram(power=None, n_fft=self.n_fft, hop_length=self.hop)
        else:
            self.power = power_mod
            self.use_torch = False
            self.spec_transform = None

    def transform(self, wav_data_prepad):
        # print(wav_data_prepad)
        # print(wav_data_prepad.shape)#(132300,)
        wav_data = librosa.util.fix_length(wav_data_prepad, wav_data_prepad.shape[-1] + self.n_fft // 2)
        #print(wav_data.shape)#(132556,)
        #exit(-1)

        if wav_data.shape[0] < 4410:
            wav_data = librosa.util.fix_length(wav_data, 4410)
        if self.use_torch:
            transformed_data = self.spec_transform(torch.from_numpy(wav_data)).numpy()
        else:
            transformed_data = np.array([librosa.stft(wav_data, n_fft=self.n_fft, hop_length=self.hop)])#[ :-1]
        #print(transformed_data)#1 257 1036
        #exit(-1)
        real_component = np.abs(transformed_data)
        #print(real_component.shape)
        #exit(-1)
        # lengh = (real_component.shape[2])
        # if (lengh < 800):
        #     a = np.ones((1, 257, 800 - lengh)) * 0
        #     print(lengh)#1
        #     real_component = np.concatenate((real_component, a), axis=2)  # 1e-4
        # else:
        #     real_component = real_component[:, :, :800]
        # new
        # print(real_component[:,:,499])
        real_component = real_component[:, :, :201]
        #print(real_component.shape)
        return np.log(real_component + 1e-3)#1e-7


spec_getter = get_spec()
scaling = 10000
try:
    f_mag.close()
    del f_mag
except:
    pass



df = pd.read_csv('/data/vision/torralba/scratch/chuang/Projects/ObjectFolder/objects.csv')
types=[]



mat_dict={'Steel':0, 'Wood':1, 'Polycarbonate':2, 'Plastic':3, 'Iron':4, 'Ceramic':5, 'Glass':6}
all_container = dict()
basepath="/data/vision/torralba/scratch/chuang/Projects/ObjectFolder/data/audio_results/"

for index, row in df.iterrows():
    #print(index+1)
    wavpath=basepath+str(row[0])+"/4.wav"
    if not os.path.exists(wavpath):
        continue
    mat_num=mat_dict[row[3]]
    scale_num = row[2]
    #print(scale_num)

    for cnt in range(1,5):
        #print(str(row[0])+"_"+str(cnt))
        all_container[str(row[0])+"_"+str(cnt)] = {"material_num": mat_num,"scale_num": scale_num}



with open("/data/vision/torralba/scratch/chuang/Projects/ObjectFolder/all_container_both.pkl", "wb") as fff:
    pickle.dump(all_container, fff)
print("all container file saved!")

f_mag = h5py.File("/data/vision/torralba/scratch/chuang/Projects/ObjectFolder/object_psd.h5", 'w')#object_psd.h5

for index, row in df.iterrows():
    #print(index+1)
    wavpath=basepath+str(row[0])+"/4.wav"
    #print(wavpath)
    if not os.path.exists(wavpath):
        continue
    for cnt in range(1,5):
        #print(str(row[0])+"_"+str(cnt))
        #print(basepath+str(row[0])+"/"+str(cnt)+".wav")
        wav_i=load_audio(basepath+str(row[0])+"/"+str(cnt)+".wav", use_torch=False)

        wc1 = welch(wav_i, fs=44100)
        #wc2 = welch(wav_i[1], fs=44100)
        #real_spec = np.stack([wc1[1], wc2[1]]) * scaling
        real_spec = np.stack([wc1[1]]) * scaling
        #print(real_spec)

        f_mag.create_dataset(str(row[0])+"_"+str(cnt),
                             data=real_spec.astype(np.single))


print("psd file saved!")
f_mag.close()

f_mag2 = h5py.File("/data/vision/torralba/scratch/chuang/Projects/ObjectFolder/object_stft.h5", 'w')#object_stft.h5
for index, row in df.iterrows():
    # print(index+1)
    wavpath = basepath + str(row[0]) + "/4.wav"
    if not os.path.exists(wavpath):
        continue
    for cnt in range(1, 5):

        wav_i = load_audio(basepath + str(row[0]) + "/" + str(cnt) + ".wav", use_torch=False)
        real_spec = spec_getter.transform(wav_i) #* scaling

        f_mag2.create_dataset(str(row[0]) + "_" + str(cnt),
                             data=real_spec.astype(np.single))

f_mag2.close()

print("stft file saved!")




