import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import math
import json
from pprint import pprint
import os
import pandas as pd
import pickle
from scipy.signal import welch
import scipy


def json_loader(input_filename):
    with open(input_filename, "rb") as f_:
        data_json_ = json.load(f_)
    return data_json_


def return_paths(input_path):
    files_ = os.listdir(input_path)
    return [os.path.join(input_path, _) for _ in files_]


def filter_obj(list_in, require):
    return [__ for __ in list_in if require in __]


class QuaternionUtils:
    """
    Helper functions for using quaternions.

    Quaternions are always numpy arrays in the following order: `[x, y, z, w]`.
    This is the order returned in all Output Data objects.

    Vectors are always numpy arrays in the following order: `[x, y, z]`.
    """

    """:class_var
    The global up directional vector.
    """
    UP = np.array([0, 1, 0])
    """:class_var
    The global forward directional vector.
    """
    FORWARD: np.array = np.array([0, 0, 1])
    """:class_var
    The quaternion identity rotation.
    """
    IDENTITY = np.array([0, 0, 0, 1])
    IDENTITY.setflags(write=0)

    @staticmethod
    def get_inverse(q: np.array) -> np.array:
        """
        Source: https://referencesource.microsoft.com/#System.Numerics/System/Numerics/Quaternion.cs

        :param q: The quaternion.

        :return: The inverse of the quaternion.
        """

        x = q[0]
        y = q[1]
        z = q[2]
        w = q[3]

        ls = x * x + y * y + z * z + w * w
        inv = 1.0 / ls

        return np.array([-x * inv, -y * inv, -z * inv, w * inv])

    @staticmethod
    def multiply(q1: np.array, q2: np.array) -> np.array:
        """
        Multiply two quaternions.
        Source: https://stackoverflow.com/questions/4870393/rotating-coordinate-system-via-a-quaternion

        :param q1: The first quaternion.
        :param q2: The second quaternion.
        :return: The multiplied quaternion: `q1 * q2`
        """

        x1 = q1[0]
        y1 = q1[1]
        z1 = q1[2]
        w1 = q1[3]

        x2 = q2[0]
        y2 = q2[1]
        z2 = q2[2]
        w2 = q2[3]

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
        return np.array([x, y, z, w])

    @staticmethod
    def get_conjugate(q: np.array) -> np.array:
        """
        Source: https://stackoverflow.com/questions/4870393/rotating-coordinate-system-via-a-quaternion

        :param q: The quaternion.

        :return: The conjugate of the quaternion: `[-x, -y, -z, w]`
        """

        x = q[0]
        y = q[1]
        z = q[2]
        w = q[3]

        return np.array([-x, -y, -z, w])

    @staticmethod
    def multiply_by_vector(q: np.array, v: np.array) -> np.array:
        """
        Multiply a quaternion by a vector.
        Source: https://stackoverflow.com/questions/4870393/rotating-coordinate-system-via-a-quaternion

        :param q: The quaternion.
        :param v: The vector.

        :return: A directional vector calculated from: `q * v`
        """

        q2 = (v[0], v[1], v[2], 0.0)
        return QuaternionUtils.multiply(QuaternionUtils.multiply(q, q2), QuaternionUtils.get_conjugate(q))[:-1]

    @staticmethod
    def world_to_local_vector(position: np.array, origin: np.array, rotation: np.array) -> np.array:
        """
        Convert a vector position in absolute world coordinates to relative local coordinates.
        Source: https://answers.unity.com/questions/601062/what-inversetransformpoint-does-need-explanation-p.html

        :param position: The position vector in world coordinates.
        :param origin: The origin vector of the local space in world coordinates.
        :param rotation: The rotation quaternion of the local coordinate space.

        :return: `position` in local coordinates.
        """

        return QuaternionUtils.multiply_by_vector(
            q=QuaternionUtils.get_invemedia / aluo / big2 / multimodal_challenge_combined / dataset / multimodal_mag_v3.h5rse(
                q=rotation), v=position - origin)

    @staticmethod
    def get_up_direction(q: np.array) -> np.array:
        """
        :param q: The rotation as a quaternion.

        :return: A directional vector corresponding to the "up" direction from the quaternion.
        """

        return QuaternionUtils.multiply_by_vector(q, QuaternionUtils.UP)

    @staticmethod
    def euler_angles_to_quaternion(euler: np.array) -> np.array:
        """
        Convert Euler angles to a quaternion.

        :param euler: The Euler angles vector.

        :return: The quaternion representation of the Euler angles.
        """

        roll = euler[0]
        pitch = euler[1]
        yaw = euler[2]
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        w = cy * cp * cr + sy * sp * sr
        x = cy * cp * sr - sy * sp * cr
        y = sy * cp * sr + cy * sp * cr
        z = sy * cp * cr - cy * sp * sr
        return np.array([x, y, z, w])

    @staticmethod
    def quaternion_to_euler_angles(quaternion: np.array) -> np.array:
        """
        Convert a quaternion to Euler angles.

        :param quaternion: A quaternion as a nump array.

        :return: The Euler angles representation of the quaternion.
        """

        x = quaternion[0]
        y = quaternion[1]
        z = quaternion[2]
        w = quaternion[3]
        ysqr = y * y

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + ysqr)
        ex = np.degrees(np.arctan2(t0, t1))

        t2 = +2.0 * (w * y - z * x)
        t2 = np.where(t2 > +1.0, +1.0, t2)

        t2 = np.where(t2 < -1.0, -1.0, t2)
        ey = np.degrees(np.arcsin(t2))

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (ysqr + z * z)
        ez = np.degrees(np.arctan2(t3, t4))

        return np.array([ex, ey, ez])

    @staticmethod
    def get_y_angle(q1: np.array, q2: np.array) -> float:
        """
        Source: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

        :param q1: The first quaternion.
        :param q2: The second quaternion.

        :return: The angle between the two quaternions in degrees around the y axis.
        """

        qd = QuaternionUtils.multiply(QuaternionUtils.get_conjugate(q1), q2)
        return np.rad2deg(2 * np.arcsin(np.clip(qd[1], -1, 1)))

    @staticmethod
    def is_left_of(origin: np.array, target: np.array, forward: np.array) -> bool:
        """
        :param origin: The origin position.
        :param target: The target position.
        :param forward: The forward directional vector.

        :return: True if `target` is to the left of `origin` by the `forward` vector; False if it's to the right.
        """

        # Get the heading.
        target_direction = target - origin
        # Normalize the heading.
        target_direction = target_direction / np.linalg.norm(target_direction)
        perpendicular: np.array = np.cross(forward, target_direction)
        direction = np.dot(perpendicular, QuaternionUtils.UP)
        return direction > 0


def map_position(object_position, robot_position, robot_rotation):
    angle = -QuaternionUtils.get_y_angle(QuaternionUtils.IDENTITY, robot_rotation)
    angle = np.radians(angle)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    relative_pos = object_position - robot_position
    return relative_pos @ rot


import torchaudio
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa
from skimage.transform import rescale, resize
from scipy.interpolate import interp1d
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
        assert str(loaded[1].dtype) == "int16"
        wave_data_loaded = np.clip(loaded[1] / np.iinfo(loaded[1].dtype).max, -10.0, 10.0).T
        sr_loaded = loaded[0]

    resampled_wave = wave_data_loaded
    assert sr_loaded == 44100
    return np.clip(resampled_wave, -10.0, 10.0)


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
        wav_data = librosa.util.fix_length(wav_data_prepad, wav_data_prepad.shape[-1] + self.n_fft // 2)
        if wav_data.shape[1] < 4410:
            wav_data = librosa.util.fix_length(wav_data, 4410)
        if self.use_torch:
            transformed_data = self.spec_transform(torch.from_numpy(wav_data)).numpy()
        else:
            transformed_data = np.array([librosa.stft(wav_data[0], n_fft=self.n_fft, hop_length=self.hop),
                                         librosa.stft(wav_data[1], n_fft=self.n_fft, hop_length=self.hop)])[:, :-1]

        real_component = np.abs(transformed_data)
        # print(real_component.shape)
        lengh = len(real_component[0][0])
        if (lengh < 600):
            a = np.ones((2, 256, 600 - lengh)) * 0
            # print(lengh)
            real_component = np.concatenate((real_component, a), axis=2)  # 1e-4
        else:
            real_component = real_component[:, :, :600]
        # new
        # print(real_component[:,:,499])
        return np.log(real_component + 1e-3)


spec_getter = get_spec()
scaling = 10000
try:
    f_mag.close()
    del f_mag
except:
    pass

f_mag = h5py.File("/home/zdp21n5/datasets/multimodal_challenge_combined//dataset/spectral_data_xxx.h5", 'w')

offset = 0
for room in sorted(return_paths("/home/zdp21n5/datasets/multimodal_challenge_combined//dataset/distractor")):
    fax = room
    if not os.path.isdir(fax):
        continue
    all_jsons = sorted(filter_obj(return_paths(fax), "json"))
    for trial in all_jsons:
        offset += 1
        if offset % 100 == 0:
            print(offset)
        loaded_wav = load_audio(trial.replace(".json", ".wav"), use_torch=False)
        # print(loaded_wav.shape)#(2, 63504)

        ### USE WELCH DATA
        wc1 = welch(loaded_wav[0], fs=44100)

        wc2 = welch(loaded_wav[1], fs=44100)

        real_spec = np.stack([wc1[1], wc2[1]])*scaling
        # print(real_spec.shape)

        #         plt.plot(real_spec[0])
        #         plt.plot(real_spec[1])
        #         plt.show()
        ## USE SPECTROGRAM DATA
        #real_spec = spec_getter.transform(loaded_wav) * scaling

        # real_spec=np.stack(real_spec[0],real_spec[1])
        #         ch1=real_spec[0][0:500]
        #         ch2=real_spec[1][0:500]
        #         print(real_spec.shape)
        #         real_spec=np.stack(ch1,ch2)
        # plt.imshow(real_spec[0]) # left ch
        #         print(real_spec[0][0:500].shape)#(256, 499)
        #         print(real_spec[0].shape)
        #         plt.show()
        #         plt.imshow(real_spec[1]) # right ch
        #         plt.show()

        # assert False

        f_mag.create_dataset(os.path.basename(room) + "__" + os.path.basename(trial) + "__distractor",
                             data=real_spec.astype(np.single))
        # print(f_mag)
        # break
for room in sorted(
        return_paths("/home/zdp21n5/datasets/multimodal_challenge_combined//dataset/non_distractor")):
    fax = room
    if not os.path.isdir(fax):
        continue
    all_jsons = sorted(filter_obj(return_paths(fax), "json"))
    for trial in all_jsons:
        offset += 1
        if offset % 100 == 0:
            print(offset)
        loaded_wav = load_audio(trial.replace(".json", ".wav"), use_torch=False)

        ### USE WELCH DATA
        wc1 = welch(loaded_wav[0], fs=44100)
        wc2 = welch(loaded_wav[1], fs=44100)
        real_spec = np.stack([wc1[1], wc2[1]])*scaling
        #         plt.plot(real_spec[0])
        #         plt.plot(real_spec[1])
        #         plt.show()

        ### USE SPECTROGRAM DATA
        #real_spec = spec_getter.transform(loaded_wav) * scaling
        #         plt.imshow(real_spec[0]) # left ch
        #         plt.show()
        #         plt.imshow(real_spec[1]) # right ch
        #         plt.show()
        # assert False
        f_mag.create_dataset(os.path.basename(room) + "__" + os.path.basename(trial) + "__non_distractor",
                             data=real_spec.astype(np.single))
f_mag.close()