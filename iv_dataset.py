import pdb  # noqa: F401

import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset

from typing import Tuple, Any, Union

import os
from glob import glob
import copy
import multiprocessing as mp


TensorArray = Union[torch.Tensor, np.ndarray]


class IVDataset(Dataset):
    L_cut_x = 1
    L_cut_y = 1

    def __init__(self, DIR:str, XNAME:str, YNAME:str,
                 N_data=-1, normalize=True):
        self.DIR = DIR
        self.XNAME = XNAME
        self.YNAME = YNAME
        self.normalize = normalize

        # for file in os.scandir(DIR):
        self.all_files = glob(os.path.join(DIR,'*.npy'))
        if N_data != -1:
            self.all_files = self.all_files[:N_data]
        for file in self.all_files[:]:
            if file.endswith('metadata.npy'):
                self.all_files.remove(file)

        # Calculate summation & no. of total frames (parallel)
        if normalize:
            N_CORES = mp.cpu_count()
            pool = mp.Pool(N_CORES)
            result = pool.map(IVDataset.sum_files,
                              [(file, XNAME, YNAME)
                               for file in self.all_files])

            sum_x = np.sum([res[0] for res in result], axis=0
                           )[np.newaxis,:,:,:]
            N_frame_x = np.sum([res[1] for res in result])
            sum_y = np.sum([res[2] for res in result], axis=0
                           )[:,np.newaxis,:]
            N_frame_y = np.sum([res[3] for res in result])

            # mean
            self.mean_x = sum_x / N_frame_x
            self.mean_y = sum_y / N_frame_y
            # self.mean_x = (sum_x+sum_y) / (N_frame_x + N_frame_y)
            # self.mean_y = (sum_x+sum_y) / (N_frame_x + N_frame_y)

            # Calculate Standard Deviation
            result = pool.map(IVDataset.sum_dev_files,
                              [(file, XNAME, YNAME, self.mean_x, self.mean_y)
                               for file in self.all_files])

            pool.close()

            sum_dev_x = np.sum([res[0] for res in result], axis=0
                               )[np.newaxis,:,:,:]
            sum_dev_y = np.sum([res[1] for res in result], axis=0
                               )[:,np.newaxis,:]

            self.std_x = np.sqrt(sum_dev_x / N_frame_x + 1e-5)
            self.std_y = np.sqrt(sum_dev_y / N_frame_y + 1e-5)
            # self.std_x = np.sqrt((sum_dev_x + sum_dev_y)
            #                      /(N_frame_x + N_frame_y)
            #                      + 1e-5)
            # self.std_y = np.sqrt((sum_dev_x + sum_dev_y)
            #                      /(N_frame_x + N_frame_y)
            #                      + 1e-5)
        else:
            self.mean_x = 0.
            self.mean_y = 0.
            self.std_x = 1.
            self.std_y = 1.

        print(f'{len(self)} data prepared from {os.path.basename(DIR)}.')

    @classmethod
    def sum_files(cls, tup:Tuple[str, str, str]) -> Tuple[Any, int, Any, int]:
        file, XNAME, YNAME = tup
        try:
            data_dict = np.load(file).item()
            x = data_dict[XNAME]
            y = data_dict[YNAME]
        except:  # noqa: E722
            pdb.set_trace()

        x_stacked = cls.stack_x(x, L_trunc=y.shape[1])

        return (x_stacked.sum(axis=0), x_stacked.shape[0],
                y.sum(axis=1), y.shape[1],
                )

    @classmethod
    def sum_dev_files(cls,
                      tup:Tuple[str, str, str, Any, Any]) -> Tuple[Any, Any]:
        file, XNAME, YNAME, mean_x, mean_y = tup
        data_dict = np.load(file).item()
        x = data_dict[XNAME]
        y = data_dict[YNAME]

        x_stacked = cls.stack_x(x, L_trunc=y.shape[1])

        return (((x_stacked - mean_x)**2).sum(axis=0),
                ((y - mean_y)**2).sum(axis=1),
                )

    def do_normalize(self, mean_x, mean_y, std_x, std_y):
        self.normalize = True
        self.mean_x = mean_x
        self.mean_y = mean_y
        self.std_x = std_x
        self.std_y = std_y

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx:int):
        # File Open
        data_dict = np.load(self.all_files[idx]).item()
        x = data_dict[self.XNAME]
        y = data_dict[self.YNAME]

        # Stack & Normalize
        x_stacked = IVDataset.stack_x(x, L_trunc=y.shape[1])
        if self.normalize:
            x_stacked = (x_stacked - self.mean_x)/self.std_x
            y = (y - self.mean_y)/self.std_y
        y_stacked = IVDataset.stack_y(y)

        x_stacked = torch.from_numpy(x_stacked).float()
        y_stacked = torch.from_numpy(y_stacked).float()
        sample = {'x_stacked': x_stacked, 'y_stacked': y_stacked}

        return sample

    # Make groups of the frames of x and stack the groups
    # x_stacked: (time_length) x (N_freq) x (L_cut_x) x (XYZ0 channel)
    @classmethod
    def stack_x(cls, x:np.ndarray, L_trunc=0) -> np.ndarray:
        if x.ndim != 3:
            raise Exception('Dimension Mismatch')
        if cls.L_cut_x == 1:
            return x

        L0, L1, L2 = x.shape

        half = cls.L_cut_x//2

        x = np.concatenate((np.zeros((L0, half, L2)),
                            x,
                            np.zeros((L0, half, L2))),
                           axis=1)

        if L_trunc != 0:
            L1 = L_trunc

        return np.stack([x[:, ii - half:ii + half + 1, :]
                         for ii in range(half, half + L1)
                         ])

    @classmethod
    def stack_y(cls, y:np.ndarray) -> np.ndarray:
        if y.ndim != 3:
            raise Exception('Dimension Mismatch')

        return y.transpose((1, 0, 2))[:,:,np.newaxis,:]

    @classmethod
    def unstack_x(cls, x:TensorArray) -> TensorArray:
        if type(x) == torch.Tensor:
            if x.dim() != 4 or x.size(2) <= cls.L_cut_x//2:
                raise Exception('Dimension/Size Mismatch')
            x = x[:,:,cls.L_cut_x//2,:].squeeze()
            return x.transpose(1, 0)
        else:
            if x.ndim != 4 or x.shape[2] <= cls.L_cut_x//2:
                raise Exception('Dimension/Size Mismatch')
            x = x[:,:,cls.L_cut_x//2,:].squeeze()
            return x.transpose((1, 0, 2))

    @classmethod
    def unstack_y(cls, y:TensorArray) -> TensorArray:
        if type(y) == torch.Tensor:
            if y.dim() != 4 or y.size(2) != 1:
                raise Exception('Dimension/Size Mismatch')
            return y.squeeze().transpose(1, 0)
        else:
            if y.ndim != 4 or y.shape[2] != 1:
                raise Exception('Dimension/Size Mismatch')
            return y.squeeze().transpose((1, 0, 2))

    @staticmethod
    def my_collate(batch):
        x_stacked = torch.cat([item['x_stacked'] for item in batch])
        y_stacked = torch.cat([item['y_stacked'] for item in batch])
        return [x_stacked, y_stacked]

    @classmethod
    def split(cls, a, ratio:Tuple):
        if type(a) != cls:
            raise TypeError
        n_split = len(ratio)
        ratio = np.array(ratio)
        mask = (ratio == -1)
        ratio[np.where(mask)] = 0
        if mask.sum() > 1:
            raise Exception("Only one element of the parameter 'ratio' "
                            "can have the value of -1")
        if ratio.sum() >= 1:
            raise Exception('The sum of ratio must be 1')
        if mask.sum() == 1:
            ratio[np.where(mask)] = 1 - ratio.sum()

        idx_data = np.cumsum(np.insert(ratio, 0, 0) * len(a), dtype=int)
        result = [copy.copy(a) for ii in range(n_split)]
        all_f_per = np.random.permutation(a.all_files)
        for ii in range(n_split):
            result[ii].all_files = all_f_per[idx_data[ii]:idx_data[ii + 1]]

        return result
