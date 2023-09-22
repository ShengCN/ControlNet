import os
from os.path import join
import h5py
import numpy as np
from glob import glob

from torch.utils.data import Dataset


class SSN_Dataset(Dataset):
    def __init__(self, ds_dict: dict):
        """ Init
        :param ds_dict: {'hdf5': path} 
        """        
        keys = ['ds_root']
        for k in keys:
            assert k in ds_dict, 'Not find key {}'.format(k)

        ds_root = ds_dict['ds_root']
        assert os.path.exists(ds_root)

        self.files = glob(join(ds_root, '*.npz'))
        self.files.sort()
        self.N = len(self.files)


    def __len__(self):
        return self.N


    def __getitem__(self, idx):
        prompt = 'shadow'

        buffers = np.load(self.files[idx])
        mask    = buffers['mask']
        ibl     = buffers['ibl']
        shadow  = buffers['shadow']

        # Normalize source images to [0, 1].
        source = np.repeat(mask[..., None], 3, axis=2)

        # Normalize target images to [-1, 1].
        target = np.repeat(shadow[..., None] * 2.0 - 1.0, 3, axis=2)

        return dict(jpg=target, txt=prompt, hint=source)

