# add parent path
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from SSN_Dataset import SSN_Dataset 
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import h5py


# Configs
resume_path = './models/control_sd21_ini.ckpt'
# batch_size = 4
batch_size = 160
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
hdf5_fname = 'Dataset/ControlNet_SSN_Exp'
ds_dict = {
	'ds_root': hdf5_fname,
}
worker = 64

dataset = SSN_Dataset(ds_dict)
# dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
dataloader = DataLoader(dataset, num_workers=worker, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)

# trainer = pl.Trainer(gpus=2, precision=32, callbacks=[logger])
trainer = pl.Trainer(strategy="ddp", accelerator="gpu", devices=2, precision=32, callbacks=[logger])


# Train!
trainer.fit(model, dataloader)
