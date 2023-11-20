'''
Mirror of trainer, done only to load the data.
Author: Virginia Fernandez
'''
import os
import moreutils as uvir
from options.train_options import TrainOptions
import data
from utils.iter_counter import IterationCounter
from utils.visualizer import Visualizer
from data.dataset_utils import clear_data
import numpy as np
import torch
from trainers.pix2pix_trainer import Pix2PixTrainer
from copy import deepcopy
import gc
from data.spadenai_v2 import SpadeNai
from data.spadenai_v2_sliced import SpadeNaiSlice
import shutil
from monai.data.dataloader import DataLoader
from utils.tensorboard_writer import BrainspadeBoard


plot_errors = False
# Parse options
opt = TrainOptions().parse()

# # Save images for discriminator training
# folder_save = "/home/vf19/Documents/brainSPADE_2D/DATA/DISCRIMINATOR_ONLY_TRAINING_VAL"
# if not os.path.isdir(folder_save):
#     os.makedirs(folder_save)

# Remove triplet
os.chdir('..')  # Change directory to the previous one

# Dataset
if opt.dataset_type == 'sliced':
    dataset_container = SpadeNaiSlice(opt, mode = 'train')
    dataset_val_container = SpadeNaiSlice(opt, mode='validation')
else:
    dataset_container = SpadeNai(opt, mode = 'train')
    dataset_val_container = SpadeNai(opt, mode = 'validation', store_and_use_slices=True)
dataloader = DataLoader(dataset_container.sliceDataset,
                                         batch_size=opt.batchSize, shuffle=False,
                                         num_workers=1, drop_last=opt.isTrain)
dataloader_val = DataLoader(dataset_val_container.sliceDataset,
                                             batch_size=opt.batchSize, shuffle=False,
                                             num_workers=1, drop_last=False)

# Iterations counter
iter_counter = IterationCounter(opt, len(dataset_container))

# Training Loop
for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for dind, data_i in enumerate(dataloader):
        continue