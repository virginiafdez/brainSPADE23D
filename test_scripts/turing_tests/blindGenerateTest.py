'''
This script generates fake images and saves them along the fakes, to perform a blind test (AKA: user saying if the image is real or fake).
After the storage, call analyseTestFiles to 1) label the images 2) get a statistical analysis.
'''

from options.turing_test_options import TuringTestOptions
from models_spade.pix2pix_model import Pix2PixModel
import os
import moreutils as uvir
import numpy as np
from data.spadenai_v2_sliced import SpadeNaiSlice
from data.spadenai_v2 import SpadeNai
import monai
from copy import deepcopy

uvir.set_deterministic(True, 1)

# Initialisation
opt = TuringTestOptions().parse()
opt_sy = deepcopy(opt)
opt_sy.label_dir = opt.fake_label_dir
opt_sy.non_corresponding_dirs = True
opt_sy.style_label_dir = opt.label_dir

how_many = 50
opt.batchSize = 1

# Create model
model = Pix2PixModel(opt)
model = model.eval()
modalities = opt.sequences

# Blind test results dir
if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.name, 'turing_test')):
    os.makedirs(os.path.join(opt.checkpoints_dir, opt.name,  'turing_test'))

# Dataset creation and shuffle.

for mod in modalities:
    # Create results dir
    if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.name,  'turing_test', mod)):
        os.makedirs(os.path.join(opt.checkpoints_dir, opt.name, 'turing_test', mod))

    # Naming
    all_subjects = ['0'*(len(str(how_many * 3)) - len(str(i))) + str(i) for i in np.arange(0, how_many * 3)]
    np.random.shuffle(all_subjects)
    all_subjects_map = {}

    counter = 0
    opt.fix_seq = mod
    opt_sy.fix_seq = mod

    # We first do the normal labels.
    if opt.dataset_type == 'volume':
        dataset_container_gt = SpadeNai(opt, mode= 'test')
        dataset_container_sy = SpadeNai(opt_sy, mode = 'test')
    else:
        dataset_container_gt = SpadeNaiSlice(opt, mode = 'test')
        dataset_container_sy = SpadeNaiSlice(opt_sy, mode = 'test')

    dataloader_gt = monai.data.dataloader.DataLoader(
        dataset_container_gt.sliceDataset,
        batch_size=opt.batchSize, shuffle=False,
        num_workers=int(opt.nThreads), drop_last=opt.isTrain)

    dataloader_sy = monai.data.dataloader.DataLoader(
        dataset_container_sy.sliceDataset,
        batch_size=opt.batchSize, shuffle=False,
        num_workers=int(opt.nThreads), drop_last=opt.isTrain)

    for i, data_i in enumerate(dataloader_gt):
        if i * opt.batchSize >= how_many:
            break
        generated = model(data_i, 'generator_test').detach().cpu()
        # We skull strip the generated image and the ground truth.
        for b in range(generated.shape[0]):  # For each image of the batch (should be one)
            # We process the path to get the output name
            name_gt = "sample_%s.png" %(all_subjects[counter])
            name_sy = "sample_%s.png" % (all_subjects[counter+1])

            # Detach label and image
            label = data_i['label'][b].detach().cpu()  # Input label
            input = data_i['image'][b, 0:1, ...].detach().cpu()
            synth = generated[b]  # Input image
            uvir.saveSingleImage(input, label, skullstrip=True, path=os.path.join(
                opt.checkpoints_dir, opt.name, 'turing_test', mod, name_gt), denormalize=False,
                                 save_as_grey=True,)
            uvir.saveSingleImage(synth, label, skullstrip=True, path=os.path.join(
                opt.checkpoints_dir, opt.name, 'turing_test', mod, name_sy), denormalize=False,
                                 save_as_grey=True, )

            # Save gt
            all_subjects_map[all_subjects[counter]] = 0 # Real images, real labels
            all_subjects_map[all_subjects[counter+1]] = 1 # Real labels, fake images

            counter += 2

    # Now we loop through the fake data.
    for i, data_i in enumerate(dataloader_sy):
        if i * opt.batchSize >= how_many:
            break
        generated = model(data_i, 'generator_test').detach().cpu()

        for b in range(generated.shape[0]):  # For each image of the batch (should be one)
            # We process the path to get the output name
            name_ssy = "sample_%s.png" % (all_subjects[counter])

            # Detach label and image
            label = data_i['label'][b].detach().cpu()  # Input label
            synth = generated[b]  # Input image
            uvir.saveSingleImage(synth, label, skullstrip=True, path=os.path.join(
                opt.checkpoints_dir, opt.name, 'turing_test', mod, name_ssy), denormalize=True,
                                 save_as_grey=True)

            # Save gt
            all_subjects_map[all_subjects[counter]] = 2  # Real images, real labels

            counter += 1

    with open(os.path.join(opt.checkpoints_dir, opt.name, 'turing_test', mod, mod + "_directory.txt"), 'w') as f:
        f.write("Image name\tFakeness Label\n")
        for sub_id, sub_tag in all_subjects_map.items():
            f.write("%s\t%d\n" % ("sample_%s.png" %sub_id, sub_tag))
        f.close()

