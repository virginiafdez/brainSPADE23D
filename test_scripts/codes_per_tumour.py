'''
Look at codes
Author: Virginia Fernandez (King's College London)
Date: March 2023
The script looks and plots mean codes for slices with and without tumours.
'''

import data
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from models_spade.pix2pix_model import Pix2PixModel
from data.spadenai_v2 import SpadeNai
from data.spadenai_v2_sliced import SpadeNaiSlice
from options.lookatcodes_options import LACodeOptions
import moreutils as uvir
import data.dataset_utils as dutils
from sklearn.manifold import TSNE, LocallyLinearEmbedding
from support_tests.hoyermetric import HoyerMetric
from sklearn.decomposition import PCA as PCA
import moviepy.video.io.ImageSequenceClip
from monai.data import DataLoader
from options.test_options import FullPipelineOptions
from copy import deepcopy
# Input
tumour_slices_labels = "/media/vf19/BigCrumb/JOURNAL_22/SOURCE_DATASETS/CODE_TUMOURS/TUMOUR_DATA/labels_train" # Path to images with tumours
tumour_slices_images = "/media/vf19/BigCrumb/JOURNAL_22/SOURCE_DATASETS/CODE_TUMOURS/TUMOUR_DATA/images_train"
non_tumour_slices_labels = "/media/vf19/BigCrumb/JOURNAL_22/SOURCE_DATASETS/CODE_TUMOURS/NON_TUMOUR_DATA/labels_train"
non_tumour_slices_images = "/media/vf19/BigCrumb/JOURNAL_22/SOURCE_DATASETS/CODE_TUMOURS/NON_TUMOUR_DATA/images_train" # Path to images without tumour
save_to = "/media/vf19/BigCrumb/JOURNAL_22/SOURCE_DATASETS/CODE_TUMOURS/tumour_codes.png"
perplexity_tsne = 25
lr_tsne = 400
path_to_model = "/media/vf19/BigCrumb/JOURNAL_22/GENERATIVE_MODELS/ABLATION/BRAINSPADEV3_25_nomodisc_diffs_selfs"
mods = ['T1', 'FLAIR', 'T2']

opt_tum = FullPipelineOptions().load_options(os.path.join(path_to_model, 'opt'))
opt_tum.isTrain = False
opt_tum.label_dir = tumour_slices_labels
opt_tum.image_dir = tumour_slices_images
# Initialise model and dataset
model = Pix2PixModel(opt_tum)

# Duplicate opt
opt_ntum = deepcopy(opt_tum)
opt_ntum.label_dir = non_tumour_slices_labels
opt_ntum.image_dir = non_tumour_slices_images

# Other variables
# Colors
colors = ['green', 'blue', 'pink', 'orange']
shapes = ["^", ".", "*", "D", "s", "+"]
colors_sub = {'green': ['darkgreen', 'lime', 'olive', 'darkolivegreen', 'lawngreen'],
              'blue': ['royalblue', 'dodgerblue', 'navy', 'turquoise', 'powderblue'],
              'pink': ['hotpink', 'magenta', 'mediumvioletred', 'lightpink', 'violet'],
              'orange': ['darksalmon', 'coral', 'orange', 'peru', 'orange']}

codes = []
tumours = []
modalities = []
dataset_container_tum = SpadeNaiSlice(opt_tum, mode='test')
dataset_container_ntum = SpadeNaiSlice(opt_ntum, mode='test')
for mod in mods:
    opt_tum.fix_seq = mod
    opt_ntum.fix_seq = mod
    dataset_container_tum.resetDatasets(fixed_modality=mod)
    dataset_container_ntum.resetDatasets(fixed_modality=mod)
    dataloader_tum =  DataLoader(dataset_container_tum.sliceDataset, batch_size=8, shuffle=False,
                                 num_workers=3, drop_last=False)
    dataloader_ntum = DataLoader(dataset_container_ntum.sliceDataset, batch_size=8, shuffle=False,
                               num_workers=3, drop_last=False)
    for i, data_i in enumerate(dataloader_tum):
        model.eval()
        with torch.no_grad():
            _, generated = model(data_i, 'encode_only')  # We use the "this_seq" decoder.
            generated = generated.detach().cpu()
        codes.append(generated.numpy())
        modalities.append(data_i['this_seq'])
        tumours+=[1]*generated.shape[0]
    for i, data_i in enumerate(dataloader_ntum):
        model.eval()
        with torch.no_grad():
            _, generated = model(data_i, 'encode_only')  # We use the "this_seq" decoder.
            generated = generated.detach().cpu()
        codes.append(generated.numpy())
        modalities.append(data_i['this_seq'])
        tumours+=[0]*generated.shape[0]

codes = np.concatenate(codes, 0)
modalities = np.concatenate(modalities, 0)
tumours = np.asarray(tumours)
Y = TSNE(n_components=2, init='pca', random_state=0, perplexity=perplexity_tsne,
         learning_rate=lr_tsne).fit(codes).fit_transform(codes)

 # Plot
plt.figure(figsize=(10, 7))
legend_ = []
for mod_ind, mod in enumerate(mods):
    if not Y[(modalities == mod) * (tumours == 1), 0].shape[0] == 0:
        plt.scatter(Y[(modalities == mod) * (tumours == 1), 0],
                    Y[(modalities == mod) * (tumours == 1), 1],
                    cmap=plt.cm.Spectral,
                    color=colors_sub[colors[mod_ind]][0],
                    marker='+')
        legend_.append("%s-tum" % (mod))

    if not Y[(modalities == mod) * (tumours == 0), 0].shape[0] == 0:
        plt.scatter(Y[(modalities == mod) * (tumours == 0), 0],
                    Y[(modalities == mod) * (tumours == 0), 1],
                    cmap=plt.cm.Spectral,
                    color=colors_sub[colors[mod_ind]][1],
                    marker='o')
        legend_.append("%s-no-tum" % (mod))

plt.legend(legend_)
plt.savefig(save_to)
dataset_container_ntum.clearCache()
dataset_container_tum.clearCache()