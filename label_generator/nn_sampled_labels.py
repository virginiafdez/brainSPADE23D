import matplotlib.pyplot as plt
import mlflow.pytorch
import torch
from util import get_test_data_loader
import os
import monai
import nibabel as nib
import pandas as pd
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import numpy as np
from monai.metrics import DiceMetric
from tqdm import tqdm
# Load model -----------------------------------------------------------------------------------------------------------
# Add here the path to your local trained final models_
save_dir = "/media/vf19/BigCrumb/JOURNAL_22/PRIVACY_ANALYSIS/test_nn_2D"
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

synt_labels = "/media/vf19/BigCrumb/JOURNAL_22/GENERATED_DATASETS/SABRE_HEALTHY_P3D_1000/train/labels"
real_labels = "/media/vf19/BigCrumb/BRAINSPADE3D/TSVS/1MMISOTROPIC_BRAINSPADE3D/sabre_adni_brats_labelgen_local_train.tsv"
max = 5
max_nns = 5
is_3d = True

# Load training dataset
if ".tsv" in real_labels:
    files = list(pd.read_csv(real_labels, sep = "\t", index_col=False)['label'])
    real_labels = [{'image': i} for i in files]
else:
    real_labels = [{'image': os.path.join(real_labels, i)} for i in os.listdir(real_labels)]
synt_labels = [{'image': os.path.join(synt_labels, i)} for i in os.listdir(synt_labels)[:max]]
np.random.shuffle(synt_labels)
if is_3d:
    transforms_syn = monai.transforms.Compose([
        monai.transforms.LoadImaged(keys = ['image']),
        monai.transforms.EnsureChannelFirstd(keys=['image'], channel_dim=-1),
        monai.transforms.SpatialPadd(keys=['image'], spatial_size=[197, 233, 189])
    ])
    transforms_real = monai.transforms.Compose([
        monai.transforms.LoadImaged(keys='image'),
        monai.transforms.EnsureChannelFirstd(keys='image', channel_dim=-1)
    ])
else:
    transforms_syn = monai.transforms.Compose([
        monai.transforms.LoadImaged(keys = ['image'], npz_keys = ['label']),
        monai.transforms.SqueezeDimd(keys=['image'], dim=2),
        monai.transforms.EnsureChannelFirstd(keys=['image'], channel_dim=-1),
        monai.transforms.SpatialPadd(keys=['image'], spatial_size=[256, 256])
    ])
    transforms_real = monai.transforms.Compose([
        monai.transforms.LoadImaged(keys = 'image'),
    ])

dataset_real = monai.data.Dataset(real_labels, transforms_real)
dataset_synt = monai.data.Dataset(synt_labels, transforms_syn)
dataloader_real = monai.data.DataLoader(dataset_real, batch_size=8, num_workers=4)
dataloader_synt = monai.data.DataLoader(dataset_synt, batch_size=1)
metric = DiceMetric(include_background=True)

for batch in dataloader_synt:
    img = batch['image']
    img = img[:,:10, ...]
    dices = []
    filenames = []
    for batch_r in tqdm(dataloader_real):
        img_r = batch_r['image']
        try:
            dices += list(torch.nanmean(metric(img_r, torch.cat([img] * img_r.shape[0], 0)), -1).numpy())
        except:
            print("S")
        filenames += batch_r['image_meta_dict']['filename_or_obj']
    best_shots = np.sort(np.argpartition(dices, len(dices) - max_nns)[-max_nns:])
    to_plot_images = [np.argmax(img[0,...], 0)]
    for bs in best_shots:
        if is_3d:
            image = np.asarray(nib.load(filenames[bs]).dataobj)
            image = np.argmax(image, -1)
        else:
            image = np.load(filenames[bs])
            image = np.argmax(image, 0)
        to_plot_images.append(image)

    if is_3d:
        to_plot_images = np.concatenate(to_plot_images, 0)
        to_plot_images[0:10, 0:10, :] = 10
        img = nib.Nifti1Image(to_plot_images.astype("uint8"), affine = np.eye(4))
        nib.save(img, os.path.join(save_dir, batch['image_meta_dict']['filename_or_obj'][0].split("/")[-1]))
    else:
        to_plot_images = np.concatenate(to_plot_images, 1)
        f = plt.figure(figsize=(15, 2))
        plt.imshow(to_plot_images, cmap = 'jet')
        plt.axis('off')
        plt.savefig(os.path.join(save_dir,
                                 "%s.png" %batch['image_meta_dict']['filename_or_obj'][0].split("/")[-1].strip(".npz")))
        plt.close(f)



