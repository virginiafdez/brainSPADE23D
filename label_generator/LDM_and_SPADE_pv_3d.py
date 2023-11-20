'''
LABEL AND IMAGE GENERATOR WITH PARTIAL VOLUME MAPS LABELS
Partial Volume Maps:
TYPICALLY (according to GIF nomenclature)
Channel 0: Background
Channel 1: CSF
Channel 2: Grey Matter
Channel 3: White Matter
Channel 4: Deep Grey Matter
Channel 5: Brainstem
Channel 6: WMH
Channel 7: Tumour
Channel 8: Edema
Subsequent: empty
'''

import matplotlib.pyplot as plt
import torch
import monai
import os
import nibabel as nib
import numpy as np
from options.test_options import FullPipelineOptions
from data.spadenai_v2 import SpadeNai
from models_spade.pix2pix_model import Pix2PixModel
import moreutils as uvir
from datetime import datetime
from conditioned_ldm.src.python.training_and_testing.models.ddim import DDIMSampler3D
from copy import deepcopy
from label_generator.conditioned_ldm_import_functions import loadmodel
from conditioned_ldm.src.python.training_and_testing.util import cond2conKey
from label_generator.utils_lbg import pad_latent
import random

# CONTROL VARIABLES
channels = {'generic': 6, 'wmh': 6, 'tumour': 7, 'edema': 8, 'gdtumour': 9}
lesion_tokens = {1: 'wmh', 2: 'tumour', 3: 'edema', 4: 'gdtumour'}
n_healthy_labels = 6

# PARAMETERS
############################################### INPUT SETTINGS
stage_1 = False # Whether to perform label generation
stage_2 = True # Whether to perform style image generation

# For stage 1
vae_model_path = "/media/vf19/BigCrumb/JOURNAL_22/GENERATIVE_MODELS/VAE_3D_FL_NO-GAN_bigres_144-176-112_z8_16CH_noKLD/best_model.pth" # Path to the model .pth file
vae_config_file = "/media/vf19/BigCrumb/JOURNAL_22/GENERATIVE_MODELS/VAE_3D_FL_NO-GAN_bigres_144-176-112_z8_16CH_noKLD/train_aekl_3D_fl_3ds.yaml" #  YAML file for the VAE
ldm_model_path = "/media/vf19/BigCrumb/JOURNAL_22/GENERATIVE_MODELS/LDM_3D_3DS_FLOAT-LES-SF3_64/final_model.pth" # Path to the model .pth file
ldm_config_file = "/media/vf19/BigCrumb/JOURNAL_22/GENERATIVE_MODELS/LDM_3D_3DS_FLOAT-LES-SF3_64/train_ldm_3D_3ds.yaml" # YAML file for the LDM

result_save_dir = "/media/vf19/BigCrumb/JOURNAL_22/GENERATED_DATASETS/TUMOUR_AND_WMH_3D"
n_samples = 100
batch_size = 3
n_downsamplings = 3
type_stage_1 = "VAE" # VQ-VAE or VAE
float_lesions = True # Whereas lesions are in float format, or ints
use_lesion_cond = True # USE LESION CONDITIONING
plot_raw_labels = True
SOFTMAX = True # Whether the results need to be softmaxed or not.
num_timesteps = 80
scaling_factor = 3
cond_map = {0: 'wmh', 1: 'tumour', 2: 'edema', 3: 'gdtumour'}
LESION_QUARTILES = {'wmh': (0.02, 0.75), # (0.0117, 0.1016),
                    'tumour': (0.05, 0.3), #(0.0242, 0.18),
                    'edema': (0.1492, 0.3509),
                    'gdtumour': (0.2, 0.7), #(0.0823, 0.4094)
                    }
i_want_these_lesions = ['wmh', 'tumour','edema','gdtumour']

slice_offset = 2 # Number of generated slices left out

if not os.path.isdir(result_save_dir):
    os.makedirs(result_save_dir)

# For stage 2

brainspade_checkpoint = "/media/vf19/BigCrumb/JOURNAL_22/GENERATIVE_MODELS/ABLATION"
# Different label directory than the one in this folder.
override_label_dir = None
override_label_subdir = "labels"
brainspade_name = "BRAINSPADEV3_25_nomodisc_diffs_selfs"
target_datasets = ["ABIDE"]
modalities = ['T1']
path_to_styles = "/media/vf19/BigCrumb/JOURNAL_22/SOURCE_DATASETS/STYLES/ABIDE"
appendix_styles = "styles" # If the folders are named other than "style"
path_to_styles_labels = "/media/vf19/BigCrumb/JOURNAL_22/SOURCE_DATASETS/STYLES/ABIDE/styles_mask"
spade_channels = {'healthy': 6, 'total': 12, 'lesions': 4, 'empty': 2}
n_passes = 1 # Number of passes through the same dataset of labels.

# INITIAL SET UP -------------------------------------------------------------------------------------------------
# Create directories
if not os.path.isdir(result_save_dir):
    os.makedirs(result_save_dir)
if not os.path.isdir(os.path.join(result_save_dir,"labels")):
    os.makedirs(os.path.join(result_save_dir,"labels"))
    with open(os.path.join(result_save_dir, 'labels_rendered.txt'), 'w') as f:
        f.write("NOLABS")
        f.close()
    rendered_labels = False
else:
    r = ""
    with open(os.path.join(result_save_dir, 'labels_rendered.txt'), 'r') as f:
        r = f.readlines()[0]
        f.close()
    if r.strip("\n") == "NOLABS":
        rendered_labels = False
    else:
        rendered_labels = True
if not os.path.isdir(os.path.join(result_save_dir,"labels")):
    os.makedirs(os.path.join(result_save_dir,"labels"))
if not os.path.isdir(os.path.join(result_save_dir,"examples")):
    os.makedirs(os.path.join(result_save_dir,"examples"))

################# Initialise and load models_ ####################################
opt = FullPipelineOptions().load_options(os.path.join(brainspade_checkpoint, brainspade_name, 'opt'),
                                         dataset_type='volume',)
opt.label_dir = os.path.join(result_save_dir, 'labels')
datasets_original = deepcopy(opt.datasets)
opt.datasets += target_datasets
device = torch.device("cuda")
brainspade = Pix2PixModel(opt).eval()

#####################################################################################################
#######################################STAGE 1#######################################################
#####################################################################################################
# We need to create the models_ and load their state dict
device = torch.device("cuda")
vae = loadmodel(vae_model_path, vae_config_file, "VAE3D")
vae = vae.eval().to(device)
if use_lesion_cond:
    ldm = loadmodel(ldm_model_path, ldm_config_file, "LDM3D_CONDITIONED")
else:
    ldm = loadmodel(ldm_model_path, ldm_config_file, "LDM3D_UNCONDITIONED")
ldm = ldm.eval().to(device)
ldm.eval().to(device)
ddim_sampler = DDIMSampler3D(ldm)

ldm.log_every_t = 5
plot_every = 1
if plot_raw_labels:
    if not os.path.isdir(os.path.join(result_save_dir, 'examples_labels_raw')):
        os.makedirs(os.path.join(result_save_dir, 'examples_labels_raw'))

# Label sampling process ----------------------------------------------------------------------------------
if stage_1 and not rendered_labels:
    processed = 0
    if os.path.isdir(os.path.join(result_save_dir, 'labels')):
        processed += len(os.listdir(os.path.join(result_save_dir, 'labels')))
    if use_lesion_cond:
        counter_condlist = 0

    # Find out which sample shape
    sample_shape_vae = [batch_size, vae.decoder.z_channels, ] + [
        int(j / (2 ** (len(vae.decoder.ch_mult) - 1))) for j in list(vae.encoder.resolution)]
    _, sample_shape_ldm = pad_latent(list(sample_shape_vae[1:]), n_downsamplings)

    if sample_shape_vae[1:] != sample_shape_ldm:
        tf_crop = monai.transforms.CenterSpatialCrop(sample_shape_vae[1:])
    else:
        tf_crop = None

    while processed < n_samples and not rendered_labels:
        # Sample label and decode
        if n_samples - processed - batch_size < 0:
            b_size = n_samples - processed # Smaller batch size because you've reached the maximum number of samples.
        else:
            b_size = batch_size

        # Conditioning flag handling
        if use_lesion_cond:
            cond_list = []
            for key_cond, cond_name in cond_map.items():
                if not float_lesions:
                    cond_list.append(np.random.choice([0, 1], batch_size, replace=True))  # Categorical 0 1
                else:
                    if random.uniform(0, 1) > 0.0 and cond_name in i_want_these_lesions:
                        if cond_name in LESION_QUARTILES.keys():
                            cond_list.append(np.round(np.random.uniform(LESION_QUARTILES[cond_name][0],
                                                               LESION_QUARTILES[cond_name][1],
                                                               batch_size), 4))
                        else:
                            cond_list.append(np.round(np.random.uniform(0, 1, batch_size), 4))
                    else:
                        cond_list.append([0.0]*batch_size)

            cond_list = np.stack(cond_list, -1).squeeze()
            if batch_size == 1:
                cond_list = np.expand_dims(cond_list, 0)
            cond = torch.FloatTensor(cond_list)  # Convert the list to tensors.
            with torch.no_grad():
                cond = cond2conKey(ldm.conditioning_key, cond, sample_shape_ldm[2:]).to(device)

            latent_vectors, _ = ddim_sampler.sample(num_timesteps,conditioning=cond[:b_size,...], batch_size=b_size,
                                                shape=sample_shape_ldm, eta=1.0)

        else:
            latent_vectors, intermediates = ddim_sampler.sample(num_timesteps,
                                                                batch_size=batch_size,
                                                                shape=sample_shape_ldm,
                                                                eta=1.0)
        # Reconstruct
        if scaling_factor > 0:
            latent_vectors = latent_vectors * scaling_factor
        with torch.no_grad():
            if tf_crop is not None:
                latent_vectors = tf_crop(latent_vectors)
            x_hat = vae.reconstruct_ldm_outputs(latent_vectors)
            if SOFTMAX:
                x_hat = torch.softmax(x_hat, 1)

        # Save label
        number_subjects = 100000 # For subject naming

        for b in range(x_hat.shape[0]):

            out_label_tmp = x_hat[b, ...].detach().cpu()
            cond_list_tmp =  cond_list[b, ...]

            out_label_tmp = torch.clamp(out_label_tmp, 0, 1.0) # We clip between 0 and 1

            # This section processes labels to make sure that they are compliant with the number of lesions,
            # and the lesions that are requested.

            if len(i_want_these_lesions) == 0: # No lesions necessary. Concatenate zeros on lesion channels.
                additional_channels = torch.zeros([spade_channels['total'] - n_healthy_labels ]
                                                  + list(out_label_tmp.shape[1:]))  # If number of channels lesion > 2 changes in SPADE, modify this
                out_label = torch.cat([out_label_tmp[:n_healthy_labels,...], additional_channels], 0)
                lesion_type = "nolesion"

            elif len(i_want_these_lesions) > 0:
                # We keep the channels belonging to lesions that are conditioned on, leave the rest to zero.
                reverse_channels = {v: k for k, v in channels.items()}
                lesion_type = ""
                for ch in range(out_label_tmp.shape[0]):
                    if ch in reverse_channels.keys() and reverse_channels[ch] not in i_want_these_lesions:
                        out_label_tmp[ch, ...] = torch.zeros(list(out_label_tmp.shape[1:]))
                    elif ch in reverse_channels.keys() and cond_list_tmp[i_want_these_lesions.index(reverse_channels[ch])] > 0.0:
                        lesion_type += "%s-" %reverse_channels[ch]
                if lesion_type == "":
                    lesion_type = "nolesion"
                elif lesion_type[-1] == "-":
                    lesion_type = lesion_type[:-1]

            if out_label_tmp.shape[0]<spade_channels['total']:
                out_label_tmp = torch.cat([out_label_tmp, torch.zeros([spade_channels['empty']]+
                                                                      list(out_label_tmp.shape[1:]))],0)

            # Argmax to plot it
            out_label = deepcopy(out_label_tmp).numpy()
            out_label_argmax = np.argmax(out_label, 0)

            # Plot labels.
            cond = cond.squeeze()
            if plot_raw_labels and processed%plot_every == 0:
                if use_lesion_cond:
                    title_ = lesion_type

                ni_img = nib.Nifti1Image(out_label_argmax.astype('float'), affine = np.eye(4))
                nib.save(ni_img, os.path.join(result_save_dir, 'examples_labels_raw', "sample_%d.nii.gz" %processed))

            out_label = np.transpose(out_label, [1, 2, 3, 0]) # From CxHxWx1 > HxWx1xC

            # Save img.
            subject_id = str(processed+1000)
            append = "0"*(len(str(number_subjects))-len(str(processed)))
            lesions_in_label = []
            for les_name, les_channel in channels.items():
                if les_name == 'generic':
                    continue
                if (out_label_argmax == les_channel).sum() > 1:
                    lesions_in_label.append(les_name)
            if len(lesions_in_label) == 0:
                lesions_in_label = "nolesion"
            else:
                lesions_in_label = "-".join(lesions_in_label)

            out_name = "Parcellation_SYNTHETIC_sub-%s_ses-v0_%s.nii.gz" %(append+subject_id, lesions_in_label)

            out_ni = nib.Nifti1Image(out_label.astype('float'), affine = np.eye(4))
            nib.save(out_ni, os.path.join(result_save_dir, 'labels', out_name))

            processed += 1

    with open(os.path.join(result_save_dir, 'labels_rendered.txt'), 'w') as f:
        f.write("LABS")
        f.close()

vae.cpu()
ldm.cpu()

# Brainspade call ----------------------------------------------------------------------------------
if stage_2:
    disp_every = 20
    brainspade.to(device)
    log_file = os.path.join(result_save_dir, "run_logs.txt")
    with open(log_file, 'w') as f:
        f.write(str(datetime.today))

    # Set working volume
    colors = uvir.get_rgb_colours()

    # We modify the settings for the dataset created for each modality
    opt.image_dir = os.path.join(path_to_styles, "%s_%s" %(appendix_styles, modalities[0])) # Dummy (we will not use this)
    opt.nThreads = 0
    if not stage_1 and override_label_dir is not None:
        if override_label_subdir is None:
            opt.label_dir = os.path.join(override_label_dir, 'labels')
        else:
            opt.label_dir = os.path.join(override_label_dir, override_label_subdir)
            tag_imgs = override_label_subdir.strip("labels_")
    else:
        if override_label_subdir is None:
            opt.label_dir = os.path.join(result_save_dir, 'labels')
        else:
            opt.label_dir = os.path.join(result_save_dir, override_label_subdir)
            tag_imgs = override_label_subdir.strip("labels_")
    opt.non_corresponding_dirs = True
    opt.style_label_dir = os.path.join(path_to_styles_labels)
    opt.cache_dir = os.path.join(result_save_dir, 'spadenai_cache_dir')
    dataset_container = SpadeNai(opt, mode="test")

    style_paths = {}
    for mod in modalities:
        style_paths[mod] = os.path.join(path_to_styles, 'styles_%s' %mod)
        # Set up place where we'll save images.
        if not os.path.isdir(os.path.join(result_save_dir, 'images%s_%s' % (tag_imgs,mod))):
            os.makedirs(os.path.join(result_save_dir, 'images%s_%s' % (tag_imgs,mod)))

    for lab in os.listdir(opt.label_dir):
        dataset_container.setWorkingVolume_noIm(label_path = os.path.join(opt.label_dir, lab),
                                                style_path = style_paths,
                                                style_path_label = path_to_styles_labels,
                                                style = target_datasets,
                                                format_data='nii.gz',
                                                fix_style = True)

        # Loop: number of instances per label
        for mod in modalities:
            for p in range(n_passes):
                stop_flag = False
                while not stop_flag:
                    output_dict, stop_flag = dataset_container.iterateThroughWV_noIm(modality=mod,
                                                                                     batch_size=batch_size)
                    gen = brainspade(output_dict, 'inference')
                    for b in range(gen.shape[0]):
                        dataset_container.working_volume.storeSlice(gen[b, ...], mod)

                # File name
                file_name = lab.replace("Parcellation", mod).replace(".nii.gz", "_%d.nii.gz" % p)
                dataset_container.working_volume.storeReconstruction(os.path.join(result_save_dir,
                                                                                  'images%s_%s' %(tag_imgs, mod),
                                                                                  file_name), mod)

                with open(log_file, 'a') as f:
                    f.write("%s: LABEL %s IMAGE %s\n" %(file_name.split(".")[0],
                                                        output_dict['label_path'][0],
                                                        output_dict['image_path'][0]))
            print("Fin de passe %d" %p)

        with open(os.path.join(result_save_dir, 'log.txt'), 'w') as f:
            f.write("%s\n" %str(datetime.today()))
            f.write("VQ-VAE model: %s\n" %vae_model_path)
            f.write("LDM model: %s\n" %ldm_model_path)
            f.write("Lesion conditioning:%s\n" %use_lesion_cond)
            f.write("Brainspade model: %s\n" %os.path.join(brainspade_checkpoint, brainspade_name))
            f.write("Number of passes per modality: %d\n" %n_passes)
            f.close()