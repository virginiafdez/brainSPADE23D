import sys
sys.path.append("/nfs/home/vfernandez/models/brainSPADE_fi/brainSPADE_f/")
sys.path.append("/nfs/home/vfernandez/models/brainSPADE_fi/brainSPADE_f/conditioned_ldm/src/python/training_and_testing/")
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
from tqdm import tqdm
import os
import random
import numpy as np
from options.test_options import FullPipelineOptions
from data.spadenai_v2_sliced import SpadeNaiSlice
from models_spade.pix2pix_model import Pix2PixModel
import moreutils as uvir
from omegaconf import OmegaConf
from monai.data.dataloader import DataLoader
from datetime import datetime
from conditioned_ldm.src.python.training_and_testing.models.ddim import DDIMSampler
from utils import util
from models_spade.modality_discrimination.modisc_v2 import Modisc
from data.sampling_ds_utils import mod_disc_pass_Accuracy
from copy import deepcopy
from test_scripts.conditioning_lists import nolesion_conditioning_list, \
    even_conditioning_list_noslice
from label_generator.conditioned_ldm_import_functions import loadmodel
from conditioned_ldm.src.python.training_and_testing.util import cond2conKey
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_model_path", type=str, help = "Path to VAE")
    parser.add_argument("--vae_config_file", type=str, help="Path to config file of VAE")
    parser.add_argument("--ldm_model_path", type=str, help="Path to LDM")
    parser.add_argument("--ldm_config_file", type=str, help= "Path to config file of LDM")
    parser.add_argument("--n_samples", type=int, help="Number of samples")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--guidance_scale", type=int, help="Guidance scale used for LDM")
    parser.add_argument("--stage_1", action='store_true', help="Generate labels")
    parser.add_argument("--stage_2", action='store_true', help="Generate images")
    parser.add_argument("--stage_purge", action='store_true', help="Delete corrupt labels")
    parser.add_argument("--use_slice_cond", action='store_true', help="Use slice conditioning")
    parser.add_argument("--use_lesion_cond", action='store_true', help="Use lesion conditioning")
    parser.add_argument("--n_timesteps_ddim", type=int, help="Number of steps of DDIM")
    parser.add_argument("--checkpoint_spade", type=str, help="Checkpoint of SPADE net")
    parser.add_argument("--name_spade", type=str, help="Name of SPADE model")
    parser.add_argument("--path_to_styles", type=str, help="Path to SPADE styles")
    parser.add_argument("--n_passes", type=int, help="number of passes through SPADE for each generated label")
    parser.add_argument("--results_save", type=str, help="Saving directory")
    parser.add_argument("--i_want_lesions", type=str, nargs="*", help="List of lesions you want in the labels,"
                                                                      "in addition to slice if slice is in as well."
                                                                      "Don't put slice here", default=[])
    parser.add_argument("--modalities", type=str, nargs="*", help="Modalities you want to generate", default=[]),
    parser.add_argument("--datasets", type=str, nargs="*", help="list of style datasets")
    parser.add_argument("--override_label_dir", type=str, help="If you want to use the labels from another directory")
    args = parser.parse_args()
    return args

def main(args):
    # CONTROL VARIABLES
    channels = {'generic': 6, 'wmh': 6, 'tumour': 7, 'edema': 8, 'gdtumour': 9}
    lesion_tokens = {1: 'wmh', 2: 'tumour', 3: 'edema', 4: 'gdtumour'}
    n_healthy_labels = 6
    # PARAMETERS
    ############################################### INPUT SETTINGS
    stage_1 = args.stage_1 # Whether to perform label generation
    stage_2 = args.stage_2 # Whether to perform style image generation
    stage_purge = args.stage_purge

    # For stage 1
    vae_model_path = args.vae_model_path
    vae_config_file = args.vae_config_file
    ldm_model_path = args.ldm_model_path
    ldm_config_file = args.ldm_config_file

    result_save_dir = args.results_save
    n_samples = args.n_samples
    batch_size = args.batch_size
    guidance_scale= args.guidance_scale
    float_lesions = True # Whereas lesions are in float format, or ints
    use_slice_cond = args.use_slice_cond # USE SLICE CONDITIONING
    use_lesion_cond = args.use_lesion_cond # USE LESION CONDITIONING
    plot_raw_labels = True
    SOFTMAX = True # Whether the results need to be softmaxed or not.
    num_timesteps = args.n_timesteps_ddim
    correct = True
    cond_map = {0: 'wmh', 1: 'tumour', 2: 'edema', 3: 'gdtumour'}
    cond_map_inv = {v: k for k, v in cond_map.items()}
    #

    LESION_QUARTILES = {'wmh': (0.0149, 0.2), #(0.8, 0.9),
                        'tumour': (0.8, 0.9),
                        'edema': (0.8, 0.9),
                        'gdtumour': (0.8, 0.9),}
    i_want_these_lesions = args.i_want_lesions

    if not os.path.isdir(result_save_dir):
        os.makedirs(result_save_dir)

    # For stage 2

    brainspade_checkpoint = args.checkpoint_spade
    # Different label directory than the one in this folder.
    override_label_dir = args.override_label_dir
    brainspade_name = args.name_spade
    target_datasets = args.datasets
    modalities = args.modalities
    path_to_styles = args.path_to_styles
    appendix_styles = "styles" # If the folders are named other than "style"
    path_to_styles_labels = os.path.join(path_to_styles, 'styles_mask')
    spade_channels = {'healthy': 6, 'total': 12, 'lesions': 4, 'empty': 2}
    n_passes = 1 # Number of passes through the same dataset of labels.
    format = 'npy' # Either "spade", "png" or "npy". Spade is a npz with keys: img, img_affine, img_header and modality.
    save_as_npy = False # Whether to, in addition to saving as NPZ (for SPADE), save as NPY

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
    opt = FullPipelineOptions().load_options(os.path.join(brainspade_checkpoint, brainspade_name, 'opt'))
    opt.checkpoints_dir = brainspade_checkpoint
    opt.name = brainspade_name
    opt.label_dir = os.path.join(result_save_dir, 'labels')
    datasets_original = deepcopy(opt.datasets)
    opt.datasets += target_datasets
    device = torch.device("cuda")
    brainspade = Pix2PixModel(opt).eval()

    # To make sure that generated images flal within the modalities
    use_modisc = True
    if use_modisc:
        modality_discriminator = Modisc(len(opt.sequences), len(datasets_original), 0.2, 2, 1)
        modality_discriminator = util.load_network(modality_discriminator, 'MD', -1, opt, strict = False).eval()
    max_attempts_per_img = 20 # Up to 200 instances per img.

    #####################################################################################################
    #######################################STAGE 1#######################################################
    #####################################################################################################
    # We need to create the models_ and load their state dict
    device = torch.device("cuda")
    vae = loadmodel(vae_model_path, vae_config_file)
    vae = vae.eval().to(device)
    if use_slice_cond or use_lesion_cond:
        ldm = loadmodel(ldm_model_path, ldm_config_file, "LDM_CONDITIONED")
    else:
        ldm = loadmodel(ldm_model_path, ldm_config_file, "LDM_UNCONDITIONED")
    ldm = ldm.eval().to(device)
    ldm.eval().to(device)
    ddim_sampler = DDIMSampler(ldm)

    ldm.log_every_t = 15
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
        while processed < n_samples and not rendered_labels:
            # Sample label and decode
            if n_samples - processed - batch_size < 0:
                b_size = n_samples - processed # Smaller batch size because you've reached the maximum number of samples.
            else:
                b_size = batch_size

            sample_shape = sample_shape = tuple([batch_size, vae.decoder.z_channels,] + [
                int(j/(2**(len(vae.decoder.ch_mult) -1))) for j in list(vae.encoder.resolution)]) # Sample shape
            # Conditioning flag handling
            cond_list = []
            if use_lesion_cond: # If we condition on lesions
                cond_list = []
                for key_cond, cond_name in cond_map.items():
                    if not float_lesions:
                        cond_list.append(np.random.choice([0, 1], b_size, replace=True))  # Categorical 0 1
                    else:
                        if random.uniform(0, 1) > 0.5 and cond_name in i_want_these_lesions:
                            if cond_name in LESION_QUARTILES.keys():
                                cond_list.append(np.round(np.random.uniform(LESION_QUARTILES[cond_name][0],
                                                                   LESION_QUARTILES[cond_name][1],
                                                                   b_size), 4))
                            else:
                                cond_list.append(np.round(np.random.uniform(0, 1, b_size), 4))
                        else:
                            cond_list.append([0.0]*b_size)
            if use_slice_cond: # If we condition on slice number
                slice_list = list(np.random.uniform(0.15859, 0.59375, size=b_size))  # Slice
                cond_list.append(slice_list)

            if use_slice_cond or use_lesion_cond: # Conditioned case
                cond_list = np.stack(cond_list, -1).squeeze()
                cond = torch.FloatTensor(cond_list)  # Convert the list to tensors.
                with torch.no_grad():
                    cond = cond2conKey(ldm.conditioning_key, cond, sample_shape[2:]).to(device)
                # Sample latent space
                latent_vectors, _ = ddim_sampler.sample(num_timesteps,conditioning=cond, batch_size=b_size,
                                                        shape=sample_shape[1:], eta=1.0,
                                                        guidance_scale = guidance_scale)
            else: # Unconditioned case
                latent_vectors, intermediates = ddim_sampler.sample(num_timesteps,
                                                                    batch_size=sample_shape[0],
                                                                    shape=sample_shape[1:],
                                                                    eta=1.0,
                                                                    )
            # Reconstruct
            with torch.no_grad():
                x_hat = vae.reconstruct_ldm_outputs(latent_vectors)
                if SOFTMAX:
                    x_hat = torch.softmax(x_hat, 1)

            # Save label
            number_subjects = 100000 # For subject naming

            for b in range(x_hat.shape[0]):

                out_label_tmp = x_hat[b, ...].detach().cpu()
                cond_list_tmp =  cond_list[b, ...]
                out_label_tmp = torch.clamp(out_label_tmp, 0, 1.0) # We clip between 0 and 1
                out_label_argmax = torch.argmax(out_label_tmp, 0)
                if len(i_want_these_lesions) == 0: # No lesions necessary. Concatenate zeros on lesion channels.
                    additional_channels = torch.zeros([spade_channels['total'] - n_healthy_labels ]
                                                      + list(out_label_tmp.shape[1:]))  # If number of channels lesion > 2 changes in SPADE, modify this
                    out_label = torch.cat([out_label_tmp[:n_healthy_labels,...], additional_channels], 0)
                    lesion_type = "nolesion"

                elif len(i_want_these_lesions) > 0:
                    # We keep the channels belonging to lesions that are conditioned on, leave the rest to zero.
                    lesion_type = ""
                    for les in i_want_these_lesions:
                        if channels[les] in out_label_argmax:
                            lesion_type += "%s-" %les
                    if lesion_type == "":
                        lesion_type = "nolesion"
                    if lesion_type[-1] == "-":
                        lesion_type = lesion_type[:-1]

                if out_label_tmp.shape[0]<spade_channels['total']:
                    out_label_tmp = torch.cat([out_label_tmp, torch.zeros([spade_channels['empty']]+
                                                                          list(out_label_tmp.shape[1:]))],0)

                out_label = deepcopy(out_label_tmp).numpy()
                out_label_argmax = np.argmax(out_label, 0)
                flag = False
                for les_wanted in i_want_these_lesions:
                    if channels[les_wanted] in out_label_argmax:
                        flag = True
                if flag:
                    print("Positive sampling.")

                # Plot labels.
                cond = cond.squeeze()
                if plot_raw_labels and processed%plot_every == 0:
                    f, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, out_label.shape[0]]},
                                               figsize=((out_label.shape[0]-spade_channels['empty']) * 6, 8))
                    a0p = a0.imshow(out_label_argmax); plt.axis('off')
                    out = [out_label[i, ...] for i in range(out_label.shape[0] - spade_channels['empty'])]
                    out = np.concatenate(out, -1)
                    a1p = a1.imshow(out, vmin=0.0, vmax=1.0)
                    f.colorbar(a1p, ax=a1, fraction=0.006); plt.axis('off')
                    title_ = ""
                    plt.tight_layout()
                    if use_lesion_cond:
                        title_ = lesion_type
                    if use_slice_cond:
                        title_ += ", %d" %(int(cond_list_tmp[-1]*256))
                    plt.title(title_, fontsize = 18)
                    plt.savefig(os.path.join(result_save_dir, 'examples_labels_raw', "sample_%s_%d.png" %(lesion_type,processed)))
                    plt.close(f)

                # One-hot encoded (not PV map label)
                out_label_ohe = np.zeros_like(out_label)
                for ch in range(out_label.shape[0]):
                    out_label_ohe[ch, ...] = out_label_argmax == ch

                out_label_npy_pv = deepcopy(out_label)
                out_label = np.expand_dims(out_label, -1) # From CxHxW > CxHxWx1
                out_label = np.transpose(out_label, [1, 2, 3, 0]) # From CxHxWx1 > HxWx1xC

                # Save img.
                subject_id = str(processed+1000)
                append = "0"*(len(str(number_subjects))-len(str(processed)))
                if use_slice_cond:
                    slice_nbr = str(int(cond[b][-1]*256))
                else:
                    slice_nbr = '0'
                out_name = "Parcellation_SYNTHETIC_sub-%s_ses-v0_%s.npz" %(append+subject_id, slice_nbr)
                if not os.path.isdir(os.path.join(result_save_dir, 'labels_NPY')):
                    os.makedirs(os.path.join(result_save_dir, 'labels_NPY'))
                if not os.path.isdir(os.path.join(result_save_dir, 'labels_pv_NPY')):
                    os.makedirs(os.path.join(result_save_dir, 'labels_pv_NPY'))
                # np.save(os.path.join(result_save_dir, 'labels_NPY', out_name.replace("npz", "npy")),
                #         out_label_ohe)
                if save_as_npy:
                    np.save(os.path.join(result_save_dir, 'labels_pv_NPY', out_name.replace("npz", "npy")),
                            out_label_npy_pv)
                np.savez(os.path.join(result_save_dir, 'labels', out_name),
                         label = out_label,
                         slice = 0,
                         dataset_name = 'SYNTHETIC')

                processed += 1

        with open(os.path.join(result_save_dir, 'labels_rendered.txt'), 'w') as f:
            f.write("LABS")
            f.close()
    vae.cpu()
    ldm.cpu()

    if stage_purge:
        print("Purging...")
        label_dir = os.path.join(result_save_dir, 'labels')
        for l in tqdm(os.listdir(label_dir)):
            try:
                _ = np.load(os.path.join(label_dir, l))['label']
            except:
                print("Deleting... %s" % l)
                os.remove(os.path.join(label_dir, l))

    # Brainspade call ----------------------------------------------------------------------------------
    if stage_2:
        disp_every = 20
        brainspade.to(device)
        if use_modisc:
            modality_discriminator.to(device)
        log_file = os.path.join(result_save_dir, "run_logs.txt")
        with open(log_file, 'w') as f:
            f.write(str(datetime.today))
        for mod in modalities:
            print("Processing modality %s" %mod)
            # We modify the settings for the dataset created for each modality
            opt.image_dir =  os.path.join(path_to_styles, "%s_%s" %(appendix_styles,mod))
            opt.fix_seq = mod
            opt.nThreads = 0
            if not stage_1 and override_label_dir is not None:
                opt.label_dir = os.path.join(override_label_dir, 'labels')
            else:
                opt.label_dir = os.path.join(result_save_dir, 'labels')
            opt.fix_seq = mod
            opt.non_corresponding_dirs = True
            opt.second_style_slice = False
            opt.batchSize = batch_size
            opt.style_label_dir = os.path.join(path_to_styles_labels)
            colors = uvir.get_rgb_colours()

            # Set up place where we'll save images.
            result_im_dir = os.path.join(result_save_dir, 'images_%s' %mod)
            if not os.path.isdir(result_im_dir):
                os.makedirs(result_im_dir)

            # Create dataset
            opt.cache_dir = os.path.join(result_save_dir, 'spadenai_cache_dir')
            dataset_container = SpadeNaiSlice(opt, mode="test")
            dataloader = DataLoader(dataset_container.sliceDataset, batch_size=opt.batchSize, shuffle=False,
                                    num_workers=int(opt.nThreads), drop_last=opt.isTrain,
                                    )

            # Loop: number of instances per label
            for p in range(n_passes):
                dataset_container.resetDatasets(fixed_modality=mod)
                dataloader = DataLoader(dataset_container.sliceDataset, batch_size=opt.batchSize, shuffle=False,
                                        num_workers=int(opt.nThreads), drop_last=opt.isTrain,
                                        )

                # Enumerate all labels
                for ind, i in enumerate(dataloader):
                    print("Batch %d / %d" % (ind, len(dataset_container)))

                    gen = brainspade(i, 'inference') # Forward
                    if use_modisc:
                        mod_accuracy = mod_disc_pass_Accuracy(modality_discriminator, gen, i,
                                                              opt.sequences) # Modality accuracy

                        # We try until our modality discriminator doesn't fail.
                        n_attempts = 0
                        accuracy_th = mod_accuracy < 0.75 # Threshold
                        while True in accuracy_th and n_attempts < max_attempts_per_img:
                            # Switch style
                            i_tuned = dataset_container.switchStyles(i, flags=accuracy_th)
                            gen = brainspade(i_tuned, 'inference')
                            mod_accuracy = mod_disc_pass_Accuracy(modality_discriminator, gen, i_tuned,
                                                              opt.sequences)
                            accuracy_th = mod_accuracy < 0.75
                            n_attempts += 1
                        if n_attempts == max_attempts_per_img:
                            print("Number of maximum attempts reached for %d out of %d images in the batch"
                                  %(np.sum(accuracy_th), opt.batchSize))

                    # Save the resulting images
                    for b in range(gen.shape[0]):
                        file_name = i['label_path'][b].split("/")[-1].replace("Parcellation", i['this_seq'][b])
                        to_save_img = gen[b, ...].detach().cpu()
                        to_save_img = util.tensor2im(to_save_img) # Scale 0-255
                        if format == 'png':
                            uvir.saveSingleImage(to_save_img, None, os.path.join(result_save_dir,
                                                                                         result_im_dir, file_name.replace(".npz", "_%d.png" %p)),
                                                 skullstrip=False)
                        elif format == 'spade':
                            np.savez(os.path.join(result_save_dir, result_im_dir, file_name.replace(".npz", "_%d.npz" %p)),
                                     img = to_save_img.unsqueeze(0).numpy(), img_affine = i['affine'][b,...],
                                     modality = i['this_seq'][b])
                        elif format == 'npy':
                            np.save(os.path.join(result_save_dir, result_im_dir, file_name.replace(".npz", "_%d.npy" %p)),
                                    to_save_img[..., 1])
                        else:
                            ValueError("Unknown format %s. It can either be spade, npy or png." %format)

                        # Plot results

                        if p == 0 and (ind*opt.batchSize + b) % disp_every == 0:
                            f = plt.figure(figsize=(15, 6))
                            repeated_img = util.tensor2im(i['style_image'][b, ...].detach().cpu())
                            out = np.concatenate([repeated_img, to_save_img,
                                                  colors[torch.argmax(i['label'][b, ...].detach().cpu(), 0).numpy()]],
                                                  1)
                            plt.imshow(out)
                            plt.title("Generated label")
                            plt.savefig(os.path.join(result_save_dir, 'examples', file_name.replace(".npz", ".png")))
                            plt.close(f)

                        with open(log_file, 'a') as f:
                            f.write("%s: LABEL %s IMAGE %s\n" %(file_name.split(".")[0],
                                                                i['label_path'][b],
                                                                i['image_path'][b]))

                dataset_container.clearCache()

                print("Fin de passe %d" %p)
            dataset_container.clearCache()
            with open(os.path.join(result_save_dir, 'log.txt'), 'w') as f:
                f.write("%s\n" %str(datetime.today()))
                f.write("VQ-VAE model: %s\n" %vae_model_path)
                f.write("LDM model: %s\n" %ldm_model_path)
                f.write("Lesion conditioning:%s\n" %use_lesion_cond)
                f.write("Slice conditioning:%s\n" %use_slice_cond)
                f.write("Brainspade model: %s\n" %os.path.join(brainspade_checkpoint, brainspade_name))
                f.write("Number of passes per modality: %d\n" %n_passes)
                f.close()

args = parse_args()
main(args)