import sys
sys.path.append("/nfs/home/vfernandez/models/brainSPADE_fi/brainSPADE_f/")
sys.path.append("/nfs/home/vfernandez/models/brainSPADE_fi/brainSPADE_f/conditioned_ldm/src/python/training_and_testing/")
sys.path.append("/nfs/home/vfernandez/models/brainSPADE_fi/brainSPADE_f/GenerativeModels")
sys.path.append("/nfs/home/vfernandez/models/brainSPADE_fi/brainSPADE_f/GenerativeModels/generative")
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

import torch
from tqdm import tqdm
import os
import nibabel as nib
import numpy as np
from options.test_options import FullPipelineOptions
from data.spadenai_v2 import SpadeNai
from models_spade.pix2pix_model import Pix2PixModel
import moreutils as uvir
from datetime import datetime
from copy import deepcopy
from label_generator.monai_generative_import_functions import define_DDPM, define_VAE
from conditioned_ldm.src.python.training_and_testing.util import cond2conKey
from label_generator.utils_lbg import pad_latent
import random
from label_generator.sizeable_inferer import SizeableInferer
import argparse
from data.dataset_utils import findSimilarName

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_model_path", type=str, help="Path to VAE")
    parser.add_argument("--vae_config_file", type=str, help="Path to config file of VAE")
    parser.add_argument("--ldm_model_path", type=str, help="Path to LDM")
    parser.add_argument("--ldm_config_file", type=str, help="Path to config file of LDM")
    parser.add_argument("--n_samples", type=int, help="Number of samples")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--stage_1", action='store_true', help="Generate labels")
    parser.add_argument("--stage_2", action='store_true', help="Generate images")
    parser.add_argument("--stage_purge", action='store_true', help="Delete corrupt labels")
    parser.add_argument("--use_lesion_cond", action='store_true', help="Use lesion conditioning")
    parser.add_argument("--n_timesteps_ddim", type=int, help="Number of steps of DDIM")
    parser.add_argument("--checkpoint_spade", type=str, help="Checkpoint of SPADE net")
    parser.add_argument("--name_spade", type=str, help="Name of SPADE model")
    parser.add_argument("--path_to_styles", type=str, help="Path to SPADE styles")
    parser.add_argument("--scaling_factor", type=float, default=1.0, help="Scaling factor for LDM")
    parser.add_argument("--hard_constraint", type=float, default=0.0, help="If 1.0, if the diseases aren't present"
                                                                           "on the output label, the label isn't saved"
                                                                           "and doesn't count")
    parser.add_argument("--n_passes", type=int, help="number of passes through SPADE for each generated label")
    parser.add_argument("--results_save", type=str, help="Saving directory")
    parser.add_argument("--i_want_lesions", type=str, nargs="*", help="List of lesions you want in the labels,"
                                                                      "in addition to slice if slice is in as well."
                                                                      "Don't put slice here", default=[])
    parser.add_argument("--modalities", type=str, nargs="*", help="Modalities you want to generate", default=[]),
    parser.add_argument("--datasets", type=str, nargs="*", help="list of style datasets")
    parser.add_argument("--override_label_dir", type=str, help="If you want to override the labels from being taken"
                                                               "from results save > labels, pass a directory here.",
                        default=None)
    parser.add_argument("--override_label_subdir", type=str, help="folder where the labels are saved in override_label_dir",
                        default="labels")

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
    vae_model_path = args.vae_model_path #"/media/vf19/BigCrumb/JOURNAL_22/GENERATIVE_MODELS/AEKL3D_1MMISO/best_model.pth" # Path to the model .pth file
    vae_config_file = args.vae_config_file #"/media/vf19/BigCrumb/JOURNAL_22/GENERATIVE_MODELS/AEKL3D_1MMISO/config.yaml" #  YAML file for the VAE
    ldm_model_path = args.ldm_model_path #"/media/vf19/BigCrumb/JOURNAL_22/GENERATIVE_MODELS/LDM3D_1MMISO/final_model.pth" # Path to the model .pth file
    ldm_config_file = args.ldm_config_file #"/media/vf19/BigCrumb/JOURNAL_22/GENERATIVE_MODELS/LDM3D_1MMISO/config.yaml" # YAML file for the LDM

    result_save_dir = args.results_save
    n_samples = args.n_samples
    batch_size = args.batch_size

    float_lesions = True # Whereas lesions are in float format, or ints
    use_lesion_cond = args.use_lesion_cond # USE LESION CONDITIONING
    plot_raw_labels = True
    SOFTMAX = True # Whether the results need to be softmaxed or not.

    cond_map = {0: 'wmh', 1: 'tumour', 2: 'edema', 3: 'gdtumour'}
    LESION_QUARTILES = {'wmh': (0.1, 0.55),
                        'tumour': (0.0224, 0.1833),
                        'edema': (0.1408, 0.3595),
                        'gdtumour': (0.0712, 0.4280)}
    i_want_these_lesions = args.i_want_lesions

    if not os.path.isdir(result_save_dir):
        os.makedirs(result_save_dir)

    # For stage 2

    brainspade_checkpoint = args.checkpoint_spade
    # Different label directory than the one in this folder.
    override_label_dir = args.override_label_dir
    override_label_subdir = args.override_label_subdir
    brainspade_name = args.name_spade
    target_datasets = args.datasets
    modalities = args.modalities
    path_to_styles = args.path_to_styles
    appendix_styles = "styles" # If the folders are named other than "style"
    path_to_styles_labels = os.path.join(path_to_styles, 'styles_mask')
    spade_channels = {'healthy': 6, 'total': 12, 'lesions': 4, 'empty': 2}
    n_passes = args.n_passes # Number of passes through the same dataset of labels.

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
    # We need to replace the checkpoints directory and the name by the ones that are new
    opt.checkpoints_dir = brainspade_checkpoint
    opt.name = brainspade_name
    device = torch.device("cuda")
    brainspade = Pix2PixModel(opt).eval()

    #####################################################################################################
    #######################################STAGE 1#######################################################
    #####################################################################################################
    # We need to create the models_ and load their state dict
    device = torch.device("cuda")
    vae, image_shape = define_VAE(vae_config_file, vae_model_path)
    vae = vae.eval().to(device)
    ldm, scheduler, scale_factor = define_DDPM(ldm_config_file, ldm_model_path, scheduler_type="ddim",
                                               num_inference_steps=args.n_timesteps_ddim)
    ldm = ldm.eval().to(device)
    z_shape_vae = [vae.latent_channels] + [s // (2 ** (len(vae.decoder.num_channels) - 1)) for s in image_shape]
    need_adjust, z_shape_ldm = pad_latent(z_shape_vae, len(ldm.block_out_channels))
    z_shape_ldm = z_shape_ldm
    inferer = SizeableInferer(scheduler=scheduler, scale_factor=scale_factor,
                              latent_shape_vae=z_shape_vae, latent_shape_ldm=
                              z_shape_ldm)
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
                    cond = cond2conKey("crossattn", cond, z_shape_ldm).to(device)
            else:
                cond = None

            samples = inferer.sample(
                input_noise=torch.randn([batch_size] + z_shape_ldm).to(device),
                autoencoder_model=vae,
                diffusion_model=ldm,
                scheduler=scheduler,
                conditioning=cond,
                save_intermediates=False,
                device=device)

            if SOFTMAX:
                samples = torch.softmax(samples, 1)

            # Save label
            number_subjects = 100000 # For subject naming

            for b in range(samples.shape[0]):

                out_label_tmp = samples[b, ...].detach().cpu()
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

                # Name and lesions
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

                ommit = False
                if args.hard_constraint != 0:
                    if 'wmh' in i_want_these_lesions and 'wmh' not in lesions_in_label:
                        ommit = True
                    if len([i for i in i_want_these_lesions if i!="wmh"]) > 0 and lesions_in_label=="wmh":
                        ommit = True

                if not ommit:
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
                    out_name = "Parcellation_SYNTHETIC_sub-%s_ses-v0_%s.nii.gz" %(append+subject_id, lesions_in_label)

                    out_ni = nib.Nifti1Image(out_label.astype('float'), affine = np.eye(4))
                    nib.save(out_ni, os.path.join(result_save_dir, 'labels', out_name))

                    processed += 1

        with open(os.path.join(result_save_dir, 'labels_rendered.txt'), 'w') as f:
            f.write("LABS")
            f.close()

    vae.cpu()
    ldm.cpu()

    # Purge
    if stage_purge:
        print("Purge...")
        list_labels = os.listdir(os.path.join(result_save_dir, "labels"))
        if override_label_dir is not None:
            list_labels = os.listdir(os.path.join(override_label_dir, override_label_subdir))
        for label_path in tqdm(list_labels):
            m_im = []
            for m in modalities:
                if os.path.isdir(os.path.join(result_save_dir, "images_%s" %m)):
                    m_im_ = findSimilarName(label_path, os.path.join(result_save_dir, "images_%s" %m),
                                                               extension=".nii.gz", slice_index=None,
                                           slice_ori_index=None, return_all=True)
                    if m_im_ is not None:
                        m_im_ = [os.path.join(os.path.join(result_save_dir, "images_%s" %m), i) for i in m_im_]
                        m_im+=m_im_
                    else:
                        print("No image found for modality %s and label %s" %(m, label_path))
            delete_flag = False
            try:
                if override_label_dir is None:
                    label = np.asarray(nib.load(os.path.join(result_save_dir, "labels", label_path)).dataobj)
                else:
                    label = np.zeros([256,256])
                for im_path in m_im:
                    img = np.asarray(nib.load(im_path).dataobj)
                    if True in np.isnan(img) or True in np.isinf(img):
                        delete_flag = True

                if True in np.isnan(label) or True in np.isinf(label) or delete_flag:
                    print("Deleting... %s & co." % label_path)
                    if override_label_dir is None:
                        print("Deleting... %s & co." % label_path)
                        os.remove(os.path.join(result_save_dir, "labels", label_path))
                    else:
                        if True in np.isnan(label) or True in np.isinf(label):
                            print("Deleting... %s & co." % label_path)
                            os.remove(os.path.join(override_label_dir, override_label_subdir, label_path))
                    for im_path in m_im:
                        os.remove(im_path)
            except:
                if override_label_dir is None:
                    print("Deleting... %s & co." %label_path)
                    os.remove(os.path.join(result_save_dir, "labels", label_path))
                else:
                    try:
                        if True in np.isnan(label) or True in np.isinf(label):
                            print("Deleting... %s & co." % label_path)
                            os.remove(os.path.join(override_label_dir, override_label_subdir, label_path))
                    except:
                        print("Deleting... %s & co." % label_path)
                        os.remove(os.path.join(override_label_dir, override_label_subdir, label_path))
                for im_path in m_im:
                    os.remove(im_path)

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

args = parse_args()
main(args)