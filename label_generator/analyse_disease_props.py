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
import os
import numpy as np
from conditioned_ldm.src.python.training_and_testing.models.ddim import DDIMSampler

from label_generator.conditioned_ldm_import_functions import loadmodel
from conditioned_ldm.src.python.training_and_testing.util import cond2conKey
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_model_path", type=str, help = "Path to VAE")
    parser.add_argument("--vae_config_file", type=str, help="Path to config file of VAE")
    parser.add_argument("--ldm_model_path", type=str, help="Path to LDM")
    parser.add_argument("--ldm_config_file", type=str, help= "Path to config file of LDM")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--guidance_scale", type=int, help="Guidance scale used for LDM")
    parser.add_argument("--n_timesteps_ddim", type=int, help="Number of steps of DDIM")
    parser.add_argument("--results_save", type=str, help="Saving directory")
    args = parser.parse_args()
    return args

def main(args):
    # CONTROL VARIABLES
    channels = {'wmh': 6, 'tumour': 7, 'edema': 8, 'gdtumour': 9}
    vae_model_path = args.vae_model_path
    vae_config_file = args.vae_config_file
    ldm_model_path = args.ldm_model_path
    ldm_config_file = args.ldm_config_file
    result_save_dir = args.results_save
    batch_size = args.batch_size
    guidance_scale= args.guidance_scale
    SOFTMAX = True # Whether the results need to be softmaxed or not.
    num_timesteps = args.n_timesteps_ddim
    save_img = True
    cond_map = {0: 'wmh', 1: 'tumour', 2: 'edema', 3: 'gdtumour'}
    cond_map_inv = {v: k for k, v in cond_map.items()}

    LESION_QUARTILES = {'wmh': (0.0149, 0.2), #(0.8, 0.9),
                        'tumour': (0.8, 0.9),
                        'edema': (0.8, 0.9),
                        'gdtumour': (0.8, 0.9),}

    if not os.path.isdir(result_save_dir):
        os.makedirs(result_save_dir)

    # INITIAL SET UP -------------------------------------------------------------------------------------------------
    # Create directories
    if not os.path.isdir(result_save_dir):
        os.makedirs(result_save_dir)

    #####################################################################################################
    #######################################STAGE 1#######################################################
    #####################################################################################################
    # We need to create the models_ and load their state dict
    device = torch.device("cuda")
    vae = loadmodel(vae_model_path, vae_config_file)
    vae = vae.eval().to(device)
    ldm = loadmodel(ldm_model_path, ldm_config_file, "LDM_CONDITIONED")
    ldm = ldm.eval().to(device)
    ldm.eval().to(device)
    ddim_sampler = DDIMSampler(ldm)

    # Label sampling process ----------------------------------------------------------------------------------
    processed = 0
    slice_crescendo = np.linspace(0.158, 0.59, num=100)
    slice_crescendo = np.concatenate([slice_crescendo]*5, 0)
    cond_list_disease = np.linspace(0.1, 0.9, num=5)
    total_diseases = {'wmh': {}, 'tumour': {}, 'edema': {},'gdtumour': {}}
    for t in total_diseases.keys():
        for c  in cond_list_disease:
            total_diseases[t][c] = {}

    n_batches_to_sample = int(np.ceil(len(slice_crescendo) * len(cond_list_disease) / batch_size))
    counter_sc = 0
    index_disease = 0
    # WMH
    for n_batch in range(n_batches_to_sample):
        if counter_sc+batch_size >= (len(slice_crescendo)) and index_disease == len(cond_list_disease)-1:
            b_size = len(slice_crescendo[counter_sc:])
            cond_slice=slice_crescendo[counter_sc:]
            cond_wmh = [cond_list_disease[index_disease]] * len(slice_crescendo[counter_sc:])
        else:
            b_size = batch_size
        if b_size==batch_size:
            if len(slice_crescendo[counter_sc:]) <= b_size:
                cond_slice=np.concatenate([slice_crescendo[counter_sc:],
                                          slice_crescendo[:b_size-len(slice_crescendo[counter_sc:])]], 0)
                cond_wmh=np.concatenate([[cond_list_disease[index_disease]]*len(slice_crescendo[counter_sc:]),
                                        [cond_list_disease[index_disease+1]]*(b_size-len(slice_crescendo[counter_sc:]))],
                                        0)
                index_disease += 1
                counter_sc = b_size-len(slice_crescendo[counter_sc:])
            else:
                cond_slice = slice_crescendo[counter_sc:counter_sc+b_size]
                cond_wmh= [cond_list_disease[index_disease]]*b_size
                counter_sc += b_size

        sample_shape = tuple([b_size, vae.decoder.z_channels,] + [
                int(j/(2**(len(vae.decoder.ch_mult) -1))) for j in list(vae.encoder.resolution)]) # Sample shape

        cond_list = [list(cond_wmh), [0.0] * b_size, [0.0] * b_size, [0.0] * b_size, list(cond_slice)]

        cond_list = np.stack(cond_list, -1) # .squeeze()
        cond = torch.FloatTensor(cond_list)  # Convert the list to tensors.
        with torch.no_grad():
            cond = cond2conKey(ldm.conditioning_key, cond, sample_shape[2:]).to(device)
        # Sample latent space
        latent_vectors, _ = ddim_sampler.sample(num_timesteps,conditioning=cond, batch_size=b_size,
                                            shape=sample_shape[1:], eta=1.0,
                                            guidance_scale = guidance_scale)

        # Reconstruct
        with torch.no_grad():
            x_hat = vae.reconstruct_ldm_outputs(latent_vectors)
            if SOFTMAX:
                x_hat = torch.softmax(x_hat, 1)

        # Loop along batches. Save imag and also compute volumes


        for b in range(x_hat.shape[0]):
            out_label = torch.argmax(x_hat[b,...].detach().cpu(), 0)
            if int(cond_slice[b]*256) not in total_diseases['wmh'][cond_wmh[b]].keys():
                total_diseases['wmh'][cond_wmh[b]][int(cond_slice[b]*256)] = []

            total_diseases['wmh'][cond_wmh[b]][int(cond_slice[b]*256)].append((out_label==channels['wmh']).sum().item())
            name_file = "sample_%d_wmh%.3f.png" %(cond_slice[b]*256, cond_wmh[b])
            plt.imshow(out_label, cmap = "jet"); plt.axis('off');
            plt.savefig(os.path.join(result_save_dir,name_file))
            plt.close('all')

    counter_sc = 0
    index_disease = 0
    # TUM
    for n_batch in range(n_batches_to_sample):
        if counter_sc+batch_size >= (len(slice_crescendo)) and index_disease == len(cond_list_disease)-1:
            b_size = len(slice_crescendo[counter_sc:])
            cond_slice=slice_crescendo[counter_sc:]
            cond_tum = [cond_list_disease[index_disease]] * len(slice_crescendo[counter_sc:])
            cond_ede = [cond_list_disease[index_disease]] * len(slice_crescendo[counter_sc:])
            cond_gdt = [cond_list_disease[index_disease]] * len(slice_crescendo[counter_sc:])
        else:
            b_size = batch_size
        if b_size==batch_size:
            if len(slice_crescendo[counter_sc:]) <= b_size:
                cond_slice=np.concatenate([slice_crescendo[counter_sc:],
                                          slice_crescendo[:b_size-len(slice_crescendo[counter_sc:])]], 0)
                cond_tum=np.concatenate([[cond_list_disease[index_disease]]*len(slice_crescendo[counter_sc:]),
                                        [cond_list_disease[index_disease+1]]*(b_size-len(slice_crescendo[counter_sc:]))],
                                        0)
                cond_ede=np.concatenate([[cond_list_disease[index_disease]]*len(slice_crescendo[counter_sc:]),
                                        [cond_list_disease[index_disease+1]]*(b_size-len(slice_crescendo[counter_sc:]))],
                                        0)
                cond_gdt=np.concatenate([[cond_list_disease[index_disease]]*len(slice_crescendo[counter_sc:]),
                                        [cond_list_disease[index_disease+1]]*(b_size-len(slice_crescendo[counter_sc:]))],
                                        0)
                index_disease += 1
                counter_sc = b_size-len(slice_crescendo[counter_sc:])
            else:
                cond_slice = slice_crescendo[counter_sc:counter_sc+b_size]
                cond_tum= [cond_list_disease[index_disease]]*b_size
                cond_ede = [cond_list_disease[index_disease]] * b_size
                cond_gdt = [cond_list_disease[index_disease]] * b_size
                counter_sc += b_size

        sample_shape = tuple([b_size, vae.decoder.z_channels,] + [
                int(j/(2**(len(vae.decoder.ch_mult) -1))) for j in list(vae.encoder.resolution)]) # Sample shape
        cond_list = [[0.0]*b_size, cond_tum, cond_ede, cond_gdt, cond_slice]
        cond_list = np.stack(cond_list, -1)
        cond = torch.FloatTensor(cond_list)  # Convert the list to tensors.
        with torch.no_grad():
            cond = cond2conKey(ldm.conditioning_key, cond, sample_shape[2:]).to(device)
        # Sample latent space
        latent_vectors, _ = ddim_sampler.sample(num_timesteps,conditioning=cond, batch_size=b_size,
                                                shape=sample_shape[1:], eta=1.0,
                                                guidance_scale = guidance_scale)
        # Reconstruct
        with torch.no_grad():
            x_hat = vae.reconstruct_ldm_outputs(latent_vectors)
            if SOFTMAX:
                x_hat = torch.softmax(x_hat, 1)

        # Loop along batches. Save imag and also compute volumes


        for b in range(x_hat.shape[0]):
            out_label = torch.argmax(x_hat[b,...].detach().cpu(), 0)
            if int(cond_slice[b]*256) not in total_diseases['tumour'][cond_tum[b]].keys():
                total_diseases['tumour'][cond_tum[b]][int(cond_slice[b]*256)] = []
            if int(cond_slice[b]*256) not in total_diseases['edema'][cond_ede[b]].keys():
                total_diseases['edema'][cond_ede[b]][int(cond_slice[b]*256)] = []
            if int(cond_slice[b] * 256) not in total_diseases['gdtumour'][cond_gdt[b]].keys():
                total_diseases['gdtumour'][cond_gdt[b]][int(cond_slice[b] * 256)] = []

            total_diseases['tumour'][cond_tum[b]][int(cond_slice[b]*256)].append((out_label==channels['tumour']).sum().item())
            total_diseases['edema'][cond_ede[b]][int(cond_slice[b] * 256)].append((out_label == channels['edema']).sum().item())
            total_diseases['gdtumour'][cond_gdt[b]][int(cond_slice[b] * 256)].append((out_label == channels['gdtumour']).sum().item())

            name_file = "sample_%d_tum%.3f_tum%.3f_tum%.3f.png" %(cond_slice[b]*256,
                                                                  cond_tum[b],
                                                                  cond_ede[b],
                                                                  cond_gdt[b])
            plt.imshow(out_label, cmap = "jet"); plt.axis('off');
            plt.savefig(os.path.join(result_save_dir,name_file))
            plt.close('all')
    vae.cpu()
    ldm.cpu()

    torch.save(total_diseases, os.path.join(result_save_dir, "total_diseases.pth"))

    # Once done
    colors = ["#ffbb33", "#ff7733", "#ff1a1a", "#cc0044"]
    patterns = [1, 1.5, 2, 2.5, 3]

    f = plt.figure(); plt.grid("#f2f2f2")
    cond_value_ctr = 0
    legend_ = []
    for cond_value, all_slices in total_diseases['wmh'].items():
        slice_plot = list(all_slices.keys())
        values = []
        for slice_no, values_ in all_slices.items():
            values.append(np.mean(values_))
        plt.plot(slice_plot, values, color = colors[0], linewidth = patterns[cond_value_ctr])
        legend_.append(cond_value)
        cond_value_ctr+=1
    plt.ylabel("disease pixels")
    plt.xlabel("slice conditioning")
    plt.title("WMH")
    plt.legend(legend_)
    plt.savefig(os.path.join(result_save_dir, "wmh_plot.png"))

    f = plt.figure();
    list_subtumour = ['tumour', 'edema', 'gdtumour']
    for st_ind, st in enumerate(list_subtumour):
        plt.subplot(1,len(list_subtumour),st_ind+1)
        plt.grid("#f2f2f2")
        cond_value_ctr = 0
        legend_ = []
        for cond_value, all_slices in total_diseases[st].items():
            slice_plot = list(all_slices.keys())
            values = []
            for slice_no, values_ in all_slices.items():
                values.append(np.mean(values_))
            plt.plot(slice_plot, values, color = colors[st_ind+1], linewidth = patterns[cond_value_ctr])
            legend_.append(np.round(cond_value, 3))
            cond_value_ctr+=1
        plt.ylabel("disease pixels")
        plt.xlabel("slice conditioning")
        plt.title(st)
        plt.legend(legend_)
    plt.savefig(os.path.join(result_save_dir, "tumour_plot.png"))
    plt.close(f)


args = parse_args()
main(args)