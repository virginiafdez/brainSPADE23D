import monai
import torch
import numpy as np
import os
import pandas as pd
import nibabel as nib

class ImageSimulator:

    def __init__(self, csv_info, tissues, modalities):

        self.csv_info_path = csv_info
        if os.path.isfile(csv_info):
            self.tissue_info = pd.read_csv(csv_info)
        else:
            self.tissue_info = None
        self.tissues = tissues
        self.modalities = modalities
        self.map_tissue = {'background': 0, 'csf': 1, 'gm': 2, 'wm': 3, 'dgm': 4, 'brainstem': 5,
                           'wmh': 6, 'tumour': 7, 'edema': 8, 'gdtumour': 9}

    def read_tissue_info(self, csv_data):

        if ".tsv" in csv_data:
            image_files = pd.read_csv(csv_data, sep = "\t")
        else:
            image_files = pd.read_csv(csv_data)

        tissues_mod = {}
        for modality in self.modalities:
            if modality in image_files.columns:
                tissues_mod[modality] = {}
                list_files = list(image_files[modality])
                for ind, img_file in enumerate(list_files):
                    print(img_file)
                    if img_file == 'none':
                        continue
                    img = np.asarray(nib.load(img_file).dataobj)
                    # Normalise
                    img = (img - img.min()) / (img.max()-img.min())
                    label = np.asarray(nib.load(image_files['label'].iloc[ind]).dataobj)
                    for tissue in self.tissues:
                        if tissue not in tissues_mod[modality].keys():
                            tissues_mod[modality][tissue] = [[],[]]
                        img_mask = img[label[..., self.map_tissue[tissue]]>0.75]
                        if len(img_mask)!=0 and not True in np.isnan(img_mask):
                            tissues_mod[modality][tissue][0].append(img_mask.mean())
                            tissues_mod[modality][tissue][1].append(img_mask.std())
                            if img_mask.max() > 100000:
                                print("Inf: %s" %img_file)
                for tissue in tissues_mod[modality].keys():
                    tissues_mod[modality][tissue][0] = np.mean(tissues_mod[modality][tissue][0])
                    tissues_mod[modality][tissue][1] = np.mean(tissues_mod[modality][tissue][1])

        column_names = {'tissue': self.tissues}
        for modality in self.modalities:
            out_data = []
            for tissue in self.tissues:
                out_data.append("%.5f-%.5f" %(tissues_mod[modality][tissue][0],
                                              tissues_mod[modality][tissue][1]))
            column_names[modality] = out_data
        out_dataframe = pd.DataFrame(column_names)
        out_dataframe.to_csv(self.csv_info_path, index=False)
        self.tissue_info = out_dataframe

    def simulate_data(self, modality, store_at, base_segmentation_path,
                      channel_dim = -1, save_as = '.npy',  n_images=None):

        if not os.path.isdir(store_at):
            os.makedirs(store_at)

        all_files = os.listdir(base_segmentation_path)
        if n_images is not None:
            all_files = all_files[:n_images]
        for f in all_files:
            if ".npy" in f or ".npz" in f:
                lab = np.load(os.path.join(base_segmentation_path, f))
                if ".npz" in f:
                    lab = lab['label']
            elif ".nii" in f:
                lab = np.asarray(nib.load(os.path.join(base_segmentation_path, f)).dataobj)
            else:
                ValueError("Unsupported extension.")

            if channel_dim == -1 or channel_dim == (len(lab.shape) - 1):
                img_shape = list(lab.shape)[:-1]
            elif channel_dim == 0:
                img_shape = list(lab.shape)[1:]
            else:
                raise ValueError("Channel dim can only be the last or first dims. Otherwise add code here to support it.")
            img = np.zeros(img_shape)
            for tissue in self.tissues:
                if channel_dim == -1 or channel_dim == (len(lab.shape) - 1):
                    tissue_val_mean = self.tissue_info.loc[self.tissue_info['tissue']==tissue][modality].iloc[0]
                    tissue_mean = float(tissue_val_mean.split("-")[0])
                    tissue_std = float(tissue_val_mean.split("-")[-1])
                    noise = np.random.normal(scale = tissue_std/3, size = img_shape)
                    if tissue == 'background':
                        noise =  np.random.normal(scale = 0.01, size = img_shape)
                    img[lab[..., self.map_tissue[tissue]]>0.35] = tissue_mean + noise[lab[..., self.map_tissue[tissue]]>0.35]

            if save_as == '.npz':
                name = "%s.npz" %f.split(".")[0].replace("Parcellation_", "%s_" %modality)
                np.savez(os.path.join(store_at, name), img = img, modality = modality,
                         slice = int(f.split(".")[0].split("_")[-1]), dataset_name = "SIMUL")
            else:
                name = "%s.npy" % f.split(".")[0].replace("Parcellation_", "%s_" % modality)
                np.save(os.path.join(store_at, name), img)


image_sim = ImageSimulator(csv_info="/media/vf19/BigCrumb/JOURNAL_22/GENERATED_DATASETS/simulated_data.csv",
                           tissues=['background', 'csf', 'gm', 'wm', 'dgm', 'brainstem',
                                    'wmh', 'tumour', 'edema', 'gdtumour'],
                           modalities=['T1', 'FLAIR', 'T2'])
#image_sim.read_tissue_info("/media/vf19/BigCrumb/BRAINSPADE3D/TSVS/sabre_adni_brats_imagegen_root_readable_local_train.tsv")
image_sim.simulate_data('T1',
                        "/media/vf19/BigCrumb/JOURNAL_22/GENERATED_DATASETS/SABRE_HEALTHY_GUID15_50K_7/train/sim_T1",
                        "/media/vf19/BigCrumb/JOURNAL_22/GENERATED_DATASETS/SABRE_HEALTHY_GUID15_50K_7/train/labels",
                        channel_dim=-1,
                        save_as=".npy",
                        n_images=5096)
image_sim.simulate_data('T2',
                        "/media/vf19/BigCrumb/JOURNAL_22/GENERATED_DATASETS/SABRE_HEALTHY_GUID15_50K_7/train/sim_T2",
                        "/media/vf19/BigCrumb/JOURNAL_22/GENERATED_DATASETS/SABRE_HEALTHY_GUID15_50K_7/train/labels",
                        channel_dim=-1,
                        save_as=".npy",
                        n_images=5096)
image_sim.simulate_data('FLAIR',
                        "/media/vf19/BigCrumb/JOURNAL_22/GENERATED_DATASETS/SABRE_HEALTHY_GUID15_50K_7/train/sim_FLAIR",
                        "/media/vf19/BigCrumb/JOURNAL_22/GENERATED_DATASETS/SABRE_HEALTHY_GUID15_50K_7/train/labels",
                        channel_dim=-1,
                        save_as=".npy",
                        n_images=5096)

