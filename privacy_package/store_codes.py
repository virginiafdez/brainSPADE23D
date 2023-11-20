import os
from models_spade.pix2pix_model import Pix2PixModel
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from data.spadenai_v2_sliced import SpadeNaiSlice
from options.test_options import FullPipelineOptions
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

path_to_model = "/media/vf19/BigCrumb/JOURNAL_22/GENERATIVE_MODELS/ABLATION/BRAINSPADEV3_25_nomodisc_diffs_selfs"
iod_train_images = "/media/vf19/BigCrumb/JOURNAL_22/SOURCE_DATASETS/SPADE_SLICED/images_sabre"
iod_train_labels = "/media/vf19/BigCrumb/JOURNAL_22/SOURCE_DATASETS/SPADE_SLICED/labels_sabre"
iod_test_images = "/media/vf19/BigCrumb/JOURNAL_22/SOURCE_DATASETS/SPADE_SLICED/images_sabre_test"
iod_test_labels = "/media/vf19/BigCrumb/JOURNAL_22/SOURCE_DATASETS/SPADE_SLICED/labels_sabre_test"
ood_tumo_images = "/media/vf19/BigCrumb/JOURNAL_22/SOURCE_DATASETS/BRATS-OTHER_SLICED/images_test"
ood_tumo_labels = "/media/vf19/BigCrumb/JOURNAL_22/SOURCE_DATASETS/BRATS-OTHER_SLICED/labels_test"
ood_test_images = "/media/vf19/BigCrumb/JOURNAL_22/SOURCE_DATASETS/OASIS_SLICED/images_train"
ood_test_labels = "/media/vf19/BigCrumb/JOURNAL_22/SOURCE_DATASETS/OASIS_SLICED/labels_train"
output_folder = "/media/vf19/BigCrumb/JOURNAL_22/PRIVACY_ANALYSIS/CODE_ANALYSIS_BRATS"
tops = 500
modalities = ['T1', 'FLAIR']
stage_1 = True
stage_2 = True
if stage_1:
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
        for modality in modalities:
            os.makedirs(os.path.join(output_folder, "codes_%s" %modality))

    opt = FullPipelineOptions().load_options(os.path.join(path_to_model, 'opt'), dataset_type='sliced')
    opt.cache_dir = os.path.join(output_folder, "cache")
    opt.datasets.append('OAS')
    opt.datasets.append('BRATS-OTHER')
    opt.max_dataset_size = tops
    brainspade = Pix2PixModel(opt).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    brainspade = brainspade.to(device)

    # Set up images
    input_image_labels = {0: {'labels': iod_train_labels, 'images': iod_train_images},
                          1: {'labels': iod_test_labels, 'images': iod_test_images},
                          2: {'labels': ood_test_labels, 'images': ood_test_images},
                          3: {'labels': ood_tumo_labels, 'images': ood_tumo_images}}

    for modality in modalities:
        outputs_grid = [['path', 'id']]
        for label_id, folders in input_image_labels.items():
            opt.image_dir = folders['images']
            opt.label_dir = folders['labels']
            dataset_container = SpadeNaiSlice(opt, mode="test")
            dataset_container.resetDatasets(fixed_modality=modality)
            dataloader = DataLoader(dataset_container.sliceDataset, batch_size=8, shuffle=False,
                                    num_workers=int(opt.nThreads), drop_last=opt.isTrain,
                                    )
            counter = 0
            for ind, i in enumerate(dataloader):
                code, _ = brainspade(i, 'encode_only')
                code = code.detach().cpu()
                for b in range(code.shape[0]):
                    path_save = os.path.join(output_folder, "codes_%s" %modality, "code_%d_%d.pth" %(label_id,
                                                                                                     ind*8+b))
                    torch.save(code[b, ...], path_save)
                    outputs_grid.append([path_save, label_id])
                counter += 1
                if counter == tops:
                    break

        dataframe = pd.DataFrame(data = outputs_grid[1:], columns = outputs_grid[0])
        dataframe.to_csv(os.path.join(output_folder, "list_items_%s.csv" %modality), index=False)

if stage_2:
    for modality in modalities:
            grid = pd.read_csv(os.path.join(output_folder, 'list_items_%s.csv' %modality))
            paths = np.asarray(grid['path'])
            ids = np.asarray(grid['id'])
            features = []
            for p in paths:
                features.append(torch.load(p).numpy())
            features = np.stack(features, 0)
            features_scaled = (features - features.min()) / (features.max() - features.min())
            features_scaled = np.concatenate([features, np.expand_dims(ids, -1)], -1)
            kmeans = KMeans(init="random", n_clusters=4, n_init=10, max_iter=300,  random_state=42)
            kmeans.fit(features_scaled[:, :-1]) # No ids!!!!!
            # accuracy in identifying
            accuracy = (kmeans.labels_ == features_scaled[:, -1]).mean()
            print(accuracy)
            labels = ['#ff0000', '#1a75ff', '#53c653', '#ff00ff']
            tsne_algo = TSNE(n_components=2, learning_rate=400, perplexity=10)
            res = tsne_algo.fit_transform(features_scaled[:, :-1])
            f = plt.figure(figsize=(9, 7))
            plt.subplot(1, 2, 1)
            for lab_ind, lab in enumerate(labels):
                plt.scatter(res[features_scaled[:, -1] == lab_ind, 0], res[features_scaled[:, -1] == lab_ind, 1], color=lab,
                            alpha=0.3, s=2)
            plt.xticks([]); plt.yticks([])
            plt.legend(['iod-train', 'iod-test', 'ood', 'ood-tumours'])
            plt.title("Actual labels")
            plt.subplot(1, 2, 2)
            for lab_ind, lab in enumerate(labels):
                plt.scatter(res[kmeans.labels_ == lab_ind, 0], res[kmeans.labels_ == lab_ind, 1], color=lab, alpha=0.3, s=2)
            plt.xticks([]); plt.yticks([])
            plt.title("K-Means labels");
            plt.legend(['label 1', 'label 2', 'label 3', 'label 4'])
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, "results_%s.png" %modality))



