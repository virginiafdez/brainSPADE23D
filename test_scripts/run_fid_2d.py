from torchvision.models import inception_v3
import torch
import monai
from GenerativeModels.generative.metrics.fid import FIDMetric
import numpy as np
import argparse
import os
from data.dataset_utils import findSimilarName
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

class PartialInceptionNetwork(nn.Module):
    '''
    Got from: https://github.com/hukkelas/pytorch-frechet-inception-distance.git
    From hukkelas, "Frechet Inception Distance (FID) from Pytorch"
    '''
    def __init__(self):
        super().__init__()
        self.inception_network = inception_v3(pretrained=True)
        self.inception_network.Mixed_7c.register_forward_hook(self.define_output_hook())

    def define_output_hook(self):
        # N x 2048 x 8 x 8
        def hook(module, input, output):
            self.mixed_7c_output = output
        return hook

    def forward(self, x):
        """
        Args:
            x: shape (N, 3, 299, 299) dtype: torch.float32 in range 0-1
        Returns:
            inception activations: torch.tensor, shape: (N, 2048), dtype: torch.float32
        """
        assert x.shape[1:] == (3, 299, 299), "Expected input shape to be: (N,3,299,299)" +\
                                             ", but got {}".format(x.shape)
        x = x * 2 -1 # Normalize to [-1, 1]

        # Trigger output hook
        self.inception_network(x)

        # Output: N x 2048 x 1 x 1
        activations = self.mixed_7c_output
        activations = torch.nn.functional.adaptive_avg_pool2d(activations, (1,1))
        activations = activations.view(x.shape[0], 2048)
        return activations

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_images_1", type = str, help = "Path to first set of images")
    parser.add_argument("--path_to_images_2", type = str, help = "Path to second set of images (ground truth)")
    parser.add_argument("--path_to_images_3", type=str, help="Path to third set of images")
    parser.add_argument("--path_to_labels_1", type = str, help = "Path to first set of labels")
    parser.add_argument("--path_to_labels_2", type = str, help = "Path to second set of labels (ground truth)")
    parser.add_argument("--path_to_labels_3", type=str, help="Path to third set of labels")
    parser.add_argument("--max_no_images", type = int, default=1000, help = "Maximum number of images to use")
    parser.add_argument("--filter_1", type=str, nargs = "*", default=[], help = "Make sure that selected images in "
                                                                                "path_to_images_1 contain these keywords")
    parser.add_argument("--filter_2", type=str, nargs = "*", default=[], help = "Make sure that selected images in "
                                                                                "path_to_images_2 contain these keywords")
    parser.add_argument("--filter_3", type=str, nargs="*", default=[], help="Make sure that selected images in "
                                                                            "path_to_images_3 contain these keywords")
    parser.add_argument("--npz_images_1", action='store_true', help="If the images on dataset 1 are npy, you assume they"
                                                                    "don't have a last channel dimension to squeeze")
    parser.add_argument("--npz_images_2", action='store_true',
                        help="If the images on dataset 2 are npy, you assume they"
                             "don't have a last channel dimension to squeeze")
    parser.add_argument("--npz_images_3", action='store_true', help="If the images on dataset 3 are npy, you assume they"
                                                                    "don't have a last channel dimension to squeeze")
    args = parser.parse_args()
    return args

def get_activations(dataloader, partial_network):

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), ncols=110)
    progress_bar.set_description(f"Passing dataset...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    partial_network = partial_network.to(device).eval()

    activations = []
    for step, el in progress_bar:
        img = el['image'].to(device)
        with torch.no_grad():
            activations.append(partial_network(img).detach().cpu())
    return torch.cat(activations,0)


def main(args):

    # Load two datasets (create MONAI dataset and loader)
    dataset_1_path = args.path_to_images_1
    dataset_1_path_labels = args.path_to_labels_1
    dataset_2_path = args.path_to_images_2
    dataset_2_path_labels = args.path_to_labels_2
    dataset_3_path = args.path_to_images_3
    dataset_3_path_labels = args.path_to_labels_3

    dataset_1_files_list = os.listdir(dataset_1_path)
    if len(args.filter_1) > 0:
        for filter in args.filter_1:
            dataset_1_files_list = [i for i in dataset_1_files_list if filter in i]
    dataset_1_files_list = [i for i in dataset_1_files_list if int(i.split(".")[0].split("_")[-2]) > 40]
    dataset_1_files_list = dataset_1_files_list[:args.max_no_images]
    np.random.shuffle(dataset_1_files_list)
    dataset_2_files_list = os.listdir(dataset_2_path)
    if len(args.filter_2) > 0:
        for filter in args.filter_2:
            dataset_2_files_list = [i for i in dataset_2_files_list if filter in i]
    dataset_2_files_list = [i for i in dataset_2_files_list if int(i.split(".")[0].split("_")[-1]) > 40]
    dataset_2_files_list_1 = dataset_2_files_list[:args.max_no_images]
   # dataset_2_files_list_2 = dataset_2_files_list_1[:-5] + dataset_2_files_list[args.max_no_images:(args.max_no_images+5)]
    dataset_2_files_list_2 = dataset_2_files_list[args.max_no_images:(2*args.max_no_images)]
    np.random.shuffle(dataset_2_files_list_1)
    np.random.shuffle(dataset_2_files_list_2)
    dataset_3_files_list = os.listdir(dataset_3_path)
    if len(args.filter_3) > 0:
        for filter in args.filter_3:
            dataset_3_files_list = [i for i in dataset_3_files_list if filter in i]
    dataset_3_files_list = dataset_3_files_list[:args.max_no_images]
    np.random.shuffle(dataset_3_files_list)
    dataset_1_files = [{'image': os.path.join(dataset_1_path, i),
                        'label': os.path.join(dataset_1_path_labels,
                                              findSimilarName(i, dataset_1_path_labels,
                                                              slice_index=-1, extension='.npz', slice_ori_index=-2))}
                       for i in dataset_1_files_list]
    dataset_2_files_1 = [{'image': os.path.join(dataset_2_path, i),
                         'label': os.path.join(dataset_2_path_labels,
                                               findSimilarName(i, dataset_2_path_labels,
                                                               slice_index=-1, extension='.npz'))}
                        for i in dataset_2_files_list_1]
    dataset_2_files_2 = [{'image': os.path.join(dataset_2_path, i),
                        'label': os.path.join(dataset_2_path_labels,
                                              findSimilarName(i, dataset_2_path_labels,
                                                              slice_index=-1, extension='.npz'))}
                       for i in dataset_2_files_list_2]
    dataset_3_files = [{'image': os.path.join(dataset_3_path, i),
                        'label': os.path.join(dataset_3_path_labels,
                                              findSimilarName(i, dataset_3_path_labels,
                                                              slice_index=-1, extension='.npz', slice_ori_index=-1))}
                       for i in dataset_3_files_list]

    transform_npz_list = [monai.transforms.LoadImaged(keys=['image'], npz_keys=['img']),
                          monai.transforms.LoadImaged(keys=['label'], npz_keys=['label']),
                          monai.transforms.EnsureChannelFirstd(keys=['label'], channel_dim=-1),
                          monai.transforms.SqueezeDimd(keys=['image', 'label'], dim=-1),
                          monai.transforms.EnsureChannelFirstd(keys=['image']),
                          monai.transforms.Lambdad(keys=['label'], func=lambda l: (l[0:1, ...] < 0.2).astype('float32')),
                          monai.transforms.SpatialPadd(keys=['label', 'image'], spatial_size=[256, 256]),
                          monai.transforms.MaskIntensityd(keys=['image'], mask_key='label'),
                          monai.transforms.Resized(keys=['image'], spatial_size=[299, 299]),
                          monai.transforms.RepeatChanneld(keys=['image'], repeats= 3),
                          monai.transforms.ScaleIntensityd(keys=['image'],minv=-1, maxv=1),
                          monai.transforms.ToTensord(keys=['image'])]

    transform_npy_list  = [monai.transforms.LoadImaged(keys=['image']),
              monai.transforms.LoadImaged(keys=['label'], npz_keys=['label']),
              monai.transforms.EnsureChannelFirstd(keys=['label'], channel_dim=-1),
              monai.transforms.SqueezeDimd(keys=['label'], dim=-1),
              monai.transforms.EnsureChannelFirstd(keys=['image']),
              monai.transforms.Lambdad(keys=['label'],
                                       func=lambda l: (l[0:1, ...] < 0.4).astype('float32')),
              monai.transforms.SpatialPadd(keys=['label', 'image'], spatial_size=[256, 256]),
              monai.transforms.MaskIntensityd(keys=['image'], mask_key='label'),
              monai.transforms.Resized(keys=['image'], spatial_size=[299, 299]),
              monai.transforms.RepeatChanneld(keys=['image'], repeats= 3),
              monai.transforms.ScaleIntensityd(keys=['image'],minv=-1, maxv=1),
              monai.transforms.ToTensord(keys=['image'])]

    transform_npy = monai.transforms.Compose(transform_npy_list)

    if args.npz_images_1:
        dataset_1 = monai.data.Dataset(dataset_1_files, transform=monai.transforms.Compose(transform_npz_list))
    else:
        dataset_1 = monai.data.Dataset(dataset_1_files, transform=transform_npy)
    if args.npz_images_2:
        transform_npz_list_aug = transform_npz_list[:-1] + [
                                                            monai.transforms.GaussianSmoothd(keys=['image'],
                                                                                             sigma = 1.0, 
                                                                                             ),
            monai.transforms.RandGaussianNoised(keys=['image'], std=0.05, prob=1.0),
        ] + \
                                 [transform_npz_list[-1]]
        dataset_2_1 = monai.data.Dataset(dataset_2_files_1, transform=monai.transforms.Compose(transform_npz_list))
        dataset_2_2 = monai.data.Dataset(dataset_2_files_2, transform=monai.transforms.Compose(transform_npz_list_aug)) # <<
    else:
        transform_npy_list_aug = transform_npy_list[:-1] + [monai.transforms.RandGaussianNoised(keys=['image'], std=0.05, prob=1.0)] + \
                                 [monai.transforms.GaussianSmoothd(keys=['image'],
                                                                                             sigma = 1.0)] + \
                                 [transform_npy_list[-1]]
        dataset_2_1 = monai.data.Dataset(dataset_2_files_1, transform=transform_npy)
        dataset_2_2 = monai.data.Dataset(dataset_2_files_2, transform=monai.transforms.Compose(transform_npy_list_aug)) # <<
    if args.npz_images_3:
        transform_npz_list_mod = transform_npz_list[:4] +  \
                                 [monai.transforms.EnsureChannelFirstd(keys=['image'], channel_dim=-1),] + \
                                 transform_npz_list[5:]
        dataset_3 = monai.data.Dataset(dataset_3_files, transform=monai.transforms.Compose(transform_npz_list_mod))
    else:
        transform_npy_list_mod = transform_npy_list[:4] +  \
                                 [monai.transforms.EnsureChannelFirstd(keys=['image'], channel_dim=-1),] + \
                                 transform_npy_list[5:]
        dataset_3 = monai.data.Dataset(dataset_3_files, transform=monai.transforms.Compose(transform_npy_list_mod))

    loader_1 = DataLoader(dataset_1, batch_size=8, num_workers=0, shuffle=False)
    loader_2_1 = DataLoader(dataset_2_1, batch_size=8, num_workers=0, shuffle=False)
    loader_2_2 = DataLoader(dataset_2_2, batch_size=8, num_workers=0, shuffle=False)
    loader_3 = DataLoader(dataset_3, batch_size=8, num_workers=0, shuffle=False)

    # Instance inception
    net = PartialInceptionNetwork()

    fid= FIDMetric()
    activations_3 = get_activations(loader_3, net)
    # Loop across dataset, forward through inceptionV3 and get features
    activations_1 = get_activations(loader_1, net)
    activations_2_2 = get_activations(loader_2_2, net)
    activations_2_1 = get_activations(loader_2_1, net)
    fid_score_gt = fid(activations_2_1, activations_2_2)

    fid_score_1 = fid(activations_1, activations_2_1)
    fid_score_3 = fid(activations_3, activations_2_2)

    print("Generated vs. ground truth: %.4f" %fid_score_1)
    print("Simulated vs. ground truth: %.4f" %fid_score_3)
    print("Ground truth vs. ground truth: %.4f" %fid_score_gt)


args = parse_args()
main(args)