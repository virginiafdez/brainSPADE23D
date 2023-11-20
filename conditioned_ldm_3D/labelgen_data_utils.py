import monai
import os
import numpy as np
import pandas as pd
import monai.transforms as transforms
from monai.data.dataset import PersistentDataset, CacheDataset, Dataset
from monai.data import DataLoader


def get_data_dicts(
        ids_path: str,
        shuffle: bool = False,
        conditioned: bool = False,
        conditionings=None,
        max_size = None,
        even_brats = False,
):
    """
    Get data dictionaries for label generator training.
    :param ids_path: path to TSV file
    :param shuffle: whether to shuffle the labels
    :param conditioned: if conditioning is required, conditioning columns will be present
    :param conditionings: list of conditioning keywords present on TSV file as columns
    :return:
    """
    df = pd.read_csv(ids_path, sep="\t")
    if shuffle:
        df = df.sample(frac=1, random_state=1)

    data_dicts = []
    for index, row in df.iterrows():
        out_dict = {
            "label": row["label"],
        }

        if conditioned:
            for conditioning in conditionings:
                if conditioning in row.keys():
                    out_dict[conditioning] = float(row[conditioning])

        data_dicts.append(out_dict)

    if even_brats:
        brats_files = [i for i in data_dicts if 'BRATS' in i['label']]
        non_brats_files = [i for i in data_dicts if 'BRATS' not in i['label']]
        counter_brats = 0
        new_data_dicts = []
        for i in non_brats_files:
            new_data_dicts.append(i)
            new_data_dicts.append(brats_files[counter_brats])
            counter_brats+=1
            if counter_brats == len(brats_files):
                counter_brats=1
        data_dicts = new_data_dicts

    print(f"Found {len(data_dicts)} subjects.")
    if max_size is not None:
        return data_dicts[:max_size]
    else:
        return data_dicts


def get_training_loaders(
        batch_size: int,
        training_ids: str,
        validation_ids: str,
        spatial_size: list,
        conditionings: list = [],
        only_val: bool = False,
        augmentation: bool = True,
        drop_last: bool = False,
        num_workers: int = 8,
        cache_dir=None,
        for_ldm=False,
        max_size = None,
        even_brats: bool = False,
        random_x_stack = [],
        one_channel = False,
):
    '''

    :param batch_size:
    :param training_ids:
    :param validation_ids:
    :param spatial_size:
    :param conditionings:
    :param only_val:
    :param augmentation:
    :param drop_last:
    :param num_workers:
    :param cache_dir:
    :param for_ldm:
    :param max_size:
    :param even_brats:
    :param random_x_stack: list with the dimensions (0: sagit., 1: coronal, 2: axial) for which you want the spatial
    size resolution to be a random crop.
    Example: spatial size 144, 144, 50, here you pass [2]. That means that random chunks of 144, 144 50 will be taken
    along the axial dimension.
    :return:
    '''

    # Create cache directory
    if  cache_dir is not None:
        if not os.path.isdir(os.path.join(cache_dir, 'cache')):
            os.makedirs(os.path.join(cache_dir, 'cache'))
        cache_dir = os.path.join(cache_dir, 'cache')

    # Define transformations
    if len(random_x_stack) != 0:
        spatial_size_random = [s if s_ind in random_x_stack else -1 for s_ind, s in enumerate(spatial_size)]
        spatial_size = [-1 if s_ind in random_x_stack else s for s_ind, s in enumerate(spatial_size)]

    if one_channel:
        channel_transform = transforms.EnsureChannelFirstd(keys=['label'])
    else:
        channel_transform = transforms.AsChannelFirstd(keys=['label'], channel_dim=-1)
    base_transforms = [
        transforms.LoadImaged(keys=['label']),  # Niftis
        channel_transform,
        transforms.CenterSpatialCropd(keys=['label'], roi_size=spatial_size)]

    if len(random_x_stack) != 0:
        base_transforms += [transforms.RandSpatialCropd(max_roi_size=spatial_size_random, roi_size=spatial_size_random,
                                                        keys=['label'])]
    if not one_channel:
        base_transforms += [
            transforms.SpatialPadd(keys=['label'], spatial_size=spatial_size, method='symmetric'),
            transforms.Lambdad(keys=['label'], func = lambda l: np.concatenate(
                [np.expand_dims(1-np.sum(l[1:, ...], 0),0), l[1:,...]], 0)),
            transforms.ToTensord(keys=["label", ] + conditionings)
        ]
    else:
        base_transforms += [
            transforms.SpatialPadd(keys=['label'], spatial_size=spatial_size, method='symmetric'),
            transforms.ToTensord(keys=["label",] + conditionings),
        ]

    val_transforms = transforms.Compose(base_transforms)

    if augmentation:
        if for_ldm:
            rotate_range = [-0.05, 0.05]
            shear_range = [0.001, 0.01]
            scale_range = [0, 0.005]
        else:
            rotate_range = [-0.1, 0.1]
            shear_range = [0.001, 0.15],
            scale_range = [-0.3, 0]

        train_transforms = transforms.Compose(
            base_transforms[:-1] + \
            [
                transforms.Rand3DElasticDict(
                sigma_range=(3, 8),
                magnitude_range=(50, 80),
                rotate_range=rotate_range,
                shear_range=shear_range,
                scale_range=scale_range,
                keys=["label"],
                prob=0.5,
                mode='nearest',
                padding_mode="border"
            ),
            ] +
            base_transforms[-3:]
        )

        # Affine
        # transforms.Rand3DElasticDict(
        #     sigma_range=[],
        #
        #     keys=["label"],
        #     prob=0.15,
        #     rotate_range=rotate_range,
        #     shear_range=shear_range,
        #     scale_range=scale_range,
        #     padding_mode='border',
        #     mode='nearest',
        #
        # ),

    else:
        train_transforms = val_transforms

    val_dicts = get_data_dicts(
        ids_path=validation_ids,
        shuffle=False,
        conditioned=True,
        conditionings=conditionings,

    )

    if cache_dir is not None:
        val_ds = PersistentDataset(
            cache_dir = cache_dir,
            data=val_dicts,
            transform=val_transforms,
        )
    else:
        val_ds = Dataset(data=val_dicts, transform=val_transforms)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=False
    )

    if only_val:
        return val_loader

    train_dicts = get_data_dicts(
        ids_path=training_ids,
        shuffle=False,
        conditioned=True,
        conditionings=conditionings,
        max_size=max_size,
        even_brats = even_brats
    )
    if cache_dir is not None:
        train_ds = PersistentDataset(
            cache_dir=cache_dir,
            data=train_dicts,
            transform=train_transforms,
        )
    else:
        train_ds = Dataset(
            data=train_dicts,
            transform=train_transforms,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=False
    )

    return train_loader, val_loader


def get_testing_loader(
        batch_size: int,
        testing_ids: str,
        spatial_size: list,
        conditionings: list = [],
        drop_last: bool = False,
        num_workers: int = 8,
        cache_dir=None,
        one_channel = False,
        max_size = None

):
    """
    Get data loaders for scenario with Partial Volume maps and conditioning.
    """

    # Create cache directory
    if cache_dir is not None:
        if not os.path.isdir(os.path.join(cache_dir, 'cache')):
            os.makedirs(os.path.join(cache_dir, 'cache'))
        cache_dir = os.path.join(cache_dir, 'cache')

    if one_channel:
        channel_transform = transforms.EnsureChannelFirstd(keys=['label'])
    else:
        channel_transform = transforms.AsChannelFirstd(keys=['label'], channel_dim=-1)
    base_transforms = [
        transforms.LoadImaged(keys=['label']),  # Niftis
        channel_transform,
        transforms.CenterSpatialCropd(keys=['label'], roi_size=spatial_size)]

    if not one_channel:
        base_transforms += [
            transforms.SpatialPadd(keys=['label'], spatial_size=spatial_size, method='symmetric'),
            transforms.Lambdad(keys=['label'], func=lambda l: np.concatenate(
                [np.expand_dims(1 - np.sum(l[1:, ...], 0), 0), l[1:, ...]], 0)),
            transforms.ToTensord(keys=["label", ] + conditionings)
        ]
    else:
        base_transforms += [
            transforms.SpatialPadd(keys=['label'], spatial_size=spatial_size, method='symmetric'),
            transforms.ToTensord(keys=["label", ] + conditionings),
        ]

    test_transforms = transforms.Compose(base_transforms)
    test_dicts = get_data_dicts(
        ids_path=testing_ids,
        shuffle=False,
        conditioned=True,
        conditionings=conditionings
    )

    if max_size is not None and max_size < len(test_dicts):
        test_dicts = test_dicts[:max_size]
    if cache_dir is not None:
        test_ds = PersistentDataset(
            cache_dir=cache_dir,
            data=test_dicts,
            transform=test_transforms,
        )
    else:
        test_ds = Dataset(
            data=test_dicts,
            transform=test_transforms,
        )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=False
    )
    return test_loader



