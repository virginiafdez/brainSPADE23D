import monai
import numpy as np
import os
from data.dataset_utils import findSimilarName

def findDataset(filename, datasets):
    out = []
    for d in datasets:
        if d in filename:
            out.append(d)
    if len(out) == 0:
        return ""
    elif len(out) > 1:
        print("Ambigous: more than one dataset name found for %s" %filename)
        return out[0]
    else:
        return out[0]

def getDataDicts(image_dir_str, label_dir_str,
                 image_dir_ste, label_dir_ste,
                 is_volume, format_data, sequence, datasets, max_dataset_size = 20000,
                 slice_index_str = -2,  slice_index_ste = -2,
                 validation_percentage = 0.05):

    '''
    Creates the MONAI-friendly data dictionary for the training dataset to train the real-train-test discriminator
    :param image_dir_str: path to the synthetic images from the training dataset
    :param label_dir_str: path to the labels from the training dataset
    :param image_dir_ste: path to the images from the test dataset
    :param label_dir_ste: path to the labels from the test dataset
    :param is_volume:  whether we are dealing with volumes or not
    :param format_data: format of data (npy or npz)
    :param sequence: which modality we are dealing with (i.e. T1, FlAIR etc.)
    :param datasets: list of datasets involved (list of strings: i.e. [SABRE, ADNI etc.]
    :param max_dataset_size: maximum training dataset size
    :param slice_index_real: slice index in the filename  if volume is not True, i.e.  in NAME_NAME_[SLICENO]_NAME.np
    slice index is -2
    :param validation_percentage: percentage of the data that goes into validation
    :return:
    '''

    if is_volume:
        slice_index_real = None

    # List images and corresponding labels
    all_images_rtr = [i for i in os.listdir(image_dir_str) if i.split("_")[0] == sequence]
    all_images_rte = [i for i in os.listdir(image_dir_ste) if i.split("_")[0] == sequence]
    np.random.shuffle(all_images_rte)
    np.random.shuffle(all_images_rtr)
    all_images_rtr = all_images_rtr[:int(max_dataset_size/2)]
    all_images_rte = all_images_rte[:int(max_dataset_size / 2)]
    all_labels_rtr = [findSimilarName(i, label_dir_str, slice_index=slice_index_real,
                              extension='.%s' %format_data, return_all=False) for i in all_images_rtr]
    all_labels_rte = [findSimilarName(i, label_dir_ste, slice_index=slice_index_real,
                                      extension='.%s' % format_data, return_all=False) for i in all_images_rte]
    # Make sure that we don't have images for which labels are None
    all_images_rtr = [i for ind, i in enumerate(all_images_rtr) if all_labels_rtr[ind] is not None]
    all_images_rte = [i for ind, i in enumerate(all_images_rte) if all_labels_rte[ind] is not None]
    all_labels_rtr = [i for i in all_labels_rtr if i is not None]
    all_labels_rte = [i for i in all_labels_rte if i is not None]
    # Datasets
    datasets_rtr = [findDataset(i, datasets) for i in all_images_rtr]
    datasets_rte = [findDataset(i, datasets) for i in all_images_rte]
    # 0 for train, 1 for test
    id_rtr = [0]*len(all_images_rtr)
    id_rte = [1]*len(all_images_rte)

    all_images = all_images_rtr + all_labels_rte
    all_labels = all_labels_rtr + all_labels_rte
    all_datasets = datasets_rtr + datasets_rte
    all_ids = id_rtr + id_rte
    all = list(zip(all_images, all_labels, all_datasets, all_ids))
    np.random.shuffle(all)

    # We create the output dictionary
    out_dicts = []
    for ind, i in all:
        out_dict = {}
        if i[3] == 0:
            out_dict['image'] = os.path.join(image_dir_str, i[0])
            out_dict['label'] = os.path.join(label_dir_str, i[1])
        else:
            out_dict['image'] = os.path.join(image_dir_ste, i[0])
            out_dict['label'] = os.path.join(label_dir_ste, i[1])

        out_dict['dataset'] = i[2]
        out_dict['id'] = i[3]
        out_dicts.append(out_dict)

    validation_dict = out_dicts[:int(len(out_dicts)*validation_percentage)]
    training_dict = out_dicts[int(len(out_dicts)*validation_percentage):]
    return training_dict, validation_dict


def getTrainingDataloaders(options):

    '''
    Obtains the data dictionaries, and their tags (labels that will be classified, train / test [0/1],
    creates CacheDatasets and dataloaders using MONAI for validation and testing.
    :param options: instance of  RTT_DiscOptions (realtraintest_disc_options)
    :return:
    '''

    # Get the dictionaries
    training_dict, validation_dict = getDataDicts(image_dir_str=options.image_dir_tr,
                                                  image_dir_ste=options.image_dir_te,
                                                  label_dir_ste=options.label_dir_te,
                                                  label_dir_str = options.label_dir_tr,
                                                  is_volume=options.volume,
                                                  format_data=options.format_data,
                                                  sequence=options.sequence,
                                                  datasets=options.datasets,
                                                  max_dataset_size=options.max_dataset_size,
                                                  validation_percentage=options.validation_percentage)
    if options.volume:
        dimension_transforms = [monai.transforms.Transposed(keys=["label"], indices=(3, 0, 1, 2)),
                                monai.transforms.AddChanneld(keys=["image", "id"])]
    else:
        dimension_transforms = [
            monai.transforms.SqueezeDimd(keys=["label"], dim=2),
            monai.transforms.Transposed(keys=["image", "label"], indices = (2, 0, 1)),
            monai.transforms.AddChanneld(keys=["id"])]

    val_transforms  = [monai.transforms.LoadImaged(keys=["image", "label"])] + \
                      dimension_transforms +\
                      [monai.transforms.SpatialPadd(keys=["image", "label"], spatial_size=options.crop_size,
                                     method="symmetric", mode="edge"),
                       monai.transforms.SpatialCropd(keys=["image", "label"], spatial_size=options.crop_size,
                                     method="symmetric", mode="edge")]

    if options.use_augmentation:
        train_transforms = val_transforms + [monai.transforms.RandAffined(keys=["image", "label"],
                                                                          prob=0.5, rotate_range=[0, 0.1],
                                                                          shear_range=[0.001, 0.15],
                                                                          scale_range=[0, 0.3], padding_mode='zeros',
                                                                          mode='nearest'),
                                             monai.transforms.RandAdjustContrast(keys=["image"], prob = 0.5,
                                                                                 gamma=2.0)]
    else:
        train_transforms = val_transforms

    train_transforms += monai.transforms.ToTensord(keys=["image", "label", "id"])
    val_transforms += monai.transforms.ToTensord(keys=["image", "label", "id"])

    train_transforms = monai.transforms.Compose(train_transforms)
    val_transforms = monai.transforms.Compose(val_transforms)

    train_dataset = monai.data.CacheDataset(data = training_dict, transform=train_transforms)
    validation_dataset = monai.data.CacheDataset(data = validation_dict, transform=val_transforms)

    train_loader = monai.data.DataLoader(train_dataset, batch_size=options.batchSize,
                                         num_workers=options.num_workers, shuffle=True)
    validation_loader = monai.data.DataLoader(validation_dataset, batch_size=options.batchSize,
                                         num_workers=options.num_workers, shuffle=False)

    return train_loader, validation_loader