import torch
import numpy
import os

def post_process_os(data_i):
    """
    From a SPADE Dataset item framework, extract the strings from this_seq and other_seqs to put them right after auto-collate fn call
    :param data_i: Data item obtained from pix2pix dataset
    :return:
    """

    if 'this_seq' in data_i.keys():
        var = data_i['this_seq']
        if not type(var) == str:
            data_i['this_seq'] = var[0]
    return data_i

def clear_data(trainer, data_i):

    """
    Detaches some data from the GPU to ensure that the training is possible
    :param data_i:
    :return:
    """

    data_i['image'] = data_i['image'].detach().cpu()
    if trainer is not None:
        if trainer.generated is not None:
            trainer.generated = trainer.generated.detach().cpu()

def findSimilarName(original, directory, slice_index=-1, extension='.png', return_all = False, additional_keywords = [],
                    slice_ori_index = None):
    all_files = os.listdir(directory)
    keywords = ["sub", "ses", "SUB", "SES",]+additional_keywords
    root = original.replace(extension, "")
    if return_all:
        return_list = []
    if slice_index is not None:
        #Prefilter! Quicker
        if slice_ori_index is not None:
            slice_no = root.split(".")[0].split("_")[slice_ori_index]
        else:
            slice_no = root.split(".")[0].split("_")[slice_index]
        all_files = [f for f in all_files if f.split(".")[0].split("_")[slice_index] == slice_no]
    for f_sup in all_files:
        f = f_sup.replace(extension, "")
        f_sp = f.split("_")
        keys = []
        for key in keywords:
            positives_targ = [sp for sp in f_sp if key in sp]
            if len(positives_targ) == 0:
                continue
            else:
                positives_targ = positives_targ[0]
                ori_sp = root.split("_")
                positives_ori = [sp for sp in ori_sp if key in sp]
                if len(positives_ori) == 0:
                    continue
                else:
                    positives_ori = positives_ori[0]
                    if positives_targ == positives_ori:
                        keys.append(True)
                    else:
                        keys.append(False)
                        break
        # Now we compare the slice number if applicable
        if slice_index is not None:
            slice = f_sp[slice_index]
            if slice_ori_index is not None:
                slice_ori = root.split(".")[0].split("_")[slice_ori_index]
            else:
                slice_ori = root.split(".")[0].split("_")[slice_index]
            if slice != slice_ori:
                keys.append(False)
        if False not in keys:
            if return_all:
                return_list.append(f+extension)
            else:
                return f + extension
    if return_all:
        if len(return_list) > 0:
            return return_list
        else:
            return  None
    else:
        return None
