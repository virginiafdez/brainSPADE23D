import numpy as np
import os
import matplotlib.pyplot as plt
import collections
from scipy.interpolate import make_interp_spline

input_labels = "/media/vf19/BigCrumb/JOURNAL_22/SOURCE_DATASETS/SABRE_HEALTHY_SLICED/train/labels"
figure_path = "/media/vf19/BigCrumb/JOURNAL_22/GENERATED_DATASETS/SABRE_HEALTHY_GUID15/train/tissue_volumes_gt.png"
channels = {0: 'background',
            1: 'csf',
            2: 'gm',
            3: 'wm',
            4: 'dgm',
            5: 'brainstem'}
            # 6: 'wmh',
            # 7: 'tumour',
            # 8: 'edema',
            # 9: 'gdtumour'}
colors = {0: "#6666ff",
          1: "#33ff99",
          2: "#ff9933",
          3: "#33cccc",
          4: "#ff3333",
          5: "#cc9966",
          6: "#ff66cc",
          7: "#9494b8",
          8: "#262626",
          9: "#944dff"}

brain_volume = False

if not brain_volume:
    volumes = {}
    for key, val in channels.items():
        volumes[key] = {}
    list_of_labels = [l for l in os.listdir(input_labels) if int(l.split(".")[0].split("_")[-1])>=40]
    for lab_path in list_of_labels:
        lab = np.load(os.path.join(input_labels, lab_path))['label']
        lab_argmax = np.argmax(lab[..., 0, :], -1)
        slice_no = int(lab_path.split("_")[-1].split(".")[0])
        for ch_ind, ch_name in channels.items():
            if ch_ind == 0:
                continue
            tissue_vol = (lab_argmax==ch_ind).sum()
            brain_vol = (lab_argmax!=0).sum()
            tissue_vol = lab[..., 0, ch_ind].sum()
            if slice_no not in volumes[ch_ind].keys():
                volumes[ch_ind][slice_no] = []
            volumes[ch_ind][slice_no].append(tissue_vol)
            # Volume
            #volumes[ch_ind][slice_no].append(tissue_vol / brain_vol)

    min_slice = 100000
    max_slice = 0
    for ch_ind, slice_volumes in volumes.items():
        for slice_no, values in slice_volumes.items():
            volumes[ch_ind][slice_no] = [np.mean(values), 2*np.std(values)]
            if slice_no > max_slice:
                max_slice = slice_no
            if slice_no < min_slice:
                min_slice = slice_no
    main_vec = np.linspace(min_slice, max_slice, max_slice-min_slice + 1).astype('int')
    f = plt.figure(figsize=(10,6))
    legend_ = []
    for ch_ind, slice_volumes in volumes.items():
        if ch_ind == 0:
            continue
        ordered = collections.OrderedDict(sorted(slice_volumes.items()))
        slice_x = list(ordered.keys())
        values_y = np.asarray(list(ordered.values()))
        slice_y = list(values_y[:, 0])
        slice_err= list(values_y[:, 1])
        plt.errorbar(slice_x, slice_y, yerr=slice_err, color = colors[ch_ind],
                alpha = 0.75, elinewidth=0.5)
        #plt.ylim([0, 0.85])
        plt.ylim([0, 12000])
        legend_.append(channels[ch_ind])
    plt.grid(c = "#f2f2f2", linestyle = '-.')
    plt.subplots_adjust(right=0.8)
    plt.xlabel("slice number", fontsize = 9)
    #plt.ylabel("region volume / brain volume")
    plt.ylabel("sum of pixel probabilities per tissue and slice", fontsize=9)
    plt.xticks(fontsize=9)
    plt.legend(legend_, loc = 'center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(figure_path)
else:
    volumes = {}
    for key, val in channels.items():
        volumes[key] = [[], []]
    for lab_path in os.listdir(input_labels):
        lab = np.load(os.path.join(input_labels, lab_path))['label']
        lab = np.argmax(lab[..., 0, :], -1)
        slice_no = int(lab_path.split("_")[-1].split(".")[0])
        for ch_ind, ch_name in channels.items():
            if ch_ind == 0:
                continue
            tissue_vol = (lab == ch_ind).sum()
            brain_vol = (lab != 0).sum()
            volumes[ch_ind][0].append(brain_vol)
            volumes[ch_ind][1].append(tissue_vol / brain_vol)

    min_slice = 100000
    max_slice = 0
    f = plt.figure(figsize=(10, 6))
    legend_ = []
    for ch_ind, slice_volumes in volumes.items():
        if ch_ind == 0:
            continue
        slice_volumes_arr = np.asarray(slice_volumes)
        ind=np.argsort(slice_volumes_arr[0, :])
        ordered = slice_volumes_arr[:, ind]
        no_dups_bv = []
        no_dups_tv = []
        bv = ordered[0, :]
        tv = ordered[1, :]
        for o_ind, o in enumerate(bv):
            alls = tv[bv == o]
            if o not in no_dups_bv:
                no_dups_bv.append(o)
                no_dups_tv.append(np.mean(alls))
        no_dups_bv = np.asarray(no_dups_bv)
        no_dups_tv = np.asarray(no_dups_tv)
        X_Y_Spline = make_interp_spline(no_dups_bv, no_dups_tv)
        brain_vol_smooth = np.linspace(no_dups_bv.min(), no_dups_bv.max(), int(no_dups_tv.shape[0]/3))
        tissue_vol_smooth = X_Y_Spline(brain_vol_smooth)
        plt.plot(brain_vol_smooth, tissue_vol_smooth, c=colors[ch_ind], alpha = 0.75)
        plt.ylim([0, 0.85])
        legend_.append(channels[ch_ind])
        plt.plot()
    plt.grid(c="#f2f2f2", linestyle='-.')
    plt.subplots_adjust(right=0.8)
    plt.xlabel("slice number")
    plt.ylabel("region volume / brain volume")
    plt.legend(legend_, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(figure_path)

