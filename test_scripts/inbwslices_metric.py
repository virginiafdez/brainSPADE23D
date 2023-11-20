"""
For nifi files resulting of reconstructing slice-by-slice inputs with SPADE (see test.py, test[4]),
that are named "GEN_XXXXXXXXX.nii.gz", calculates the in-between slice structural similarity index
and MSE.
"""
import numpy as np
import itertools
import os
import nibabel as nib
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import scipy.stats as stats

cases = {'ALL_LOSSES':"/media/vf19/BigCrumb/JOURNAL_22/GENERATIVE_MODELS/BRAINSPADEV3_25/TEST/test_3D",
         'SELFS': "/media/vf19/BigCrumb/JOURNAL_22/GENERATIVE_MODELS/ABLATION/BRAINSPADEV3_25_nomodisc_nodiffs_selfs/TEST/test_3D",
         'NADA': "/media/vf19/BigCrumb/JOURNAL_22/GENERATIVE_MODELS/ABLATION/BRAINSPADEV3_25_nomodisc_nodiffs_noselfs/TEST/test_3D",
         'DIFFS-SELFS': "/media/vf19/BigCrumb/JOURNAL_22/GENERATIVE_MODELS/ABLATION/BRAINSPADEV3_25_nomodisc_diffs_selfs/TEST/test_3D",
         'DIFFS': "/media/vf19/BigCrumb/JOURNAL_22/GENERATIVE_MODELS/ABLATION/BRAINSPADEV3_25_nomodisc_diffs_noselfs/TEST/test_3D",
         'MODISC': "/media/vf19/BigCrumb/JOURNAL_22/GENERATIVE_MODELS/ABLATION/BRAINSPADEV3_25_modisc_nodiffs_noselfs/TEST/test_3D"}

save_to = "/media/vf19/BigCrumb/JOURNAL_22/GENERATIVE_MODELS/ABLATION/COMPARATIVE"
modality = "T2_"
ssims = {}
mses = {}

# We aggregate structural similarity indices between slices of all the images.
for case, case_dir in cases.items():
    all_files = [i for i in os.listdir(case_dir) if "GEN_" in i and ".nii.gz" in i and modality in i]
    ssims[case] = []
    mses[case] = []
    for f in all_files:
        img = np.asarray(nib.load(os.path.join(case_dir, f)).dataobj)
        gt_f = [g for g in os.listdir(case_dir) if f.strip("GEN_") in g]
        if len(gt_f)>0:
            gt_img = np.asarray(nib.load(os.path.join(case_dir, gt_f[0])).dataobj)[..., 0]
            ssim_img = [ssim(img[..., i], img[..., i+1]) for i in range(img.shape[-1]-1)]
            ssim_img_gt = [ssim(gt_img[..., i], gt_img[..., i+1]) for i in range(gt_img.shape[-1]-1)]
            ssim_img = [i/ssim_img_gt[ind] for ind, i in enumerate(ssim_img)]
            ssims[case]+=ssim_img
            mses_img = [np.sqrt((img[..., i]-img[..., i+1])**2).sum() for i in range(img.shape[-1]-1)]
            mses[case]+=mses_img

# We build a boxplot
f, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize=(20, 12))
bplot1 = ax1.boxplot(ssims.values(), labels = ssims.keys(), patch_artist = True)
ax1.set_xticklabels(labels=ssims.keys(), fontsize=11, rotation = 45)
ax1.set_title("Structural similarity index")
ax1.set_ylabel("SSIM (%)")
bplot2 = ax2.boxplot(mses.values(), labels = mses.keys(), patch_artist = True)
ax2.set_xticklabels(labels=mses.keys(), fontsize=11, rotation = 45)
ax2.set_title("MSE")
ax2.set_ylabel("MSE")
f.tight_layout()
# Style
for bplot in (bplot1, bplot2):
    for patch, color in zip(bplot['boxes'], ['navajowhite']*len(ssims.keys())):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for median in bplot['medians']:
        median.set_color('black')
        median.set_linewidth(1.5)
plt.savefig(os.path.join(save_to, 'inbwslices_comparative.png'))
plt.close(f)

# T-TEST
pairs = list(itertools.product(list(cases.keys()), list(cases.keys())))
t_test_results = ""
for pair in pairs:
    if pair[0] != pair[1]:
        statistic_ssim, pvalue_ssim = stats.ttest_ind(ssims[pair[0]], ssims[pair[1]], equal_var=False,
                                                      alternative='greater')
        statistic_mse, pvalue_mse = stats.ttest_ind(mses[pair[0]], mses[pair[1]], equal_var=False,
                                                    alternative='greater')
        t_test_results+= "SSIM: %s > %s, p-value:%.5f\n" %(pair[0], pair[1], pvalue_ssim)
        t_test_results += "MSE: %s > %s, p-value:%.5f\n" % (pair[0], pair[1], pvalue_mse)

with open(os.path.join(save_to, "inbwslices_comparative_tt_os.txt"), 'w') as f:
    f.write(t_test_results)
