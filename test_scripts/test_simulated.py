from moreutils import structural_Similarity, l2
import os
import numpy as np
import monai

images_ori = "/media/vf19/BigCrumb/JOURNAL_22/SOURCE_DATASETS/SPADE_SLICED/images_test"
images_sim = "/media/vf19/BigCrumb/JOURNAL_22/SIMULATED_IMAGES"
labels = "/media/vf19/BigCrumb/JOURNAL_22/SOURCE_DATASETS/SPADE_SLICED/labels_test"
modalities = ['T1', 'FLAIR', 'T2']
save_at = "/media/vf19/BigCrumb/JOURNAL_22/RESULTS/quality_metrics/simulated_images_test.txt"

normaliser = monai.transforms.NormalizeIntensity()
out_string = "ID\tMOD\tSSIM\tMSE\n"
for mod in modalities:
    metrics = {'ssim': [], 'mse': []}
    mod_images = [i for i in os.listdir(images_ori) if "%s_" %mod in i]
    for i in mod_images:
        image_ori = np.load(os.path.join(images_ori, i))['img']
        image_sim = np.load(os.path.join(images_sim, i.replace(".npz", ".npy")))
        label = np.load(os.path.join(labels, i.replace("%s_" %mod, "Parcellation_")))['label']
        # Skullstrip
        image_ori[label[..., 0] > 0.35] = 0.0
        image_sim[label[..., 0] > 0.35] = 0.0
        # Now we have all three
        image_ori = np.transpose(image_ori, [-1, 0, 1])
        image_sim = np.transpose(image_sim, [-1, 0, 1])
        image_ori = normaliser(image_ori)
        image_sim = normaliser(image_sim)
        # SSIM
        metrics['ssim'].append(structural_Similarity(image_ori.unsqueeze(0), image_sim.unsqueeze(0))[0])
        metrics['mse'].append(l2(image_ori, image_sim)[0])
    for ind, i in enumerate(metrics['ssim']):
        out_string += "%d\t%s\t%.3f\t%.3f\n" %(ind, mod, i, metrics['mse'][ind])
    out_string += "%s\tMEAN\t%.3f\t%.3f\n" %(mod, np.mean(metrics['ssim']), np.mean(metrics['mse']))
    out_string += "%s\tSTD\t%.3f\t%.3f\n" % (mod, np.std(metrics['ssim']), np.std(metrics['mse']))

with open(save_at, 'w') as f:
    f.write(out_string)
    f.close()



