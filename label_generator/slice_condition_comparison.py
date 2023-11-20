import matplotlib.pyplot as plt
import numpy as np
import os

labels_gt_path = "/media/vf19/BigCrumb/JOURNAL_22/SOURCE_DATASETS/SPADE_SLICED/labels_train"
labels_path = "/media/vf19/BigCrumb/JOURNAL_22/GENERATED_DATASETS/MULTI-DISEASE/train/labels"
save_folder = "/media/vf19/BigCrumb/JOURNAL_22/GENERATED_DATASETS/MULTI-DISEASE/train/slice_cond_eval"
if not os.path.isdir(save_folder):
    os.makedirs(save_folder)
labels_gt = os.listdir(labels_gt_path)
labels = os.listdir(labels_path)
n_slices = 100
max_gts = 3
done_slices = []
for n in range(n_slices):
    gen_label_path = np.random.choice(labels)
    slice = int(gen_label_path.split("_")[-1].split(".")[0])
    choices = [i for i in labels_gt if "_%d.npz" %slice in i]
    if len(choices) > max_gts:
        choices = choices[:max_gts]
    gen_label = np.load(os.path.join(labels_path, gen_label_path))['label']
    gen_label = np.squeeze(gen_label, -2)
    gen_label = np.argmax(gen_label, -1)
    plot_images = [gen_label]
    for i in choices:
        ground_truth_ = np.load(os.path.join(labels_gt_path, i))['label']
        ground_truth_ = np.squeeze(ground_truth_, -2)
        ground_truth_ = np.argmax(ground_truth_,-1)
        plot_images.append(ground_truth_)
    f = plt.figure()
    for ind, i in enumerate(plot_images):
        plt.subplot(1, len(plot_images), ind+1)
        plt.imshow(i)
        plt.axis('off')
        if ind == 0:
            plt.title("synthetic")
        else:
            plt.title("gt %d" %(ind))
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, gen_label_path.replace(".npz", ".png")))
    plt.close(f)
