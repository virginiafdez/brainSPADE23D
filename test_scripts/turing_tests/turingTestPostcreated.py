'''
This script generates fake images and saves them along the fakes, to perform a blind test (AKA: user saying if the image is real or fake).
After the storage, call analyseTestFiles to 1) label the images 2) get a statistical analysis.
'''

from options.test_options import TuringTestOptions
import os
import moreutils as uvir
import numpy as np
from data.rrt_dataset import getTrainingDataloaders
import matplotlib.pyplot as plt
import  itertools

uvir.set_deterministic(True, 1)

# Initialisation
opt = TuringTestOptions().parse()
if not os.path.isdir(opt.results_dir):
    os.makedirs(opt.results_dir)
opt.validation_percentage = 0.0
modalities = opt.sequences
#

ground_truth = {}
voted = {}
for mod in modalities:
    # Create datasets
    opt.sequence = mod
    ground_truth[mod] = []
    voted[mod] = []
    train_loader, _ = getTrainingDataloaders(opt)
    for ind, i in enumerate(train_loader):
        img = i["image"]
        if opt.skullstrip:
            img[i["label"][:, 0:1, ...]>0.3] = img.min()
        for b in range(img.shape[0]):
            f = plt.figure(figsize = (20,20))
            plt.imshow(img[b, ...], cmap = "grey")
            plt.show()
            correct_input = False
            while not correct_input:
                input_tag = input("Real (1) or Fake (0)?\n")
                try:
                    suggested_tag = int(input_tag)
                    if suggested_tag not in [0, 1]:
                        print("Please, input 0 or 1\n")
                    else:
                        correct_input = True
                        voted[mod].append(suggested_tag)
                        ground_truth[mod].append(int(i['tag']))
                except:
                    continue
            plt.close(f)

    ground_truth[mod] = np.asarray(ground_truth)
    voted[mod] = np.asarray(voted)

# Reporting
accuracies = {}
precision = {}
recall = {}
mislabels = []

with open(os.path.join(opt.results_dir, 'results_turing_test.txt'), 'w') as f:
    f.write("Turing test. Number of images per modality: %d\n" %opt.n_images)

for mod, results in voted.items():
    accuracies[mod] = np.mean(results == ground_truth[mod])*100
    precision[mod] = (results[ground_truth[mod]==1.0] == 1.0).sum()/((results[ground_truth[mod]==1.0]==1.0).sum()+ \
                                                                     (results[ground_truth[mod]==0.0] == 1.0).sum())
    recall[mod] = (results[ground_truth[mod]==1.0] == 1.0).sum()/((results[ground_truth[mod]==1.0]==1.0).sum()+ \
                                                                     (results[ground_truth[mod]==1.0] == 0.0).sum())

    with open(os.path.join(opt.results_dir, 'results_turing_test.txt'), 'a') as f:
        f.write("%s\tAccuracy:%.3f\tPrecision:%.3f\tRecall:%.3f\n" %(mod, accuracies[mod],
                                                                     precision[mod], recall[mod]))
        f.close()

    mislabels.append([(results[ground_truth[mod] == 1.0] == 0.0).sum(),
                      (results[ground_truth[mod] == 0.0] == 1.0).sum()])

# Nest bar chart with mislabelled
fig, ax = plt.subplots()
size = 0.4
cmap = plt.colormaps["tab20c"]
outer_colors = cmap(np.arange(len(modalities)) * 4)
inner_colors = cmap(list(itertools.chain(
    *[[i, i+1] for i in np.arange(0, 4*len(modalities), 4)])))

ax.pie(mislabels.sum(axis=1), radius=1, colors=outer_colors,
       wedgeprops=dict(width=size, edgecolor='w'))
ax.pie(mislabels.flatten(), radius=1 - size, colors=inner_colors,
       wedgeprops=dict(width=size, edgecolor='w'))
ax.legend(modalities + ["GT", "SYN"]*len(modalities), bbox_to_anchor=(0,1.0))
ax.set(aspect="equal", title='Mislabelled images')
plt.savefig(os.path.join(opt.results_dir, "mislabelled.png"))
plt.close(fig)





