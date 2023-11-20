'''
SCRIPT that samples images from a certain modality and stores ground truths in another folder, then
you are show each and you have to rate whether it's real or fake.
'''


import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import time

# The file structure is considered to be the same you use when you run blindGenerateTest.py, which
# generates the files you have to use here.

parent_dir = "/media/vf19/BigCrumb/JOURNAL_22/TURING_TEST"
modality = 'FLAIR'
file_ground_truth = os.path.join(parent_dir, 'turing_test', modality, modality + "_directory.txt")
file_rating = os.path.join(parent_dir, 'turing_test', modality, modality + "_rating.txt")
folder = os.path.join(parent_dir, 'turing_test', modality)
save_to = os.path.join(parent_dir, 'turing_test', modality, modality + "_summary.txt")

def show(file_ground_truth, file_rating, folder, fake_label = None):
    if fake_label:
        tag = 2
    else:
        tag = 1
    print("Beginning test for modality %s...\n" %modality)
    # Initialise file rating
    with open(file_rating, 'w') as f:
        f.write("ID\tGT\tRATED\n")

    with open(file_ground_truth, 'r') as f:
        all_files = f.readlines()
    all_files = [[i.split("\t")[0], int(i.split("\t")[-1].strip("\n"))] for i in all_files[1:]]
    all_files = [i for i in all_files if i[1] == tag or i[1] == 0]
    np.random.shuffle(all_files)
    for ind, im in enumerate(all_files):
        img = Image.open(os.path.join(folder, im[0]))
        f = plt.figure()
        plt.imshow(img)
        plt.axis('off')
        plt.show(block = False)
        correct_input = False
        while not correct_input:
            input_tag = input("Real (1) or Fake (0)?\n")
            if input_tag in ["R", "r"]:
                f = plt.figure()
                plt.imshow(img)
                plt.show()
            else:
                try:
                    suggested_tag = int(input_tag)
                    if suggested_tag not in [0, 1]:
                        print("Please, input 0 or 1\n")
                    else:
                        correct_input = True
                except:
                    continue

        with open(file_rating, 'a') as fl:
            fl.write("%d\t%d\t%d\t" %(ind, im[1], int(suggested_tag)))


        plt.close(f)

    fl.close()


def analyseTestFiles(file_rating, save_to):

    '''
    For two symmetric files containing one sample name + 1 or 0 separated from the sample line
    by a \t sign, one file being the ground truth and the other file having the user's ratings,
    calculates the true positive rate, true negative rate etc.
    :param file_rating:
    :param save_to:
    :return:
    '''

    with open(file_rating, 'r') as f:
        lines_rate = f.readlines()
        f.close()

    samples = {}
    for s_ind, s in enumerate(lines_rate[1:]):
        samples[s.split("\t")[0]] = {
            'gt': int(s.split("\t")[1].strip("\n")),
            'rate': int(s.split("\t")[2].strip("\n"))
        }

    # Now we calculate the TPR, FPR, FNR, TNR
    results = {'tp': 0,
               'fp': 0,
               'tn': 0,
               'fn': 0}
    for s_id, s in samples.items():
        if s['gt'] == 0 and s['rate'] == 0:
            results['tn'] += 1
        if s['gt'] == 1 and s['rate'] == 1:
            results['tp'] += 1
        if s['gt'] == 0 and s['rate'] == 1:
            results['fp'] += 1
        if s['gt'] == 1 and s['rate'] == 0:
            results['fn'] += 1

    tpr = 100*(results['tp'] / (results['tp']+results['fn']))
    fpr = 100*(results['fp'] / (results['tn']+results['fp']))
    tnr = 100 * (results['tn'] / (results['tn'] + results['fp']))
    precision = 100*(results['tp']/(results['tp']+results['fp']))
    accuracy = 100*((results['tp']+results['tn'])/(results['tn']+results['tp']+
                                              results['fn']+results['fp']))
    recall = 100*(results['tp']/(results['tp']+results['fn']))


    with open(save_to, 'w') as f:
        f.write("True positive rate %.2f\n" %(tpr))
        f.write("False positive rate %.2f\n" % (fpr))
        f.write("True negative rate %.2f\n" % (tnr))
        f.write("Precision (TP/(TP+FP)) %.2f\n" % (precision))
        f.write("Recall (TP/(TP+FN)) %.2f\n" % (recall))
        f.write("Accuracy ((TP+TN)/(FP+FN+TP+TN))) %.2f\n" % (accuracy))


#show(file_ground_truth, file_rating, folder, fake_label=False)
analyseTestFiles(file_rating, save_to)

