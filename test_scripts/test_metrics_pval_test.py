import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import itertools

def run_DistributionVisualizer(file_paths, save_data_dir, metric_name, restrict_modality = None):

    if type(restrict_modality) is not list:
       restrict_modality = [restrict_modality]

    OUTPUT_FRAMES = {}

    for mod in restrict_modality:
        # If the files are txt, convert them into CSV
        all_data = None
        counter = 0
        for key, f in file_paths.items():
            this_csv = pd.read_csv(f, sep = "\t")
            if mod is not None:
                this_csv = this_csv[this_csv["Sequence"]==mod]
            if counter == 0:
                all_data = this_csv
                all_data.rename({"%s" %metric_name: key}, axis=1, inplace = True) # We replace the column name by
                columns = set(all_data.columns)
                columns2 = columns.copy()
                for c in columns:
                    if c != key:
                        columns2.remove(c)
                all_data = all_data[columns2]
                all_data = all_data.astype(float)

                # the name of the data column
            else:
                this_csv.rename({"%s" %metric_name: key}, axis=1, inplace = True)
                columns = set(this_csv.columns)
                columns2 = columns.copy()
                for c in columns:
                    if c != key:
                        columns2.remove(c)
                this_csv = this_csv[columns2]
                this_csv = this_csv.astype(float)
                all_data = pd.concat([all_data, this_csv], axis = 1)

            counter += 1

        # We should have a data frame, with the columns reflecting the different values.

        if not os.path.exists(save_data_dir):
            os.makedirs(save_data_dir)


        # Normality tests
        names_distrib = all_data.columns.values
        normality_stats = np.zeros([len(names_distrib), 2])
        for d_ind, d in enumerate(names_distrib):
            statistic, pval = stats.normaltest(all_data.values[:, d_ind])
            normality_stats[d_ind, 0] = statistic
            normality_stats[d_ind, 1] = pval
            if pval < 0.05:
                print(d + " is significantly not normal.")

        OUTPUT_FRAMES[mod] = all_data

    # Plot
    i = 1
    f = plt.figure(figsize=(len(restrict_modality)*5, 10))
    for mod_name, frame in OUTPUT_FRAMES.items():
        plt.subplot(1,len(restrict_modality), i)
        boxplot = frame.boxplot()
        plt.title("%s_%s.png" % (metric_name, mod_name))
        boxplot.set_xticklabels(boxplot.get_xticklabels(), rotation=90)
        if mod is None:
            mod = "all"
        i+=1
    plt.savefig(os.path.join(save_data_dir, "boxplot_%s.png" % (metric_name)), bbox_inches='tight')
    plt.close(f)
    f = plt.figure()
    i=1
    for mod_name, frame in OUTPUT_FRAMES.items():
        plt.subplot(1, len(restrict_modality), i)
        frame.hist(bins=30)
        plt.ylabel("Counts")
        if mod is None:
            mod = "all"
        plt.xlabel("%s_%s.png" % (metric_name, mod_name))
        i+=1
    plt.savefig(os.path.join(save_data_dir, "histogram_%s.png" % (metric_name)))
    plt.close(f)

    return OUTPUT_FRAMES

def run_pairedTTest(dataframe, indices):

    if len(indices) != 2:
        ValueError("Paired t-test requires 2 indices, not less, not more.")

    statistic, pvalue = stats.ttest_ind(dataframe.values[:-2, indices[0]], dataframe.values[:-2, indices[1]], equal_var = False)

    return statistic, pvalue

# MAIN
modalities = ["T1", "FLAIR", "T2"]
metrics = ["SSIM", "L2"]
relative = "TEST/generated_im/metrics_results.txt"
all_results = {"all_losses": "/media/vf19/BigCrumb/JOURNAL_22/GENERATIVE_MODELS/BRAINSPADEV3_25",
               "modisc_nodiffs_noselfs" : "/media/vf19/BigCrumb/JOURNAL_22/GENERATIVE_MODELS/ABLATION/BRAINSPADEV3_25_modisc_nodiffs_noselfs",
               "nomodisc_diffs_noselfs" : "/media/vf19/BigCrumb/JOURNAL_22/GENERATIVE_MODELS/ABLATION/BRAINSPADEV3_25_nomodisc_diffs_noselfs",
               "nomodisc_nodiffs_selfs": "/media/vf19/BigCrumb/JOURNAL_22/GENERATIVE_MODELS/ABLATION/BRAINSPADEV3_25_nomodisc_nodiffs_selfs",
               "nomodisc_nodiffs_noselfs": "/media/vf19/BigCrumb/JOURNAL_22/GENERATIVE_MODELS/ABLATION/BRAINSPADEV3_25_nomodisc_nodiffs_noselfs",}
all_paths = {}
for key, val in all_results.items():
    all_paths[key] = os.path.join(val, relative)
save_data_dir="/media/vf19/BigCrumb/JOURNAL_22/GENERATIVE_MODELS/ABLATION/COMPARATIVE"

for metric in metrics:
    output_frames = run_DistributionVisualizer(file_paths= all_paths,
                                               save_data_dir=save_data_dir,
                                               metric_name="SSIM",
                                               restrict_modality=modalities)
    combis = {}
    idx = 0
    for k in all_paths.keys():
        combis[idx] = k
        idx+=1
        pairs = list(itertools.combinations(list(combis.keys()), 2))
        t_tests = {}
        for mod in modalities:
            t_tests[mod] = []
            for p in pairs:
                s, p = run_pairedTTest(output_frames[mod], [p[0], p[1]])
                t_tests[mod].append(p)

        t_test_results = "T-Test\n"
        for mod in modalities:
            t_test_results+= "-----%s-----\n" %mod
            for ind, p in enumerate(pairs):
                t_test_results+= "%s vs %s, p-value:%.5f\n" %(combis[p[0]], combis[p[1]], t_tests[mod][ind])

with open(os.path.join(save_data_dir, 't_test.txt'), 'w') as f:
    f.write(t_test_results)

print(t_test_results)

