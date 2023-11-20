"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
import argparse
import os
from utils import util
import torch
import models_spade
import data
import pickle
import shutil

class RTT_DiscOptions():
    def __init__(self):
        self.initialized = False

    def help(self):
        max_char_name =  30
        max_char_value = 300

        message = ''
        title = "REAL-TRAIN-TEST-DISCRIMINATOR OPTIONS"
        message += '%s%s\n' %(title, "-"*(max_char_name + max_char_value - 3 - len(title)))
        options = {
            'DATA': {

            },

            'MODEL' : {

            },
            }

        for SUPERKEY, SUPERVALUE in options.items():
            sub_title = "%s" %SUPERKEY
            sub_message = ""
            sub_message += '%s%s\n' %(sub_title, "-"*(max_char_name + max_char_value - 3 - len(sub_title)))
            sorted_super =  dict(sorted(SUPERVALUE.items(), key=lambda x: x[0].lower()))
            for key, value in sorted_super.items():
                sub_message += "|%s%s\t%s%s|\n" %(key, "-"*(max_char_name-len(key)), value, "-"*(max_char_value-len(value)))
            message += sub_message

        message += "-"*(max_char_name + max_char_value + 3)
        print(message)

    def initialize(self, parser):

        # experiment specifics
        parser.add_argument('--name', type=str, default='brainspade', help='name of the experiment. It decides where to store samples and models_')
        parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='models_ are saved here')
        parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        parser.add_argument('--num_workers', type=int, default=1, help='mber of workers for the dataloader')
        parser.add_argument('--uncertainty', action='store_true', help="Whether you want test time augmentation uncertainty estimation")
        parser.add_argument('--lr_init', type=float, default = 0.002, help = "Initial learning rate")
        parser.add_argument('--lr_end', type=float, default=0.002, help="Last learning rate")

        # for model
        parser.add_argument('--init_features', type=int, default=8, help='Number of features in first filter (multiplicative afterwards.'
                                                                         'See MONAI documentation for DenseNet169.')


        # DATASET features
        parser.add_argument('--label_dir_tr', type=str, required=True,
                            help='path to the directory that contains the training label images.'
                                 'Only used for skullstrip')
        parser.add_argument('--image_dir_tr', type=str, default=None,
                            help='path to the directory that contains the training images')
        parser.add_argument('--label_dir_te', type=str, required=True,
                            help='path to the directory that contains the test label images.'
                                 'Only used for skullstrip')
        parser.add_argument('--image_dir_te', type=str, default=None,
                            help='path to the directory that contains the test  images')
        parser.add_argument('--skullstrip', action = 'store_true', help= "Whether to skull-strip or not the images.")

        parser.add_argument('--use_augmentation', action='store_true', help = 'Activate '
                                                                                          'intensity augmentation transform')
        parser.add_argument('--volume', action='store_true', help = "Work with volumes or with 2D slices.")
        parser.add_argument('--cache_dir', type =str, default = None, help = "Directory where you can cache images"
                                                                     "and label volumes with PersistentDataset")
        parser.add_argument('--cache_type', type = str, default = "none", help = "If ram, CacheDataset is used (caching to RAM),"
                                                                               "if cache, PeristentDataset is used (caching to disk),"
                                                                                 "if none, Dataset is used (no caching)")
        parser.add_argument('--format_data', type=str, default='npz', help = "Format of the image data. Can be either"
                                                                             "NPZ or NPY. If it's NPY, some features"
                                                                             "- affine transform requiring - will not"
                                                                             "be available.")
        parser.add_argument('--sequence', type = str, help = "Sequence of the images. Must meet the first element of the file name: T1_[rest].np*")
        parser.add_argument('--datasets', type = str, nargs='*',help = "Dataset of the images. Must meet the first element of the file name: T1_[rest].np*")
        parser.add_argument('--max_dataset_size', type=int, default=None,
                            help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--crop_size', type=int, nargs='*',
                            help='New height and width [and depth if volume is true] to resize the images to.')
        parser.add_argument('--validation_percentage', type=float, default=0.05,
                            help='Percentage of the images that go into validation fold.')

        self.initialized = True

        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, unknown = parser.parse_known_args()

        if opt.format_data not in ['npy', 'npz']:
            ValueError("The format of the data can only be npy or npz files.")

        opt = parser.parse_args()
        self.parser = parser

        return opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def option_file_path(self, opt, makedir=False):
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if makedir:
            util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt')
        return file_name

    def save_options(self, opt):
        file_name = self.option_file_path(opt, makedir=True)
        with open(file_name + '.txt', 'wt') as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

        with open(file_name + '.pkl', 'wb') as opt_file:
            pickle.dump(opt, opt_file)

    def load_options(self, opt):
        file_name = self.option_file_path(opt, makedir=False)
        new_opt = pickle.load(open(file_name + '.pkl', 'rb'))
        return new_opt

    def parse(self, save=False):
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test
        self.print_options(opt)
        self.opt = opt
        return self.opt


class RTT_DiscOptionsTrain(RTT_DiscOptions):
    def __init__(self):
        self.initialized = False

    def help(self):
        max_char_name = 30
        max_char_value = 300

        message = ''
        title = "REAL-TRAIN-TEST-DISCRIMINATOR OPTIONS"
        message += '%s%s\n' % (title, "-" * (max_char_name + max_char_value - 3 - len(title)))
        options = {
            'DATA': {
                'label_dir': 'type: string, help: path to the directory that contains label images',
                'image_dir': 'type: string, help: ppath to the directory that contains photo images',
                'skullstrip': 'type: flag, help: Whether to skull-strip or not the images',
                'intensity_transform': 'type: bool, default: False,  help: Activate intensity augmentation transform',

                'dataset_type': 'type: string, help: Type of dataset, loading from volumes or slices. Can only be volume or sliced.',
                'cut': 'type: string, default: a, help: Slice cut used (sagittal, coronal or axial',
                'cache_dir': 'type: string, default: None, help: Directory where you can cache images and label volumes with PersistentDataset',
                'cache_type': 'type: string, default: none, help: If ram, CacheDataset is used (caching to RAM), if cache, '
                              'PeristentDataset is used (caching to disk),if none, Dataset is used (no caching)',
                'dataset_mode': 'type: string, default: a, help: Slice cut used (sagittal, coronal or axial',
                'diff_slice': 'type: flag, help: Use different slices as style input.Both slices come from the same volume',
                'diff_style_volume': 'type: flag, help: dd another volume to the persistent dataset volumes of '
                                     'an image from a different style (dataset). The slices come from different images,'
                                     'and have, if can be, a different style',
                'non_corresponding_dirs': 'type: flag, help: Whether the label directory is different from the image directory.'
                                          'Since the files are not equivalent, non_corresponding_style is True.'
                                          'You need to provide style_label_dir for skullstrip mode',
                'style_label_dir': 'type: string, default: None,  help: Directory leading to the labels of the '
                                   'images (different from label_dir because non_corresponding_dirs is True',
                'bound_normalization': 'type: flag, help: With this flag, the default normalisation '
                                       'is replaced by a -1 to 1 normalisation.',
                'format_data': 'type: string, default: npz, help:   Format of the image data. Can be either, '
                               'NPZ or NPY. If it s NPY, some features, - affine transform requiring - '
                               'will not be available. ',
                'sequences': 'type: list of strings, default: [T1, FLAIR], help: List of sequences that make up the dataset. Must coincide with the label.',
                'fix_seq': 'type: str, default: None, help: List of sequences that make up the dataset. Must coincide with the label.',
                'datasets': 'type: list of strings, help: List of datasets included in the images. Must coincide with the label.',
                'max_dataset_size': 'type: int, default: none, help: Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.',
                'new_size': 'type: list of int, default: [256, 256], help: New height and width to resize the images to.',
                'label_nc': 'type: int, default: 182, help: # of input label classes without unknown class. If you have '
                            'unknown class as class label, specify --contain_dopntcare_label',
                'contain_dontcare_label': 'DEPRECATED type: flag, help: if the label map contains dontcare label (dontcare=255)'
            },

            'MODEL': {
                'model': 'type: string, default: pix2pix, help: which model to use',
                'norm_G': 'type: string, default: spectralinstance, help: instance normalization or batch normalization',
                'norm_E': 'type: string, default: spectralinstance, help: instance normalization or batch normalization',
                'netG': 'type: string, default: spade, help: selects model to use for netG (pix2pixhd | spade)',
                'ngf': 'type: int, default: 64, help: # of gen filters in first conv layer',
                'init_type': 'type: string, default: xavier, help: network initialization [normal|xavier|kaiming|orthogonal]',
                'init_variance': 'type: float, default: 0.02, help: variance of the initialization distribution',
                'z_dim': 'type: int, default: 256, help: dimension of the latent z vector',
                'n_decoders': 'type: string, default: dontcare, help: Names of the decoder tails, with dashes in between: i.e. FLAIR-T1',
                'upsampling_type': 'type: Type of convolution type: transposed upsample subpixel',
                'nef': 'type: int, defualt: 16, help: # of encoder filters in the first conv layer',
                'use_vae': 'type: flag, help: enable training with an image encoder.',
                'type_prior': 'type: string, default: N,  help: Type of prior, S - spherical, uniform, or N - normal',
            },

            'EXPERIMENT': {
                'name': 'type: string, default: brainspade, help: name of the experiment. It decides where to store samples and models_',
                'gpu_ids': 'type: string, default: 0, help: gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU',
                'checkpoints_dir': 'type: string, default: checkpoints, help: models_ are saved here',
                'phase': 'type: string, default: checkpoints, help: train, val, test, etc',
                'batchSize': 'type: int, default: 1, help: input batch size',
                'output_nc': 'type: int, default: 3, help: # of output image channels',
                'serial_batches': 'type: flag, default: checkpoints, help: if true, takes images in order to make batches, otherwise takes them randomly',
                'nThreads': 'type: int, default: 0, help: number of threads for loading data',
                'load_from_opt_file': 'DEPRECATED type: flag, help: load the options from checkpoints and use that as default',
                'cache_filelist_write': 'DEPRECATED type: flag, help:saves the current filelist into a text file, so that it loads faster',
                'cache_filelist_read': 'DEPRECATED type: flag, help: reads from the file list cache',
                'display_winsize': 'type: int, default: 400, help: display window size',

            },
        }

        for SUPERKEY, SUPERVALUE in options.items():
            sub_title = "%s" % SUPERKEY
            sub_message = ""
            sub_message += '%s%s\n' % (sub_title, "-" * (max_char_name + max_char_value - 3 - len(sub_title)))
            sorted_super = dict(sorted(SUPERVALUE.items(), key=lambda x: x[0].lower()))
            for key, value in sorted_super.items():
                sub_message += "|%s%s\t%s%s|\n" % (
                key, "-" * (max_char_name - len(key)), value, "-" * (max_char_value - len(value)))
            message += sub_message

        message += "-" * (max_char_name + max_char_value + 3)
        print(message)

    def initialize(self, parser):

        RTT_DiscOptions.initialize(self, parser)
        parser.add_argument('--num_epochs', type = int, default = 100, help = "Number of epochs to train the model for.")
        parser.add_argument('--use_tboard', action='store_true', help = "Use tensorboard to report the results.")
        parser.add_argument('--lr', type = float, lr = 0.001, help = "Learning rate")
        parser.add_argument('--beta1', type = float, default=0.9, help = "Beta1 for Adam optimizer")
        parser.add_argument('--beta2', type = float, default=0.999, help = "Beta2 for Adam optimizer")
        parser.add_argument('--validation_csv', type=str, help = "Validation path to the TSV files with paths.")
        parser.add_argument('--frequency_save', type = int, help = "Every how many epochs we save the model")

        parser.add_argument()
        self.initialized = True

        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, unknown = parser.parse_known_args()

        opt.format_data = opt.format_data.lower().replace(".", "")
        if opt.format_data not in ['npy', 'npz']:
            ValueError("The format of the data can only be npy or npz files.")

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models_spade.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)

        opt, unknown = parser.parse_known_args()

        # if there is opt_file, load it.
        # The previous default options will be overwritten
        if opt.load_from_opt_file:
            parser = self.update_options_from_file(parser, opt)
        opt = parser.parse_args()
        self.parser = parser

        return opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def option_file_path(self, opt, makedir=False):
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if makedir:
            util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt')
        return file_name

    def save_options(self, opt):
        file_name = self.option_file_path(opt, makedir=True)
        # Save previous options before
        with open(file_name + '.txt', 'wt') as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

        with open(file_name + '.pkl', 'wb') as opt_file:
            pickle.dump(opt, opt_file)

    def update_options_from_file(self, parser, opt):
        new_opt = self.load_options(opt)
        for k, v in sorted(vars(opt).items()):
            if hasattr(new_opt, k) and v != getattr(new_opt, k):
                new_val = getattr(new_opt, k)
                parser.set_defaults(**{k: new_val})
        return parser

    def load_options(self, opt):
        file_name = self.option_file_path(opt, makedir=False)
        new_opt = pickle.load(open(file_name + '.pkl', 'rb'))
        return new_opt

    def parse(self, save=False):
        opt = self.gather_options()
        self.print_options(opt)
        self.save_options(opt)
        self.opt = opt
        return self.opt