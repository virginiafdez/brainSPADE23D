"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from .base_options import BaseOptions
import pickle

class TestOptions(BaseOptions):
    def help(self):
        BaseOptions().help()

        max_char_name =  30
        max_char_value = 300
        message = ''
        message += 'TestOptions%s\n' %("-"*(max_char_name + max_char_value - 10))
        options = {'results_dir': "type: string; default: ./results/; help: saves results here.",
                   'which_epocch': 'type: string, default: latest, help: which epoch to load? set to latest to use latest cached model',
                   'lesion': 'type: string, default: "", help: Path to test the lesion included images',
                   'mod_classifier': 'type: int, default: None, help: Epoch of the modality classifier, stored in the checkpoints folder and named X_net_MD.pth',
                   'mod_disc_dir': 'type: string, default: None, help: Optional path specified for the modality discriminator. Otherwise, checkpoints is picked.',
                   'mode': 'type: string, default: eval, help: mode to put the netowrk in (train or eval)',
                   'guest_mods': 'type: list of strings, default: [], help: List of new modalities for the new_sequences test',
                   'guest_images_dir': 'type: string, default: "", help: Guest-images directory. For test using guest modalities.',
                   'guest_labels_dir': 'type: string, default: "", help: Guest labels directory. For test using guest modalities.',
                   'nifti_path': 'type: string, default: None, required: False, '
                                 'help: Path to a nifti file that we borrow the affine transform and the header from if'
                                 'npy is the data format, to create volumes and save them as niftis. '
                                 'Only used when instancing OfflineVolume class.'

                   }

        sorted_options = dict(sorted(options.items(), key=lambda x: x[0].lower()))
        for key, value in sorted_options.items():
             message += "|%s%s\t%s%s|\n" %(key, "-"*(max_char_name-len(key)), value, "-"*(max_char_value-len(value)))

        message += "-"*(max_char_name + max_char_value + 3)
        print(message)
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)

        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--which_epoch', type=str, default='latest',
                            help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--lesion', type=str, default='', help="Path to test the lesion included images")
        parser.add_argument('--mod_classifier', type=int, default=None, help="Epoch of the modality classifier,"
                                                                             "stored in the checkpoints folder and"
                                                                             "named X_net_MD.pth")
        parser.set_defaults(preprocess_mode='scale_width_and_crop', crop_size=256, load_size=256, display_winsize=256)
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(no_flip=True)
        parser.set_defaults(phase='test')
        parser.add_argument('--mod_disc_dir', type=str, default=None,
                            help="Optional path specified for the modality discriminator."
                                 "Otherwise, checkpoints is picked. ")
        parser.add_argument('--mode', type=str, default='eval', help="Mode to put the network in. ")
        parser.add_argument('--guest_mods', type=str, nargs='*', default=[],
                            help="List of new modalities for the new_sequences test")
        parser.add_argument('--guest_images_dir', type=str, default='',
                            help="Guest-images directory. ")
        parser.add_argument('--guest_labels_dir', type=str, default='',
                            help="Guest-labels directory. ")
        parser.add_argument('--second_style_slice', action = 'store_true',
                            help="")
        parser.add_argument('--nifti_path', type=str, default=None, required=False,
                            help="Path to a nifti file that we borrow the affine transform and the header from if"
                                 "npy is the data format, to create volumes and save them as niftis. "
                                 "Only used when instancing OfflineVolume class.")
        self.isTrain = False
        return parser

class SamplingOptions(TestOptions):
    def initialize(self, parser):
        super(SamplingOptions, self).initialize(parser)
        parser.add_argument('--store_dir', type=str, help='Directory in which the images are to be stored.')
        parser.add_argument('--max_glevel', type=int,required = True, help = "Maximum granularity level")
        parser.add_argument('--parc_dir', type=str, help = "Root of all directories containing parcellation files."
                                                           "The parcellation files will be root_i, where"
                                                           "i is the granularity level.")
        parser.add_argument('--style_dir', type = str, nargs ='*', default = None,
                            help = "List of style directories: either it's one folder containing the"
                                   "style images, or a folder per modality.")
        parser.add_argument('--style_dir_mask', type=str, help = "Path to the mask volumes "
                                                                         "corresponding to the style images.")
        parser.add_argument('--skip_gt', default = False, type = bool, help = "If true, ground truth images"
                                                                              "aren't saved.")
        parser.add_argument('--store_root_gt', default = 'gt', type = str,
                            help = "Root of the ground truth images, to which modality names will be appended.")
        parser.add_argument('--store_root_lab', default = 'labels', type = str,
                            help = "Root of the ground truth images, to which granularity levels will be appended.")
        parser.add_argument('--store_same', default = False, type = bool,
                            help = "Whether the ground truth images are separated into different folders or not.")
        parser.add_argument('--use_mod_disc', default = False, type = bool,
                            help = "Whether to use the modality ")
        parser.add_argument('--mod_disc_th', type = float, default = 0.85,
                            help = "Accuracy threshold at which we consider the resulting image to be accurate")
        parser.add_argument('--mod_disc_attempts', type = int, default = 50,
                            help = "Number of attempts until modality discrimination constraint is applied.")
        parser.add_argument('--allowed_styles', type = str, nargs ='*', default = None,
                            help = "List of styles that you want ot consider for the style images."
                                   "Must be embedded in the image files name.")
        parser.add_argument('--sample_from', type = str, default = None, help = "Directory from which you want to base new slices."
                                                                                "Must be a directory with png files (NAME_slice.png")
        parser.add_argument('--lesion_ids_ind', type = int, nargs = '*', default = [], help = "Indices of the channels of the PV maps of the lesions"
                                                                                               "that need to be saved")
        parser.add_argument('--lesion_ids_names', type = str, nargs = '*', default = [], help = "Names of the lesions corresponding to those indices")

        return parser

class TuringTestOptions():
    def help(self):
        max_char_name =  30
        max_char_value = 300
        message = ''
        message += 'TestOptions%s\n' %("-"*(max_char_name + max_char_value - 10))
        options = {'image_dir_te': "type: Synthetic or target image directory.",
                   'image_dir_tr': "type: Real or source image directory.",
                   'label_dir_tr': "type: Real or source label directory.",
                   'label_dir_te': "type: Synthetic or target label directory.",
                   'volume': "Whether images are volumetric or not.",
                   'sequences': "List of modalities.",
                   'datasets': 'List of datasets',
                   'max_dataset_size': "Maximum dataset size.",
                   'n_images': "How many images to show for each modality",
                   'use_augmentation': "Whether to use augmentation",
                   'batchSize': "Number of images per batch",
                   'num_workers': "Number of workers",
                   'results_file': "Path to the results file",
                   'skullstrip': "Whether to skullstrip the image"
                   }

        sorted_options = dict(sorted(options.items(), key=lambda x: x[0].lower()))
        for key, value in sorted_options.items():
             message += "|%s%s\t%s%s|\n" %(key, "-"*(max_char_name-len(key)), value, "-"*(max_char_value-len(value)))

        message += "-"*(max_char_name + max_char_value + 3)
        print(message)

    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--image_dir_te', type=str,  help='Synthetic or target image directory')
        parser.add_argument('--image_dir_tr', type=str, help='Real or source image directory.')
        parser.add_argument('--label_dir_tr', type=str, help='Real or source label directory.')
        parser.add_argument('--label_dir_te', type=str, help='Synthetic or target label directory.')
        parser.add_argument('--volume', action = 'store_true', help = "Whether images are volumetric or not")
        parser.add_argument('--sequences', type = str, nargs='*', help = "List of modalities")
        parser.add_argument('--datasets', type=str, nargs='*', help="List of datasets")
        parser.add_argument('--max_dataset_size', type=int, default=10000,  help='Maximum dataset size.')
        parser.add_argument('--n_images', type = int, default = 300, help = "How many images to show for each modality")
        parser.add_argument('--use_augmentation', action = 'store_true', help = "Whether to use augmentation")
        parser.add_argument('--batchSize', type = int, default = 8, help = "Number of images per batch")
        parser.add_argument('--num_workers', type = int, default = 0, help = "Number of workers")
        parser.add_argument('--results_dir', type=str, help = "Results directory")
        parser.add_argument('--skullstrip', action = 'store_true', help = "Whether to skull strip image")

        self.isTrain = False
        return parser


class FullPipelineOptions(BaseOptions):

    def load_options(self, file, dataset_type = 'sliced'):

        new_opt = pickle.load(open(file + '.pkl', 'rb'))

        #Remove training parameters
        new_opt.isTrain = False
        new_opt.semantic_nc = new_opt.label_nc + \
            (1 if new_opt.contain_dontcare_label else 0) + \
            (0)
        # set gpu ids
        str_ids = new_opt.gpu_ids.split(',')
        new_opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                new_opt.gpu_ids.append(id)

        assert len(new_opt.gpu_ids) == 0 or new_opt.batchSize % len(new_opt.gpu_ids) == 0, \
            "Batch size %d is wrong. It must be a multiple of # GPUs %d." \
            % (new_opt.batchSize, len(new_opt.gpu_ids))

        if new_opt.phase == 'train' and new_opt.batchSize < 4:
            new_opt.latent_triplet_loss = False
            print("Triplet loss deactivated because batch size is insufficient.")


        if dataset_type == 'volume' and new_opt.dataset_type == 'sliced':
            # Volumetric dataset - specific parameters.
            new_opt.dataset_type = 'volume'
            new_opt.intensify_lesions = 0
            new_opt.store_and_use_slices = True
            new_opt.lesion_sampling_mode = 'threshold'
            new_opt.threshold_low = 100
            new_opt.ruleout_offset = 0.25
            new_opt.sample_lesions = False
            new_opt.continuous_slices = True

        return new_opt

