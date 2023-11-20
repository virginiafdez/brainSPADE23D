"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from .base_options import BaseOptions
import pickle

class TuringTestOptions(BaseOptions):
    def help(self):
        BaseOptions().help()
        max_char_name =  30
        max_char_value = 300
        message = ''
        message += 'TestOptions%s\n' %("-"*(max_char_name + max_char_value - 10))
        options = {'results_dir': "type: string; default: ./results/; help: saves results here.",
                   'which_epocch': 'type: string, default: latest, help: which epoch to load? set to latest to use latest cached model',
                   'fake_label_dir': 'type, string, help = Directory containing fake NPZ labels to generate images with.'
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
        parser.set_defaults(preprocess_mode='scale_width_and_crop', crop_size=256, load_size=256, display_winsize=256)
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(no_flip=True)
        parser.set_defaults(phase='test')
        parser.add_argument('--fake_label_dir', type = str, required = True, help = "Directory containing fake NPZ"
                                                                                    "labels to generate images with.")
        parser.add_argument('--second_style_slice', action='store_true', help = "Second style slice")
        self.isTrain = False
        return parser

