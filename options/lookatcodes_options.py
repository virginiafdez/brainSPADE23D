""""""

from .base_options import BaseOptions


class LACodeOptions(BaseOptions):
    def help(self):
        max_char_name =  30
        max_char_value = 300
        BaseOptions().help()
        message = ''
        title = "LOOK-AT-CODE TEST OPTIONS"
        message += '%s%s\n' %(title, "-"*(max_char_name + max_char_value - 3 - len(title)))
        options = {
            'GENERAL': {
                'savecode_dir': 'type: string, default: ./savecode_dir, help: saves results in this folder',
                'which_epoch': 'type: stirng, default: latest, help: which epoch to load? set to latest to use latest cached model',
                'do_reduction': 'type: flag, help: Do reduction test (projection to a 2D space with TSNE or PCA)',
                'do_heatmaps': 'type: flag, help: Do heatmaps test. Removing element of the code and decoding to see impact',
                'do_sparsity': 'type: flag, help: calcualte mean sparsity of latent space vectors',
                'do_volume': 'type: flag, help: plot codes  slice-by-slice of a volume',
                'do_new_seq': 'type: flag, help: plot latent space of unseen modalities',
                'mode': 'type: string, help: eval or train'
             },

            'REDUCTION_TEST' : {
                'reduction_algorithm': 'type: string, default: pca, help: tsne or pca',
                'reduction_itemizing': 'type: flag, help: In the reduction task, save individual codes plots + images with the image code colored.'
            },

            'NEW_SEQ_EXPERIMENT': {
                'guest_mods': 'type: list of strings, help: List of additional modalities',
                'guest_images_dir': 'type: string, default: "", help: List of image directory with guest modalities',
                'guest_labels_dir': 'type: string, default: "", help: List of label directory with guest modalities',
                'guest_label': 'type: string, default: guest, help: dataset identifying the filenames of the guest modality files',

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
        BaseOptions.initialize(self, parser)
        parser.add_argument('--savecode_dir', type=str, default='./savecode_dir/', help='saves results here.')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--whole_volume', type=str, default='', help="Path to test whole 3D volume nii file")
        parser.add_argument('--do_reduction', action='store_true',  help="Flag. Run reduction test.")
        parser.add_argument('--do_heatmaps', action='store_true', help="Flag. Do heatmaps removing each element of"
                                                                         "code.")
        parser.add_argument('--do_sparsity', action='store_true', help="Flag. Calculate mean sparsity of vectors"
                                                                         "(T1 and FlAIR and both)")
        parser.add_argument('--reduction_algorithm', type = str, default = 'pca', help = 'tsne or pca')
        parser.add_argument('--do_volume', action='store_true', help = "Plot codes equivalent to whole volume")
        parser.add_argument('--do_new_seq', action='store_true', help = "Flag. Do code plots with additional sequences.")
        parser.add_argument('--do_std', action='store_true', help="Look at standard deviation across generated codes.")
        parser.add_argument('--guest_mods', type=str, nargs='*', default = [], help = 'List of additional modalities')
        parser.add_argument('--guest_images_dir', type=str, default="", help='List of image directory with guest modalities')
        parser.add_argument('--guest_labels_dir', type=str, default="", help='List of label directory with guest modalities')
        parser.add_argument('--guest_label', type=str, default="guest",
                            help='Label identifying the "guest" dataset')
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(phase='test')
        parser.add_argument('--second_style_slice', action='store_true', help="Loss similar to contrastive between style of "
                                                                              "diffrent slices.")
        parser.add_argument('--mod_disc_dir', type = str, default=None, help = "Optional path specified for the modality discriminator."
                                                                               "Otherwise, checkpoints is picked. ")
        parser.add_argument('--mode', type =str, default = 'eval', help = "Mode to put the network in. ")
        parser.add_argument('--reduction_itemizing', action='store_true',
                            help="In the reduction task, save individual codes plots + images with the image code colored.")

        self.isTrain = False
        return parser
