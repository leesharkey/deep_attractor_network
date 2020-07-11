import argparse
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import lib.networks.models as models
from lib.data import SampleBuffer, Dataset
import lib.utils
import lib.managers as managers


def calc_enrg_masks(args):
    m = [divl(args.state_sizes[0][1:], x[1:]).item() for x in args.state_sizes]
    print(m)
    return m

def dict_len_check(args):
    # check the dicts are the right len
    num_layers = len(args.state_sizes)
    for l in range(num_layers):
        inps           = args.arch_dict['mod_connect_dict'][l]
        cct_statuses   = args.arch_dict['mod_cct_status_dict'][l]
        base_kern_pads = args.arch_dict['base_kern_pad_dict'][l]

        if not len(cct_statuses)==len(inps) or not \
            len(inps)==len(base_kern_pads):
            str1 = "Layer %i architecture dictionaries invalid. " % l
            raise ValueError(str1 +
                             "cct_statuses, inp_state_shapes, and " +
                             "base_kern_pads must be the same length. Check " +
                             "that the architecture dictionary defines these "+
                             "correctly.")


def finalize_args(parser):

    args = parser.parse_args()

    # Generate random args, if any
    if args.randomize_args is not []:
        args = lib.utils.random_arg_generator(parser, args)

    # Determine the correct device
    vars(args)['use_cuda'] = args.use_cuda and torch.cuda.is_available()

    # Give a very short description of what is special about this run
    if args.require_special_name:
        vars(args)['special_name'] = input("Special name: ") or "None"

    # Set architecture-specific hyperparams
    if args.network_type == 'BengioFischer':
        if args.architecture == 'BFN_small_4_layers':
                vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                             [args.batch_size, 500],
                                             [args.batch_size, 100],
                                             [args.batch_size, 10]]

        elif args.architecture == 'BFN_med_4_layers':
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size, 500],
                                         [args.batch_size, 500],
                                         [args.batch_size, 200]]

        elif args.architecture == 'BFN_large_5_layers':
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size, 1000],
                                         [args.batch_size, 1000],
                                         [args.batch_size, 300],
                                         [args.batch_size, 300]]
    if args.network_type == 'ConvBFN':
        if args.architecture == 'ConvBFN_med_6_layers':
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size, 64, 28, 28],
                                         [args.batch_size, 64, 28, 28],
                                         [args.batch_size, 64, 28, 28],
                                         [args.batch_size, 16, 12, 12],
                                         [args.batch_size, 16, 12, 12]]

            mod_connect_dict = {0: [],
                                1: [0],
                                2: [0,1],
                                3: [1,2,3],
                                4: [3],
                                5: [4]}
            mod_kernel_dict = {0: [],
                                1: [3],
                                2: [3,3],
                                3: [3,3,3,3],
                                4: [7],
                                5: [3]}
            mod_padding_dict = {0: [],
                                1: [1],
                                2: [1,1],
                                3: [1,1,1,1],
                                4: [1],
                                5: [1]}
            mod_strides_dict = {0: [],
                                1: [1],
                                2: [1,1],
                                3: [1,1,1,1],
                                4: [2],
                                5: [1]}

            vars(args)['arch_dict'] = {'num_ch': 32,
                                       'num_ch_initter': 32,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': mod_kernel_dict,
                                       'strides': mod_strides_dict,
                                       'padding': mod_padding_dict,
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 32}
            vars(args)['energy_weight_mask'] = calc_enrg_masks(args)

        elif args.architecture == 'ConvBFN_med_2_dense_3layers_strides':
            #Messed this one up. Keeping as is for posterity.
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size, 16, 28, 28],
                                         [args.batch_size, 16, 28, 28],
                                         [args.batch_size, 16, 28, 28],
                                         [args.batch_size, 16, 12, 12],
                                         [args.batch_size, 16, 12, 12],
                                         [args.batch_size, 16, 12, 12],
                                         ]

            mod_connect_dict = {0: [],
                                1: [0],
                                2: [0,1],
                                3: [0,1,2],
                                4: [3],
                                5: [3,4],
                                6: [3,4,5]}
            mod_kernel_dict = {0: [],
                                1: [3],
                                2: [3,3],
                                3: [3,3,3,3],
                                4: [7],
                                5: [7,3],
                                6: [7,3,3]}
            mod_padding_dict = {0: [],
                                1: [1],
                                2: [1,1],
                                3: [1,1,1],
                                4: [1],
                                5: [1,1],
                                6: [1,1,1]}
            mod_strides_dict = {0: [],
                                1: [1],
                                2: [1,1],
                                3: [1,1,1,1],
                                4: [2],
                                5: [2,1],
                                6: [2,1,1]}

            vars(args)['arch_dict'] = {'num_ch': 32,
                                       'num_ch_initter': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': mod_kernel_dict,
                                       'strides': mod_strides_dict,
                                       'padding': mod_padding_dict,
                                       'mod_connect_dict': mod_connect_dict}
            vars(args)['energy_weight_mask'] = calc_enrg_masks(args)
        elif args.architecture == 'ConvBFN_med_3_dense_3layers_base':#Untested and incomplete
            #Messed this one up. Keeping as is for posterity.
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size, 64, 22, 22],
                                         [args.batch_size, 16, 22, 22],
                                         [args.batch_size, 16, 22, 22],
                                         [args.batch_size, 16, 22, 22],
                                         [args.batch_size, 16, 16, 16],
                                         [args.batch_size, 16, 16, 16],
                                         [args.batch_size, 16, 16, 16],
                                         [args.batch_size, 16, 10, 10],
                                         [args.batch_size, 16, 10, 10],
                                         [args.batch_size, 16, 10, 10],
                                         ]
            mod_connect_dict = {0: [],
                                1: [0],
                                2: [1],
                                3: [1,2],
                                4: [1,2,3],
                                5: [4],
                                6: [4,5],
                                7: [4,5,6],
                                8: [7],
                                9: [7, 8],
                                10: [7, 8, 9]
                                }
            mod_kernel_dict = {0: [],
                               1: [7],
                                2: [3],
                                3: [3,3],
                                4: [3,3,3,3],
                                5: [7],
                                6: [7,3],
                                7: [7,3,3],
                                8: [7],
                                9: [7,3],
                                10: [7,3,3]}
            mod_padding_dict = {0: [],
                                1: [0],
                                2: [1],
                                3: [1,1],
                                4: [1,1,1],
                                5: [0],
                                6: [0,1],
                                7: [0,1,1],
                                8: [0],
                                9: [0, 1],
                                10:[0, 1, 1]
                                }
            mod_strides_dict = {0: [],
                                1: [1],
                                2: [1],
                                3: [1,1],
                                4: [1,1,1],
                                5: [1],
                                6: [1,1],
                                7: [1,1,1],
                                8: [1],
                                9: [1, 1],
                                10: [1, 1, 1]}

            vars(args)['arch_dict'] = {'num_ch': 32,
                                       'num_ch_initter': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': mod_kernel_dict,
                                       'strides': mod_strides_dict,
                                       'padding': mod_padding_dict,
                                       'mod_connect_dict': mod_connect_dict}
            # vars(args)['energy_weight_mask'] = calc_enrg_masks(args)
        elif args.architecture == 'ConvBFN_small_3_layers':
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size, 32, 22, 22],
                                         [args.batch_size, 32, 16, 16],
                                         [args.batch_size, 32, 10, 10]
                                         ]

            mod_connect_dict = {0: [],
                                1: [0],
                                2: [1],
                                3: [2]
                                }
            mod_kernel_dict = {0: [],
                                1: [7],
                                2: [7],
                                3: [7]
                                }
            mod_padding_dict = {0: [],
                                1: [0],
                                2: [0],
                                3: [0]
                                }
            mod_strides_dict = {0: [],
                                1: [1],
                                2: [1],
                                3: [1]
                                }

            vars(args)['arch_dict'] = {'num_ch': 32,
                                       'num_ch_initter': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': mod_kernel_dict,
                                       'strides': mod_strides_dict,
                                       'padding': mod_padding_dict,
                                       'mod_connect_dict': mod_connect_dict,
                                       'spec_norm_reg': False}
            vars(args)['energy_weight_mask'] = calc_enrg_masks(args)
        elif args.architecture == 'ConvBFN_small_4_layers':
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                         [args.batch_size, 32, 22, 22],
                                         [args.batch_size, 32, 16, 16],
                                         [args.batch_size, 32, 10, 10],
                                         [args.batch_size, 32, 4, 4]# Think this layer causes box artefacts in the image
                                         ]

            mod_connect_dict = {0: [],
                                1: [0],
                                2: [1],
                                3: [2],
                                4: [3]
                                }
            mod_kernel_dict = {0: [],
                               1: [7],
                               2: [7],
                               3: [7],
                               4: [7],
                               }
            mod_padding_dict = {0: [],
                                1: [0],
                                2: [0],
                                3: [0],
                                4: [0]
                                }
            mod_strides_dict = {0: [],
                                1: [1],
                                2: [1],
                                3: [1],
                                4: [1]
                                }

            vars(args)['arch_dict'] = {'num_ch': 32,
                                       'num_ch_initter': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': mod_kernel_dict,
                                       'strides': mod_strides_dict,
                                       'padding': mod_padding_dict,
                                       'mod_connect_dict': mod_connect_dict}
            vars(args)['energy_weight_mask'] = calc_enrg_masks(args)

    if args.network_type == 'VectorField' or args.network_type == 'VFEBMLV':

        if args.architecture == 'VF_small_2_layers_toy':
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size, 2]]

            mod_connect_dict = {0: [1],
                                1: [0]}

            vars(args)['arch_dict'] = {'num_ch': 64,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [3, 3, 3],
                                       'strides': [1,1],
                                       'padding': 1,
                                       'mod_connect_dict': mod_connect_dict}
            vars(args)['energy_weight_mask'] = [1.0, 8.0, 32.0, 36, 144.0]

        elif args.architecture == 'VF_small_3_layers':
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size, 100],
                                         [args.batch_size, 50]]

            mod_connect_dict = {0: [1],
                                1: [0,2],
                                2: [1]} # no self connections, just a FF-like net

            vars(args)['arch_dict'] = {'num_ch': 32,
                                       'num_ch_initter': 32,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [[3, 3],
                                                        [3, 3],
                                                        [3, 3]],
                                       'strides': [1,1],
                                       'padding': [[1, 1],
                                                   [1, 1], [1, 1],
                                                   [1, 1], [1, 1],
                                                   [1, 1]],
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 32}

            vars(args)['energy_weight_mask'] = calc_enrg_masks(args)

        elif args.architecture == 'VF_largelayer1_3_layers_for_EMasktesting':
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size, 4000],
                                         [args.batch_size, 16]]

            mod_connect_dict = {0: [1],
                                1: [0,2],
                                2: [1]} # no self connections, just a FF-like net

            vars(args)['arch_dict'] = {'num_ch': 32,
                                       'num_ch_initter': 32,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [[3, 3],
                                                        [3, 3],
                                                        [3, 3]],
                                       'strides': [1,1],
                                       'padding': [[1, 1],
                                                   [1, 1], [1, 1],
                                                   [1, 1], [1, 1],
                                                   [1, 1]],
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 32}

            vars(args)['energy_weight_mask'] = calc_enrg_masks(args)

        elif args.architecture == 'VF_largelayer1_3_layers_for_EMasktesting2':
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size, 4000],
                                         [args.batch_size, 16]]

            mod_connect_dict = {0: [1],
                                1: [0,2],
                                2: [1]}

            vars(args)['arch_dict'] = {'num_ch': 32,
                                       'num_ch_initter': 32,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [[3, 3],
                                                        [3, 3],
                                                        [3, 3]],
                                       'strides': [1,1],
                                       'padding': [[1, 1],
                                                   [1, 1], [1, 1],
                                                   [1, 1], [1, 1],
                                                   [1, 1]],
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 32}

            vars(args)['energy_weight_mask'] = [1., 1., 1.]

        elif args.architecture == 'VF_cifar_med_5_layers':
            vars(args)['state_sizes'] = [[args.batch_size,  3, 32, 32],
                                         [args.batch_size, 1024],
                                         [args.batch_size, 512],
                                         [args.batch_size, 128],
                                         [args.batch_size, 32]
                                         ]

            mod_connect_dict = {0: [1],
                                1: [0,2],
                                2: [1,3],
                                3: [2,4],
                                4: [3]} # no self connections, just a FF-like net

            vars(args)['arch_dict'] = {'num_ch': 64,
                                       'num_ch_initter': 64,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [[3, 3],
                                                        [3, 3],
                                                        [3, 3]],
                                       'strides': [1,1],
                                       'padding': [[1, 1],
                                                   [1, 1], [1, 1],
                                                   [1, 1], [1, 1],
                                                   [1, 1]],
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 64}

            vars(args)['energy_weight_mask'] = calc_enrg_masks(args)

        elif args.architecture == 'VF_small_5_layers':
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size, 256],
                                         [args.batch_size, 128],
                                         [args.batch_size, 64],
                                         [args.batch_size, 32]
                                         ]

            mod_connect_dict = {0: [1],
                                1: [0,2],
                                2: [1,3],
                                3: [2,4],
                                4: [3]} # no self connections, just a FF-like net

            vars(args)['arch_dict'] = {'num_ch': 32,
                                       'num_ch_initter': 32,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [[3, 3],
                                                        [3, 3],
                                                        [3, 3]],
                                       'strides': [1,1],
                                       'padding': [[1, 1],
                                                   [1, 1], [1, 1],
                                                   [1, 1], [1, 1],
                                                   [1, 1]],
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 32}

            vars(args)['energy_weight_mask'] = calc_enrg_masks(args)

        elif args.architecture == 'VF_small_4_layers':
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size, 300],
                                         [args.batch_size, 100],
                                         [args.batch_size, 50]]

            mod_connect_dict = {0: [1],
                                1: [0,2],
                                2: [1,3],
                                3: [2]} # no self connections, just a FF-like net

            vars(args)['arch_dict'] = {'num_ch': 64,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [3, 3, 3],
                                       'strides': [1,1],
                                       'padding': 1,
                                       'mod_connect_dict': mod_connect_dict}
            vars(args)['energy_weight_mask'] = [1.0, 8.0, 32.0, 36, 144.0]

        elif args.architecture == 'VF_small_4_layers_self':
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size, 300],
                                         [args.batch_size, 100],
                                         [args.batch_size, 50]]

            mod_connect_dict = {0: [1],
                                1: [0,1,2],
                                2: [1,2,3],
                                3: [2,3]}

            vars(args)['arch_dict'] = {'num_ch': 64,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [3, 3, 3],
                                       'strides': [1,1],
                                       'padding': 1,
                                       'mod_connect_dict': mod_connect_dict}
            vars(args)['energy_weight_mask'] = [10.45, 1.0, 81.92, 163.84] #incorrect
    if args.network_type == 'SVF':
        if args.architecture == 'SVF_small_3_layers':
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size, 8, 8, 8],
                                         [args.batch_size, 128]]

            mod_connect_dict = {0: [1],
                                1: [0,2],
                                2: [1]}

            vars(args)['arch_dict'] = {'num_ch_initter': 32,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [[3, 3],
                                                        [3, 3],
                                                        [3, 3]],
                                       'strides': [1,1],
                                       'padding': [[1, 1],
                                                   [1, 1], [1, 1],
                                                   [1, 1], [1, 1],
                                                   [1, 1]],
                                       'mod_connect_dict': mod_connect_dict}

            vars(args)['energy_weight_mask'] = calc_enrg_masks(args)

        elif args.architecture == 'SVF_small_flat_4_layers':
            vars(args)['state_sizes'] = [[args.batch_size,   1, 28, 28],
                                         [args.batch_size, 784],
                                         [args.batch_size, 256],
                                         [args.batch_size, 128]]

            mod_connect_dict = {0: [1],
                                1: [0,2],
                                2: [1,3],
                                3: [2]}

            vars(args)['arch_dict'] = {'num_ch_initter': 32,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [[3, 3],
                                                        [3, 3],
                                                        [3, 3],
                                                        [3, 3]],
                                       'strides': [1,1,1],
                                       'padding': [[1, 1],
                                                   [1, 1], [1, 1],
                                                   [1, 1], [1, 1],
                                                   [1, 1]],
                                       'mod_connect_dict': mod_connect_dict}

            vars(args)['energy_weight_mask'] = calc_enrg_masks(args)

        elif args.architecture == 'SVF_med_5_layers':
            vars(args)['state_sizes'] = [[args.batch_size,   1, 28, 28],
                                         [args.batch_size,  32, 28, 28],
                                         [args.batch_size,  32, 16, 16],
                                         [args.batch_size,  32,  8,  8],
                                         [args.batch_size, 128]]

            mod_connect_dict = {0: [1],
                                1: [0,2],
                                2: [1,3],
                                3: [2,4],
                                4: [3,4]}

            vars(args)['arch_dict'] = {'num_ch_initter': 32,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [[3, 3],
                                                        [3, 3],
                                                        [3, 3],
                                                        [3, 3],
                                                        [3, 3],
                                                        [3, 3]],
                                       'strides': [1,1,1,1],
                                       'padding': [[1, 1],
                                                   [1, 1], [1, 1],
                                                   [1, 1], [1, 1],
                                                   [1, 1]],
                                       'mod_connect_dict': mod_connect_dict}

            vars(args)['energy_weight_mask'] = calc_enrg_masks(args)

        elif args.architecture == 'SVF_med_6_layers_flat_base':
            vars(args)['state_sizes'] = [[args.batch_size,   1, 28, 28],
                                         [args.batch_size, 784],
                                         [args.batch_size,  32, 28, 28],
                                         [args.batch_size,  32, 16, 16],
                                         [args.batch_size,  32,  8,  8],
                                         [args.batch_size, 128]]

            mod_connect_dict = {0: [1],
                                1: [0,2],
                                2: [1,3],
                                3: [2,4],
                                4: [3,4]}

            vars(args)['arch_dict'] = {'num_ch_initter': 32,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [[3, 3],
                                                        [3, 3],
                                                        [3, 3],
                                                        [3, 3],
                                                        [3, 3],
                                                        [3, 3]],
                                       'strides': [1,1,1,1],
                                       'padding': [[1, 1],
                                                   [1, 1], [1, 1],
                                                   [1, 1], [1, 1],
                                                   [1, 1]],
                                       'mod_connect_dict': mod_connect_dict}

            vars(args)['energy_weight_mask'] = calc_enrg_masks(args)

        elif args.architecture == 'SVF_small_flat_4_layers_experimental': #untested
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                         [args.batch_size, 64],
                                         [args.batch_size, 64],
                                         [args.batch_size, 64],
                                         [args.batch_size, 256],
                                         [args.batch_size, 128]]

            mod_connect_dict = {0: [1,2,3],
                                1: [0,1,2,3,4],
                                2: [0,1,2,3,4],
                                3: [0,1,2,3,4],
                                4: [1,2,3,4,5],
                                5: [4,5]}

            vars(args)['arch_dict'] = {'num_ch_initter': 32,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [[3, 3],
                                                        [3, 3],
                                                        [3, 3],
                                                        [3, 3]],
                                       'strides': [1,1,1],
                                       'padding': [[1, 1],
                                                   [1, 1], [1, 1],
                                                   [1, 1], [1, 1],
                                                   [1, 1]],
                                       'mod_connect_dict': mod_connect_dict}

            vars(args)['energy_weight_mask'] = calc_enrg_masks(args)

        elif args.architecture == 'SVF_med_4_layers':
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size,  16, 28, 28],
                                         [args.batch_size, 8, 8, 8],
                                         [args.batch_size, 128]]

            mod_connect_dict = {0: [1],
                                1: [0,2],
                                2: [1,3],
                                3: [2]}

            vars(args)['arch_dict'] = {'num_ch_initter': 32,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [[3, 3],
                                                        [3, 3],
                                                        [3, 3],
                                                        [3, 3]],
                                       'strides': [1,1],
                                       'padding': [[1, 1],
                                                   [1, 1], [1, 1],
                                                   [1, 1], [1, 1],
                                                   [1, 1]],
                                       'mod_connect_dict': mod_connect_dict}

            vars(args)['energy_weight_mask'] = calc_enrg_masks(args)
    if args.network_type == 'EBMLV':
        if args.architecture == 'EBMLV_very_small_4_layers_self':
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size, 16, 8, 8],
                                         [args.batch_size, 100],
                                         [args.batch_size, 50]]

            mod_connect_dict = {0: [0,1],
                                1: [0,1,2],
                                2: [1,2,3],
                                3: [2,3]}

            vars(args)['arch_dict'] = {'num_ch': 16,
                                       'num_ch_initter': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [[3, 3], [3, 3], [3, 3]],
                                       'strides': [1,1,1],
                                       'padding': [[1,1], [1,1], [1,1]],
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 16}
            vars(args)['energy_weight_mask'] = calc_enrg_masks(args)
            #[1.0, 0.784, 7.84, 15.68] [1.0, 1.0, 1.0, 1.0]

        elif args.architecture == 'EBMLV_very_small_4_layers_topself':
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size, 16, 8, 8],
                                         [args.batch_size, 100],
                                         [args.batch_size, 50]]

            mod_connect_dict = {0: [1],
                                1: [0,2],
                                2: [1,3],
                                3: [2,3]}

            vars(args)['arch_dict'] = {'num_ch': 32,
                                       'num_ch_initter': 32,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [[3, 3],
                                                        [3, 3],
                                                        [3, 3],
                                                        [3, 3]],
                                       'strides': [1, 1, 1],
                                       'padding': [[1, 1],
                                                   [1, 1],
                                                   [1, 1],
                                                   [1, 1]],
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 32}
            vars(args)['energy_weight_mask'] = calc_enrg_masks(args)
            #[1.0, 0.765, 7.84, 15.68] [1.0, 1.0, 1.0, 1.0]
        elif args.architecture == 'EBMLV_small_4_layers_topself':
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                         [args.batch_size, 16, 28, 28],
                                         [args.batch_size, 8, 8, 8],
                                         [args.batch_size, 50]]

            mod_connect_dict = {0: [1],
                                1: [0, 2],
                                2: [1, 3],
                                3: [2, 3]}

            vars(args)['arch_dict'] = {'num_ch': 32,
                                       'num_ch_initter': 32,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [[3, 3],
                                                        [3, 3],
                                                        [3, 3],
                                                        [3, 3]],
                                       'strides': [1, 1, 1],
                                       'padding': [[1, 1],
                                                   [1, 1],
                                                   [1, 1],
                                                   [1, 1]],
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 32}
            vars(args)['energy_weight_mask'] = calc_enrg_masks(args)

        elif args.architecture == 'EBMLV_small_4_layers_topself':
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                         [args.batch_size, 16, 28, 28],
                                         [args.batch_size, 8, 8, 8],
                                         [args.batch_size, 50]]

            mod_connect_dict = {0: [1],
                                1: [0, 2],
                                2: [1, 3],
                                3: [2, 3]}

            vars(args)['arch_dict'] = {'num_ch': 16,
                                       'num_ch_initter': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [[3, 3],
                                                        [3, 3],
                                                        [3, 3],
                                                        [3, 3]],
                                       'strides': [1, 1, 1],
                                       'padding': [[1, 1],
                                                   [1, 1],
                                                   [1, 1],
                                                   [1, 1]],
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 32}
            vars(args)['energy_weight_mask'] = calc_enrg_masks(args)
    if args.network_type == 'DAN':
        if args.architecture == 'DAN_small_2_layers_allself':
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size, 128]]

            mod_connect_dict = {0: [0,1],
                                1: [0,1]}
            vars(args)['arch_dict'] = {'num_ch': 16,
                                       'num_ch_initter': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [[3, 3],
                                                        [3, 3],
                                                        [3, 3],
                                                        [3, 3]],
                                       'strides': [1, 1],
                                       'padding': [[1, 1],
                                                   [1, 1],
                                                   [1, 1],
                                                   [1, 1]],
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 64}

            vars(args)['energy_weight_mask'] = calc_enrg_masks(args)
        elif args.architecture == 'DAN_small_2_layers_convself':
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size,  16, 28, 28]]

            mod_connect_dict = {0: [0,1],
                                1: [0,1]}
            vars(args)['arch_dict'] = {'num_ch': 32,
                                       'num_ch_initter': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [[3, 3],
                                                        [3, 3],
                                                        [3, 3],
                                                        [3, 3]],
                                       'strides': [1, 1],
                                       'padding': [[1, 1],
                                                   [1, 1],
                                                   [1, 1],
                                                   [1, 1]],
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 64}

            vars(args)['energy_weight_mask'] = calc_enrg_masks(args)
        elif args.architecture == 'DAN_small_3_layers_allself':
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size, 1, 28, 28],
                                         [args.batch_size, 128]]

            mod_connect_dict = {0: [0,1],
                                1: [0,1,2],
                                2: [1,2]}
            vars(args)['arch_dict'] = {'num_ch': 32,
                                       'num_ch_initter': 32,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [[3, 3],
                                                        [3, 3],
                                                        [3, 3],
                                                        [3, 3]],
                                       'strides': [1, 1],
                                       'padding': [[1, 1],
                                                   [1, 1],
                                                   [1, 1],
                                                   [1, 1]],
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 64}

            vars(args)['energy_weight_mask'] = calc_enrg_masks(args)

        elif args.architecture == 'DAN_smallish_3_layers_allself':
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size, 16, 28, 28],
                                         [args.batch_size, 128]]

            mod_connect_dict = {0: [0,1],
                                1: [0,1,2],
                                2: [1,2]}
            vars(args)['arch_dict'] = {'num_ch': 2,
                                       'num_ch_initter': 2,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [[3, 3],
                                                        [3, 3],
                                                        [3, 3],
                                                        [3, 3]],
                                       'strides': [1, 1],
                                       'padding': [[1, 1],
                                                   [1, 1],
                                                   [1, 1],
                                                   [1, 1]],
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 64}

            vars(args)['energy_weight_mask'] = calc_enrg_masks(args)

        elif args.architecture == 'DAN_smallish_3_layers_topself':
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size, 16, 28, 28],
                                         [args.batch_size, 128]]

            mod_connect_dict = {0: [1],
                                1: [0,2],
                                2: [1,2]}
            vars(args)['arch_dict'] = {'num_ch': 32,
                                       'num_ch_initter': 32,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [[3, 3],
                                                        [3, 3],
                                                        [3, 3],
                                                        [3, 3]],
                                       'strides': [1, 1],
                                       'padding': [[1, 1],
                                                   [1, 1],
                                                   [1, 1],
                                                   [1, 1]],
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 64}

            vars(args)['energy_weight_mask'] = calc_enrg_masks(args)

        elif args.architecture == 'DAN_very_small_4_layers_new_selftop_larger_initr':
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size, 16, 8, 8],
                                         [args.batch_size, 100],
                                         [args.batch_size, 50]]

            mod_connect_dict = {0: [1],
                                1: [0,2],
                                2: [1,3],
                                3: [2,3]}

            vars(args)['arch_dict'] = {'num_ch': 16,
                                       'num_ch_initter': 64,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [[3, 3],
                                                        [3, 3],
                                                        [3, 3],
                                                        [3, 3]],
                                       'strides': [1, 1],
                                       'padding': [[1, 1],
                                                   [1, 1],
                                                   [1, 1],
                                                   [1, 1]],
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 16}
            vars(args)['energy_weight_mask'] = [1.0, 0.784, 7.84, 15.68]

        elif args.architecture == 'DAN_very_small_4_layers_new_btself_larger_initr':
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size, 16, 8, 8],
                                         [args.batch_size, 100],
                                         [args.batch_size, 50]]

            mod_connect_dict = {0: [0,1],
                                1: [0,2],
                                2: [1,3],
                                3: [2,3]}

            vars(args)['arch_dict'] = {'num_ch': 16,
                                       'num_ch_initter': 64,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [[3, 3],
                                                        [3, 3],
                                                        [3, 3],
                                                        [3, 3]],
                                       'strides': [1, 1],
                                       'padding': [[1, 1],
                                                   [1, 1],
                                                   [1, 1],
                                                   [1, 1]],
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 16}
            vars(args)['energy_weight_mask'] = [1.0, 0.784, 7.84, 15.68]

        if args.architecture == 'DAN_small_4_layers_experimental':
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size, 1, 56, 56],
                                         [args.batch_size, 16, 16, 16],
                                         [args.batch_size, 32, 8, 8],
                                         [args.batch_size, 100],
                                         [args.batch_size, 50]]

            mod_connect_dict = {0: [1],
                                1: [0,1,2],
                                2: [1,2,3],
                                3: [1, 2, 4],
                                4: [2, 4],
                                5: [0,1,2,3,4]} # no self connections, just a FF-like net

            vars(args)['arch_dict'] = {'num_ch': 64,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [3, 3, 3],
                                       'strides': [1,1],
                                       'padding': 1,
                                       'mod_connect_dict': mod_connect_dict}
            vars(args)['energy_weight_mask'] = [1.0, 8.0, 32.0, 36, 144.0]

        elif args.architecture == 'DAN_very_small_3_layers':
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size, 100],
                                         [args.batch_size, 50]]

            mod_connect_dict = {0: [1],
                                1: [0,2],
                                2: [1,3]}
            vars(args)['arch_dict'] = {'num_ch': 16,
                                       'num_ch_initter': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [[3, 3], [3, 3], [3, 3]],
                                       'strides': [1,1,1],
                                       'padding': [[1,1], [1,1], [1,1]],
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 16}
            vars(args)['energy_weight_mask'] = calc_enrg_masks(args)

        elif args.architecture == 'DAN_very_small_4_layers_selftop':
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size, 16, 8, 8],
                                         [args.batch_size, 100],
                                         [args.batch_size, 50]]

            mod_connect_dict = {0: [1],
                                1: [0,2],
                                2: [1,3],
                                3: [2,3]}

            vars(args)['arch_dict'] = {'num_ch': 16,
                                       'num_ch_initter': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [[3, 3], [3, 3], [3, 3]],
                                       'strides': [1,1,1],
                                       'padding': [[1,1], [1,1], [1,1]],
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 16}
            vars(args)['energy_weight_mask'] = [1.0, 0.784, 7.84, 15.68]

        elif args.architecture == 'DAN_very_small_4_layers_selftop_larger_initr':
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size, 16, 8, 8],
                                         [args.batch_size, 100],
                                         [args.batch_size, 50]]

            mod_connect_dict = {0: [1],
                                1: [0,2],
                                2: [1,3],
                                3: [2,3]}

            vars(args)['arch_dict'] = {'num_ch': 16,
                                       'num_ch_initter': 64,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [3, 3, 3],
                                       'strides': [1,1],
                                       'padding': 1,
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 16}
            vars(args)['energy_weight_mask'] = [1.0, 0.784, 7.84, 15.68]

        elif args.architecture == 'DAN_small_4_layers_selftop3_larger_initr':
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size, 16, 8, 8],
                                         [args.batch_size, 100],
                                         [args.batch_size, 50]]

            mod_connect_dict = {0: [1],
                                1: [0,1,2],
                                2: [1,2,3],
                                3: [2,3]}

            vars(args)['arch_dict'] = {'num_ch': 64,
                                       'num_ch_initter': 64,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [3, 3, 3],
                                       'strides': [1,1],
                                       'padding': 1,
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 32}
            vars(args)['energy_weight_mask'] = [1.0, 0.784, 7.84, 15.68]

        elif args.architecture == 'DAN_very_small_4_layers_selftop2':
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size, 16, 8, 8],
                                         [args.batch_size, 100],
                                         [args.batch_size, 50]]

            mod_connect_dict = {0: [1],
                                1: [0,2],
                                2: [1,2,3],
                                3: [2,3]}

            vars(args)['arch_dict'] = {'num_ch': 16,
                                       'num_ch_initter': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [3, 3, 3],
                                       'strides': [1,1],
                                       'padding': 1,
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 16}
            vars(args)['energy_weight_mask'] = [1.0, 0.784, 7.84, 15.68]

        elif args.architecture == 'DAN_very_small_4_layers_selftop_smallworld':
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size, 16, 8, 8],
                                         [args.batch_size, 100],
                                         [args.batch_size, 50],
                                         [args.batch_size, 50]]

            mod_connect_dict = {0: [1],
                                1: [0,2,4],
                                2: [1,3,4],
                                3: [2,3,4],
                                4: [0,1,2,3,4]}

            vars(args)['arch_dict'] = {'num_ch': 16,
                                       'num_ch_initter': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [3, 3, 3],
                                       'strides': [1,1],
                                       'padding': 1,
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 16}
            vars(args)['energy_weight_mask'] = [1.0, 0.784, 7.84, 15.68, 1.]# just 1. for small world layer in order to place soft influence over the rest

        elif args.architecture == 'DAN_small_4_layers':
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size, 32, 16, 16],
                                         [args.batch_size, 100],
                                         [args.batch_size, 50]]

            mod_connect_dict = {0: [1],
                                1: [0,2],
                                2: [1,3],
                                3: [2]} # no self connections, just a FF-like net

            vars(args)['arch_dict'] = {'num_ch': 32,
                                       'num_ch_initter': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [3, 3, 3],
                                       'strides': [1,1],
                                       'padding': 1,
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 64}
            vars(args)['energy_weight_mask'] = [10.45, 1.0, 81.92, 163.84]

        elif args.architecture == 'DAN_small_4_layers_self':
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                         [args.batch_size, 32, 16, 16],
                                         [args.batch_size, 100],
                                         [args.batch_size, 50]]

            mod_connect_dict = {0: [1],
                                1: [0, 1, 2],
                                2: [1, 2, 3],
                                3: [2, 3]}

            vars(args)['arch_dict'] = {'num_ch': 32,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [3, 3, 3],
                                       'strides': [1, 1],
                                       'padding': 1,
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 128}
            vars(args)['energy_weight_mask'] = [1.0, 1.4, 32.0, 36,
                                                144.0]  # WRONG NEEDS FIXING BEFORE USE
        elif args.architecture == 'DAN_med_4_layers_sides':
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                           [args.batch_size, 32, 28, 28],
                                         [args.batch_size, 32, 16, 16],
                                           [args.batch_size, 32, 4, 4],
                                         [args.batch_size, 16, 8, 8],
                                           [args.batch_size, 16, 4, 4],
                                         [args.batch_size, 100]]

            mod_connect_dict = {0: [2,       1],
                                1: [0],
                                2: [0, 4,    3],
                                3: [2],
                                4: [2, 6,    5],
                                5: [4],
                                6: [4, 6]}

            vars(args)['arch_dict'] = {'num_ch': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'num_ch_initter': 16,
                                       'kernel_sizes': [[3, 3],
                                                        [3, 3],
                                                        [3, 3],
                                                        [3, 3],
                                                        [3, 3],
                                                        [3, 3],
                                                        [3, 3],
                                                        [3, 3]],
                                       'strides': [1,1,1,1,1,1],
                                       'padding': [[1,1],
                                                   [1,1],
                                                   [1, 1],
                                                   [1, 1],
                                                   [1,1],
                                                   [1,1],
                                                   [1,1],
                                                   [1,1]],
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 128}
            vars(args)['energy_weight_mask'] = calc_enrg_masks(args)
        elif args.architecture == 'DAN_med_4_layers_smallsides':
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                           [args.batch_size, 32, 4, 4],
                                         [args.batch_size, 32, 16, 16],
                                           [args.batch_size, 32, 4, 4],
                                         [args.batch_size, 16, 8, 8],
                                           [args.batch_size, 16, 4, 4],
                                         [args.batch_size, 100]]

            mod_connect_dict = {0: [2,       1],
                                1: [0],
                                2: [0, 4,    3],
                                3: [2],
                                4: [2, 6,    5],
                                5: [4],
                                6: [4, 6]}

            vars(args)['arch_dict'] = {'num_ch': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'num_ch_initter': 16,
                                       'kernel_sizes': [[3, 3],
                                                        [3, 3],
                                                        [3, 3],
                                                        [3, 3],
                                                        [3, 3],
                                                        [3, 3],
                                                        [3, 3],
                                                        [3, 3]],
                                       'strides': [1,1,1,1,1,1],
                                       'padding': [[1,1],
                                                   [1,1],
                                                   [1, 1],
                                                   [1, 1],
                                                   [1,1],
                                                   [1,1],
                                                   [1,1],
                                                   [1,1]],
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 128}
            vars(args)['energy_weight_mask'] = calc_enrg_masks(args)
        elif args.architecture == 'DAN_med_4_layers_smallconnectedsides':
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                           [args.batch_size, 32, 4, 4],
                                         [args.batch_size, 32, 16, 16],
                                           [args.batch_size, 32, 4, 4],
                                         [args.batch_size, 16, 8, 8],
                                           [args.batch_size, 16, 4, 4],
                                         [args.batch_size, 100]]

            mod_connect_dict = {0: [2,       1],
                                1: [0, 3],
                                2: [0, 4,    3],
                                3: [2, 1, 5],
                                4: [2, 6,    5],
                                5: [4, 3],
                                6: [4, 6]}

            vars(args)['arch_dict'] = {'num_ch': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'num_ch_initter': 16,
                                       'kernel_sizes': [[3, 3],
                                                        [3, 3],
                                                        [3, 3],
                                                        [3, 3],
                                                        [3, 3],
                                                        [3, 3],
                                                        [3, 3],
                                                        [3, 3]],
                                       'strides': [1,1,1,1,1,1],
                                       'padding': [[1,1],
                                                   [1,1],
                                                   [1, 1],
                                                   [1, 1],
                                                   [1,1],
                                                   [1,1],
                                                   [1,1],
                                                   [1,1]],
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 128}
            vars(args)['energy_weight_mask'] = calc_enrg_masks(args)
        elif args.architecture == 'ConvBFN_med_4_layers_convfit': #In progress. Might need to change DAN code so that backwards nets are conv_transposes...
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size, 32, 22, 22],
                                         [args.batch_size, 32, 16, 16],
                                         [args.batch_size, 32, 10, 10]
                                         ]

            mod_connect_dict = {0: [1],
                                1: [0, 2],
                                2: [1, 3],
                                3: [2, 3]}
            # mod_kernel_dict = {0: [],
            #                     1: [7],
            #                     2: [7],
            #                     3: [7]
            #                     }
            # mod_padding_dict = {0: [],
            #                     1: [0],
            #                     2: [0],
            #                     3: [0]
            #                     }

            vars(args)['arch_dict'] = {'num_ch': 32,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'num_ch_initter': 32,
                                       'kernel_sizes': [[7, 3],
                                                        [7, 3],
                                                        [7, 3],
                                                        [7, 3]],
                                       'strides': [1,1,1,1],
                                       'padding': [[0,1], [0,1], [0,1], [0,1]],
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 32}
        elif args.architecture == 'DAN_med_5_layers_nocompress':
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                         [args.batch_size, 16, 28, 28],
                                         [args.batch_size, 16, 28, 28],
                                         [args.batch_size, 16, 28, 28],
                                         [args.batch_size, 16, 28, 28]]

            mod_connect_dict = {0: [1],
                                1: [0, 2],
                                2: [1, 3],
                                3: [2, 4],
                                4: [3,4]}

            vars(args)['arch_dict'] = {'num_ch': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'num_ch_initter': 16,
                                       'kernel_sizes': [[3, 3], [3, 3], [3, 3], [3, 3],[3, 3], [3, 3], [3, 3]],
                                       'strides': [1,1,1,1,1,1,1],
                                       'padding': [[1,1], [1,1], [1,1], [1,1],
                                                   [1,1], [1,1], [1,1]],
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 32}
            vars(args)['energy_weight_mask'] = calc_enrg_masks(args)
        elif args.architecture == 'DAN_med_5_layers':
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                         [args.batch_size, 16, 28, 28],
                                         [args.batch_size, 16, 8, 8],
                                         [args.batch_size, 100],
                                         [args.batch_size, 50]]

            mod_connect_dict = {0: [1],
                                1: [0, 2],
                                2: [1, 3],
                                3: [2, 3, 4],
                                4: [3,4]}

            vars(args)['arch_dict'] = {'num_ch': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'num_ch_initter': 16,
                                       'kernel_sizes': [[3, 3], [3, 3], [3, 3], [3, 3]],
                                       'strides': [1,1,1,1],
                                       'padding': [[1,1], [1,1], [1,1], [1,1]],
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 32}
            vars(args)['energy_weight_mask'] = calc_enrg_masks(args)

        elif args.architecture == 'DAN_med_5_layers_deep':
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                         [args.batch_size, 64, 28, 28],
                                         [args.batch_size, 32, 8, 8],
                                         [args.batch_size, 100],
                                         [args.batch_size, 50]]

            mod_connect_dict = {0: [1],
                                1: [0, 2],
                                2: [1, 3],
                                3: [2, 4],
                                4: [3]}

            vars(args)['arch_dict'] = {'num_ch': 64,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'num_ch_initter': 64,
                                       'kernel_sizes': [[3, 3], [3, 3], [3, 3], [3, 3]],
                                       'strides': [1,1,1,1],
                                       'padding': [[1,1], [1,1], [1,1], [1,1]],
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 32}
            vars(args)['energy_weight_mask'] = calc_enrg_masks(args)

        elif args.architecture == 'DAN_med_5_layers_allself_highcap':
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                         [args.batch_size, 16, 28, 28],
                                         [args.batch_size, 16, 8, 8],
                                         [args.batch_size, 100],
                                         [args.batch_size, 50]]

            mod_connect_dict = {0: [0, 1],
                                1: [0, 1, 2],
                                2: [1, 2, 3],
                                3: [2, 3, 4],
                                4: [3, 4]}

            vars(args)['arch_dict'] = {'num_ch': 32,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'num_ch_initter': 32,
                                       'kernel_sizes': [[3, 3], [3, 3], [3, 3], [3, 3]],
                                       'strides': [1,1,1,1],
                                       'padding': [[1,1], [1,1], [1,1], [1,1]],
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 32}
            vars(args)['energy_weight_mask'] = calc_enrg_masks(args)

        elif args.architecture == 'DAN_med_5_layers_noself':
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                         [args.batch_size, 16, 28, 28],
                                         [args.batch_size, 16, 8, 8],
                                         [args.batch_size, 100],
                                         [args.batch_size, 50]]

            mod_connect_dict = {0: [1],
                                1: [0, 2],
                                2: [1, 3],
                                3: [2, 4],
                                4: [3]}

            vars(args)['arch_dict'] = {'num_ch': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'num_ch_initter': 16,
                                       'kernel_sizes': [[3, 3], [3, 3], [3, 3], [3, 3]],
                                       'strides': [1,1,1,1],
                                       'padding': [[1,1], [1,1], [1,1], [1,1]],
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 32}
            #vars(args)['energy_weight_mask'] = calc_enrg_masks(args)

        elif args.architecture == 'DAN_very_small_5_layers_selftop':
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size, 16, 8, 8],
                                         [args.batch_size, 16, 8, 8],
                                         [args.batch_size, 100],
                                         [args.batch_size, 50]]

            mod_connect_dict = {0: [1],
                                1: [0,2],
                                2: [1,3],
                                3: [2,4],
                                4: [3,4]}

            vars(args)['arch_dict'] = {'num_ch': 16,
                                       'num_ch_initter': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [[3, 3], [3, 3], [3, 3], [3, 3]],
                                       'strides': [1,1,1,1],
                                       'padding': [[1,1], [1,1], [1,1], [1,1]],
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 16}
            vars(args)['energy_weight_mask'] = [1.0, 0.784, 0.784, 7.84, 15.68]
        elif args.architecture == 'DAN_med_5_layers_self':
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size, 16, 28, 28],
                                         [args.batch_size, 16, 8, 8],
                                         [args.batch_size, 100],
                                         [args.batch_size, 50]]

            mod_connect_dict = {0: [0,1],
                                1: [0,1,2],
                                2: [1,2,3],
                                3: [2,3,4],
                                4: [3,4]}

            vars(args)['arch_dict'] = {'num_ch': 16,
                                       'num_ch_initter': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [[3, 3], [3, 3], [3, 3], [3, 3]],
                                       'strides': [1,1,1,1],
                                       'padding': [[1,1], [1,1], [1,1], [1,1]],
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 16}
            vars(args)['energy_weight_mask'] = [1.0, 0.0625, 0.784, 7.84, 15.68]

        elif args.architecture == 'DAN_med_5_layers_selftop':
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size, 32, 16, 16],
                                         [args.batch_size, 32, 8, 8],
                                         [args.batch_size, 100],
                                         [args.batch_size, 50]]

            mod_connect_dict = {0: [1],
                                1: [0,2],
                                2: [1,3],
                                3: [2,4],
                                4: [3,4]}

            vars(args)['arch_dict'] = {'num_ch': 32,
                                       'num_ch_initter': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [3, 3, 3],
                                       'strides': [1,1],
                                       'padding': 1,
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 32}
            vars(args)['energy_weight_mask'] = [1.0, 0.09, 0.383, 7.84, 15.68]
        elif args.architecture == 'DAN_med_5_layers_btself':
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size, 16, 16, 16],
                                         [args.batch_size, 32, 8, 8],
                                         [args.batch_size, 128],
                                         [args.batch_size, 64]]

            mod_connect_dict = {0: [0,1],
                                1: [0,2],
                                2: [1,3],
                                3: [2,4],
                                4: [3,4]}
            vars(args)['arch_dict'] = {'num_ch': 16,
                                       'num_ch_initter': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [[3, 3],
                                                        [3, 3],
                                                        [3, 3],
                                                        [3, 3],
                                                        [3, 3]],
                                       'strides': [1, 1],
                                       'padding': [[1, 1],
                                                   [1, 1], [1, 1],
                                                   [1, 1], [1, 1],
                                                   [1, 1]],
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 16}

            vars(args)['energy_weight_mask'] = [1.0, 0.1914, 0.3828, 6.125, 12.25]

        elif args.architecture == 'DAN_med_4_layers_allself': #Lee want to do on 20200307 after very small 2 l allself
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size, 32, 28, 28],
                                         [args.batch_size, 32, 8, 8],
                                         [args.batch_size, 128]]

            mod_connect_dict = {0: [0,1],
                                1: [0,1,2],
                                2: [1,2,3],
                                3: [2,3]}
            vars(args)['arch_dict'] = {'num_ch': 32,
                                       'num_ch_initter': 32,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [[3, 3],
                                                        [3, 3],
                                                        [3, 3],
                                                        [3, 3]],
                                       'strides': [1, 1],
                                       'padding': [[1, 1],
                                                   [1, 1],
                                                   [1, 1],
                                                   [1, 1]],
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 32}

            vars(args)['energy_weight_mask'] = calc_enrg_masks(args)

        elif args.architecture == 'DAN_large_6_layers_allself':
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size, 32, 28, 28],
                                         [args.batch_size, 32, 16, 16],
                                         [args.batch_size, 32, 8, 8],
                                         [args.batch_size, 128],
                                         [args.batch_size, 64]]

            mod_connect_dict = {0: [0,1],
                                1: [0,1,2],
                                2: [1,2,3],
                                3: [2,3,4],
                                4: [3,4,5],
                                5: [4,5]}
            vars(args)['arch_dict'] = {'num_ch': 32,
                                       'num_ch_initter': 32,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [[3, 3],
                                                        [3, 3],
                                                        [3, 3],
                                                        [3, 3],
                                                        [3, 3]],
                                       'strides': [1, 1],
                                       'padding': [[1, 1],
                                                   [1, 1], [1, 1],
                                                   [1, 1], [1, 1],
                                                   [1, 1]],
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 32}

            vars(args)['energy_weight_mask'] = calc_enrg_masks(args)



        if args.architecture == 'DAN_cifar10_large_5_layers_self':
            vars(args)['state_sizes'] = [[args.batch_size, 3, 32, 32],  # 3072
                                         [args.batch_size, 64, 16, 16], # 16384
                                         [args.batch_size, 64, 10, 10], # 6400
                                         [args.batch_size, 516],
                                         [args.batch_size, 128]]

            mod_connect_dict = {0: [0, 1],
                                1: [0, 1, 2],
                                2: [1, 2, 3],
                                3: [2, 3, 4],
                                4: [3, 4]}

            vars(args)['arch_dict'] = {'num_ch': 64,
                                       'num_ch_initter': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [3, 3, 3],
                                       'strides': [1, 1],
                                       'padding': 1,
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 128}
            vars(args)['energy_weight_mask'] = [1.0, 0.18, 0.48, 5.95, 24.0]

        if args.architecture == 'DAN_cifar10_large_5_layers_top2self_fcconvconnect':
            vars(args)['state_sizes'] = [[args.batch_size, 3, 32, 32],  # 3072
                                         [args.batch_size, 64, 16, 16], # 16384
                                         [args.batch_size, 32, 8, 8], # 6400
                                         [args.batch_size, 516],
                                         [args.batch_size, 128]]

            mod_connect_dict = {0: [1],
                                1: [0, 2],
                                2: [1, 3, 4],
                                3: [2, 3, 4],
                                4: [3, 4]}

            vars(args)['arch_dict'] = {'num_ch': 64,
                                       'num_ch_initter': 64,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [[3,3],
                                                        [3,3],
                                                        [3,3],
                                                        [3,3],
                                                        [3,3]],
                                       'strides': [1, 1],
                                       'padding': [[1,1],
                                                   [1,1],[1,1],
                                                   [1,1],[1,1],
                                                   [1,1]],
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 64}
            vars(args)['energy_weight_mask'] = [1.0, 0.18, 0.48, 5.95, 24.0]



        if args.architecture == 'DAN_cifar10_large_5_layers_btop2self_fcconvconnect':
            vars(args)['state_sizes'] = [[args.batch_size, 3, 32, 32],  # 3072
                                         [args.batch_size, 64, 16, 16], # 16384
                                         [args.batch_size, 32, 8, 8], # 6400
                                         [args.batch_size, 516],
                                         [args.batch_size, 128]]

            mod_connect_dict = {0: [0, 1],
                                1: [0, 2],
                                2: [1, 3, 4],
                                3: [2, 3, 4],
                                4: [3, 4]}

            vars(args)['arch_dict'] = {'num_ch': 64,
                                       'num_ch_initter': 64,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [[3,3],
                                                        [3,3],
                                                        [3,3],
                                                        [3,3],
                                                        [3,3]],
                                       'strides': [1, 1],
                                       'padding': [[1,1],
                                                   [1,1],[1,1],
                                                   [1,1],[1,1],
                                                   [1,1]],
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 64}
            vars(args)['energy_weight_mask'] = [1.0, 0.18, 0.48, 5.95, 24.0]
        if args.architecture == 'DAN_cifar10_5layers_all_self_filtermix':
            vars(args)['state_sizes'] = [[args.batch_size, 3, 32, 32],  # 3072
                                         [args.batch_size, 32, 32, 32],  # 32768
                                         [args.batch_size, 32, 16, 16],  # 8192
                                         [args.batch_size, 32, 8, 8],  # 2048
                                         [args.batch_size, 256]]

            mod_connect_dict = {0: [0, 1],
                                1: [0, 1, 2],
                                2: [1, 2, 3, 4],
                                3: [2, 3, 4],
                                4: [3, 4]}

            vars(args)['arch_dict'] = {'num_ch': 64,
                                       'num_ch_initter': 64,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [[3,3],
                                                        [3,3], [7,7],
                                                        [3,3],
                                                        [3,3]],
                                       'strides': [1, 1],
                                       'padding': [[1,1],
                                                   [1,1],[3,3],
                                                   [1,1],[3,3],
                                                   [1,1]],
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 64}
            vars(args)['energy_weight_mask'] = [1.0,
                                                0.09375,
                                                0.375,
                                                1.5,
                                                12.0]

        if args.architecture == 'DAN_cifar10_5layers_btself':
            vars(args)['state_sizes'] = [[args.batch_size, 3, 32, 32],  # 3072
                                         [args.batch_size, 16, 16, 16],  # 32768
                                         [args.batch_size, 16, 8, 8],  # 2048
                                         [args.batch_size, 256],
                                         [args.batch_size, 64]]

            mod_connect_dict = {0: [0, 1],
                                1: [0, 2],
                                2: [1, 3],
                                3: [2, 3, 4],
                                4: [3, 4]}

            vars(args)['arch_dict'] = {'num_ch': 64,
                                       'num_ch_initter': 64,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [[3,3],
                                                        [3,3],
                                                        [3,3],
                                                        [3,3],
                                                        [3,3]],
                                       'strides': [1, 1],
                                       'padding': [[1,1],
                                                   [1,1],[1,1],
                                                   [1,1],[1,1],
                                                   [1,1]],
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 64}
            vars(args)['energy_weight_mask'] = [1.0,
                                                0.75,
                                                0.75,
                                                1.5,
                                                12.0]

        if args.architecture == 'DAN_cifar10_med_6layers_btself':
            vars(args)['state_sizes'] = [[args.batch_size, 3, 32, 32],  # 3072
                                         [args.batch_size, 16, 16, 16],  # 32768
                                         [args.batch_size, 16, 16, 16],  # 8192
                                         [args.batch_size, 16, 8, 8],  # 2048
                                         [args.batch_size, 256],
                                         [args.batch_size, 64]]

            mod_connect_dict = {0: [0, 1],
                                1: [0, 2],
                                2: [1, 3],
                                3: [2, 4],
                                4: [3, 4, 5],
                                5: [4, 5]}

            vars(args)['arch_dict'] = {'num_ch': 64,
                                       'num_ch_initter': 64,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [[3,3],
                                                        [3,3],
                                                        [3,3],
                                                        [3,3],
                                                        [3,3]],
                                       'strides': [1, 1],
                                       'padding': [[1,1],
                                                   [1,1],[1,1],
                                                   [1,1],[1,1],
                                                   [1,1]],
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 64}
            vars(args)['energy_weight_mask'] = [1.0,
                                                0.75,
                                                0.75,
                                                1.5,
                                                12.0,
                                                48.0]
        if args.architecture == 'DAN_cifar10_large_6_layers_top3self_fcconvconnect':
            vars(args)['state_sizes'] = [[args.batch_size, 3, 32, 32],  # 3072
                                         [args.batch_size, 32, 32, 32], # 16384
                                         [args.batch_size, 32, 16, 16], # 6400
                                         [args.batch_size, 32, 8, 8],
                                         [args.batch_size, 256],
                                         [args.batch_size, 128]]

            mod_connect_dict = {0: [1],
                                1: [0, 2],
                                2: [1, 3],
                                3: [2, 3, 4],
                                4: [3, 4, 5],
                                5: [4, 5]
                                }

            vars(args)['arch_dict'] = {'num_ch': 64,
                                       'num_ch_initter': 64,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [[3,3],
                                                        [3,3],
                                                        [3,3],
                                                        [3,3],
                                                        [3,3]],
                                       'strides': [1, 1, 1, 1, 1],
                                       'padding': [[1,1],
                                                   [1,1],[1,1],
                                                   [1,1],[1,1],
                                                   [1,1]],
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 64}
            #vars(args)['energy_weight_mask'] = [1.0, 0.18, 0.48, 5.95, 24.0]
        if args.architecture == 'DAN_cifar10_large_6layers_btself':
            vars(args)['state_sizes'] = [[args.batch_size, 3, 32, 32],  # 3072
                                         [args.batch_size, 32, 32, 32],  # 32768
                                         [args.batch_size, 32, 16, 16],  # 8192
                                         [args.batch_size, 32, 8, 8],  # 2048
                                         [args.batch_size, 256],
                                         [args.batch_size, 64]]

            mod_connect_dict = {0: [0, 1],
                                1: [0, 2],
                                2: [1, 3],
                                3: [2, 4],
                                4: [3, 4, 5],
                                5: [4, 5]}

            vars(args)['arch_dict'] = {'num_ch': 64,
                                       'num_ch_initter': 64,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [[3,3],
                                                        [3,3],
                                                        [3,3],
                                                        [3,3],
                                                        [3,3]],
                                       'strides': [1, 1],
                                       'padding': [[1,1],
                                                   [1,1],[1,1],
                                                   [1,1],[1,1],
                                                   [1,1]],
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 64}
            vars(args)['energy_weight_mask'] = [1.0,
                                                0.0938,
                                                0.3750,
                                                1.5,
                                                12.0,
                                                48.0]

        if args.architecture == 'DAN_cifar10_large_6layers_allself':
            vars(args)['state_sizes'] = [[args.batch_size, 3, 32, 32],
                                         [args.batch_size, 64, 32, 32],
                                         [args.batch_size, 32, 16, 16],
                                         [args.batch_size, 32, 8, 8],
                                         [args.batch_size, 256],
                                         [args.batch_size, 64]]

            mod_connect_dict = {0: [0, 1],
                                1: [0, 1, 2],
                                2: [1, 2, 3],
                                3: [2, 3, 4],
                                4: [3, 4, 5],
                                5: [4, 5]}

            vars(args)['arch_dict'] = {'num_ch': 64,
                                       'num_ch_initter': 64,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [[3,3],
                                                        [3,3],
                                                        [3,3],
                                                        [3,3],
                                                        [3,3]],
                                       'strides': [1, 1],
                                       'padding': [[1,1],
                                                   [1,1],[1,1],
                                                   [1,1],[1,1],
                                                   [1,1]],
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 64}
            # vars(args)['energy_weight_mask'] = [1.0,
            #                                     0.0938,
            #                                     0.3750,
            #                                     1.5,
            #                                     12.0,
            #                                     48.0]
            vars(args)['energy_weight_mask'] = calc_enrg_masks(args)

        if args.architecture == 'DAN_cifar10_very_large_6_layers_top2self':
            vars(args)['state_sizes'] = [[args.batch_size, 3, 32, 32],  # 3072
                                         [args.batch_size, 64, 32, 32], # 65536
                                         [args.batch_size, 64, 16, 16], # 16384
                                         [args.batch_size, 64, 10, 10], # 6400
                                         [args.batch_size, 516],
                                         [args.batch_size, 128]]

            mod_connect_dict = {0: [1],
                                1: [0, 2],
                                2: [1, 3],
                                3: [2, 4],
                                4: [3, 4, 5],
                                5: [4, 5]}

            vars(args)['arch_dict'] = {'num_ch': 32,
                                       'num_ch_initter': 32,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [[3,3],
                                                        [3,3],
                                                        [3,3],
                                                        [3,3],
                                                        [3,3]],
                                       'strides': [1, 1,1,1,1],
                                       'padding': [[1,1],
                                                   [1,1],[1,1],
                                                   [1,1],[1,1],
                                                   [1,1]],
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 64}
            vars(args)['energy_weight_mask'] = calc_enrg_masks(args)

        if args.architecture == 'DAN_cifar10_very_large_7_layers_top2self':
            vars(args)['state_sizes'] = [[args.batch_size, 3, 32, 32],  # 3072
                                         [args.batch_size, 64, 32, 32], # 65536
                                         [args.batch_size, 64, 32, 32],
                                         [args.batch_size, 64, 16, 16], # 16384
                                         [args.batch_size, 64, 10, 10], # 6400
                                         [args.batch_size, 516],
                                         [args.batch_size, 128]]

            mod_connect_dict = {0: [1],
                                1: [0, 2],
                                2: [1, 3],
                                3: [2, 4],
                                4: [3, 5],
                                5: [4, 5, 6],
                                6: [5, 6]}

            vars(args)['arch_dict'] = {'num_ch': 32,
                                       'num_ch_initter': 32,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [[3,3],
                                                        [3,3],
                                                        [3,3],
                                                        [3,3],
                                                        [3, 3],
                                                        [3,3]],
                                       'strides': [1, 1,1,1,1],
                                       'padding': [[1,1],
                                                   [1,1],[1,1],
                                                   [1,1],[1,1],
                                                   [1,1]],
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 64}
            vars(args)['energy_weight_mask'] = calc_enrg_masks(args)

        if args.architecture == 'DAN_cifar10_large_6_layers_top2self_fcconvconnect':
            vars(args)['state_sizes'] = [[args.batch_size, 3, 32, 32],  # 3072
                                         [args.batch_size, 64, 32, 32], # 65536
                                         [args.batch_size, 64, 16, 16], # 16384
                                         [args.batch_size, 32, 8, 8], # 6400
                                         [args.batch_size, 516],
                                         [args.batch_size, 128]]

            mod_connect_dict = {0: [1],
                                1: [0, 2],
                                2: [1, 3],
                                3: [2, 4, 5],
                                4: [3, 4, 5],
                                5: [4, 5]}

            vars(args)['arch_dict'] = {'num_ch': 64,
                                       'num_ch_initter': 64,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [3, 3, 3],
                                       'strides': [1, 1],
                                       'padding': 1,
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 64}
            vars(args)['energy_weight_mask'] = [1.0, 0.046875, 0.18, 0.48, 5.95, 24.0]

        if args.architecture == 'DAN_cifar10_8layers_huge_filtermix':
            vars(args)['state_sizes'] = [[args.batch_size, 3, 32, 32],  # 3072

                                         [args.batch_size, 32, 32, 32],  # 32768
                                         [args.batch_size, 32, 32, 32],  # 32768

                                         [args.batch_size, 32, 16, 16],  # 8192
                                         [args.batch_size, 32, 16, 16],  # 8192

                                         [args.batch_size, 32, 8, 8],  # 2048
                                         [args.batch_size, 32, 8, 8],  # 2048

                                         [args.batch_size, 256]]

            mod_connect_dict = {0: [1,2],
                                1: [0, 1, 2, 3],
                                2: [0, 1, 2, 4],
                                3: [1, 3, 4, 5],
                                4: [2, 3, 4, 6],
                                5: [4, 5, 6, 7],
                                6: [5, 5, 6, 7],
                                7: [5, 6, 7]}

            vars(args)['arch_dict'] = {'num_ch': 64,
                                       'num_ch_initter': 64,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [[3,3],
                                                        [3,3], [7,7],
                                                        [3,3], [7,7],
                                                        [3,3], [7,7],
                                                        [3,3]],
                                       'strides': [1, 1],
                                       'padding': [[1,1],
                                                   [1,1],[3,3],
                                                   [1,1],[3,3],
                                                   [1,1],[3,3],
                                                   [1,1]],
                                       'mod_connect_dict': mod_connect_dict,
                                       'num_fc_channels': 128}
            vars(args)['energy_weight_mask'] = [1.0,
                                                0.09375,  0.09375,
                                                0.48, 0.48,
                                                1.5, 1.5,
                                                12.0] #Fails due to memory issues...
    if args.network_type == 'DAN2':
        if args.architecture == 'DAN2_very_small_1SL_self':
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28]]

            mod_connect_dict = {0: [0]}
            mod_cct_status_dict = {0: [0] # 0 for cct, 1 for oc, 2 for oct
                                   }
            mod_num_lyr_dict =    {0: 2, # 0 to have no dense block
                                   }
            base_kern_pad_dict = {0: [[7,3]]}
            main_kern_dict = {0: 3}
            vars(args)['arch_dict'] = {'num_ch_base': 8,#Feeling a bit restricted by not being able to specify that the base of the bottom layer should be different (since I predict that it will only have dense block rarely so needs more in the base).
                                       'growth_rate': 8,
                                       'num_ch_initter': 32,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'base_kern_pad_dict': base_kern_pad_dict,
                                       'main_kern_dict': main_kern_dict,
                                       'mod_connect_dict': mod_connect_dict,
                                       'mod_cct_status_dict': mod_cct_status_dict,
                                       'mod_num_lyr_dict': mod_num_lyr_dict}
        if args.architecture == 'DAN2_small_4SL_allself':
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size,  32, 28, 28],
                                         [args.batch_size,  32, 28, 28],
                                         [args.batch_size,  32, 28, 28]]

            mod_connect_dict = {0: [0,1],
                                1: [0,1,2],
                                2: [1,2,3],
                                3: [2,3]}
            mod_cct_status_dict = {0: [0,1], # 0 for cct, 1 for oc, 2 for oct
                                   1: [2,0,0],
                                   2: [0,0,0],
                                   3: [0,0]}
            mod_num_lyr_dict =    {0: 0, # 0 to have no dense block
                                   1: 2,
                                   2: 2,
                                   3: 2}
            base_kern_pad_dict = {0: [[3,1],[7,3]],
                                  1: [[7,3],[3,1],[3,1]],
                                  2: [[3,1],[3,1],[3,1]],
                                  3: [[3,1],[3,1]]}
            main_kern_dict = {0: 3,
                              1: 3,
                              2: 3,
                              3: 3}
            vars(args)['arch_dict'] = {'num_ch_base': 32,#Feeling a bit restricted by not being able to specify that the base of the bottom layer should be different (since I predict that it will only have dense block rarely so needs more in the base).
                                       'growth_rate': 16,
                                       'num_ch_initter': 32,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'base_kern_pad_dict': base_kern_pad_dict,
                                       'main_kern_dict': main_kern_dict,
                                       'mod_connect_dict': mod_connect_dict,
                                       'mod_cct_status_dict': mod_cct_status_dict,
                                       'mod_num_lyr_dict': mod_num_lyr_dict}
        if args.architecture == 'DAN2_small_4SL_allself_nocompress':
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size,  32, 28, 28],
                                         [args.batch_size,  32, 28, 28],
                                         [args.batch_size,  32, 28, 28]]

            mod_connect_dict = {0: [0,1],
                                1: [0,1,2],
                                2: [1,2,3],
                                3: [2,3]}
            mod_cct_status_dict = {0: [0,1], # 0 for cct, 1 for oc, 2 for oct
                                   1: [2,0,0],
                                   2: [0,0,0],
                                   3: [0,0]}
            mod_num_lyr_dict =    {0: 0, # 0 to have no dense block
                                   1: 2,
                                   2: 2,
                                   3: 2}
            base_kern_pad_dict = {0: [[3,1],[7,3]],
                                  1: [[7,3],[3,1],[3,1]],
                                  2: [[3,1],[3,1],[3,1]],
                                  3: [[3,1],[3,1]]}
            main_kern_dict = {0: 3,
                              1: 3,
                              2: 3,
                              3: 3}
            vars(args)['arch_dict'] = {'num_ch_base': 32,#Feeling a bit restricted by not being able to specify that the base of the bottom layer should be different (since I predict that it will only have dense block rarely so needs more in the base).
                                       'growth_rate': 16,
                                       'num_ch_initter': 32,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'base_kern_pad_dict': base_kern_pad_dict,
                                       'main_kern_dict': main_kern_dict,
                                       'mod_connect_dict': mod_connect_dict,
                                       'mod_cct_status_dict': mod_cct_status_dict,
                                       'mod_num_lyr_dict': mod_num_lyr_dict}
        if args.architecture == 'DAN2_small_4SL_topself_cleanbottom_bigkerns':
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size,  32, 28, 28],
                                         [args.batch_size,  32, 28, 28],
                                         [args.batch_size,  32, 28, 28]]

            mod_connect_dict = {0: [1],
                                1: [0,1,2],
                                2: [1,2,3],
                                3: [2,3]}
            mod_cct_status_dict = {0: [2], # 0 for cct, 1 for oc, 2 for oct
                                   1: [1,0,0],
                                   2: [0,0,0],
                                   3: [0,0]}
            mod_num_lyr_dict =    {0: 0, # 0 to have no dense block
                                   1: 2,
                                   2: 2,
                                   3: 2}
            base_kern_pad_dict = {0: [[3,1],[7,3]],
                                  1: [[7,3],[3,1],[3,1]],
                                  2: [[3,1],[3,1],[3,1]],
                                  3: [[3,1],[3,1]]}
            main_kern_dict = {0: 7,
                              1: 7,
                              2: 7,
                              3: 7}
            vars(args)['arch_dict'] = {'num_ch_base': 32,#Feeling a bit restricted by not being able to specify that the base of the bottom layer should be different (since I predict that it will only have dense block rarely so needs more in the base).
                                       'growth_rate': 8,
                                       'num_ch_initter': 32,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'base_kern_pad_dict': base_kern_pad_dict,
                                       'main_kern_dict': main_kern_dict,
                                       'mod_connect_dict': mod_connect_dict,
                                       'mod_cct_status_dict': mod_cct_status_dict,
                                       'mod_num_lyr_dict': mod_num_lyr_dict,
                                       'spec_norm_reg': True}
        elif args.architecture == 'DAN2_small_6SL_allself_compress':
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size,  32, 28, 28],
                                         [args.batch_size,  32, 22, 22],
                                         [args.batch_size,  32, 16, 16],
                                         [args.batch_size,  16, 10, 10],
                                         [args.batch_size,  16, 1, 1]]

            mod_connect_dict = {0: [0,1],
                                1: [0,1,2],
                                2: [1,2,3],
                                3: [2,3,4],
                                4: [3,4,5],
                                5: [4,5]}
            mod_cct_status_dict = {0: [0,1], # 0 for cct, 1 for oc, 2 for oct
                                   1: [2,0,2],
                                   2: [1,0,2],
                                   3: [1,0,2],
                                   4: [1,0,2],
                                   5: [1,1]}
            mod_num_lyr_dict =    {0: 0, # 0 to have no dense block
                                   1: 2,
                                   2: 2,
                                   3: 2,
                                   4: 2,
                                   5: 0}
            base_kern_pad_dict = {0: [[3,1],[7,3]],
                                  1: [[7,3],[3,1],[7,0]],
                                  2: [[7,0],[3,1],[7,0]],
                                  3: [[7,0],[3,1],[7,0]],
                                  4: [[7,0],[3,1],[10,0]],
                                  5: [[10,0],[1,0]]}
            main_kern_dict = {0: 3,
                              1: 3,
                              2: 3,
                              3: 3,
                              4: 3,
                              5: 1}
            vars(args)['arch_dict'] = {'num_ch_base': 32,#Feeling a bit restricted by not being able to specify that the base of the bottom layer should be different (since I predict that it will only have dense block rarely so needs more in the base).
                                       'growth_rate': 16,
                                       'num_ch_initter': 32,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'base_kern_pad_dict': base_kern_pad_dict,
                                       'main_kern_dict': main_kern_dict,
                                       'mod_connect_dict': mod_connect_dict,
                                       'mod_cct_status_dict': mod_cct_status_dict,
                                       'mod_num_lyr_dict': mod_num_lyr_dict}
        elif args.architecture == 'DAN2_small_4SL_compress_bigkern':
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size,  32, 28, 28],
                                         [args.batch_size,  32, 22, 22],
                                         [args.batch_size,  32, 16, 16]]
            mod_connect_dict = {0: [1],
                                1: [0,1,2],
                                2: [1,2,3],
                                3: [2,3]}
            mod_cct_status_dict = {0: [2], # 0 for cct, 1 for oc, 2 for oct
                                   1: [1,0,2],
                                   2: [1,0,2],
                                   3: [1,0,2]}
            mod_num_lyr_dict =    {0: 0, # 0 to have no dense block
                                   1: 2,
                                   2: 2,
                                   3: 2}
            base_kern_pad_dict = {0: [[7,3]],
                                  1: [[7,3],[7,3],[7,0]],
                                  2: [[7,0],[7,3],[7,0]],
                                  3: [[7,0],[7,3]]}
            main_kern_dict = {0: 7,
                              1: 7,
                              2: 7,
                              3: 7}
            vars(args)['arch_dict'] = {'num_ch_base': 32,#Feeling a bit restricted by not being able to specify that the base of the bottom layer should be different (since I predict that it will only have dense block rarely so needs more in the base).
                                       'growth_rate': 8,
                                       'num_ch_initter': 32,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'base_kern_pad_dict': base_kern_pad_dict,
                                       'main_kern_dict': main_kern_dict,
                                       'mod_connect_dict': mod_connect_dict,
                                       'mod_cct_status_dict': mod_cct_status_dict,
                                       'mod_num_lyr_dict': mod_num_lyr_dict,
                                       'spec_norm_reg': True}
        elif args.architecture == 'DAN2_small_light_4SL_compress_noself_bigkern_cct':
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                         [args.batch_size, 32, 28, 28],
                                         [args.batch_size, 32, 22, 22],
                                         [args.batch_size, 16, 14, 14]]
            mod_connect_dict = {0: [1],
                                1: [0,2],
                                2: [1,3],
                                3: [2]}
            mod_cct_status_dict = {0: [0], # 0 for cct, 1 for oc, 2 for oct
                                   1: [1,2],
                                   2: [1,2],
                                   3: [1]}
            mod_num_lyr_dict = {0: 0,  # 0 to have no dense block
                                1: 0,
                                2: 0,
                                3: 0}
            base_kern_pad_dict = {0: [[7,3]],
                                  1: [[7,3],[7,0]],
                                  2: [[7,0],[9,0]],
                                  3: [[9,0]]}
            main_kern_dict = {0: 7,
                              1: 7,
                              2: 7,
                              3: 7}
            vars(args)['arch_dict'] = {'num_ch_base': 16,
                                       # Feeling a bit restricted by not being able to specify that the base of the bottom layer should be different (since I predict that it will only have dense block rarely so needs more in the base).
                                       'growth_rate': 8,
                                       'num_ch_initter': 32,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'base_kern_pad_dict': base_kern_pad_dict,
                                       'main_kern_dict': main_kern_dict,
                                       'mod_connect_dict': mod_connect_dict,
                                       'mod_cct_status_dict': mod_cct_status_dict,
                                       'mod_num_lyr_dict': mod_num_lyr_dict,
                                       'spec_norm_reg': True}
        elif args.architecture == 'DAN2_small_light_4SL_nospecnormreg_compress_noself_bigkern_cct':
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                         [args.batch_size, 32, 28, 28],
                                         [args.batch_size, 32, 22, 22],
                                         [args.batch_size, 16, 14, 14]]
            mod_connect_dict = {0: [1],
                                1: [0,2],
                                2: [1,3],
                                3: [2]}
            mod_cct_status_dict = {0: [0], # 0 for cct, 1 for oc, 2 for oct
                                   1: [1,2],
                                   2: [1,2],
                                   3: [1]}
            mod_num_lyr_dict = {0: 0,  # 0 to have no dense block
                                1: 0,
                                2: 0,
                                3: 0}
            base_kern_pad_dict = {0: [[7,3]],
                                  1: [[7,3],[7,0]],
                                  2: [[7,0],[9,0]],
                                  3: [[9,0]]}
            main_kern_dict = {0: 7,
                              1: 7,
                              2: 7,
                              3: 7}
            vars(args)['arch_dict'] = {'num_ch_base': 16,
                                       # Feeling a bit restricted by not being able to specify that the base of the bottom layer should be different (since I predict that it will only have dense block rarely so needs more in the base).
                                       'growth_rate': 8,
                                       'num_ch_initter': 32,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'base_kern_pad_dict': base_kern_pad_dict,
                                       'main_kern_dict': main_kern_dict,
                                       'mod_connect_dict': mod_connect_dict,
                                       'mod_cct_status_dict': mod_cct_status_dict,
                                       'mod_num_lyr_dict': mod_num_lyr_dict,
                                       'spec_norm_reg': False}
        elif args.architecture == 'DAN2_small_light_4SL_lotofcompress':
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                         [args.batch_size, 32, 28, 28],
                                         [args.batch_size, 32, 16, 16],
                                         [args.batch_size, 16, 7, 7]]
            mod_connect_dict = {0: [1],
                                1: [0,2],
                                2: [1,3],
                                3: [2]}
            mod_cct_status_dict = {0: [1], # 0 for cct, 1 for oc, 2 for oct
                                   1: [1,1],
                                   2: [1,1],
                                   3: [1]}
            mod_num_lyr_dict = {0: 0,  # 0 to have no dense block
                                1: 0,
                                2: 0,
                                3: 0}
            base_kern_pad_dict = {0: [[7,3]],
                                  1: [[7,3],[7,3]],
                                  2: [[7,0],[7,3]],
                                  3: [[7,0]]}
            main_kern_dict = {0: 7,
                              1: 7,
                              2: 7,
                              3: 7}
            vars(args)['arch_dict'] = {'num_ch_base': 16,
                                       # Feeling a bit restricted by not being able to specify that the base of the bottom layer should be different (since I predict that it will only have dense block rarely so needs more in the base).
                                       'growth_rate': 8,
                                       'num_ch_initter': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'base_kern_pad_dict': base_kern_pad_dict,
                                       'main_kern_dict': main_kern_dict,
                                       'mod_connect_dict': mod_connect_dict,
                                       'mod_cct_status_dict': mod_cct_status_dict,
                                       'mod_num_lyr_dict': mod_num_lyr_dict,
                                       'spec_norm_reg': False}
        elif args.architecture == 'DAN2_small_light_4SL_top2self_lotofcompress':
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                         [args.batch_size, 32, 28, 28],
                                         [args.batch_size, 32, 16, 16],
                                         [args.batch_size, 16, 7, 7]]
            mod_connect_dict = {0: [1],
                                1: [0,2],
                                2: [1,2,3],
                                3: [2,3]}
            mod_cct_status_dict = {0: [1], # 0 for cct, 1 for oc, 2 for oct
                                   1: [1,1],
                                   2: [1,1,1],
                                   3: [1,1]}
            mod_num_lyr_dict = {0: 0,  # 0 to have no dense block
                                1: 0,
                                2: 0,
                                3: 0}
            base_kern_pad_dict = {0: [[7,3]],
                                  1: [[7,3],[7,3]],
                                  2: [[7,0],[7,3],[7,3]],
                                  3: [[7,0], [7,3]]}
            main_kern_dict = {0: 7,
                              1: 7,
                              2: 7,
                              3: 7}
            vars(args)['arch_dict'] = {'num_ch_base': 16,
                                       # Feeling a bit restricted by not being able to specify that the base of the bottom layer should be different (since I predict that it will only have dense block rarely so needs more in the base).
                                       'growth_rate': 8,
                                       'num_ch_initter': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'base_kern_pad_dict': base_kern_pad_dict,
                                       'main_kern_dict': main_kern_dict,
                                       'mod_connect_dict': mod_connect_dict,
                                       'mod_cct_status_dict': mod_cct_status_dict,
                                       'mod_num_lyr_dict': mod_num_lyr_dict,
                                       'spec_norm_reg': False}
        elif args.architecture == 'DAN2_small_light_5SL_top3self_lotofcompress':
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                         [args.batch_size, 32, 28, 28],
                                         [args.batch_size, 32, 16, 16],
                                         [args.batch_size, 16, 7, 7],
                                         [args.batch_size, 16, 1, 1]]
            mod_connect_dict = {0: [1],
                                1: [0,2],
                                2: [1,2,3],
                                3: [2,3,4],
                                4: [3,4]}
            mod_cct_status_dict = {0: [1], # 0 for cct, 1 for oc, 2 for oct
                                   1: [1,1],
                                   2: [1,1,1],
                                   3: [1,1,1],
                                   4: [1,1]}
            mod_num_lyr_dict = {0: 0,  # 0 to have no dense block
                                1: 0,
                                2: 0,
                                3: 0,
                                4: 0}
            base_kern_pad_dict = {0: [[7,3]],
                                  1: [[7,3],[7,3]],
                                  2: [[7,0],[7,3],[7,3]],
                                  3: [[7,0],[7,3],[1,0]],
                                  4: [[1,0],[1,0]]}
            main_kern_dict = {0: 7,
                              1: 7,
                              2: 7,
                              3: 7,
                              4: 1}
            vars(args)['arch_dict'] = {'num_ch_base': 16,
                                       # Feeling a bit restricted by not being able to specify that the base of the bottom layer should be different (since I predict that it will only have dense block rarely so needs more in the base).
                                       'growth_rate': 8,
                                       'num_ch_initter': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'base_kern_pad_dict': base_kern_pad_dict,
                                       'main_kern_dict': main_kern_dict,
                                       'mod_connect_dict': mod_connect_dict,
                                       'mod_cct_status_dict': mod_cct_status_dict,
                                       'mod_num_lyr_dict': mod_num_lyr_dict,
                                       'spec_norm_reg': False}
        elif args.architecture == 'DAN2_small_light_4SL_top3self_compress':
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                         [args.batch_size, 16, 28, 28],
                                         [args.batch_size, 16, 10, 10],
                                         [args.batch_size, 16, 3, 3]]
            mod_connect_dict = {0: [1],
                                1: [0, 1, 2],
                                2: [1, 2, 3],
                                3: [2, 3]}
            mod_cct_status_dict = {0: [1],  # 0 for cct, 1 for oc, 2 for oct
                                   1: [1, 0, 1],
                                   2: [1, 0, 1],
                                   3: [1, 0]}
            mod_num_lyr_dict = {0: 0,  # 0 to have no dense block
                                1: 0,
                                2: 0,
                                3: 0}
            base_kern_pad_dict = {0: [[7, 3]],
                                  1: [[7, 3], [7, 3], [7, 3]],
                                  2: [[7, 0], [7, 3], [7, 3]],
                                  3: [[7, 0], [3, 1]]}
            main_kern_dict = {0: 7,
                              1: 7,
                              2: 7,
                              3: 3}
            vars(args)['arch_dict'] = {'num_ch_base': 16,
                                       # Feeling a bit restricted by not being able to specify that the base of the bottom layer should be different (since I predict that it will only have dense block rarely so needs more in the base).
                                       'growth_rate': 8,
                                       'num_ch_initter': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'base_kern_pad_dict': base_kern_pad_dict,
                                       'main_kern_dict': main_kern_dict,
                                       'mod_connect_dict': mod_connect_dict,
                                       'mod_cct_status_dict': mod_cct_status_dict,
                                       'mod_num_lyr_dict': mod_num_lyr_dict,
                                       'spec_norm_reg': False}

        elif args.architecture == 'DAN2_small_light_4SL_top3self_compress_dense':
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                         [args.batch_size, 16, 28, 28],
                                         [args.batch_size, 16, 10, 10],
                                         [args.batch_size, 16, 3, 3]]
            mod_connect_dict = {0: [1],
                                1: [0, 1, 2, 3],
                                2: [1, 2, 3],
                                3: [1, 2, 3]}
            mod_cct_status_dict = {0: [1],  # 0 for cct, 1 for oc, 2 for oct
                                   1: [1, 0, 1, 1],
                                   2: [1, 0, 1],
                                   3: [1, 1, 0]}
            mod_num_lyr_dict = {0: 0,  # 0 to have no dense block
                                1: 0,
                                2: 0,
                                3: 0}
            base_kern_pad_dict = {0: [[7, 3]],
                                  1: [[7, 3], [7, 3], [7, 3], [3,1]],
                                  2: [[7, 0], [7, 3], [7, 3]],
                                  3: [[10, 0], [7, 0], [3, 1]]}
            main_kern_dict = {0: 7,
                              1: 7,
                              2: 7,
                              3: 3}
            vars(args)['arch_dict'] = {'num_ch_base': 16,
                                       # Feeling a bit restricted by not being able to specify that the base of the bottom layer should be different (since I predict that it will only have dense block rarely so needs more in the base).
                                       'growth_rate': 8,
                                       'num_ch_initter': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'base_kern_pad_dict': base_kern_pad_dict,
                                       'main_kern_dict': main_kern_dict,
                                       'mod_connect_dict': mod_connect_dict,
                                       'mod_cct_status_dict': mod_cct_status_dict,
                                       'mod_num_lyr_dict': mod_num_lyr_dict,
                                       'spec_norm_reg': False}
        elif args.architecture == 'DAN2_small_light_3SL_seriouscompress':
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                         [args.batch_size, 16, 16, 16],
                                         [args.batch_size, 16, 5, 5]]
            mod_connect_dict = {0: [1],
                                1: [0, 1, 2],
                                2: [1, 2]}
            mod_cct_status_dict = {0: [1],  # 0 for cct, 1 for oc, 2 for oct
                                   1: [1, 0, 1],
                                   2: [1, 0]}
            mod_num_lyr_dict = {0: 0,  # 0 to have no dense block
                                1: 0,
                                2: 0,
                                3: 0}
            base_kern_pad_dict = {0: [[7, 3]],
                                  1: [[7, 3], [7, 3], [5, 3]],
                                  2: [[7, 0], [7, 3]]}
            main_kern_dict = {0: 7,
                              1: 7,
                              2: 3}
            vars(args)['arch_dict'] = {'num_ch_base': 16,
                                       # Feeling a bit restricted by not being able to specify that the base of the bottom layer should be different (since I predict that it will only have dense block rarely so needs more in the base).
                                       'growth_rate': 8,
                                       'num_ch_initter': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'base_kern_pad_dict': base_kern_pad_dict,
                                       'main_kern_dict': main_kern_dict,
                                       'mod_connect_dict': mod_connect_dict,
                                       'mod_cct_status_dict': mod_cct_status_dict,
                                       'mod_num_lyr_dict': mod_num_lyr_dict,
                                       'spec_norm_reg': False}
        elif args.architecture == 'DAN2_small_light_3SL_seriouscompress_heavybottom':
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                         [args.batch_size, 32, 28, 28],
                                         [args.batch_size, 16, 5, 5]]
            mod_connect_dict = {0: [1],
                                1: [0, 1, 2],
                                2: [1, 2]}
            mod_cct_status_dict = {0: [1],  # 0 for cct, 1 for oc, 2 for oct
                                   1: [1, 0, 1],
                                   2: [1, 0]}
            mod_num_lyr_dict = {0: 0,  # 0 to have no dense block
                                1: 0,
                                2: 0,
                                3: 0}
            base_kern_pad_dict = {0: [[7, 3]],
                                  1: [[7, 3], [7, 3], [5, 3]],
                                  2: [[7, 0], [7, 3]]}
            main_kern_dict = {0: 7,
                              1: 7,
                              2: 3}
            vars(args)['arch_dict'] = {'num_ch_base': 16,
                                       # Feeling a bit restricted by not being able to specify that the base of the bottom layer should be different (since I predict that it will only have dense block rarely so needs more in the base).
                                       'growth_rate': 8,
                                       'num_ch_initter': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'base_kern_pad_dict': base_kern_pad_dict,
                                       'main_kern_dict': main_kern_dict,
                                       'mod_connect_dict': mod_connect_dict,
                                       'mod_cct_status_dict': mod_cct_status_dict,
                                       'mod_num_lyr_dict': mod_num_lyr_dict,
                                       'spec_norm_reg': False}
        elif args.architecture == 'DAN2_small_light_4SL_seriouscompress':
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                         [args.batch_size, 16, 16, 16],
                                         [args.batch_size, 16, 10, 10],
                                         [args.batch_size, 16, 5, 5]]
            mod_connect_dict = {0: [1],
                                1: [0, 1, 2],
                                2: [1, 2, 3],
                                3: [2, 3]}
            mod_cct_status_dict = {0: [1],  # 0 for cct, 1 for oc, 2 for oct
                                   1: [1, 0, 1],
                                   2: [1, 0, 1],
                                   3: [1, 0]}
            mod_num_lyr_dict = {0: 0,  # 0 to have no dense block
                                1: 0,
                                2: 0,
                                3: 0}
            base_kern_pad_dict = {0: [[7, 3]],
                                  1: [[7, 0], [7, 3], [7, 3]],
                                  2: [[7, 0], [7, 3], [5, 3]],
                                  3: [[7, 0], [7, 3]]} #Note, no 7,3 from layer below, which isn't recommended anymore (for slightly speculative reasons, but little difference between 7,3 and 7,0 so just go with 7,3 since it worked well in the lowest layer)
            main_kern_dict = {0: 7,
                              1: 7,
                              2: 7,
                              3: 3}
            vars(args)['arch_dict'] = {'num_ch_base': 16,
                                       # Feeling a bit restricted by not being able to specify that the base of the bottom layer should be different (since I predict that it will only have dense block rarely so needs more in the base).
                                       'growth_rate': 8,
                                       'num_ch_initter': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'base_kern_pad_dict': base_kern_pad_dict,
                                       'main_kern_dict': main_kern_dict,
                                       'mod_connect_dict': mod_connect_dict,
                                       'mod_cct_status_dict': mod_cct_status_dict,
                                       'mod_num_lyr_dict': mod_num_lyr_dict,
                                       'spec_norm_reg': False}
        elif args.architecture == 'DAN2_small_light_4SL_seriouscompress_heavybottom':
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                         [args.batch_size, 32, 28, 28],
                                         [args.batch_size, 16, 10, 10],
                                         [args.batch_size, 16, 5, 5]]
            mod_connect_dict = {0: [1],
                                1: [0, 1, 2],
                                2: [1, 2, 3],
                                3: [2, 3]}
            mod_cct_status_dict = {0: [1],  # 0 for cct, 1 for oc, 2 for oct
                                   1: [1, 0, 1],
                                   2: [1, 0, 1],
                                   3: [1, 0]}
            mod_num_lyr_dict = {0: 0,  # 0 to have no dense block
                                1: 0,
                                2: 0,
                                3: 0}
            base_kern_pad_dict = {0: [[7, 3]],
                                  1: [[7, 3], [7, 3], [7, 3]],
                                  2: [[7, 3], [7, 3], [5, 3]],
                                  3: [[7, 3], [7, 3]]}
            main_kern_dict = {0: 7,
                              1: 7,
                              2: 7,
                              3: 3}
            vars(args)['arch_dict'] = {'num_ch_base': 16,
                                       # Feeling a bit restricted by not being able to specify that the base of the bottom layer should be different (since I predict that it will only have dense block rarely so needs more in the base).
                                       'growth_rate': 8,
                                       'num_ch_initter': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'base_kern_pad_dict': base_kern_pad_dict,
                                       'main_kern_dict': main_kern_dict,
                                       'mod_connect_dict': mod_connect_dict,
                                       'mod_cct_status_dict': mod_cct_status_dict,
                                       'mod_num_lyr_dict': mod_num_lyr_dict,
                                       'spec_norm_reg': False}

        elif args.architecture == 'DAN2_small_4SL_seriouscompress':
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                         [args.batch_size, 32, 28, 28],
                                         [args.batch_size, 32, 10, 10],
                                         [args.batch_size, 32, 5, 5]]
            mod_connect_dict = {0: [1],
                                1: [0, 1, 2],
                                2: [1, 2, 3],
                                3: [2, 3]}
            mod_cct_status_dict = {0: [1],  # 0 for cct, 1 for oc, 2 for oct
                                   1: [1, 0, 1],
                                   2: [1, 0, 1],
                                   3: [1, 0]}
            mod_num_lyr_dict = {0: 0,  # 0 to have no dense block
                                1: 0,
                                2: 0,
                                3: 0}
            base_kern_pad_dict = {0: [[7, 3]],
                                  1: [[7, 3], [7, 3], [7, 3]],
                                  2: [[7, 3], [7, 3], [5, 3]],
                                  3: [[7, 3], [7, 3]]}
            main_kern_dict = {0: 7,
                              1: 7,
                              2: 7,
                              3: 3}
            vars(args)['arch_dict'] = {'num_ch_base': 16,
                                       # Feeling a bit restricted by not being able to specify that the base of the bottom layer should be different (since I predict that it will only have dense block rarely so needs more in the base).
                                       'growth_rate': 8,
                                       'num_ch_initter': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'base_kern_pad_dict': base_kern_pad_dict,
                                       'main_kern_dict': main_kern_dict,
                                       'mod_connect_dict': mod_connect_dict,
                                       'mod_cct_status_dict': mod_cct_status_dict,
                                       'mod_num_lyr_dict': mod_num_lyr_dict,
                                       'spec_norm_reg': False}
            vars(args)['energy_weight_mask'] = calc_enrg_masks(args)

        elif args.architecture == 'DAN2_small_light_4SL_FC_top1.5':
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                         [args.batch_size, 32, 28, 28],
                                         [args.batch_size, 16, 10, 10],
                                         [args.batch_size, 16, 5, 5]]
            mod_connect_dict = {0: [1],
                                1: [0, 1, 2],
                                2: [1, 2, 3],
                                3: [2, 3]}
            mod_cct_status_dict = {0: [1],  # 0 for cct, 1 for oc, 2 for oct
                                   1: [1, 0, 1],
                                   2: [1, 0, 3],
                                   3: [3, 3]}
            mod_num_lyr_dict = {0: 0,  # 0 to have no dense block
                                1: 0,
                                2: 0,
                                3: 0}
            base_kern_pad_dict = {0: [[7, 3]],
                                  1: [[7, 3], [7, 3], [7, 3]],
                                  2: [[7, 3], [7, 3], [5, 3]],
                                  3: [[7, 3], [7, 3]]}
            main_kern_dict = {0: 7,
                              1: 7,
                              2: 7,
                              3: 3}
            vars(args)['arch_dict'] = {'num_ch_base': 16,
                                       # Feeling a bit restricted by not being able to specify that the base of the bottom layer should be different (since I predict that it will only have dense block rarely so needs more in the base).
                                       'growth_rate': 8,
                                       'num_ch_initter': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'base_kern_pad_dict': base_kern_pad_dict,
                                       'main_kern_dict': main_kern_dict,
                                       'mod_connect_dict': mod_connect_dict,
                                       'mod_cct_status_dict': mod_cct_status_dict,
                                       'mod_num_lyr_dict': mod_num_lyr_dict,
                                       'spec_norm_reg': False}
        elif args.architecture == 'DAN2_small_light_3SL_all_FC_no_self':
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                         [args.batch_size, 5, 5, 5],
                                         [args.batch_size, 3, 3, 3]]
            mod_connect_dict = {0: [1],
                                1: [0, 2],
                                2: [1]}
            mod_cct_status_dict = {0: [3],  # 0 for cct, 1 for oc, 2 for oct
                                   1: [3, 3],
                                   2: [3]}
            mod_num_lyr_dict = {0: 0,  # 0 to have no dense block
                                1: 0,
                                2: 0,
                                3: 0}
            base_kern_pad_dict = {0: [[7, 3]],
                                  1: [[7, 3], [7, 3], [7, 3]],
                                  2: [[7, 3], [7, 3], [5, 3]],
                                  3: [[7, 3], [7, 3]]}
            main_kern_dict = {0: 7,
                              1: 7,
                              2: 7,
                              3: 3}
            vars(args)['arch_dict'] = {'num_ch_base': 16,
                                       # Feeling a bit restricted by not being able to specify that the base of the bottom layer should be different (since I predict that it will only have dense block rarely so needs more in the base).
                                       'growth_rate': 8,
                                       'num_ch_initter': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'base_kern_pad_dict': base_kern_pad_dict,
                                       'main_kern_dict': main_kern_dict,
                                       'mod_connect_dict': mod_connect_dict,
                                       'mod_cct_status_dict': mod_cct_status_dict,
                                       'mod_num_lyr_dict': mod_num_lyr_dict,
                                       'spec_norm_reg': False}
        elif args.architecture == 'DAN2_combo_convfc_5SL_asymm':
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                         [args.batch_size, 32, 28, 28],
                                         [args.batch_size, 16, 10, 10],
                                         [args.batch_size, 16, 5, 5],
                                         [args.batch_size, 5, 5, 5]]
            mod_connect_dict = {0: [1, 4],
                                1: [0, 2],
                                2: [1, 3],
                                3: [2, 4],
                                4: [3]}
            mod_cct_status_dict = {0: [1, 3],  # 0 for cct, 1 for oc, 2 for oct
                                   1: [1, 1],
                                   2: [1, 1],
                                   3: [1, 3],
                                   4: [3]}
            mod_num_lyr_dict = {0: 0,  # 0 to have no dense block
                                1: 0,
                                2: 0,
                                3: 0,
                                4: 0}
            base_kern_pad_dict = {0: [[7, 3],[]],
                                  1: [[7, 3],[7, 3]],
                                  2: [[7, 3],[5, 3]],
                                  3: [[7, 3],[]],
                                  4: []}
            main_kern_dict = {0: 7,
                              1: 7,
                              2: 7,
                              3: 3,
                              4: 0}
            vars(args)['arch_dict'] = {'num_ch_base': 16,
                                       # Feeling a bit restricted by not being able to specify that the base of the bottom layer should be different (since I predict that it will only have dense block rarely so needs more in the base).
                                       'growth_rate': 8,
                                       'num_ch_initter': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'base_kern_pad_dict': base_kern_pad_dict,
                                       'main_kern_dict': main_kern_dict,
                                       'mod_connect_dict': mod_connect_dict,
                                       'mod_cct_status_dict': mod_cct_status_dict,
                                       'mod_num_lyr_dict': mod_num_lyr_dict,
                                       'spec_norm_reg': False}
        elif args.architecture == 'DAN2_combo_convfc_5SL_symm':
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                         [args.batch_size, 32, 28, 28],
                                         [args.batch_size, 16, 10, 10],
                                         [args.batch_size, 16, 5, 5],
                                         [args.batch_size, 5, 5, 5]]
            mod_connect_dict = {0: [1, 4],
                                1: [0, 2],
                                2: [1, 3],
                                3: [2, 4],
                                4: [3,0]}
            mod_cct_status_dict = {0: [1, 3],  # 0 for cct, 1 for oc, 2 for oct
                                   1: [1, 1],
                                   2: [1, 1],
                                   3: [1, 3],
                                   4: [3, 3]}
            mod_num_lyr_dict = {0: 0,  # 0 to have no dense block
                                1: 0,
                                2: 0,
                                3: 0,
                                4: 0}
            base_kern_pad_dict = {0: [[7, 3],[]],
                                  1: [[7, 3],[7, 3]],
                                  2: [[7, 3],[5, 3]],
                                  3: [[7, 3],[]],
                                  4: [[],[]]}
            main_kern_dict = {0: 7,
                              1: 7,
                              2: 7,
                              3: 3,
                              4: 0}
            vars(args)['arch_dict'] = {'num_ch_base': 16,
                                       # Feeling a bit restricted by not being able to specify that the base of the bottom layer should be different (since I predict that it will only have dense block rarely so needs more in the base).
                                       'growth_rate': 8,
                                       'num_ch_initter': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'base_kern_pad_dict': base_kern_pad_dict,
                                       'main_kern_dict': main_kern_dict,
                                       'mod_connect_dict': mod_connect_dict,
                                       'mod_cct_status_dict': mod_cct_status_dict,
                                       'mod_num_lyr_dict': mod_num_lyr_dict,
                                       'spec_norm_reg': False}
        elif args.architecture == 'DAN2_combo_convfc_5SL_asymm_densebackw':
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                         [args.batch_size, 32, 28, 28],
                                         [args.batch_size, 32, 10, 10],
                                         [args.batch_size, 32, 5, 5],
                                         [args.batch_size, 5, 5, 5]]
            mod_connect_dict = {0: [1, 2,3,4],
                                1: [0, 1, 2],
                                2: [1, 2],
                                3: [2, 3],
                                4: [3]}
            mod_cct_status_dict = {0: [1, 1,1,3],  # 0 for cct, 1 for oc, 2 for oct
                                   1: [1, 1],
                                   2: [1, 1],
                                   3: [1, 3], #TODO wtf why is this a diff len from connect dict?
                                   4: [3]}
            mod_num_lyr_dict = {0: 0,  # 0 to have no dense block
                                1: 0,
                                2: 0,
                                3: 0,
                                4: 0}
            base_kern_pad_dict = {0: [[7, 3],[7, 3],[7, 3],[]],
                                  1: [[7, 3],[7, 3],[7, 3]],
                                  2: [[7, 3],[5, 3]],
                                  3: [[7, 3],[]],
                                  4: []}
            main_kern_dict = {0: 7,
                              1: 7,
                              2: 7,
                              3: 3,
                              4: 0}
            vars(args)['arch_dict'] = {'num_ch_base': 16,
                                       # Feeling a bit restricted by not being able to specify that the base of the bottom layer should be different (since I predict that it will only have dense block rarely so needs more in the base).
                                       'growth_rate': 8,
                                       'num_ch_initter': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'base_kern_pad_dict': base_kern_pad_dict,
                                       'main_kern_dict': main_kern_dict,
                                       'mod_connect_dict': mod_connect_dict,
                                       'mod_cct_status_dict': mod_cct_status_dict,
                                       'mod_num_lyr_dict': mod_num_lyr_dict,
                                       'spec_norm_reg': False}
            #dict_len_check(args)
        elif args.architecture == 'DAN2_combo_convfc_5SL_asymm_densebackw_corrected':
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                         [args.batch_size, 32, 28, 28],
                                         [args.batch_size, 32, 10, 10],
                                         [args.batch_size, 32, 5, 5],
                                         [args.batch_size, 5, 5, 5]]
            mod_connect_dict = {0: [1, 2, 3, 4],
                                1: [0, 1, 2],
                                2: [1, 2, 3],
                                3: [2, 3, 4],
                                4: [3,4]}
            mod_cct_status_dict = {0: [1, 1, 1, 3],
                                   # 0 for cct, 1 for oc, 2 for oct
                                   1: [1, 1, 1],
                                   2: [1, 1, 1],
                                   3: [1, 1, 3],
                                   4: [3,3]}
            mod_num_lyr_dict = {0: 0,  # 0 to have no dense block
                                1: 0,
                                2: 0,
                                3: 0,
                                4: 0}
            base_kern_pad_dict = {0: [[7, 3], [7, 3], [7, 3], []],
                                  1: [[7, 3], [7, 3], [7, 3]],
                                  2: [[7, 3], [7, 3], [5, 3]],
                                  3: [[7, 3], [7, 3], []],
                                  4: [[],[]]}
            main_kern_dict = {0: 7,
                              1: 7,
                              2: 7,
                              3: 3,
                              4: 0}
            vars(args)['arch_dict'] = {'num_ch_base': 16,
                                       # Feeling a bit restricted by not being able to specify that the base of the bottom layer should be different (since I predict that it will only have dense block rarely so needs more in the base).
                                       'growth_rate': 8,
                                       'num_ch_initter': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'base_kern_pad_dict': base_kern_pad_dict,
                                       'main_kern_dict': main_kern_dict,
                                       'mod_connect_dict': mod_connect_dict,
                                       'mod_cct_status_dict': mod_cct_status_dict,
                                       'mod_num_lyr_dict': mod_num_lyr_dict,
                                       'spec_norm_reg': False}
            dict_len_check(args)
        elif args.architecture == 'DAN2_combo_convfc_5SL_asymm_densebackw_with_convt':
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                         [args.batch_size, 32, 28, 28],
                                         [args.batch_size, 32, 10, 10],
                                         [args.batch_size, 32, 5, 5],
                                         [args.batch_size, 5, 5, 5]]
            mod_connect_dict = {0: [1, 2,3,4],
                                1: [0, 1, 2],
                                2: [1, 2],
                                3: [2, 3],
                                4: [3]}
            mod_cct_status_dict = {0: [1, 1,1,3],  # 0 for cct, 1 for oc, 2 for oct
                                   1: [1, 1, 2],
                                   2: [1, 1],
                                   3: [1, 3], #TODO wtf why is this a diff len from connect dict?
                                   4: [3]}
            mod_num_lyr_dict = {0: 0,  # 0 to have no dense block
                                1: 0,
                                2: 0,
                                3: 0,
                                4: 0}
            base_kern_pad_dict = {0: [[7, 3],[7, 3],[7, 3],[]],
                                  1: [[7, 3],[7, 3],[7, 3]],
                                  2: [[7, 3],[5, 3]],
                                  3: [[7, 3],[]],
                                  4: []}
            main_kern_dict = {0: 7,
                              1: 7,
                              2: 7,
                              3: 3,
                              4: 0}
            vars(args)['arch_dict'] = {'num_ch_base': 16,
                                       # Feeling a bit restricted by not being able to specify that the base of the bottom layer should be different (since I predict that it will only have dense block rarely so needs more in the base).
                                       'growth_rate': 8,
                                       'num_ch_initter': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'base_kern_pad_dict': base_kern_pad_dict,
                                       'main_kern_dict': main_kern_dict,
                                       'mod_connect_dict': mod_connect_dict,
                                       'mod_cct_status_dict': mod_cct_status_dict,
                                       'mod_num_lyr_dict': mod_num_lyr_dict,
                                       'spec_norm_reg': False}
            #dict_len_check(args)
        elif args.architecture == 'DAN2_combo_convfc_5SL_asymm_bigkern':
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                         [args.batch_size, 32, 28, 28],
                                         [args.batch_size, 32, 10, 10],
                                         [args.batch_size, 32, 5, 5],
                                         [args.batch_size, 5, 5, 5]]
            mod_connect_dict = {0: [1, 4],
                                1: [0, 2],
                                2: [1, 3],
                                3: [2, 4],
                                4: [3]}
            mod_cct_status_dict = {0: [1, 3],  # 0 for cct, 1 for oc, 2 for oct
                                   1: [1, 1],
                                   2: [1, 1],
                                   3: [1, 3],
                                   4: [3]}
            mod_num_lyr_dict = {0: 0,  # 0 to have no dense block
                                1: 0,
                                2: 0,
                                3: 0,
                                4: 0}
            base_kern_pad_dict = {0: [[11, 5], []],
                                  1: [[11, 5], [11, 5]],
                                  2: [[11, 5], [11, 5]],
                                  3: [[11, 5], []],
                                  4: []}
            main_kern_dict = {0: 11,
                              1: 11,
                              2: 11,
                              3: 3,
                              4: 0}
            vars(args)['arch_dict'] = {'num_ch_base': 16,
                                       # Feeling a bit restricted by not being able to specify that the base of the bottom layer should be different (since I predict that it will only have dense block rarely so needs more in the base).
                                       'growth_rate': 8,
                                       'num_ch_initter': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'base_kern_pad_dict': base_kern_pad_dict,
                                       'main_kern_dict': main_kern_dict,
                                       'mod_connect_dict': mod_connect_dict,
                                       'mod_cct_status_dict': mod_cct_status_dict,
                                       'mod_num_lyr_dict': mod_num_lyr_dict,
                                       'spec_norm_reg': False}
            dict_len_check(args)
        elif args.architecture == 'DAN2_combo_convfc_5SL_asymm_above2all':
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                         [args.batch_size, 32, 28, 28],
                                         [args.batch_size, 32, 10, 10],
                                         [args.batch_size, 32, 5, 5],
                                         [args.batch_size, 5, 5, 5]]
            mod_connect_dict = {0: [1, 2,3,4],
                                1: [0, 1, 2, 3, 4],
                                2: [1, 2, 3, 4],
                                3: [2, 3, 4],
                                4: [3, 4]}
            mod_cct_status_dict = {0: [1, 1,1,3],  # 0 for cct, 1 for oc, 2 for oct
                                   1: [1, 1, 1, 1, 1],
                                   2: [1, 1, 1, 1],
                                   3: [1, 3, 3],
                                   4: [3, 3]}
            mod_num_lyr_dict = {0: 0,  # 0 to have no dense block
                                1: 0,
                                2: 0,
                                3: 0,
                                4: 0}
            base_kern_pad_dict = {0: [[7, 3],[7, 3],[7, 3],[]],
                                  1: [[7, 3],[7, 3],[7, 3],[5, 3],[5, 3]],
                                  2: [[7, 3],[7, 3],[5, 3],[5, 3]],
                                  3: [[7, 3],[]],
                                  4: []}
            main_kern_dict = {0: 7,
                              1: 7,
                              2: 7,
                              3: 3,
                              4: 0}
            vars(args)['arch_dict'] = {'num_ch_base': 16,
                                       # Feeling a bit restricted by not being able to specify that the base of the bottom layer should be different (since I predict that it will only have dense block rarely so needs more in the base).
                                       'growth_rate': 8,
                                       'num_ch_initter': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'base_kern_pad_dict': base_kern_pad_dict,
                                       'main_kern_dict': main_kern_dict,
                                       'mod_connect_dict': mod_connect_dict,
                                       'mod_cct_status_dict': mod_cct_status_dict,
                                       'mod_num_lyr_dict': mod_num_lyr_dict,
                                       'spec_norm_reg': False}
            dict_len_check(args)
        elif args.architecture == 'DAN2_combo_convfc_4SL_asymm_incorrected_loop':
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                         [args.batch_size, 32, 28, 28],
                                         [args.batch_size, 32, 7, 7],
                                         [args.batch_size, 5, 5, 5]]
            mod_connect_dict = {0: [1, 2, 3],
                                1: [0, 1],
                                2: [1, 2],
                                3: [2, 3]}
            mod_cct_status_dict = {0: [1, 1, 3],
                                   # 0 for cct, 1 for oc, 2 for oct
                                   1: [1, 1],
                                   2: [1, 3],
                                   3: [3, 3]}
            mod_num_lyr_dict = {0: 0,  # 0 to have no dense block
                                1: 0,
                                2: 0,
                                3: 0,
                                4: 0}
            base_kern_pad_dict = {0: [[7, 3], [7, 3], []],
                                  1: [[7, 3], [7, 3]],
                                  2: [[7, 3],  []],
                                  3: [[7, 3],  []]}
            main_kern_dict = {0: 7,
                              1: 7,
                              2: 7,
                              3: 3}
            vars(args)['arch_dict'] = {'num_ch_base': 16,
                                       # Feeling a bit restricted by not being able to specify that the base of the bottom layer should be different (since I predict that it will only have dense block rarely so needs more in the base).
                                       'growth_rate': 8,
                                       'num_ch_initter': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'base_kern_pad_dict': base_kern_pad_dict,
                                       'main_kern_dict': main_kern_dict,
                                       'mod_connect_dict': mod_connect_dict,
                                       'mod_cct_status_dict': mod_cct_status_dict,
                                       'mod_num_lyr_dict': mod_num_lyr_dict,
                                       'spec_norm_reg': False}
            #dict_len_check(args)
        elif args.architecture == 'DAN2_combo_convfc_4SL_all2all':
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                         [args.batch_size, 32, 28, 28],
                                         [args.batch_size, 32, 7, 7],
                                         [args.batch_size, 5, 5, 5]]
            mod_connect_dict = {0: [1, 2, 3],
                                1: [0, 1, 2, 3],
                                2: [0, 1, 2, 3],
                                3: [0, 1, 2, 3]}
            mod_cct_status_dict = {0: [1, 1, 3],
                                   # 0 for cct, 1 for oc, 2 for oct
                                   1: [1, 1, 1, 1],
                                   2: [1, 1, 1, 3],
                                   3: [1, 1, 3, 3]}
            mod_num_lyr_dict = {0: 0,  # 0 to have no dense block
                                1: 0,
                                2: 0,
                                3: 0,
                                4: 0}
            base_kern_pad_dict = {0: [[7, 3], [7, 3], []],
                                  1: [[7, 3], [7, 3], [7, 3], [7, 3]],
                                  2: [[7, 3], [7, 3], [7, 3], []],
                                  3: [[7, 3], [7, 3], [],     []]}
            main_kern_dict = {0: 7,
                              1: 7,
                              2: 7,
                              3: 3}
            vars(args)['arch_dict'] = {'num_ch_base': 16,
                                       # Feeling a bit restricted by not being able to specify that the base of the bottom layer should be different (since I predict that it will only have dense block rarely so needs more in the base).
                                       'growth_rate': 8,
                                       'num_ch_initter': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'base_kern_pad_dict': base_kern_pad_dict,
                                       'main_kern_dict': main_kern_dict,
                                       'mod_connect_dict': mod_connect_dict,
                                       'mod_cct_status_dict': mod_cct_status_dict,
                                       'mod_num_lyr_dict': mod_num_lyr_dict,
                                       'spec_norm_reg': False}
            dict_len_check(args)
        elif args.architecture == 'DAN2_combo_convfc_3SL_small':
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                         [args.batch_size, 16, 28, 28],
                                         [args.batch_size, 5, 5, 5]]
            mod_connect_dict = {0: [1, 2],
                                1: [0, 1, 2],
                                2: [1, 2]}
            mod_cct_status_dict = {0: [1, 3],
                                   # 0 for cct, 1 for oc, 2 for oct
                                   1: [1, 1, 3],
                                   2: [3, 3]}
            mod_num_lyr_dict = {0: 0,  # 0 to have no dense block
                                1: 0,
                                2: 0}
            base_kern_pad_dict = {0: [[7, 3], []],
                                  1: [[7, 3], [7, 3], []],
                                  2: [[], []]}
            main_kern_dict = {0: 7,
                              1: 7,
                              2: 7}
            vars(args)['arch_dict'] = {'num_ch_base': 16,
                                       # Feeling a bit restricted by not being able to specify that the base of the bottom layer should be different (since I predict that it will only have dense block rarely so needs more in the base).
                                       'growth_rate': 8,
                                       'num_ch_initter': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'base_kern_pad_dict': base_kern_pad_dict,
                                       'main_kern_dict': main_kern_dict,
                                       'mod_connect_dict': mod_connect_dict,
                                       'mod_cct_status_dict': mod_cct_status_dict,
                                       'mod_num_lyr_dict': mod_num_lyr_dict,
                                       'spec_norm_reg': False}
            dict_len_check(args)
        elif args.architecture == 'DAN2_CIFAR_6SL_vanilla_densebackw_skip1_4':
            vars(args)['state_sizes'] = [[args.batch_size, 3, 32, 32],
                                         [args.batch_size, 32, 32, 32],
                                         [args.batch_size, 32, 16, 16],
                                         [args.batch_size, 32, 8, 8],
                                         [args.batch_size, 16, 8, 8],
                                         [args.batch_size, 5, 5, 5]]
            mod_connect_dict = {0: [1, 2, 3, 4, 5],
                                1: [0, 1, 2],
                                2: [1, 2, 3],
                                3: [2, 3, 4],
                                4: [1, 3, 4, 5],#Conider a later archi that inputs 5 to all layers(DAN2_CIFAR_6SL_vanilla_densebackw)
                                5: [4, 5]}
            mod_cct_status_dict = {0: [1, 1, 1, 1, 3],
                                   # 0 for cct, 1 for oc, 2 for oct
                                   1: [1, 1, 1],
                                   2: [1, 1, 1],
                                   3: [1, 1, 1],
                                   4: [1, 1, 1, 3],
                                   5: [3, 3]}
            mod_num_lyr_dict = {0: 0,  # 0 to have no dense block
                                1: 0,
                                2: 0,
                                3: 0,
                                4: 0,
                                5: 0}
            base_kern_pad_dict = {0: [[7, 3], [7, 3], [7, 3], [7, 3], []],
                                  1: [[7, 3], [7, 3], [7, 3]],
                                  2: [[7, 3], [7, 3], [7, 3]],
                                  3: [[7, 3], [7, 3], [7, 3]],
                                  4: [[7, 3], [7, 3], [7, 3], []],
                                  5: [[], []],}
            main_kern_dict = {0: 7,
                              1: 7,
                              2: 7,
                              3: 7,
                              4: 7,
                              5: 0}
            vars(args)['arch_dict'] = {'num_ch_base': 32,
                                       # Feeling a bit restricted by not being able to specify that the base of the bottom layer should be different (since I predict that it will only have dense block rarely so needs more in the base).
                                       'growth_rate': 8,
                                       'num_ch_initter': 32,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'base_kern_pad_dict': base_kern_pad_dict,
                                       'main_kern_dict': main_kern_dict,
                                       'mod_connect_dict': mod_connect_dict,
                                       'mod_cct_status_dict': mod_cct_status_dict,
                                       'mod_num_lyr_dict': mod_num_lyr_dict,
                                       'spec_norm_reg': False}
            dict_len_check(args)
        elif args.architecture == 'DAN2_CIFAR_6SL_vanilla_densebackw_skip1_4_withwideSL1base':
            vars(args)['state_sizes'] = [[args.batch_size, 3, 32, 32],
                                         [args.batch_size, 32, 32, 32],
                                         [args.batch_size, 32, 16, 16],
                                         [args.batch_size, 32, 8, 8],
                                         [args.batch_size, 16, 8, 8],
                                         [args.batch_size, 5, 5, 5]]
            mod_connect_dict = {0: [1, 2, 3, 4, 5],
                                1: [0, 1, 2],
                                2: [1, 2, 3],
                                3: [2, 3, 4],
                                4: [1, 3, 4, 5],#Conider a later archi that inputs 5 to all layers(DAN2_CIFAR_6SL_vanilla_densebackw)
                                5: [4, 5]}
            mod_cct_status_dict = {0: [1, 1, 1, 1, 3],
                                   # 0 for cct, 1 for oc, 2 for oct
                                   1: [1, 1, 1],
                                   2: [1, 1, 1],
                                   3: [1, 1, 1],
                                   4: [1, 1, 1, 3],
                                   5: [3, 3]}
            mod_num_lyr_dict = {0: 0,  # 0 to have no dense block
                                1: 0,
                                2: 0,
                                3: 0,
                                4: 0,
                                5: 0}
            base_kern_pad_dict = {0: [[7, 3], [7, 3], [7, 3], [7, 3], []],
                                  1: [[7, 3], [11, 5], [7, 3]],
                                  2: [[7, 3], [7, 3], [7, 3]],
                                  3: [[7, 3], [7, 3], [7, 3]],
                                  4: [[7, 3], [7, 3], [7, 3], []],
                                  5: [[], []],}
            main_kern_dict = {0: 7,
                              1: 7,
                              2: 7,
                              3: 7,
                              4: 7,
                              5: 0}
            vars(args)['arch_dict'] = {'num_ch_base': 32,
                                       # Feeling a bit restricted by not being able to specify that the base of the bottom layer should be different (since I predict that it will only have dense block rarely so needs more in the base).
                                       'growth_rate': 8,
                                       'num_ch_initter': 32,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'base_kern_pad_dict': base_kern_pad_dict,
                                       'main_kern_dict': main_kern_dict,
                                       'mod_connect_dict': mod_connect_dict,
                                       'mod_cct_status_dict': mod_cct_status_dict,
                                       'mod_num_lyr_dict': mod_num_lyr_dict,
                                       'spec_norm_reg': False}
            dict_len_check(args)
        elif args.architecture == 'DAN2_CIFAR_4SL_vanilla_densebackw_skip1_4_truncated_with_wideSL1base':
            vars(args)['state_sizes'] = [[args.batch_size, 3, 32, 32],
                                         [args.batch_size, 32, 32, 32],
                                         [args.batch_size, 32, 16, 16],
                                         [args.batch_size, 5, 5, 5]]
            mod_connect_dict = {0: [1, 2, 3],
                                1: [0, 1, 2],
                                2: [1, 2, 3],
                                3: [2, 3]}
            mod_cct_status_dict = {0: [1, 1, 3],
                                   # 0 for cct, 1 for oc, 2 for oct
                                   1: [1, 1, 1],
                                   2: [1, 1, 3],
                                   3: [3, 3]}
            mod_num_lyr_dict = {0: 0,  # 0 to have no dense block
                                1: 0,
                                2: 0,
                                3: 0}
            base_kern_pad_dict = {0: [[7, 3], [7, 3], []],
                                  1: [[7, 3], [11, 5], [7, 3]],
                                  2: [[7, 3], [7, 3], []],
                                  3: [[], []]}
            main_kern_dict = {0: 7,
                              1: 7,
                              2: 7,
                              3: 0}
            vars(args)['arch_dict'] = {'num_ch_base': 32,
                                       # Feeling a bit restricted by not being able to specify that the base of the bottom layer should be different (since I predict that it will only have dense block rarely so needs more in the base).
                                       'growth_rate': 8,
                                       'num_ch_initter': 32,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'base_kern_pad_dict': base_kern_pad_dict,
                                       'main_kern_dict': main_kern_dict,
                                       'mod_connect_dict': mod_connect_dict,
                                       'mod_cct_status_dict': mod_cct_status_dict,
                                       'mod_num_lyr_dict': mod_num_lyr_dict,
                                       'spec_norm_reg': False}
            dict_len_check(args)
        elif args.architecture == 'DAN2_CIFAR_6SL_densebackw_ff_loopy_w_side':
            vars(args)['state_sizes'] = [[args.batch_size, 3, 32, 32],
                                         [args.batch_size, 32, 32, 32],
                                         [args.batch_size, 32, 16, 16],
                                         [args.batch_size, 64, 8, 8],
                                         [args.batch_size, 32, 8, 8],
                                         [args.batch_size, 5, 5, 5],
                                         [args.batch_size, 16, 8, 8],]
            mod_connect_dict = {0: [1, 2, 3, 4, 5, 6],
                                1: [0, 1, 2, 6],
                                2: [1, 2],
                                3: [2, 3],
                                4: [1, 3, 4],
                                5: [4, 5],
                                6: [1, 5]}
            mod_cct_status_dict = {0: [1, 1, 1, 1, 3, 1],
                                   # 0 for cct, 1 for oc, 2 for oct
                                   1: [1, 1, 1, 1],
                                   2: [1, 1],
                                   3: [1, 1],
                                   4: [1, 1, 1],
                                   5: [3, 3],
                                   6: [1, 3]}
            mod_num_lyr_dict = {0: 0,  # 0 to have no dense block
                                1: 0,
                                2: 0,
                                3: 0,
                                4: 0,
                                5: 0,
                                6: 0}
            base_kern_pad_dict = {0: [[7, 3], [7, 3], [7, 3], [7, 3], [], [7, 3]],
                                  1: [[7, 3], [7, 3], [7, 3], [7, 3]],
                                  2: [[7, 3], [7, 3]],
                                  3: [[7, 3], [7, 3]],
                                  4: [[7, 3], [7, 3], [7, 3]],
                                  5: [[], []],
                                  6: [[7,3], []]}
            main_kern_dict = {0: 7,
                              1: 7,
                              2: 7,
                              3: 7,
                              4: 7,
                              5: 0,
                              6: 7}
            vars(args)['arch_dict'] = {'num_ch_base': 32,
                                       # Feeling a bit restricted by not being able to specify that the base of the bottom layer should be different (since I predict that it will only have dense block rarely so needs more in the base).
                                       'growth_rate': 8,
                                       'num_ch_initter': 24,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'base_kern_pad_dict': base_kern_pad_dict,
                                       'main_kern_dict': main_kern_dict,
                                       'mod_connect_dict': mod_connect_dict,
                                       'mod_cct_status_dict': mod_cct_status_dict,
                                       'mod_num_lyr_dict': mod_num_lyr_dict,
                                       'spec_norm_reg': False}
            dict_len_check(args)
    if args.network_type == 'FHG':
        if args.architecture == 'FHG_5SL_densebackw':
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                         [args.batch_size, 32, 28, 28],
                                         [args.batch_size, 32, 10, 10],
                                         [args.batch_size, 32, 5, 5],
                                         [args.batch_size, 5, 5, 5]]
            mod_connect_dict = {0: [1, 2, 3, 4],
                                1: [0, 1, 2],
                                2: [1, 2],
                                3: [2, 3],
                                4: [3]}
            mod_cct_status_dict = {0: [1, 1, 1, 3],
                                   # 0 for cct, 1 for oc, 2 for oct
                                   1: [1, 1, 1],
                                   2: [1, 1],
                                   3: [1, 3],
                                   4: [3]}
            mod_num_lyr_dict = {0: 0,  # 0 to have no dense block
                                1: 0,
                                2: 0,
                                3: 0,
                                4: 0}
            base_kern_pad_dict = {0: [[7, 3], [7, 3], [7, 3], []],
                                  1: [[7, 3], [7, 3], [7, 3]],
                                  2: [[7, 3], [5, 3]],
                                  3: [[7, 3], []],
                                  4: []}
            main_kern_dict = {0: 7,
                              1: 7,
                              2: 7,
                              3: 3,
                              4: 0}
            # Gain
            cubic_mod_connect_dict = {0: [],
                                      1: [2],
                                      2: [2, 3],
                                      3: [3, 4],
                                      4: [3, 4]}
            cubic_mod_cct_status_dict = {0: [],
                                         # 0 for cct, 1 for oc, 2 for oct
                                         1: [1],
                                         2: [1, 1],
                                         3: [1, 3],
                                         4: [3]}
            cubic_mod_num_lyr_dict = {0: 0,  # 0 to have no dense block
                                      1: 0,
                                      2: 0,
                                      3: 0,
                                      4: 0}
            cubic_base_kern_pad_dict = {0: [],
                                        1: [[7, 3]],
                                        2: [[7, 3], [5, 3]],
                                        3: [[7, 3], []],
                                        4: []}
            cubic_main_kern_dict = {0: 7,
                                    1: 7,
                                    2: 7,
                                    3: 3,
                                    4: 0}
            vars(args)['arch_dict'] = {'num_ch_base': 16,
                                       # Feeling a bit restricted by not being able to specify that the base of the bottom layer should be different (since I predict that it will only have dense block rarely so needs more in the base).
                                       'growth_rate': 8,
                                       'num_ch_initter': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'base_kern_pad_dict': base_kern_pad_dict,
                                       'main_kern_dict': main_kern_dict,
                                       'mod_connect_dict': mod_connect_dict,
                                       'mod_cct_status_dict': mod_cct_status_dict,
                                       'mod_num_lyr_dict': mod_num_lyr_dict,
                                       'cubic_base_kern_pad_dict': cubic_base_kern_pad_dict,
                                       'cubic_main_kern_dict': cubic_main_kern_dict,
                                       'cubic_mod_connect_dict': cubic_mod_connect_dict,
                                       'cubic_mod_cct_status_dict': cubic_mod_cct_status_dict,
                                       'cubic_mod_num_lyr_dict': cubic_mod_num_lyr_dict,
                                       'spec_norm_reg': False}
            # dict_len_check(args)
        if args.architecture == 'FHG_1CtxL':
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                         [args.batch_size, 32, 28, 28],
                                         [args.batch_size, 32, 10, 10],
                                         [args.batch_size, 32, 5, 5],
                                         [args.batch_size, 5, 5, 5]]
            mod_connect_dict = {0: [1, 2, 3, 4],
                                1: [0, 1],
                                2: [1, 2, 3],
                                3: [0, 2, 3],
                                4: [3]}
            mod_cct_status_dict = {0: [1, 1, 1, 3],
                                   # 0 for cct, 1 for oc, 2 for oct
                                   1: [1, 1],
                                   2: [1, 1, 3],
                                   3: [1, 1, 3],
                                   4: [3]}
            mod_num_lyr_dict = {0: 0,  # 0 to have no dense block
                                1: 0,
                                2: 0,
                                3: 0,
                                4: 0}
            base_kern_pad_dict = {0: [[7, 3], [7, 3], [7, 3], []],
                                  1: [[7, 3], [7, 3]],
                                  2: [[7, 3], [5, 3], [7, 3]],
                                  3: [[7, 3], [7, 3], []],
                                  4: []}
            main_kern_dict = {0: 7,
                              1: 7,
                              2: 7,
                              3: 3,
                              4: 0}
            # Gain
            cubic_mod_connect_dict = {0: [],
                                      1: [4],
                                      2: [4],
                                      3: [2, 4],
                                      4: []}
            cubic_mod_cct_status_dict = {0: [],
                                         # 0 for cct, 1 for oc, 2 for oct
                                         1: [1],
                                         2: [1],
                                         3: [1, 3],
                                         4: []}
            cubic_mod_num_lyr_dict = {0: 0,  # 0 to have no dense block
                                      1: 0,
                                      2: 0,
                                      3: 0,
                                      4: 0}
            cubic_base_kern_pad_dict = {0: [],
                                        1: [[3, 1]],
                                        2: [[3, 1]],
                                        3: [[7, 3], []],
                                        4: []}
            cubic_main_kern_dict = {0: 7,
                                    1: 7,
                                    2: 7,
                                    3: 3,
                                    4: 0}
            vars(args)['arch_dict'] = {'num_ch_base': 16,
                                       # Feeling a bit restricted by not being able to specify that the base of the bottom layer should be different (since I predict that it will only have dense block rarely so needs more in the base).
                                       'growth_rate': 8,
                                       'num_ch_initter': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'base_kern_pad_dict': base_kern_pad_dict,
                                       'main_kern_dict': main_kern_dict,
                                       'mod_connect_dict': mod_connect_dict,
                                       'mod_cct_status_dict': mod_cct_status_dict,
                                       'mod_num_lyr_dict': mod_num_lyr_dict,
                                       'cubic_base_kern_pad_dict': cubic_base_kern_pad_dict,
                                       'cubic_main_kern_dict': cubic_main_kern_dict,
                                       'cubic_mod_connect_dict': cubic_mod_connect_dict,
                                       'cubic_mod_cct_status_dict': cubic_mod_cct_status_dict,
                                       'cubic_mod_num_lyr_dict': cubic_mod_num_lyr_dict,
                                       'spec_norm_reg': False}
            # dict_len_check(args)
    if len(args.sampling_step_size) == 1:
        vars(args)['sampling_step_size'] = args.sampling_step_size * len(args.state_sizes)
    if len(args.sigma) == 1:
        vars(args)['sigma'] = args.sigma * len(args.state_sizes)
    if len(args.momentum_param) == 1:
        vars(args)['momentum_param'] = args.momentum_param * len(args.state_sizes)


    # Print final values for args
    for k, v in zip(vars(args).keys(), vars(args).values()):
        print(str(k) + '\t' * 2 + str(v))

    return args


def main():
    ### Parse CLI arguments.
    parser = argparse.ArgumentParser(description='Deep Attractor Network.')
    #TODO before github publication, put options in here for directory strings
    # so that it's obvious where people need to put in local-dependent input.
    # Note that you should already have instructed them to make a directory
    # structure, so may be able to use the same strings as in your system.
    sgroup = parser.add_argument_group('Sampling options')
    sgroup.add_argument('--sampling_step_size', type=float, default=10, nargs='+',
                        help='The amount that the network is moves forward ' +
                             'according to the activity gradient defined by ' +
                             'the partial derivative of the Hopfield-like ' +
                             'energy. Default: %(default)s.'+
                             'When randomizing, the following options define'+
                             'a range of indices and the random value ' +
                             'assigned to the argument will be 10 to the ' +
                             'power of the float selected from the range. ' +
                             'Options: [-3, 0.5].')
    sgroup.add_argument('--num_it_neg', type=int, metavar='N', default=30,
                        help='The default number of iterations the networks' +
                             'runs in the negative (sampling) phase when ' +
                             'no adaptive iteration length is used. ' +
                             'Default: %(default)s.'+
                             'When randomizing, the following options define'+
                             'a range of integers from which the random value'+
                             'will be sampled. Options: [3, 300]. ')
    sgroup.add_argument('--randomize_neg_its', action='store_true',
                        help='If true, samples the number of negative  '+
                             'iterations every batch from a Poisson distrib'
                             '(but with a minimum of 1) using num_it_neg '+
                             'as the mean. Default: %(default)s.')
    parser.set_defaults(randomize_neg_its=False)
    sgroup.add_argument('--num_it_pos', type=int, metavar='N', default=30,
                        help='The default number of iterations the networks' +
                             'runs in the positive (inference) phase when ' +
                             'no adaptive iteration length is used. ' +
                             'Default: %(default)s. ' +
                             'When randomizing, the following options define' +
                             'a range of integers from which the random value'+
                             'will be sampled. Options: [2, 100]. ')


    tgroup = parser.add_argument_group('Training options')
    tgroup.add_argument('--require_special_name', action='store_true',
                        help='If true, asks for a description of what is ' +
                             'special about the '+
                             'experiment, if anything. Default: %(default)s.')
    parser.set_defaults(require_special_name=False)
    tgroup.add_argument('--epochs', type=int, metavar='N', default=2,
                        help='Number of training epochs. ' +
                             'Default: %(default)s.')
    tgroup.add_argument('--batch_size', type=int, metavar='N', default=128,
                        help='Training batch size. Default: %(default)s.')
    tgroup.add_argument('--lr', type=float, default=1e-3, nargs='+',
                        help='Learning rate of optimizer. Default: ' +
                             '%(default)s.' +
                             'When randomizing, the following options define'+
                             'a range of indices and the random value assigned'+
                             'to the argument will be 10 to the power of the'+
                             'float selected from the range. Options: [-3, 0.2].')
    tgroup.add_argument('--lr_decay_gamma', type=float, default=0.97,
                        help='The rate fo decay for the learning rate. Default: ' +
                             '%(default)s.')

    tgroup.add_argument('--dataset', type=str, default="CIFAR10",
                        help='The dataset the network will be trained on.' +
                             ' Default: %(default)s.')
    tgroup.add_argument('--l2_reg_energy_param', type=float, default=1.0,
                        help='Scaling parameter for the L2 regularisation ' +
                             'term placed on the energy values. Default: ' +
                             '%(default)s.'+
                             'When randomizing, the following options define' +
                             'a range of indices and the random value assigned' +
                             'to the argument will be 10 to the power of the' +
                             'float selected from the range. '+
                             'Options: [-6, -2].')
    tgroup.add_argument('--clip_grad', action='store_true',
                        help='If true, the gradients used in inference ' +
                             'are clipped to the value set by the param' +
                             '"clip_state_grad_norm".')
    parser.set_defaults(clip_grad=False)
    tgroup.add_argument('--clip_state_grad_norm', type=float, default=0.01,
                        help='The maximum norm value to clip ' +
                             'the state gradients at. Default: ' +
                             '%(default)s.')
    tgroup.add_argument('--initializer', type=str, default="random",
                        help='The type of initializer used to init the state'+
                             ' variables at the start of each minibatch.' +
                             'Options:  [zeros, random, previous, ' +
                             'persistent_particles]. ' +
                             ' Default: %(default)s.')
    tgroup.add_argument('--initter_network_lr', type=float, default=0.01,
                        help='Learning rate to pass to the Adam optimizer ' +
                             'used to train the InitializerNetwork. Default: ' +
                             '%(default)s.')
    tgroup.add_argument('--initter_network_layer_norm', action='store_true',
                        help='Puts layer normalization on the layers of the initter network'+
                             'Default: %(default)s.') #TODO consider removing
    parser.set_defaults(initter_network_layer_norm=False)
    tgroup.add_argument('--initter_network_weight_norm', action='store_true',
                        help='Puts weight normalization on the layers of the inittier network'+
                             'Default: %(default)s.')
    parser.set_defaults(initter_network_weight_norm=False)
    tgroup.add_argument('--pretrain_initializer', action='store_true',
                        help='If true, trains the feedforward initializer '+
                             'for a given number of steps before the training '
                             'of the main network starts. '+
                             'Default: %(default)s.')
    parser.set_defaults(pretrain_initializer=False)
    tgroup.add_argument('--num_initter_pretraining_batches', type=int,
                        default=30,
                        help='Learning rate to pass to the Adam optimizer ' +
                             'used to train the InitializerNetwork. Default: '+
                             '%(default)s.')
    tgroup.add_argument('--cd_mixture', action='store_true',
                        help='If true, some samples from the positive phase ' +
                             'are used to initialise the negative phase. ' +
                             'Default: %(default)s.')
    parser.set_defaults(cd_mixture=False)
    tgroup.add_argument('--pos_buffer_frac', type=float, default=0.0,
                        help='The fraction of images from the positive buffer to use to initialize negative samples.'
                             'Default: %(default)s.')
    tgroup.add_argument('--shuffle_pos_frac', type=float, default=0.0,
                        help='The fraction of images from the positive buffer that will be shuffled before initializing negative samples. The motivation for this is for experiments when a new image will initialize the state but the previous values for latent variables will be used, as would happen when an animal is presented with a new image.'
                             'Default: %(default)s.')
    tgroup.add_argument('--neg_ff_init', action='store_true',
                        help='If true, use the initializer network (if it ' +
                             'to initialise the states in the negative phase. ' +
                             'Default: %(default)s.')
    parser.set_defaults(neg_ff_init=False)
    tgroup.add_argument('--truncate_pos_its', action='store_true',
                        help='If true, the positive phase is cut short if ' +
                             'the energy stops decreasing. ' +
                             'Default: %(default)s.')
    parser.set_defaults(truncate_pos_its=False)
    tgroup.add_argument('--truncate_pos_its_threshold', type=float, default=20.,
                        help='If true, the positive phase is cut short if ' +
                             'the energy stops decreasing. ' +
                             'Default: %(default)s.')


    ngroup = parser.add_argument_group('Network and states options')
    ngroup.add_argument('--network_type', type=str, default="BengioFischer",
                        help='The type of network that will be used. Options: ' +
                             '[BengioFischer, VectorField, DAN]'
                             'Default: %(default)s.')
    ngroup.add_argument('--architecture', type=str, default="cifar10_2_layers",
                        help='The type of architecture that will be built. Options: ' +
                             '[mnist_2_layers_small, cifar10_2_layers, mnist_1_layer_small]'
                             'Default: %(default)s.')
    ngroup.add_argument('--states_activation', type=str, default="hardsig",
                        help='The activation function. Options: ' +
                             '[hardsig, hardtanh, relu, swish]'
                             'Default: %(default)s.')
    ngroup.add_argument('--activation', type=str, default="leaky_relu",
                        help='The activation function. Options: ' +
                             '[relu, swish, leaky_relu]'
                             'Default: %(default)s.')
    ngroup.add_argument('--sigma', type=float, default=0.005, nargs='+',
                        help='Sets the scale of the noise '
                             'in the network.'+
                             'When randomizing, the following options define' +
                             'a range of indices and the random value assigned' +
                             'to the argument will be 10 to the power of the' +
                             'float selected from the range. '+
                             'Options: [-3, 0].')
    ngroup.add_argument('--energy_scaling', action='store_true',
                        help='Whether or not scale the energy outputs.')
    parser.set_defaults(energy_scaling=False)
    ngroup.add_argument('--energy_scaling_noise', action='store_true',
                        help='Whether or not add noise to the energy scaling.')
    parser.set_defaults(energy_scaling_noise=False)
    ngroup.add_argument('--energy_scaling_noise_var', type=float, default=0.5,
                        help='The variance of the noise that will be added ' +
                        'to the energy mask, if noise is added at all.')
    ngroup.add_argument('--energy_weight_mask', type=int, nargs='+',
                        default=[1,1,1], help='A list that will be used to' +
                        'define a Boolean mask over the energy weights, ' +
                        'allowing you to silence the energy contributions' +
                        ' of certain state layers selectively.' +
                        ' Default: %(default)s.')
    ngroup.add_argument('--state_optimizer', type=str, default='sgd',
                        help='The kind of optimizer to use to descend the '+
                        'energy landscape. You can implement Langevin '+
                        'dynamics by choosing "sgd" and setting the right '+
                        'noise and step size. Note that in the IGEBM paper, '+
                        'I don\'t think they used true Langevin dynamics due'+
                        ' to their choice of noise and step size.')

    ngroup.add_argument('--printing_grad_mom_info', action='store_true',
                        help='Whether or not to print gradient and mom info.')
    parser.set_defaults(printing_grad_mom_info=False)
    ngroup.add_argument('--add_gradient_noise', action='store_true',
                        help='Whether or not add noise to the states gradients.')
    parser.set_defaults(add_gradient_noise=False)
    ngroup.add_argument('--momentum_param', type=float, default=1.0, nargs='+',
                        help='')
    ngroup.add_argument('--dampening_param', type=float, default=0.0,
                        help='')
    ngroup.add_argument('--mom_clip', action='store_true',
                        help='Whether or not clip the sghmc norm.')
    parser.set_defaults(mom_clip=False)
    ngroup.add_argument('--mom_clip_vals', type=float, default=[2.0,
                            10., 14.676934, 3.0, 5., 2.], nargs='+',
                        help='The maximum norm of the momentum permitted.')
    ngroup.add_argument('--maxminstate_to_zeromom', action='store_true',
                        help='Whether or not to cut the momentum when ' +
                        'the state hits a max or min allowed value. ')
    parser.set_defaults(maxminstate_to_zeromom=False)
    ngroup.add_argument('--non_diag_inv_mass', action='store_true',
                        help='Whether or not to use a non diagonal mass ' +
                        'matrix. The analogy is imposing lateral inhibition' +
                        'on the dynamics of the network.')
    parser.set_defaults(non_diag_inv_mass=False)



    # ngroup.add_argument('--no_spec_norm_reg', action='store_true',
    #                     help='If true, networks are NOT subjected to ' +
    #                          'spectral norm regularisation. ' +
    #                          'Default: %(default)s.')
    # parser.set_defaults(no_spec_norm_reg=False)
    ngroup.add_argument('--no_forced_symmetry', action='store_true',
                        help='If true, the backwards nets in ConvBFN are' +
                             'spectral norm regularisation. ' +
                             'Default: %(default)s.')
    parser.set_defaults(no_forced_symmetry=False)
    ngroup.add_argument('--no_end_layer_activation', action='store_true',
                        help='If true, there is no activation place on the ' +
                             'final layer of the quadratic nets in the DAN. ' +
                             'Default: %(default)s.') #For old DAN
    parser.set_defaults(no_end_layer_activation=False)
    ngroup.add_argument('--model_weight_norm', action='store_true',
                        help='If true, weight normalization is placed on ' +
                             'the quadratic networks of the DAN. ' +
                             'Default: %(default)s.')
    parser.set_defaults(model_weight_norm=False)
    ngroup.add_argument('--scale_grad', type=float, default=1.0,
                        help='The scale_grad parameter for the adaptive '+
                             'sghmc optimizer')
    ngroup.add_argument('--num_burn_in_steps', type=float, default=0.0,
                        help='The number of burnin steps for the adaptive ' +
                             'sghmc optimizer')
    ngroup.add_argument('--min_sq_sigma', type=float, default=1e-16,
                        help='The minimum variance of the noise added in ' +
                             'every step of the sghmc optimizer')
    ngroup.add_argument('--state_scales', type=float, default=[1,1,1,1,1,1], nargs='+',
                        help='The amount by which to scale the [0,1] noise ' +
                             'used to initialise the ')

    vgroup = parser.add_argument_group('Visualization options')
    vgroup.add_argument('--viz', action='store_true',
                        help='Whether or not to do visualizations. The exact'
                             'type of visualization is defined in the'
                             '"viz_type" argument. Default: %(default)s.')
    parser.set_defaults(viz=False)
    vgroup.add_argument('--viz_tempered_annealing', action='store_true',
                        help='Whether or not to use tempered annealing during'
                             ' viz. Default: %(default)s.')
    parser.set_defaults(viz_tempered_annealing=False)
    vgroup.add_argument('--viz_temp_decay', type=float, default=0.995,
                        help='The rate of decay for the temperature in '
                             'simulated annealing process during ' +
                             'vizualisation. Default: %(default)s.')
    vgroup.add_argument('--viz_type', type=str, default='standard',
                        help='The type of visualization you want to perform.'
                        ' "standard": Generates random samples with no'
                             'restrictions.\n\n "neurons": Generates samples'
                             ' where there is an extra gradient that seeks to '
                             'maximise the energy of a certain neuron while the '
                             'value of other neurons is free to find a local '
                             'minimum.\n\n "channels": Generates samples'
                             ' where there is an extra gradient that seeks to '
                             'maximise the energy of a certain feature layer in '
                             'energy functions that are conv nets while the '
                             'value of other neurons is free to find a local '
                             'minimum'
                              )
    vgroup.add_argument('--num_viz_samples', type=int,
                        help='The number of samples that should be generated'
                             'and visualized. ' +
                             'Default: %(default)s.')
    vgroup.add_argument('--num_it_viz', type=int,
                        help='The number of steps to use to sample images. ' +
                             'Default: %(default)s.')
    vgroup.add_argument('--viz_img_logging_step_interval', type=int, default=1,
                        help='The interval at which to save images that ' +
                             'are being sampled during visualization. '+
                             'Default: %(default)s.')
    vgroup.add_argument('--viz_start_layer', type=int, default=1,
                        help='The state layer on which to start visualization. ' +
                             'Default: %(default)s.')


    vgroup = parser.add_argument_group('Weight Visualization options')
    vgroup.add_argument('--weight_viz', action='store_true',
                        help='Whether or not to do visualizations of the '
                             'weight matrices of the network.'
                             'Default: %(default)s.')
    parser.set_defaults(weight_viz=False)


    mgroup = parser.add_argument_group('Miscellaneous options')
    mgroup.add_argument('--randomize_args', type=str, nargs='+', default=[],
                        help='List of CLI args to pass to the random arg ' +
                             'generator. Default: %(default)s.',
                        required=False)
    mgroup.add_argument('--override_loaded', type=str, nargs='+', default=[],
                        help='List of CLI args to that will take the current'+
                             'values and not the values of the loaded '+
                             'argument dictionary. Default: %(default)s.',
                        required=False) # when viz, must use: num_it_viz viz_img_logging_step_interval viz_type
    ngroup.add_argument('--sample_buffer_prob', type=float, default=0.95,
                        help='The probability that the network will be ' +
                             'initialised from the buffer instead of from '+
                             'random noise.')
    mgroup.add_argument('--tensorboard_log_dir', type=str,
                        default='exps/tblogs',
                        help='The path of the directory into which '+
                             'tensorboard logs are saved. Default:'+
                             ' %(default)s.',
                        required=False)
    mgroup.add_argument('--log_spec_neurons', action='store_true',
                        help='Whether or not to log values for specific ' +
                             'neurons and their momenta.')
    parser.set_defaults(log_spec_neurons=False)
    mgroup.add_argument('--log_histograms', action='store_true',
                        help='Whether or not to log histograms of weights ' +
                             'and other variables. Warning: Storage intensive.')
    parser.set_defaults(log_histograms=False)
    mgroup.add_argument('--histogram_logging_interval', type=int, default=40,
                        help='The size of the intervals between the logging ' +
                             'of histogram data.') #On Euler do around 1000
    mgroup.add_argument('--scalar_logging_interval', type=int, default=1,
                        help='The size of the intervals between the logging ' +
                             'of scalar data.') #On Euler do around 100
    mgroup.add_argument('--img_logging_interval', type=int, default=100,
                        help='The size of the intervals between the logging ' +
                             'of image samples.')
    mgroup.add_argument('--save_pos_images', action='store_true',
                        help='Whether or not to save images from the ' +
                             'positive phases.')
    parser.set_defaults(save_pos_images=False)
    mgroup.add_argument('--model_save_interval', type=int, default=100,
                        help='The size of the intervals between the model '+
                             'saves.')
    mgroup.add_argument('--load_model', type=str,
                        help='The name of the model that you want to load.'+
                        'The file extension should not be included.')
    ngroup.add_argument('--no_train_model', action='store_true',
                        help='Whether or not to train the model ')
    parser.set_defaults(no_train_model=False)
    ngroup.add_argument('--gen_exp_stim', action='store_true',
                        help='Whether or not to generate the artificial '+
                        'stimuli for experiment.')
    parser.set_defaults(gen_exp_stim=False)
    ngroup.add_argument('--experiment', action='store_true',
                        help='Whether or not to run experiments ')
    parser.set_defaults(experiment=False)



    xgroup = parser.add_argument_group('Options that will be determined ' +
                                       'post hoc')
    xgroup.add_argument('--use_cuda', action='store_true',
                        help='Flag to enable GPU usage.')
    xgroup.add_argument('--special_name', type=str, metavar='N',
                        default="None",
                        help='A description of what is special about the ' +
                             'experiment, if anything. Default: %(default)s.')
    xgroup.add_argument('--state_sizes', type=list, nargs='+', default=[[]],#This will be filled by default. it's here for saving
                        help='Number of units in each hidden layer of the ' +
                             'network. Default: %(default)s.')
    xgroup.add_argument('--arch_dict', type=dict, default={})

    args = finalize_args(parser)

    if args.use_cuda:
        device = 'cuda'
    else:
        device = 'cpu'

    # Set up the tensorboard summary writer and log dir
    model_name = lib.utils.datetimenow() + '__rndidx_' + str(np.random.randint(0,99999))
    print(model_name)
    writer = SummaryWriter(args.tensorboard_log_dir + '/' + model_name)
    sample_log_dir = os.path.join('exps', 'samples', model_name)
    if not os.path.isdir(sample_log_dir):
        os.mkdir(sample_log_dir)

    # Set up model
    if args.network_type == 'BengioFischer':
        model = models.BengioFischerNetwork(args, device, model_name, writer).to(
        device)
    elif args.network_type == 'ConvBFN':
        model = models.ConvBengioFischerNetwork(args, device, model_name, writer).to(
            device)
    elif args.network_type == 'VectorField':
        model = models.VectorFieldNetwork(args, device, model_name, writer).to(
            device)
    elif args.network_type == 'DAN':
        model = models.DeepAttractorNetwork(args, device, model_name, writer).to(
            device)
    elif args.network_type == 'DAN2':
        model = models.DeepAttractorNetworkTakeTwo(args, device, model_name, writer).to(
            device)
    elif args.network_type == 'FHG':
        model = models.FactorHyperGraph(args, device, model_name, writer).to(
            device)
    elif args.network_type == 'SVF':
        model = models.StructuredVectorFieldNetwork(args, device, model_name,
                                                writer).to(device)
    else:
        raise ValueError("Invalid CLI argument for argument 'network_type'. ")

    # Set up dataset
    data = Dataset(args)
    buffer = SampleBuffer(args, device=device)

    if not args.no_train_model:
        # Train the model
        tm = managers.TrainingManager(args, model, data, buffer, writer, device,
                             sample_log_dir)
        tm.train()
    if args.viz:
        vm = managers.VisualizationManager(args, model, data, buffer, writer, device,
                                  sample_log_dir)
        vm.visualize()
    if args.weight_viz:
        args = finalize_args(parser)
        wvm = managers.WeightVisualizationManager(args, model, data, buffer,
                                                  writer,
                                                  device,
                                                  sample_log_dir)
        wvm.visualize_base_weights()
        #wvm.visualize_weight_pretrained()

    if args.gen_exp_stim:
        esgm = managers.ExperimentalStimuliGenerationManager()
        # esgm.generate_double_gabor_dataset__loc_and_angles()
        esgm.generate_single_gabor_dataset__just_angle()
        #esgm.generate_single_gabor_dataset__contrast_and_angle()

    if args.experiment:
        # Re-instantiate dataset now using no randomness so that the same batch
        # is used for all experiments
        expm = managers.ExperimentsManager(args, model, data, buffer,
                                                  writer,
                                                  device,
                                                  sample_log_dir)
        expm.orientations_present_single_gabor()
        #expm.observe_cifar_pos_phase()

        # # Reset parameters and create new model so that
        # # previous experiment isn't overwritten
        # vars(args)['state_optimizer'] = 'sgd'
        # vars(args)['momentum_param']  = 0.0
        # model_name = lib.utils.datetimenow() + '__rndidx_' + str(
        #     np.random.randint(0, 99999))
        #
        # model = models.DeepAttractorNetworkTakeTwo(args, device, model_name,
        #                                            writer).to(device)
        # expm = managers.ExperimentsManager(args, model, data, buffer,
        #                                           writer,
        #                                           device,
        #                                           sample_log_dir)
        # expm.observe_cifar_pos_phase()

shapes = lambda x : [y.shape for y in x]
nancheck = lambda x : (x != x).any()
listout = lambda y : [x for x in y]
gradcheck = lambda  y : [x.requires_grad for x in y]
leafcheck = lambda  y : [x.is_leaf for x in y]
existgradcheck = lambda  y : [(x.grad is not None) for x in y]
existgraddatacheck = lambda  y : [(x.grad.data is not None) for x in y]
divl = lambda l1, l2: torch.prod(torch.tensor(l1)).float()/torch.prod(torch.tensor(l2)).float()

if __name__ == '__main__':
    main()
    #TODO go through all the code and fix the comments and docstrings. Some
    # of the comments are inaccurate.



# When doing similar experiments in future, use the following infrastructure
# For keeping track of experiments, you should have these variables:
# Session name (unique for every time you press 'run')
# Model name (If a brand new model, it is the session name; if it is reloading for training purposes, it is the session name; if reloading for analysis or viz purposes, it is the original model name)
# Model history (keeps track of the history of models/session used in training this model)

# Also use config files rather than having to go into the pycharm configs every time. It's not very user friendly. And it'll be easier to save config files.
# Plus when you make new configs during development, make them default to some value that nullifies them