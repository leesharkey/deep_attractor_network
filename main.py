import argparse
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import lib.networks.models as models
from lib.data import SampleBuffer, Dataset
import lib.utils
import lib.managers as managers


def dict_len_check(args):
    # check the dicts are the right len
    num_layers = len(args.state_sizes)
    for l in range(num_layers):
        inps           = args.arch_dict['mod_connect_dict'][l]
        # base_kern_pads = args.arch_dict['base_kern_pad_dict'][l]
        #
        # if not \
        #     len(inps)==len(base_kern_pads):
        #     str1 = "Layer %i architecture dictionaries invalid. " % l
        #     raise ValueError(str1 +
        #                      "inp_state_shapes, and " +
        #                      "base_kern_pads must be the same length. Check " +
        #                      "that the architecture dictionary defines these "+
        #                      "correctly.")


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
    elif args.architecture == 'SSN_development1':
        vars(args)['state_sizes'] = [[args.batch_size, 3, 32, 32],
                                     [args.batch_size, 32, 32, 32],
                                     [args.batch_size, 16, 8, 8]]
        mod_connect_dict = {0: [1, 2],
                            1: [0, 1, 2],
                            2: [1, 2]} # all must have self connections
        exc_kern_pad_dict = {0: [[7, 3], [7, 3]],
                              1: [[7, 3], [11, 5], [7, 3]],
                              2: [[3, 1], [3, 1]]}
        inh_kern_pad_dict = {0: [[7, 3], [7, 3]],
                              1: [[7, 3], [7, 3], [7, 3]],
                              2: [[3, 1], [3, 1]]}
        vars(args)['arch_dict'] = {'num_ch_base': 32,
                                   # Feeling a bit restricted by not being able to specify that the base of the bottom layer should be different (since I predict that it will only have dense block rarely so needs more in the base).
                                   'num_ch_initter': 32,
                                   'num_sl': len(args.state_sizes) - 1,
                                   'exc_kern_pad_dict': exc_kern_pad_dict,
                                   'inh_kern_pad_dict': inh_kern_pad_dict,
                                   'mod_connect_dict': mod_connect_dict,
                                   'spec_norm_reg': False}
        dict_len_check(args)
    elif args.architecture == 'SSN_development2':
        vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                     [args.batch_size, 32, 28, 28],
                                     [args.batch_size, 16, 8, 8]]
        mod_connect_dict = {0: [1, 2],
                            1: [0, 1, 2],
                            2: [1, 2]} # all must have self connections
        exc_kern_pad_dict = {0: [[7, 3], [7, 3]],
                              1: [[7, 3], [11, 5], [7, 3]],
                              2: [[3, 1], [3, 1]]}
        inh_kern_pad_dict = {0: [[7, 3], [7, 3]],
                              1: [[7, 3], [7, 3], [7, 3]],
                              2: [[3, 1], [3, 1]]}
        vars(args)['arch_dict'] = {'num_ch_base': 32,
                                   # Feeling a bit restricted by not being able to specify that the base of the bottom layer should be different (since I predict that it will only have dense block rarely so needs more in the base).
                                   'num_ch_initter': 32,
                                   'num_sl': len(args.state_sizes) - 1,
                                   'exc_kern_pad_dict': exc_kern_pad_dict,
                                   'inh_kern_pad_dict': inh_kern_pad_dict,
                                   'mod_connect_dict': mod_connect_dict,
                                   'spec_norm_reg': False}
        dict_len_check(args)


    if len(args.sampling_step_size) == 1:
        vars(args)['sampling_step_size'] = args.sampling_step_size * len(args.state_sizes)
    if len(args.sigma) == 1:
        vars(args)['sigma'] = args.sigma * len(args.state_sizes)

    # Print final values for args
    for k, v in zip(vars(args).keys(), vars(args).values()):
        print(str(k) + '\t' * 2 + str(v))

    return args

def main():
    ### Parse CLI arguments.
    parser = argparse.ArgumentParser(description='Stabilised Supralinear network in the style of an EBM.')
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
    sgroup.add_argument('--ff_dynamics', action='store_true',
                        help='If true, then the state updates are calculated  '+
                             'using ff nets rather than as the gradients of '
                             'those nets'+
                             'Default: %(default)s.')
    parser.set_defaults(ff_dynamics=False)


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

    tgroup.add_argument('--weights_optimizer', type=str, default="adam",
                        help='The optimizer used to train the weights and ' +
                             'biases (as opposed to the one used during ' +
                             'sampling. Default: %(default)s.')
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
    tgroup.add_argument('--trunc_pos_history_len', type=int, default=300,
                        help='The length of the history that is used to ' +
                             'determine whether the positive iterations  ' +
                             'will be truncated. Default: %(default)s.')


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
    ngroup.add_argument('--supra_k', type=float, default=0.005, nargs='+',
                        help='Sets the scale activation function'
                             'in the network.')
    ngroup.add_argument('--supra_n', type=float, default=3, nargs='+',
                        help='Sets the scale of the supralinearity of the ' +
                             'activation function in the network.')
    ngroup.add_argument('--weights_gamma_alpha', type=float, default=2, nargs='+',
                        help='Sets the shape parameter of the gamma ' +
                             'distribution used to initalise conv nets.')
    ngroup.add_argument('--weights_gamma_beta', type=float, default=0.5, nargs='+',
                        help='Sets the rate parameter of the gamma ' +
                             'distribution used to initalise conv nets.')
    ngroup.add_argument('--weights_mag_scale', type=float, default=0.04, nargs='+',
                        help='Sets the sacle of the initialised weights of ' +
                             'the EI conv nets.')
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
    ngroup.add_argument('--non_diag_inv_mass', action='store_true',
                        help='Whether or not to use a non diagonal mass ' +
                        'matrix. The analogy is imposing lateral inhibition' +
                        'on the dynamics of the network.')
    parser.set_defaults(non_diag_inv_mass=False)
    ngroup.add_argument('--mean_batch_minv_t', action='store_true',
                        help='Whether or not to use the mean minv_t' +
                        'value for all batches. This should make it more' +
                        'consistent.')
    parser.set_defaults(mean_batch_minv_t=False)


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
    ngroup.add_argument('--max_sq_sigma', type=float, default=[100.], nargs='+',
                        help='The maximum variance of the noise added in ' +
                             'every step of the sghmc optimizer')
    ngroup.add_argument('--min_sq_sigma', type=float, default=1e-16,
                        help='The minimum variance of the noise added in ' +
                             'every step of the sghmc optimizer')

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
    mgroup.add_argument('--exp_data_root_path', type=str,
                        default='/media/lee/DATA/DDocs/AI_neuro_work/DAN/exp_data',
                        help='The path of the directory into which '+
                             'experimental data are saved, e.g. traces. Default:'+
                             ' %(default)s.',
                        required=False)

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
    model = models.SSNEBM(args, device, model_name, writer).to(
        device)

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
        #TODO remove hashtags for final version

        esgm.generate_single_gabor_dataset__just_angle()
        #esgm.generate_single_gabor_dataset__contrast_and_angle()
        # esgm.generate_single_gabor_dataset__just_angle_few_angles()
        # esgm.generate_single_gabor_dataset__long_just_fewangles()
        # esgm.generate_double_gabor_dataset__fewlocs_and_fewerangles()


    if args.experiment:
        # Re-instantiate dataset now using no randomness so that the same batch
        # is used for all experiments
        expm = managers.ExperimentsManager(args, model, data, buffer,
                                                  writer,
                                                  device,
                                                  sample_log_dir)
        # expm.orientations_present("single", "just_angle")
        # expm.orientations_present("single", "contrast_and_angle")
        expm.orientations_present("single", "just_angle_few_angles")
        expm.orientations_present("double", "fewlocs_and_fewerangles")
        expm.orientations_present("single", "long_just_fewangles")

        ##Just putting this here so I can leave running ovenight. todo remove
        # vars(args)['viz_type'] = 'standard'
        # vm = managers.VisualizationManager(args, model, data, buffer, writer, device,
        #                           sample_log_dir)
        # vm.visualize()
        # vars(args)['viz_type'] = 'channels_energy'
        # vm = managers.VisualizationManager(args, model, data, buffer, writer, device,
        #                           sample_log_dir)
        # vm.visualize()
        # vars(args)['viz_type'] = 'channels_state'
        # vm = managers.VisualizationManager(args, model, data, buffer, writer, device,
        #                           sample_log_dir)
        # vm.visualize()
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

if __name__ == '__main__':
    main()
    #TODO go through all the code and fix the comments and docstrings. Some
    # of the comments are inaccurate.



# When doing similar experiments in future, use the following infrastructure
# For keeping track of experiments, you should have these variables:
# Session name (unique for every time you press 'run')
# Model name (If a brand new model, it is the session name; if it is reloading for training purposes, it is the session name; if reloading for analysis or viz purposes, it is the original model name)
# Model history (keeps track of the history of models/session used in training this model)

# Also use config files rather than having to go into the pycharm configs every time. It's not very user friendly. And it'll be easier to save config files. Have a master config file that gets copied and saved to the experimental session.
# Plus when you make new configs during development, make them default to some value that nullifies them