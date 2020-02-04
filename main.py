import argparse
import os
import random
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import lib.networks as nw
from lib.data import SampleBuffer, sample_data, Dataset
import lib.utils


class TrainingManager():
    def __init__(self, args, model, data, buffer, writer, device, sample_log_dir):
        self.args = args
        self.model = model
        self.data = data
        self.writer = writer
        self.buffer = buffer
        self.device = device
        self.sample_log_dir = sample_log_dir
        self.parameters = model.parameters()
        self.optimizer = optim.Adam(self.parameters, lr=args.lr, betas=(0.9, 0.999))
        self.noises = lib.utils.generate_random_states(self.args.state_sizes, self.device)
        self.global_step = 0
        self.batch_num = 0

        if args.initializer == 'ff_init':
            self.initter = nw.InitializerNetwork(args, writer, device)
            self.initter.to(device)
        else:
            self.initter = None

        if self.args.load_model:
            loaded_model_name = str(self.args.load_model)
            path = 'exps/models/' + loaded_model_name + '.pt'
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['model_optimizer'])

            if args.initializer == 'ff_init':
                self.initter.load_state_dict(checkpoint['initializer'])
                self.initter.optimizer.load_state_dict(
                    checkpoint['initializer_optimizer'])

            self.args = checkpoint['args'] #TODO ensure that you don't want to implement something that defines which of the previous args to overwrite
            self.batch_num = checkpoint['batch_num']
            print("Loaded model " + loaded_model_name + ' successfully')
            lib.utils.save_configs_to_csv(self.args, loaded_model_name)
        else:
            lib.utils.save_configs_to_csv(self.args, self.model.model_name)

    def train(self):
        save_dict = self.make_save_dict()
        path = 'exps/models/' + self.model.model_name + '.pt'
        torch.save(save_dict, path)
        prev_states = None
        for batch, (pos_img, pos_id) in self.data.loader:

            pos_states, pos_id = self.positive_phase(pos_img, pos_id,
                                                     prev_states)

            neg_states, neg_id = self.negative_phase()
            self.param_update_phase(neg_states, neg_id, pos_states, pos_id)

            prev_states = pos_states # In case pos init uses prev states

            if self.batch_num % self.args.img_logging_interval == 0:
                utils.save_image(neg_states[0].detach().to('cpu'), os.path.join(self.sample_log_dir, str(self.batch_num).zfill(6) + '.png'), nrow=16, normalize=True, range=(0, 1))
                if self.args.save_pos_images:
                    utils.save_image(pos_states[0].detach().to('cpu'), os.path.join(self.sample_log_dir, 'p0_' + str(self.batch_num).zfill(6) + '.png'), nrow=16, normalize=True, range=(0, 1))
            if self.batch_num % self.args.model_save_interval == 0:
                # Save
                save_dict = self.make_save_dict()
                path = 'exps/models/' + self.model.model_name + '.pt'
                torch.save(save_dict, path)

            self.batch_num += 1
            if self.batch_num >= 1e6:
                # at batchsize=128 there are 390 batches
                # per epoch, so at 1e6 max batches that's 2.56k epochs.
                break

        # Save when training is complete too
        save_dict = self.make_save_dict()
        path = 'exps/models/' +self.model.model_name + '.pt'
        torch.save(save_dict, path)

    def initialize_pos_states(self, pos_img=None, pos_id=None,
                              prev_states=None):
        """Initializes positive states"""

        pos_states = [pos_img]

        if self.args.initializer == 'zeros': # probs don't use this now that you're initting all states
            zero_states = [torch.zeros(size, device=self.device,
                                       requires_grad=True)
                           for size in self.args.state_sizes[1:]]
            pos_states.extend(zero_states)
        #requires_grad(pos_states, True)
        if self.args.initializer == 'ff_init':
            # Later consider implementing a ff_init that is trained as normal
            # and gives the same output but that only a few (maybe random)
            # state neurons are changed/clamped by the innitter so that you
            # can still use the 'previous' initialisation in your experiments
            self.initter.train()
            pos_states_new = self.initter.forward(pos_img, pos_id)
            pos_states.extend(pos_states_new)
        elif self.args.initializer == 'middle':
            raise NotImplementedError("You need to find the mean pixel value and then use the value as the value to init all pixels.")
        elif self.args.initializer == 'mix_prev_middle':
            raise NotImplementedError
        elif self.args.initializer == 'previous' and prev_states is not None:
            pos_states.extend(prev_states[1:])
        else:  # self.args.initializer == 'random':
            rand_states = lib.utils.generate_random_states(
                self.args.state_sizes[1:], self.device)
            pos_states.extend(rand_states)

        return pos_states

    def positive_phase(self, pos_img, pos_id, prev_states=None):

        print('\nStarting positive phase...')
        # Get the loaded pos samples and put them on the correct device
        pos_img, pos_id = pos_img.to(self.device), pos_id.to(self.device)

        # Gets the values of the pos states by running a short inference phase
        # conditioned on a particular state layer
        pos_states_init = self.initialize_pos_states(pos_img=pos_img,
                                                prev_states=prev_states) #Maybe remove cond_layer arg if keeping pos_phase0 as initter
        pos_states = [psi.clone().detach() for psi in pos_states_init]

        # Freeze network parameters and take grads w.r.t only the inputs
        lib.utils.requires_grad(pos_states, True)
        lib.utils.requires_grad(self.parameters, False)
        if self.args.initializer == 'ff_init':
            lib.utils.requires_grad(self.initter.parameters(), False)
        self.model.eval()

        # Get an optimizer for each statelayer
        self.state_optimizers = lib.utils.get_state_optimizers(self.args,
                                                               pos_states)

        # Positive phase sampling
        for k in tqdm(range(self.args.num_it_pos)):
            self.sampler_step(pos_states, pos_id, positive_phase=True,
                              step=self.global_step)
            self.global_step += 1

        # Stop calculting grads w.r.t. images
        for pos_state in pos_states:
            pos_state.detach_()

        # Update initializer network if present
        if self.args.initializer == 'ff_init':
            lib.utils.requires_grad(self.initter.parameters(), True)
            loss = self.initter.update_weights(outs=pos_states_init[1:],
                                               targets=pos_states[1:],
                                               step=self.global_step)

        if self.args.cd_mixture:
            print("Adding pos states to pos buffer")
            self.buffer.push(pos_states, pos_id, pos=True)

        return pos_states, pos_id

    def negative_phase(self):
        print('\nStarting negative phase...')
        # Initialize the chain (either as noise or from buffer)
        neg_states, neg_id = self.buffer.sample_buffer()

        # Freeze network parameters and take grads w.r.t only the inputs
        lib.utils.requires_grad(neg_states, True)
        lib.utils.requires_grad(self.parameters, False)
        self.model.eval()

        # Set up state optimizer
        self.state_optimizers = lib.utils.get_state_optimizers(self.args,
                                                               neg_states)

        # Negative phase sampling
        for k in tqdm(range(self.args.num_it_neg)):
            self.sampler_step(neg_states, neg_id, step=self.global_step)
            self.global_step += 1

        # Stop calculting grads w.r.t. images
        for neg_state in neg_states:
            neg_state.detach_()

        # Send negative samples to the buffer (on cpu)
        self.buffer.push(neg_states, neg_id)

        return neg_states, neg_id

    def sampler_step(self, states, ids, positive_phase=False, step=None):
        # Calculate the gradient for the Langevin step
        energies = self.model(states, ids, step)  # Outputs energy of neg sample
        total_energy = energies.sum()
        if positive_phase:
            print('\nPos Energy: ' + str(total_energy.cpu().detach().numpy()))
            if self.global_step % self.args.scalar_logging_interval == 0:
                self.writer.add_scalar('train/PosSamplesEnergy', total_energy,
                                       self.global_step)
        else:
            print('\nNeg Energy: ' + str(total_energy.cpu().detach().numpy()))
            if self.global_step % self.args.scalar_logging_interval == 0:
                self.writer.add_scalar('train/NegSamplesEnergy', total_energy,
                                       self.global_step)

        # Take gradient wrt states (before addition of noise)
        total_energy.backward()
        torch.nn.utils.clip_grad_norm_(states,
                                       self.args.clip_state_grad_norm,
                                       norm_type=2)

        # Adding noise in the Langevin step (only for non conditional
        # layers in positive phase)
        for layer_idx, (noise, state) in enumerate(zip(self.noises, states)):
            noise.normal_(0, self.args.sigma)
            if positive_phase and layer_idx == 0:
                pass
            else:
                state.data.add_(noise.data)

        # The gradient step in the Langevin/SGHMC step
        # It goes through each statelayer and steps back using its associated
        # optimizer. The gradients may pass through any network unless
        # you're using an appropriate energy mask that zeroes the grad through
        # that network in particular.
        for layer_idx, state in enumerate(states):
            if positive_phase and layer_idx == 0:
                pass
            else:
                self.state_optimizers[layer_idx].step()

        # Prepare gradients and sample for next sampling step
        for state in states:
            state.grad.detach_()
            state.grad.zero_()
            state.data.clamp_(0, 1)

    def calc_energ_and_loss(self, neg_states, neg_id, pos_states, pos_id):

        # Get energies of positive and negative samples
        pos_energy = self.model(pos_states, pos_id)
        neg_energy = self.model(neg_states, neg_id)

        # Calculate the loss
        ## L2 penalty on energy magnitudes
        loss_l2 = self.args.l2_reg_energy_param * sum([pos_energy**2,
                                                       neg_energy**2])
        loss_ml = pos_energy - neg_energy  # Maximum likelihood loss
        loss = loss_ml + loss_l2
        loss = loss.mean()

        # Calculate gradients for the network params
        loss.backward()

        # Print loss
        self.writer.add_scalar('train/loss', loss.item(), self.global_step)

        return neg_energy, pos_energy, loss

    def update_weights(self, loss):

        # Honestly don't understand what this does or where it came from
        clip_grad(self.parameters, self.optimizer)

        # Update the network params
        self.optimizer.step()

        # Ensure energy weights don't go below minimum value
        for energy_weight in self.model.energy_weights.parameters():
            energy_weight.data.clamp_(self.args.energy_weight_min)

        # Print loss
        self.data.loader.set_description(f'loss: {loss.item():.5f}')

    def param_update_phase(self, neg_states, neg_id, pos_states, pos_id):

        # Log energy weight histograms
        if self.args.log_histograms and \
        self.batch_num % self.args.histogram_logging_interval == 0:
            layer_string = 'train/energy_weights'
            self.writer.add_histogram(layer_string, self.model.energy_weights.weight,
                                      self.batch_num)

        # Put model in training mode and prepare network parameters for updates
        lib.utils.requires_grad(self.parameters, True)
        self.model.train()  # Not to be confused with self.TrainingManager.train
        self.model.zero_grad()

        # Calc energy and loss
        neg_energy, pos_energies, loss = \
            self.calc_energ_and_loss(neg_states, neg_id,
                                     pos_states, pos_id)

        # Update weights on the basis of that loss
        self.update_weights(loss)


    def make_save_dict(self):
        """Assembles the dictionary of objects to be saved"""
        save_dict = {'model': self.model.state_dict(),
                     'model_optimizer': self.optimizer.state_dict(),
                     'args': self.args,
                     'batch_num': self.batch_num}
        if self.args.initializer =='ff_init':
            initter_dict = {'initializer': self.initter.state_dict(),
                            'initializer_optimizer': self.initter.optimizer.state_dict()}
            save_dict    = {**save_dict, **initter_dict}
        return save_dict

    def pre_train_initializer(self):
        i = 0
        for batch, (pos_img, pos_id) in self.data.loader:
            print("Pretraining step")
            pos_states, pos_id = self.positive_phase(pos_img, pos_id)
            i += 1
            if i > self.args.num_pretraining_batches:
                break
        print("\nPretraining of initializer complete")


def clip_grad(parameters, optimizer):
    with torch.no_grad():
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]

                if 'step' not in state or state['step'] < 1:
                    continue

                step = state['step']
                exp_avg_sq = state['exp_avg_sq']
                _, beta2 = group['betas']

                bound = 3 * torch.sqrt(exp_avg_sq / (1 - beta2 ** step)) + 0.1
                p.grad.data.copy_(torch.max(torch.min(p.grad.data, bound), -bound))


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
    if args.dataset == 'MNIST':
        if args.architecture == 'mnist_1_layer_small':
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                         [args.batch_size, 3, 3, 3]]
            vars(args)['arch_dict'] = {'num_ch': 32,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [3, 3],
                                       'strides': [1,1],
                                       'padding': 1}
        elif args.architecture == 'mnist_2_layers_small':
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                         [args.batch_size, 9, 28, 28],
                                         [args.batch_size, 9, 3, 3]]
            vars(args)['arch_dict'] = {'num_ch': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [3, 3],
                                       'strides': [1,1],
                                       'padding': 1}
        elif args.architecture == 'mnist_2_layers_small_equal':
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                         [args.batch_size, 6, 16, 16],
                                         [args.batch_size, 256, 3, 3]]
            vars(args)['arch_dict'] = {'num_ch': 16,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [3, 3],
                                       'strides': [1, 0],
                                       'padding': 1}
        elif args.architecture == 'mnist_3_layers_small':
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                         [args.batch_size, 9, 28, 28], #scale 1
                                         [args.batch_size, 9, 3, 3], # scale 87.1
                                         [args.batch_size, 9, 1, 1]] # scale 784
            vars(args)['arch_dict'] = {'num_ch': 64,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [3, 3, 1],
                                       'strides': [1,1,1],
                                       'padding': 1}
        elif args.architecture == 'mnist_3_layers_med':
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                         [args.batch_size, 64, 28, 28],
                                         [args.batch_size, 32, 9, 9],
                                         [args.batch_size, 10, 3, 3]]
            vars(args)['arch_dict'] = {'num_ch': 64,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [3, 3, 3],
                                       'strides': [1,1,1],
                                       'padding': 0}
        elif args.architecture == 'mnist_3_layers_large': # Have roughly equal amount of 'potential energy' (i.e. neurons) in each layer
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                         [args.batch_size, 8, 28, 28],  # 6272
                                         [args.batch_size, 24, 16, 16], # 6144
                                         [args.batch_size, 180, 6, 6]]  # 4608
            vars(args)['arch_dict'] = {'num_ch': 64,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [3, 3, 3],
                                       'strides': [1,1],
                                       'padding': 1}
        elif args.architecture == 'mnist_4_layers_med': # Have roughly equal amount of 'potential energy' (i.e. neurons) in each layer
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size, 16, 28, 28],  # 12544
                                         [args.batch_size, 24, 16, 16],  # 6144
                                         [args.batch_size, 32,  9,  9],  # 2592
                                         [args.batch_size, 180, 3,  3]]  # 1620
            vars(args)['arch_dict'] = {'num_ch': 64,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [3, 3, 3],
                                       'strides': [1,1],
                                       'padding': 1}
            vars(args)['energy_weight_mask'] = [1, 2, 4.84, 7.743]
        elif args.architecture == 'mnist_5_layers_med_fc_top2': # Have roughly equal amount of 'potential energy' (i.e. neurons) in each layer
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size, 16, 28, 28],  # 12544
                                         [args.batch_size, 24, 16, 16],  # 6144
                                         [args.batch_size, 32,  9,  9],  # 2592
                                         [args.batch_size, 1024],
                                         [args.batch_size, 256]]
            mod_connect_dict = {0: [],
                                1: [0,1],
                                2: [1,2],
                                3: [2,3],
                                4: [3,4],
                                5: [4,5]}
            vars(args)['arch_dict'] = {'num_ch': 64,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [3, 3, 3],
                                       'strides': [1,1],
                                       'padding': 1,
                                       'mod_connect_dict': mod_connect_dict}
            vars(args)['energy_weight_mask'] = [1, 2, 4.84, 12.25, 49]
        elif args.architecture == 'mnist_3_layers_med_fc_top1': # Have roughly equal amount of 'potential energy' (i.e. neurons) in each layer
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size, 16, 28, 28],  # 12544
                                         [args.batch_size, 32,  9,  9],  # 2592
                                         [args.batch_size, 256]]

            mod_connect_dict = {0: [],
                                1: [0,1],
                                2: [1,2],
                                3: [2,3]}

            vars(args)['arch_dict'] = {'num_ch': 64,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [3, 3, 3],
                                       'strides': [1,1],
                                       'padding': 1,
                                       'mod_connect_dict': mod_connect_dict}
            vars(args)['energy_weight_mask'] = [1, 4.84, 49]
        elif args.architecture == 'mnist_3_layers_med_fc_top1_upim_wild':
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size, 16, 48, 48],  # 36864
                                         [args.batch_size, 32, 12, 12],  # 4608
                                         [args.batch_size, 256]]

            mod_connect_dict = {0: [],
                                1: [0,1],
                                2: [1,2],
                                3: [2,3]}

            vars(args)['arch_dict'] = {'num_ch': 64,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [3, 3, 3],
                                       'strides': [1,1],
                                       'padding': 1,
                                       'mod_connect_dict': mod_connect_dict}
            vars(args)['energy_weight_mask'] = [1.0, 8.0, 144.0]
        elif args.architecture == 'mnist_5_layers_med_fc_top1_upim':
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size, 16, 48, 48],  # 36864
                                         [args.batch_size, 32, 12, 12],  # 4608
                                         [args.batch_size, 32, 6, 6],    # 1152
                                         [args.batch_size, 1024],
                                         [args.batch_size, 256]]


            mod_connect_dict = {0: [],
                                1: [0,1],
                                2: [1,2],
                                3: [2,3],
                                4: [3,4],
                                5: [4,5]}

            vars(args)['arch_dict'] = {'num_ch': 64,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [3, 3, 3],
                                       'strides': [1,1],
                                       'padding': 1,
                                       'mod_connect_dict': mod_connect_dict}
            vars(args)['energy_weight_mask'] = [1.0, 8.0, 32.0, 36, 144.0]

        elif args.architecture == 'mnist_4_layers_med_fc_top1_upim':
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size, 16, 48, 48],  # 36864
                                         [args.batch_size, 32, 12, 12],  # 4608
                                         [args.batch_size, 1024],
                                         [args.batch_size, 256]]

            mod_connect_dict = {0: [],
                                1: [0,1],
                                2: [1,2],
                                3: [2,3],
                                4: [3,4]}

            vars(args)['arch_dict'] = {'num_ch': 64,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [3, 3, 3],
                                       'strides': [1,1],
                                       'padding': 1,
                                       'mod_connect_dict': mod_connect_dict}
            vars(args)['energy_weight_mask'] = [1.0, 8.0, 36, 144.0]

    if args.dataset == "CIFAR10":
        if args.architecture == 'cifar10_2_layers':
            vars(args)['state_sizes'] = [[args.batch_size, 3, 32, 32],
                                         [args.batch_size, 3, 32, 32],
                                         [args.batch_size, 9, 16, 16],
                                         [args.batch_size, 9, 8, 8],
                                         [args.batch_size, 9, 2, 2]]  # ,#[args.batch_size, 18, 8, 8]]
            vars(args)['arch_dict'] = {'num_ch': 64,
                                       'num_sl': len(args.state_sizes) - 1,
                                       'kernel_sizes': [3, 3],
                                       'strides': [1,1],
                                       'padding': 1}

    if len(args.energy_weight_mask) != len(args.state_sizes)-1:
        raise RuntimeError("Number of energy_weight_mask args is different"+
                           " from the number of state layers")

    # Print final values for args
    for k, v in zip(vars(args).keys(), vars(args).values()):
        print(str(k) + '\t' * 2 + str(v))

    return args


def main():
    ### Parse CLI arguments.
    parser = argparse.ArgumentParser(description='MNIST classification with ' +
                                     'EP-trained Hopfield neural networks.')
    sgroup = parser.add_argument_group('Sampling options')
    sgroup.add_argument('--sampling_step_size', type=float, default=10,
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
                        help='If true, asks for a description of what is '+
                             'special about the '+
                             'experiment, if anything. Default: %(default)s.')
    parser.set_defaults(require_special_name=False)
    tgroup.add_argument('--epochs', type=int, metavar='N', default=2,
                        help='Number of training epochs. ' +
                             'Default: %(default)s.')
    tgroup.add_argument('--batch_size', type=int, metavar='N', default=128,
                        help='Training batch size. Default: %(default)s.')
    tgroup.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate of optimizer. Default: ' +
                             '%(default)s.' +
                             'When randomizing, the following options define'+
                             'a range of indices and the random value assigned'+
                             'to the argument will be 10 to the power of the'+
                             'float selected from the range. Options: [-3, 0.2].')
    tgroup.add_argument('--dataset', type=str, default="CIFAR10",
                        help='The dataset the network will be trained on.' +
                             ' Default: %(default)s.')
    # tgroup.add_argument('--l2_reg_w_param', type=float, default=0.0,
    #                     help='Scaling parameter for the L2 regularisation ' +
    #                          'term placed on the weight values. Default: ' +
    #                          '%(default)s.'+
    #                          'When randomizing, the following options define' +
    #                          'a range of indices and the random value assigned' +
    #                          'to the argument will be 10 to the power of the' +
    #                          'float selected from the range. '+
    #                          'Options: [-6, -2].')
    tgroup.add_argument('--l2_reg_energy_param', type=float, default=1.0,
                        help='Scaling parameter for the L2 regularisation ' +
                             'term placed on the energy values. Default: ' +
                             '%(default)s.'+
                             'When randomizing, the following options define' +
                             'a range of indices and the random value assigned' +
                             'to the argument will be 10 to the power of the' +
                             'float selected from the range. '+
                             'Options: [-6, -2].')
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
    tgroup.add_argument('--pos_buffer_frac', type=float, default=0.01,
                        help='Learning rate to pass to the Adam optimizer ' +
                             'used to train the InitializerNetwork. Default: ' +
                             '%(default)s.')

    ngroup = parser.add_argument_group('Network and states options')
    ngroup.add_argument('--architecture', type=str, default="cifar10_2_layers",
                        help='The type of architecture that will be built. Options: ' +
                             '[mnist_2_layers_small, cifar10_2_layers, mnist_1_layer_small]'
                             'Default: %(default)s.')
    ngroup.add_argument('--activation', type=str, default="leaky_relu",
                        help='The activation function. Options: ' +
                             '[relu, swish, leaky_relu]'
                             'Default: %(default)s.')
    ngroup.add_argument('--sigma', type=float, default=0.005,
                        help='Sets the scale of the noise '
                             'in the network.'+
                             'When randomizing, the following options define' +
                             'a range of indices and the random value assigned' +
                             'to the argument will be 10 to the power of the' +
                             'float selected from the range. '+
                             'Options: [-3, 0].')
    ngroup.add_argument('--energy_weight_min', type=float, default=0.0,
                        help='The minimum value that weights in the energy ' +
                             'weights layer may take.')
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
    ngroup.add_argument('--momentum_param', type=float, default=1.0,
                        help='')
    ngroup.add_argument('--dampening_param', type=float, default=0.0,
                        help='')

    vgroup = parser.add_argument_group('Visualization options')
    vgroup.add_argument('--viz_neurons', action='store_true',
                        help='Whether or not to visualise the neurons.')
    parser.set_defaults(viz_neurons=False)
    vgroup.add_argument('--num_it_viz', type=int,
                        help='The number of steps to use to sample images. ' +
                             'Default: %(default)s.')
    vgroup.add_argument('--viz_img_logging_step_interval', type=int, default=1,
                        help='The interval at which to save images that ' +
                             'are being sampled during visualization. '+
                             'Default: %(default)s.')

    mgroup = parser.add_argument_group('Miscellaneous options')
    mgroup.add_argument('--randomize_args', type=str, nargs='+', default=[],
                        help='List of CLI args to pass to the random arg ' +
                             'generator. Default: %(default)s.',
                        required=False)
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
    model = nw.DeepAttractorNetwork(args, device, model_name, writer).to(device)



    # Set up dataset
    data = Dataset(args)
    buffer = SampleBuffer(args,

                          device=device)

    if not args.no_train_model:
        # Train the model
        tm = TrainingManager(args, model, data, buffer, writer, device,
                             sample_log_dir)

        if args.initializer == 'ff_init' and args.pretrain_initializer:
            tm.pre_train_initializer()

        tm.train()
    # if args.viz_neurons:
    #     vm = VisualizationManager(args, model, data, buffer, writer, device,
    #                               sample_log_dir)
    #     vm.visualize()

shapes = lambda x : [y.shape for y in x]
nancheck = lambda x : (x != x).any()
listout = lambda y : [x for x in y]
gradcheck = lambda  y : [x.requires_grad for x in y]
leafcheck = lambda  y : [x.is_leaf for x in y]
existgradcheck = lambda  y : [(x.grad is not None) for x in y]
existgraddatacheck = lambda  y : [(x.grad.data is not None) for x in y]

if __name__ == '__main__':
    main()


###########################################################################
###########################################################################
###########################################################################

# class VisualizationManager(TrainingManager):
#
#     """A new class because I need to redefine sampler step to clamp certain
#     neurons, and need to define a new type of sampling phase."""
#     def __init__(self, args, model, data, buffer, writer, device, sample_log_dir):
#         super(TrainingManager, self).__init__()
#         self.args = args
#         self.model = model
#         self.data = data
#         self.writer = writer
#         self.buffer = buffer
#         self.device = device
#         self.sample_log_dir = os.path.join(sample_log_dir, 'viz')
#         if not os.path.isdir(self.sample_log_dir):
#             os.mkdir(self.sample_log_dir)
#         self.parameters = model.parameters()
#         self.optimizer = optim.Adam(self.parameters, lr=args.lr,
#                                     betas=(0.9, 0.999))
#         self.viz_batch_sizes = self.calc_viz_batch_sizes()
#
#         self.global_step = 0
#         self.batch_num = 0
#         if args.initializer == 'ff_init':
#             self.initter = nw.InitializerNetwork(args, writer, device)
#             self.initter.to(device)
#         else:
#             self.initter = None
#
#         if self.args.load_model:
#             path = 'exps/models/' + self.args.load_model + '.pt'
#             model_name = str(self.args.load_model)
#             checkpoint = torch.load(path)
#             self.model.load_state_dict(checkpoint['model'])
#             self.optimizer.load_state_dict(checkpoint['model_optimizer'])
#             if args.initializer == 'ff_init':
#                 self.initter.load_state_dict(checkpoint['initializer'])
#                 self.initter.optimizer.load_state_dict(
#                     checkpoint['initializer_optimizer'])
#
#             # Update args dict correctly
#             curr_num_it_viz = self.args.num_it_viz # collect the current arg for num_it_viz, because it would be overwritten by the value that was given during training
#             vars(self.args).update(vars(checkpoint['args']))
#             vars(self.args)['num_it_viz'] = curr_num_it_viz
#
#             self.batch_num = checkpoint['batch_num']
#             print("Loaded model " + model_name + ' successfully')
#
#         # self.args.num_it_viz = 13
#
#     def calc_viz_batch_sizes(self):
#         """The batch size is now the number of pixels in the image, and
#         there is only one channel because we only visualize one at a time."""
#         return [self.args.state_sizes[i][-1] * self.args.state_sizes[i][-2]
#                  for i in range(len(self.args.state_sizes))]
#
#     def update_state_size_bs(self, sl_idx):
#         self.model.batch_size = self.viz_batch_sizes[sl_idx]
#         new_state_sizes = [(self.viz_batch_sizes[sl_idx],
#                  self.args.state_sizes[i][1],
#                  self.args.state_sizes[i][2],
#                  self.args.state_sizes[i][3])
#                  for i in range(len(self.args.state_sizes))]
#         self.args.state_sizes = new_state_sizes
#         self.model.state_sizes = new_state_sizes
#         self.model.calc_energy_weight_masks()
#         return
#
#     def visualize(self):
#         """Clamps each neuron while sampling the others
#
#         Goes through each of the state layers, and each of the channels"""
#         for state_layer_idx in range(len(self.args.state_sizes)):
#             for channel_idx in range(self.args.state_sizes[state_layer_idx][1]): #[1] for num of channels
#                 print("Visualizing channel %s of state layer %s" % \
#                       (str(channel_idx), state_layer_idx))
#                 self.update_state_size_bs(state_layer_idx)
#                 self.noises = lib.utils.generate_random_states(
#                     self.args.state_sizes,
#                     self.device)
#                 # a kind of negative phase where it settles for a long time
#                 # and clamps a different neuron for each image
#                 clamp_array = torch.zeros(size=self.args.state_sizes[state_layer_idx],
#                                           dtype=torch.uint8,
#                                           device=self.device)
#                 mg = np.meshgrid(np.arange(0,self.args.state_sizes[state_layer_idx][2]),
#                                  np.arange(0,self.args.state_sizes[state_layer_idx][3]))
#                 idxs = list(zip(mg[1].flatten(), mg[0].flatten()))
#                 # Sets to 1 the next pixel in each batch element
#                 print("Setting indices")
#                 for i0 in range(self.args.state_sizes[state_layer_idx][0]):
#                     clamp_array[i0, 0][idxs[i0]] = 1.0
#                 self.visualization_phase(state_layer_idx,
#                                          channel_idx,
#                                          clamp_array)
#
#     def visualization_phase(self, state_layer_idx, channel_idx, clamp_array,
#                             clamp_value=1.):
#         states = lib.utils.generate_random_states(self.args.state_sizes,
#                                                   self.device)
#         id = None
#         # Freeze network parameters and take grads w.r.t only the inputs
#         requires_grad(states, True)
#         requires_grad(self.parameters, False)
#         self.model.eval()
#
#         # Set up state optimizer if approp
#         if self.args.state_optimizer is not 'langevin':
#             self.state_optimizers = get_state_optimizers(self.args, states)
#
#         # Viz phase sampling
#         for k in tqdm(range(self.args.num_it_viz)):
#             self.viz_sampler_step(states, id, state_layer_idx, channel_idx,
#                                   clamp_array, clamp_value)
#             if k % self.args.viz_img_logging_step_interval == 0:
#                 utils.save_image(
#                     states[0].detach().to('cpu'),
#                     os.path.join(self.sample_log_dir,
#                                  str(channel_idx) + '_' +
#                                  str(state_layer_idx) + '_' +
#                                  str(k).zfill(6)  + '.png'),
#                     nrow=self.args.state_sizes[state_layer_idx][2],
#                     normalize=True,
#                     range=(0, 1))
#             self.global_step += 1
#
#         # Stop calculting grads w.r.t. images
#         for neg_state in states:
#             neg_state.detach_()
#
#         return states, id
#
#     def viz_sampler_step(self, states, ids, state_layer_idx, channel_idx,
#                          clamp_array, clamp_value):
#         energies = self.model(states, ids)  # Outputs energy of neg sample
#         total_energy = energies.sum()
#
#         total_energy.backward()
#         torch.nn.utils.clip_grad_norm_(states,
#                                        self.args.clip_state_grad_norm,
#                                        norm_type=2)
#
#         for i, noise in enumerate(self.noises):
#             noise.normal_(0, self.args.sigma)
#             if i == state_layer_idx:
#                 noise = torch.where(clamp_array,
#                                                   torch.zeros_like(noise),
#                                                   noise)
#         for i, (noise, state) in enumerate(zip(self.noises, states)):
#             state.data.add_(noise.data)
#
#
#         # The gradient step in the Langevin step (only for upper layers)
#         for i, state in enumerate(states):
#             if self.args.state_optimizer != 'langevin':
#                 if i == state_layer_idx:
#                     # Zero the momentum of the optimizer, if exists
#                     # This is the momentum buffer (difficult to access because the key is the tensor of params)
#                     state_key = self.state_optimizers[
#                         state_layer_idx].param_groups[0]['params'][0]
#                     if state_key in self.state_optimizers[state_layer_idx].state:
#                         mom_buffer = self.state_optimizers[state_layer_idx].state[state_key]['momentum_buffer']
#                         mom_buffer = torch.where(clamp_array, torch.zeros_like(mom_buffer), mom_buffer)
#
#                     # Zero the gradient of selected neurons # TODO Check that the neurons are actually staying clamped clamped
#                     state.grad.data = torch.where(clamp_array,
#                                                   torch.zeros_like(state.grad.data),
#                                                   state.grad.data)
#
#                     # Finally, make sure the state value is at the clamp value
#                     # TBH zeroing the grad is probs unnecessary
#                     state.data = torch.where(clamp_array,
#                                              torch.ones_like(state.data) * \
#                                              torch.tensor(clamp_value),
#                                              state.data)
#                 self.state_optimizers[i].step()
#             else:
#                 state.data.add_(-self.args.sampling_step_size,
#                             state.grad.data)
#
#
#
#             # Prepare gradients and sample for next sampling step
#             state.grad.detach_()
#             state.grad.zero_()
#             state.data.clamp_(0, 1)

