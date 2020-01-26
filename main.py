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
        self.state_sizes = args.state_sizes
        self.parameters = model.parameters()
        self.optimizer = optim.Adam(self.parameters, lr=args.lr, betas=(0.9, 0.999))

        # initter_optimizer = optim.Adam(parameters, lr=1e-3, betas=(0.0, 0.999))

        self.noises = lib.utils.generate_random_states(self.state_sizes, self.device)

        self.global_step = 0
        self.batch_num = 0

        if args.initializer == 'ff_init':
            self.initter = nw.InitializerNetwork(args, writer, device)
            self.initter.to(device)
        else:
            self.initter = None

        if self.args.load_model:
            path = 'exps/models/' + self.args.load_model + '.pt'
            model_name = str(self.args.load_model)
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['model_optimizer'])
            if args.initializer == 'ff_init':
                self.initter.load_state_dict(checkpoint['initializer'])
                self.initter.optimizer.load_state_dict(
                    checkpoint['initializer_optimizer'])
            self.args = checkpoint['args']
            self.batch_num = checkpoint['batch_num']
            print("Loaded model " + model_name + ' successfully')

        lib.utils.save_configs_to_csv(self.args, self.model.model_name)

    def train(self):
        save_dict = self.make_save_dict()
        path = 'exps/models/' + self.model.model_name + '.pt'
        torch.save(save_dict, path)
        for batch, (pos_img, pos_id) in self.data.loader:

            pos_states_list, pos_id = self.positive_phase(pos_img, pos_id)
            if self.batch_num % self.args.img_logging_interval == 0:
                if self.args.save_pos_images:
                    utils.save_image(
                        pos_states_list[0][0].detach().to('cpu'),
                        os.path.join(self.sample_log_dir,
                                     'p0_' + str(self.batch_num).zfill(6) + '.png'),
                        nrow=16,
                        normalize=True,
                        range=(0, 1))
                    utils.save_image(
                        pos_states_list[1][0].detach().to('cpu'),
                        os.path.join(self.sample_log_dir,
                                     'p1_' + str(self.batch_num).zfill(6) + '.png'),
                        nrow=16,
                        normalize=True,
                        range=(0, 1))

            neg_states, neg_id = self.negative_phase()

            self.param_update_phase(neg_states, neg_id, pos_states_list, pos_id)

            if self.batch_num % self.args.img_logging_interval == 0:
                utils.save_image(
                    neg_states[0].detach().to('cpu'),
                    os.path.join(self.sample_log_dir,
                                 str(self.batch_num).zfill(6) + '.png'),
                    nrow=16,
                    normalize=True,
                    range=(0, 1))

            if self.batch_num % self.args.img_logging_interval == 0:
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

    def initialize_pos_states(self, pos_img=None, cond_layer=0):

        if cond_layer == 0:
            pos_states = [pos_img]
            init_start = 1
        else:
            pos_states = []
            init_start = 0

        #requires_grad(pos_states, True)
        if self.args.initializer == 'ff_init':
            print("Don't forget that you have disabled the ff_init option now!!!!!!!!!!!")
            # self.initter.train()
            # pos_states_init = self.initter.forward(pos_img, pos_id)
            # pos_states_new = [psi.clone().detach() for psi in pos_states_init]
            # pos_states.extend(pos_states_new)
        elif self.args.initializer == 'zeros': # probs don't use this now that you're initting all states
            pos_states.extend(
                [torch.zeros(size, device=self.device, requires_grad=True)
                 for size in self.state_sizes[init_start:]])
        elif self.args.initializer == 'random' or \
                self.args.initializer == 'pos0':
            rand_states = lib.utils.generate_random_states(
                self.state_sizes[init_start:], self.device)
            pos_states.extend(rand_states)

        return pos_states

    def positive_phase(self, pos_img, pos_id):

        print('\nStarting positive phase...')
        # Get the loaded pos samples and put them on the correct device
        pos_img, pos_id = pos_img.to(self.device), pos_id.to(self.device)

        # Gets the values of the pos states by running a short inference phase
        # conditioned on a particular state layer
        pos_states_list = []

        # I'm feeling like it makes most sense to use the codes from the 0th
        # pos phase to lock as the code to be used in the subsequent phases
        # because this one is the one that is constrained most by the data
        # distribution. #TODO consider whether you want to change this! Think!
        pos_states = self.initialize_pos_states(pos_img=pos_img, cond_layer=0) #Maybe remove cond_layer arg if keeping pos_phase0 as initter
        pos_states0 = None

        requires_grad(self.parameters, False)
        # if self.args.initializer == 'ff_init':
        #     requires_grad(self.initter.parameters(), False)
        self.model.eval()

        for pos_phase_idx in range(len(self.state_sizes)): # we have a positive phase for each cond_layer
            if pos_phase_idx != 0:
                if self.args.initializer == 'pos0':
                    pos_states = pos_states0
                elif self.args.initializer == 'random':
                    pos_states = self.initialize_pos_states(pos_img=None,
                                                            cond_layer=pos_phase_idx)
            # Freeze network parameters and take grads w.r.t only the inputs
            requires_grad(pos_states, True)

            if self.args.state_optimizer is not 'langevin':
                self.state_optimizers = get_state_optimizers(self.args, pos_states)

            # Positive phase sampling
            for k in tqdm(range(self.args.num_it_pos)): #TODO ensure that this only changes the unlocked state_layer in each pos_phase_i
                self.sampler_step(pos_states, pos_id, positive_phase=True,
                                  cond_layer=pos_phase_idx)
                self.global_step += 1

            # Stop calculting grads w.r.t. images
            for pos_state in pos_states:
                pos_state.detach_()

            if pos_phase_idx == 0:
                pos_states0 = pos_states
            pos_states_list.append(pos_states)

            # Update initializer network if present
            # if self.args.initializer == 'ff_init':
            #     requires_grad(self.initter.parameters(), True)
            #     loss = self.initter.update_weights(outs=pos_states_init,
            #                                        targets=pos_states[1:],
            #                                        step=self.global_step)

            if self.args.cd_mixture: #Consider whether to move this into "if pos_phase_idx == 0:"
                print("Adding pos states to pos buffer")
                self.buffer.push(pos_states, pos_id, pos=True)

        return pos_states_list, pos_id


    def negative_phase(self):
        print('\nStarting negative phase...')
        # Initialize the chain (either as noise or from buffer)
        neg_states, neg_id = self.buffer.sample_buffer()
        # Freeze network parameters and take grads w.r.t only the inputs
        requires_grad(neg_states, True)
        requires_grad(self.parameters, False)
        self.model.eval()

        # Set up state optimizer if approp
        if self.args.state_optimizer is not 'langevin':
            self.state_optimizers = get_state_optimizers(self.args, neg_states)

        # Negative phase sampling
        for k in tqdm(range(self.args.num_it_neg)):
            self.sampler_step(neg_states, neg_id, cond_layer=None)
            self.global_step += 1

        # Stop calculting grads w.r.t. images
        for neg_state in neg_states:
            neg_state.detach_()

        # Send negative samples to the buffer (on cpu)
        self.buffer.push(neg_states, neg_id)
        return neg_states, neg_id

    def sampler_step(self, states, ids, positive_phase=False, cond_layer=None):
        """Cond layer is the layer that is the conditional variable (the
        one that is fixed and conditions the distribution of the other
        variable. """
        # Calculate the gradient for the Langevin step
        energies = self.model(states, ids)  # Outputs energy of neg sample
        total_energy = energies.sum()
        if positive_phase:
            print('\nPos Energy %s: '% str(cond_layer) + str(total_energy.cpu().detach().numpy()))
            self.writer.add_scalar('train/PosSamplesEnergy_%s' % str(cond_layer), total_energy,
                                   self.global_step)
        else:
            print('\nNeg Energy: ' + str(total_energy.cpu().detach().numpy()))
            self.writer.add_scalar('train/NegSamplesEnergy', total_energy,
                                   self.global_step)

        # Take gradient (before addition of noise)
        total_energy.backward()
        torch.nn.utils.clip_grad_norm_(states,
                                       self.args.clip_state_grad_norm,
                                       norm_type=2)

        # Adding noise in the Langevin step, but only to latent variables
        # in positive phase
        for noise in self.noises:
            noise.normal_(0, self.args.sigma)
        for i, (noise, state) in enumerate(zip(self.noises, states)):
            if positive_phase and i == cond_layer:
                pass
            else:
                state.data.add_(noise.data)

        # The gradient step in the Langevin step (only for upper layers)
        for i, state in enumerate(states):
            if positive_phase and i == cond_layer:
                pass
            else:
                if self.args.state_optimizer != 'langevin':
                    self.state_optimizers[i].step()
                else:
                    state.data.add_(-self.args.sampling_step_size,
                                state.grad.data)

            # Prepare gradients and sample for next sampling step
            state.grad.detach_()
            state.grad.zero_()
            state.data.clamp_(0, 1)

    def calc_energ_and_loss(self, neg_states, neg_id, pos_states_list, pos_id):
        # Get energies of positive and negative samples
        pos_energies = []
        for pos_states in pos_states_list:
            pos_energies += [self.model(pos_states, pos_id)]
        neg_energy = self.model(neg_states, neg_id)
        energies = pos_energies + [neg_energy]

        # Calculate the loss and the gradients for the network params
        loss_l2 = self.args.l2_reg_energy_param * sum([enrg**2 for enrg in energies])  # L2 penalty on energy magnitudes
        loss_ml = sum(pos_energies) - neg_energy  # Maximum likelihood loss
        loss = loss_ml + loss_l2
        loss = loss.mean()
        loss.backward()
        self.writer.add_scalar('train/loss', loss.item(), self.global_step)
        return neg_energy, pos_energies, loss

    def update_weights(self, loss):
        clip_grad(self.parameters, self.optimizer)

        # Update the network params
        self.optimizer.step()
        for energy_weight in self.model.energy_weights.parameters():
            energy_weight.data.clamp_(self.args.energy_weight_min)

        self.data.loader.set_description(f'loss: {loss.item():.5f}')

    def param_update_phase(self, neg_states, neg_id, pos_states_list, pos_id):

        if self.args.log_histograms and \
        self.batch_num % self.args.histogram_logging_interval == 0:
            layer_string = 'train/energy_weights'
            self.writer.add_histogram(layer_string, self.model.energy_weights.weight,
                                      self.batch_num)


        # Put model in training mode and prepare network parameters for updates
        requires_grad(self.parameters, True)
        self.model.train()  # Not to be confused with self.TrainingManager.train
        self.model.zero_grad()

        neg_energy, pos_energies, loss = \
            self.calc_energ_and_loss(neg_states, neg_id,
                                     pos_states_list, pos_id)

        self.update_weights(loss)


    def make_save_dict(self):
        save_dict = {'model': self.model.state_dict(),
                     'model_optimizer': self.optimizer.state_dict(),
                     'args': self.args,
                     'batch_num': self.batch_num}
        if self.args.initializer =='ff_init':
            initter_dict = {'initializer': self.initter.state_dict(),
                            'initializer_optimizer': self.initter.optimizer.state_dict()}
            save_dict    = {**save_dict, **initter_dict}
        return save_dict

###########################################################################
###########################################################################
###########################################################################

class VisualizationManager(TrainingManager):

    """A new class because I need to redefine sampler step to clamp certain
    neurons, and need to define a new type of sampling phase."""
    def __init__(self, args, model, data, buffer, writer, device, sample_log_dir):
        super(TrainingManager, self).__init__()
        self.args = args
        self.model = model
        self.data = data
        self.writer = writer
        self.buffer = buffer
        self.device = device
        self.sample_log_dir = os.path.join(sample_log_dir, 'viz')
        if not os.path.isdir(self.sample_log_dir):
            os.mkdir(self.sample_log_dir)
        self.state_sizes = args.state_sizes
        self.parameters = model.parameters()
        self.optimizer = optim.Adam(self.parameters, lr=args.lr,
                                    betas=(0.9, 0.999))
        self.viz_batch_sizes = self.calc_viz_batch_sizes()

        self.global_step = 0
        self.batch_num = 0
        if args.initializer == 'ff_init':
            self.initter = nw.InitializerNetwork(args, writer, device)
            self.initter.to(device)
        else:
            self.initter = None

        if self.args.load_model:
            path = 'exps/models/' + self.args.load_model + '.pt'
            model_name = str(self.args.load_model)
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['model_optimizer'])
            if args.initializer == 'ff_init':
                self.initter.load_state_dict(checkpoint['initializer'])
                self.initter.optimizer.load_state_dict(
                    checkpoint['initializer_optimizer'])

            # Update args dict correctly
            curr_num_it_viz = self.args.num_it_viz # collect the current arg for num_it_viz, because it would be overwritten by the value that was given during training
            vars(self.args).update(vars(checkpoint['args']))
            vars(self.args)['num_it_viz'] = curr_num_it_viz

            self.batch_num = checkpoint['batch_num']
            print("Loaded model " + model_name + ' successfully')

        # self.args.num_it_viz = 13

    def calc_viz_batch_sizes(self):
        """The batch size is now the number of pixels in the image, and
        there is only one channel because we only visualize one at a time."""
        return [self.state_sizes[i][-1] * self.state_sizes[i][-2]
                 for i in range(len(self.state_sizes))]

    def update_state_size_bs(self, sl_idx):
        self.model.batch_size = self.viz_batch_sizes[sl_idx]
        new_state_sizes = [(self.viz_batch_sizes[sl_idx],
                 self.state_sizes[i][1],
                 self.state_sizes[i][2],
                 self.state_sizes[i][3])
                 for i in range(len(self.state_sizes))]
        self.state_sizes = new_state_sizes
        self.model.state_sizes = new_state_sizes
        self.model.calc_energy_weight_masks()
        return

    def visualize(self):
        """Clamps each neuron while sampling the others

        Goes through each of the state layers, and each of the channels"""
        for state_layer_idx in range(len(self.state_sizes)):
            for channel_idx in range(self.state_sizes[state_layer_idx][1]): #[1] for num of channels
                print("Visualizing channel %s of state layer %s" % \
                      (str(channel_idx), state_layer_idx))
                self.update_state_size_bs(state_layer_idx)
                self.noises = lib.utils.generate_random_states(
                    self.state_sizes,
                    self.device)
                # a kind of negative phase where it settles for a long time
                # and clamps a different neuron for each image
                clamp_array = torch.zeros(size=self.state_sizes[state_layer_idx],
                                          dtype=torch.uint8,
                                          device=self.device)
                mg = np.meshgrid(np.arange(0,self.state_sizes[state_layer_idx][2]),
                                 np.arange(0,self.state_sizes[state_layer_idx][3]))
                idxs = list(zip(mg[1].flatten(), mg[0].flatten()))
                # Sets to 1 the next pixel in each batch element
                print("Setting indices")
                for i0 in range(self.state_sizes[state_layer_idx][0]):
                    clamp_array[i0, 0][idxs[i0]] = 1.0
                self.visualization_phase(state_layer_idx,
                                         channel_idx,
                                         clamp_array)

    def visualization_phase(self, state_layer_idx, channel_idx, clamp_array,
                            clamp_value=1.):
        states = lib.utils.generate_random_states(self.state_sizes,
                                                  self.device)
        id = None
        # Freeze network parameters and take grads w.r.t only the inputs
        requires_grad(states, True)
        requires_grad(self.parameters, False)
        self.model.eval()

        # Set up state optimizer if approp
        if self.args.state_optimizer is not 'langevin':
            self.state_optimizers = get_state_optimizers(self.args, states)

        # Viz phase sampling
        for k in tqdm(range(self.args.num_it_viz)):
            self.viz_sampler_step(states, id, state_layer_idx, channel_idx,
                                  clamp_array, clamp_value)
            if k % self.args.viz_img_logging_step_interval == 0:
                utils.save_image(
                    states[0].detach().to('cpu'),
                    os.path.join(self.sample_log_dir,
                                 str(channel_idx) + '_' +
                                 str(state_layer_idx) + '_' +
                                 str(k).zfill(6)  + '.png'),
                    nrow=self.state_sizes[state_layer_idx][2],
                    normalize=True,
                    range=(0, 1))
            self.global_step += 1

        # Stop calculting grads w.r.t. images
        for neg_state in states:
            neg_state.detach_()

        return states, id

    def viz_sampler_step(self, states, ids, state_layer_idx, channel_idx,
                         clamp_array, clamp_value):
        energies = self.model(states, ids)  # Outputs energy of neg sample
        total_energy = energies.sum()

        total_energy.backward()
        torch.nn.utils.clip_grad_norm_(states,
                                       self.args.clip_state_grad_norm,
                                       norm_type=2)

        for i, noise in enumerate(self.noises):
            noise.normal_(0, self.args.sigma)
            if i == state_layer_idx:
                noise = torch.where(clamp_array,
                                                  torch.zeros_like(noise),
                                                  noise)
        for i, (noise, state) in enumerate(zip(self.noises, states)):
            state.data.add_(noise.data)


        # The gradient step in the Langevin step (only for upper layers)
        for i, state in enumerate(states):
            if self.args.state_optimizer != 'langevin':
                if i == state_layer_idx:
                    # Zero the momentum of the optimizer, if exists
                    # This is the momentum buffer (difficult to access because the key is the tensor of params)
                    state_key = self.state_optimizers[
                        state_layer_idx].param_groups[0]['params'][0]
                    if state_key in self.state_optimizers[state_layer_idx].state:
                        mom_buffer = self.state_optimizers[state_layer_idx].state[state_key]['momentum_buffer']
                        mom_buffer = torch.where(clamp_array, torch.zeros_like(mom_buffer), mom_buffer)

                    # Zero the gradient of selected neurons # TODO Check that the neurons are actually staying clamped clamped
                    state.grad.data = torch.where(clamp_array,
                                                  torch.zeros_like(state.grad.data),
                                                  state.grad.data)

                    # Finally, make sure the state value is at the clamp value
                    # TBH zeroing the grad is probs unnecessary
                    state.data = torch.where(clamp_array,
                                             torch.ones_like(state.data) * \
                                             torch.tensor(clamp_value),
                                             state.data)
                self.state_optimizers[i].step()
            else:
                state.data.add_(-self.args.sampling_step_size,
                            state.grad.data)



            # Prepare gradients and sample for next sampling step
            state.grad.detach_()
            state.grad.zero_()
            state.data.clamp_(0, 1)




def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag


def get_state_optimizers(args, params):
    if args.state_optimizer == 'langevin':
        return None
    if args.state_optimizer == 'sgd':
        return [optim.SGD([prm], args.sampling_step_size) for prm in params]
    if args.state_optimizer == 'sgd_momentum':
        return [optim.SGD([prm], args.sampling_step_size,
                         momentum=args.momentum_param,
                         dampening=args.dampening_param) for prm in params]
    if args.state_optimizer == 'nesterov':
        return [optim.SGD([prm], args.sampling_step_size,
                         momentum=args.momentum_param, nesterov=True) for prm in params]
    if args.state_optimizer == 'adam':
        return [optim.Adam([prm], args.sampling_step_size, betas=(0.9,0.999)) for prm in params]


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

    if args.dataset == "CIFAR10":
        vars(args)['state_sizes'] = [[args.batch_size, 3, 32, 32],
                                     [args.batch_size, 9, 16, 16]]  #,#[args.batch_size, 18, 8, 8]]
    elif args.dataset == 'MNIST':
        if args.architecture == 'mnist_2_layers_small':
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
                                          [args.batch_size, 4, 3, 3]]
        if args.architecture == 'mnist_1_layer_small':
            vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28]]

    # Print final values for args
    for k, v in zip(vars(args).keys(), vars(args).values()):
        print(str(k) + '\t' * 2 + str(v))

    return args


def main():
    ### Parse CLI arguments.
    parser = argparse.ArgumentParser(description='MNIST classification with ' +
                                     'EP-trained Hopfield neural networks.')

    sgroup = parser.add_argument_group('Sampling options')
    # sgroup.add_argument('--alphas', type=float, nargs='+', default=1e-3,
    #                     help='Individual learning rates for each layer. '+
    #                          ' Default: %(default)s. '+
    #                          'When randomizing, the following options define'+
    #                          'a range of indices and the random value assigned'+
    #                          'to the argument will be 10 to the power of the'+
    #                          'float selected from the range. ' +
    #                          'Options: [-1.25, -0.25]. ' +
    #                          'The next option defines the multiplier for the'+
    #                          ' alphas in subsequent layers: ' +
    #                          'Opt2: {0.05, 0.5}.')
    sgroup.add_argument('--sampling_step_size', type=float, default=10,
                        help='The amount that the network is moves forward ' +
                             'according to the activity gradient defined by ' +
                             'the partial derivative of the Hopfield-like ' +
                             'energy. Default: %(default)s.'+
                             'When randomizing, the following options define'+
                             'a range of indices and the random value assigned'+
                             'to the argument will be 10 to the power of the'+
                             'float selected from the range. Options: [-3, 0.5].')

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
    tgroup.add_argument('--special_name', type=str, metavar='N',
                        default="None",
                        help='A description of what is special about the ' +
                             'experiment, if anything. Default: %(default)s.')
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
    tgroup.add_argument('--l2_reg_w_param', type=float, default=0.0,
                        help='Scaling parameter for the L2 regularisation ' +
                             'term placed on the weight values. Default: ' +
                             '%(default)s.'+
                             'When randomizing, the following options define' +
                             'a range of indices and the random value assigned' +
                             'to the argument will be 10 to the power of the' +
                             'float selected from the range. '+
                             'Options: [-6, -2].')
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
                             'Options:  [zeros, random, previous, ff_init, ' +
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
    tgroup.add_argument('--num_pretraining_batches', type=int, default=30,
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
    # ngroup.add_argument('--network_architecture', type=str, default="ff",
    #                     help='The type of network. Options: [ff, ctx]'
    #                          'Default: %(default)s.')
    ngroup.add_argument('--stochastic_add', action='store_true',
                        help='Adds gaussian white noise to the state at every'+
                             'timestep (it is deterministic by default)' +
                             'Default: %(default)s.')
    parser.set_defaults(stochastic_add=False)
    ngroup.add_argument('--stochastic_mult', action='store_true',
                        help='Multiplies the state by one-centred guassian '+
                             'white noise at each timestep (it is ' +
                             'deterministic by default)' +
                             'Default: %(default)s.')
    parser.set_defaults(stochastic_mult=False)
    ngroup.add_argument('--sigma', type=float, default=0.005,
                        help='Sets the scale of the noise '
                             'in the network.'+
                             'When randomizing, the following options define' +
                             'a range of indices and the random value assigned' +
                             'to the argument will be 10 to the power of the' +
                             'float selected from the range. '+
                             'Options: [-3, 0].')
    ngroup.add_argument('--state_sizes', type=list, nargs='+', default=[[]],#This will be filled by default. it's here for saving
                        help='Number of units in each hidden layer of the ' +
                             'network. Default: %(default)s.')
    ngroup.add_argument('--w_dropout_prob', type=float, default=1.0,
                        help='The fraction of ones in the dropout masks ' +
                             'placed on the weights.'+
                             'When randomizig, the following options define' +
                             'a range of indices and the random value assigned' +
                             'to the argument will be 10 to the power of the' +
                             'float selected from the range. ' +
                             'Options: [-0.07, 0].')
    ngroup.add_argument('--energy_weight_min', type=float, default=0.0,
                        help='The minimum value that weights in the energy ' +
                             'weights layer may take.')
    ngroup.add_argument('--energy_weight_mask', type=int, nargs='+',
                        default=[1,1,1], help='A list that will be used to' +
                        'define a Boolean mask over the energy weights, ' +
                        'allowing you to silence the energy contributions' +
                        ' of certain state layers selectively.' +
                        ' Default: %(default)s.')#TODO ensure this remains constant
    ngroup.add_argument('--state_optimizer', type=str, default='langevin',
                        help='The kind of optimizer to use to descend the '+
                        'energy landscape. Only Langevin is guaranteed to '+
                        'converge to the true stationary distribution in '+
                        'theory.')
    ngroup.add_argument('--momentum_param', type=float, default=1.0,
                        help='')
    ngroup.add_argument('--dampening_param', type=float, default=0.0,
                        help='')
    ngroup.add_argument('--mult_gauss_noise', action='store_true',
                        help='Implements multiplicative, 1-centred gaussian '+
                             'white noise inputs of every conv net (except '+
                             'the base one)')
    parser.set_defaults(mult_gauss_noise=False)


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
    mgroup.add_argument('--use_cuda', action='store_true',
                        help='Flag to enable GPU usage.')
    mgroup.add_argument('--weights_and_biases_histograms', action='store_true',
                        help='Plots the weights and biases in tensorboard.')
    mgroup.add_argument('--randomize_args', type=str, nargs='+', default=[],
                        help='List of CLI args to pass to the random arg ' +
                             'generator. Default: %(default)s.',
                        required=False)
    ngroup.add_argument('--sample_buffer_prob', type=float, default=0.95,
                        help='The probability that the network will be ' +
                             'initialised from the buffer instead of from '+
                             'random noise.')
    # Locally it should be 'exps/tblogs'
    # On Euler it should be '/cluster/scratch/sharkeyl/HH/tblogs'
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
    mgroup.add_argument('--model_save_interval', type=int, default=10000,
                        help='The size of the intervals between the model '+
                             'saves.')
    mgroup.add_argument('--load_model', type=str,
                        help='The name of the model that you want to load.'+
                        'The file extension should not be included.')
    ngroup.add_argument('--no_train_model', action='store_true',
                        help='Whether or not to train the model ')
    parser.set_defaults(no_train_model=False)


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
    #model = nw.DeepAttractorNetwork(args, device, model_name).to(device)
    model = nw.DeepAttractorNetwork(args, device, model_name).to(device)


    # Set up dataset
    data = Dataset(args)
    buffer = SampleBuffer(args,
                          batch_size=args.batch_size,
                          p=args.sample_buffer_prob,
                          device=device)



    if not args.no_train_model:
        # Train the model
        tm = TrainingManager(args, model, data, buffer, writer, device,
                             sample_log_dir)
        if args.initializer == 'ff_init' and args.pretrain_initializer:
            tm.pre_train_initializer()
        tm.train()
    if args.viz_neurons:
        vm = VisualizationManager(args, model, data, buffer, writer, device,
                                  sample_log_dir)
        vm.visualize()



shapes = lambda x : [y.shape for y in x]
nancheck = lambda x : (x != x).any()
listout = lambda y : [x for x in y]
gradcheck = lambda  y : [x.requires_grad for x in y]
leafcheck = lambda  y : [x.is_leaf for x in y]
existgradcheck = lambda  y : [(x.grad is not None) for x in y]
existgraddatacheck = lambda  y : [(x.grad.data is not None) for x in y]


if __name__ == '__main__':
    main()



# def pre_train_initializer(self):
#     i = 0
#     for batch, (pos_img, pos_id) in self.data.loader:
#         print("Pretraining step")
#         pos_states, pos_id = self.positive_phase(pos_img, pos_id)
#         i += 1
#         if i > self.args.num_pretraining_batches:
#             break
#     print("\nPretraining of initializer complete")