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
from lib.data import SampleBuffer, Dataset #sample_data
import lib.utils


class TrainingManager():
    def __init__(self, args, model, data, buffer, writer, device,
                 sample_log_dir):
        self.args = args
        self.model = model
        self.data = data
        self.writer = writer
        self.buffer = buffer
        self.device = device
        self.sample_log_dir = sample_log_dir
        self.parameters = model.parameters()
        self.optimizer = optim.Adam(self.parameters,
                                    lr=args.lr,
                                    betas=(0.0, 0.999)) # betas=(0.9, 0.999))
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size=1,
                                                   gamma=0.999)
        self.noises = lib.utils.generate_random_states(self.args.state_sizes,
                                                       self.device)
        self.global_step = 0
        self.batch_num = 0
        self.epoch = 0

        # Load initializer network (initter)
        if args.initializer == 'ff_init':
            self.initter = nw.InitializerNetwork(args, writer, device)
            self.initter.to(device)
        else:
            self.initter = None

        # Load old networks and settings if loading an old model
        if self.args.load_model:
            loaded_model_name = str(self.args.load_model)
            path = 'exps/models/' + loaded_model_name + '.pt'
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['model_optimizer'])

            # Decide which current settings should override old settings
            new_args = checkpoint['args']
            for val in self.args.override_loaded:
                print('Overriding old value for %s' % val)
                vars(new_args)[val] = vars(self.args)[val]
            self.args = new_args

            # Reload initter if it was used during training and currently using
            if self.args.initializer == 'ff_init':
                self.initter.load_state_dict(checkpoint['initializer'])
                self.initter.optimizer.load_state_dict(
                    checkpoint['initializer_optimizer'])

            self.batch_num = checkpoint['batch_num']
            print("Loaded model " + loaded_model_name + ' successfully')

            # Save new settings (doesn't overwrite old csv values)
            lib.utils.save_configs_to_csv(self.args, loaded_model_name)
        else:
            lib.utils.save_configs_to_csv(self.args, self.model.model_name)

        # Print out param sizes to ensure you aren't using something stupidly
        # large
        param_sizes = [torch.prod(torch.tensor(sz)).item() for sz in
                                   shapes(self.model.parameters())]
        param_sizes.sort()
        top10_params = param_sizes[-10:]
        print("Top 10 network param sizes: \n %s" % str(top10_params))

        # Set the rest of the hyperparams for iteration scheduling
        self.pos_history = []
        self.neg_history = []
        self.max_history_len = 5000
        self.mean_neg_pos_margin = 5000
        self.neg_it_schedule_cooldown = 0
        self.cooldown_len = 10 #epochs
        self.latest_pos_enrg = None
        self.latest_neg_enrg = None
        self.num_it_neg_mean = self.args.num_it_neg
        self.num_it_neg = self.args.num_it_neg

    def train(self):
        save_dict = self.make_save_dict()
        path = 'exps/models/' + self.model.model_name + '.pt'
        torch.save(save_dict, path)
        prev_states = None

        # Main training loop
        #for batch, (pos_img, pos_id) in self.data.loader:
        for e in range(10000):
            for pos_img, pos_id in self.data.loader:
                print("Epoch:        %i" % e)
                print("Batch num:    %i" % self.batch_num)
                print("Global step:  %i" % self.global_step)

                pos_states, pos_id = self.positive_phase(pos_img,
                                                         pos_id,
                                                         prev_states)

                neg_states, neg_id = self.negative_phase()

                self.param_update_phase(neg_states,
                                        neg_id,
                                        pos_states,
                                        pos_id)

                prev_states = pos_states  # In case pos init uses prev states

                if self.batch_num % self.args.img_logging_interval == 0:
                    self.log_images(pos_img, pos_states, neg_states)

                self.save_energies_to_histories()
                self.log_mean_energy_histories()

                # Save network(s) and settings
                if self.batch_num % self.args.model_save_interval == 0:
                    save_dict = self.make_save_dict()
                    path = 'exps/models/' + self.model.model_name + '.pt'
                    torch.save(save_dict, path)

                self.batch_num += 1

                if self.num_it_neg_mean > 1000:
                    stop = True
                else:
                    stop = False

            self.neg_iterations_schedule_update()

            print("End of epoch %i" % self.epoch)
            self.epoch += 1

            # Schedule lr and log changes
            print("Decrementing learning rate.")
            lr = self.scheduler.get_lr()[0]
            self.writer.add_scalar('train/lr', lr, self.epoch)
            self.scheduler.step()
            new_lr = self.scheduler.get_lr()[0]
            self.writer.add_scalar('train/lr', new_lr, self.epoch)

            if stop:
                break

        # Save network(s) and settings when training is complete too
        save_dict = self.make_save_dict()
        path = 'exps/models/' +self.model.model_name + '.pt'
        torch.save(save_dict, path)




    def initialize_pos_states(self, pos_img=None, pos_id=None,
                              prev_states=None):
        """Initializes positive states"""

        pos_states = [pos_img]

        if self.args.initializer == 'zeros':
            zero_states = [torch.zeros(size, device=self.device,
                                       requires_grad=True)
                           for size in self.args.state_sizes[1:]]
            pos_states.extend(zero_states)
        if self.args.initializer == 'ff_init':
            # Later consider implementing a ff_init that is trained as normal
            # and gives the same output but that only a few (maybe random)
            # state neurons are changed/clamped by the innitter so that you
            # can still use the 'previous' initialisation in your experiments
            self.initter.train()
            pos_states_new = self.initter.forward(pos_img, pos_id)
            pos_states.extend(pos_states_new)
        elif self.args.initializer == 'middle':
            raise NotImplementedError("You need to find the mean pixel value" +
                                      " and then use the value as the value " +
                                      "to init all pixels.")
        elif self.args.initializer == 'mix_prev_middle':
            raise NotImplementedError
        elif self.args.initializer == 'previous' and prev_states is not None:
            pos_states.extend(prev_states[1:])
        else:  # i.e. if self.args.initializer == 'random':
            rand_states = lib.utils.generate_random_states(
                self.args.state_sizes[1:], self.device)
            pos_states.extend(rand_states)

        return pos_states

    def positive_phase(self, pos_img, pos_id, prev_states=None):

        print('\nStarting positive phase...')
        # Get the loaded pos samples and put them on the correct device
        pos_img, pos_id = pos_img.to(self.device), pos_id.to(self.device)

        # Gets the values of the pos states by running an inference phase
        # with the image state_layer clamped
        pos_states_init = self.initialize_pos_states(pos_img=pos_img,
                                                     prev_states=prev_states)
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
        for _ in tqdm(range(self.args.num_it_pos)):
            self.sampler_step(pos_states, pos_id, positive_phase=True,
                              step=self.global_step)
            self.global_step += 1

        # Stop calculting grads w.r.t. images
        for pos_state in pos_states:
            pos_state.detach_()

        # Update initializer network if present
        if self.args.initializer == 'ff_init':
            lib.utils.requires_grad(self.initter.parameters(), True)
            initr_loss = self.initter.update_weights(outs=pos_states_init[1:],
                                                     targets=pos_states[1:],
                                                     step=self.batch_num)

        # Add positive states to positive buffer if using CD mixture
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

        if self.args.randomize_neg_its:
            self.num_it_neg = max(1, np.random.poisson(self.num_it_neg_mean))

        # Negative phase sampling
        for _ in tqdm(range(self.num_it_neg)):
            self.sampler_step(neg_states, neg_id, step=self.global_step)
            self.global_step += 1

        # Stop calculting grads w.r.t. images
        for neg_state in neg_states:
            neg_state.detach_()

        # Send negative samples to the negative buffer
        self.buffer.push(neg_states, neg_id)

        return neg_states, neg_id

    def sampler_step(self, states, ids, positive_phase=False, step=None):

        # Get total energy and energy outputs for indvdual neurons
        energy, outs = self.model(states, ids, step)

        # Calculate the gradient wrt states for the Langevin step (before
        # addition of noise)
        energy.backward()
        torch.nn.utils.clip_grad_norm_(states,
                                       self.args.clip_state_grad_norm,
                                       norm_type=2)

        # Adding noise in the Langevin step (only for non conditional
        # layers in positive phase)
        for layer_idx, (noise, state) in enumerate(zip(self.noises, states)):
            noise.normal_(0, self.args.sigma)
            # Note: Just set sigma to a very small value if you don't want to
            # add noise. It's so inconsequential that it's not worth the
            # if-statements to accommodate sigma==0.0
            if positive_phase and layer_idx == 0:
                pass
            else:
                state.data.add_(noise.data)

        # The gradient step in the Langevin/SGHMC step
        # It goes through each statelayer and steps back using its associated
        # optimizer.
        for layer_idx, optimizer in enumerate(self.state_optimizers):
            if positive_phase and layer_idx == 0:
                pass
            else:
                optimizer.step()

        # Log data to tensorboard
        # Energies for layers (mean scalar and all histogram)
        if step % self.args.scalar_logging_interval == 0 and step is not None:
            for i, enrg in enumerate(outs):
                mean_layer_string = 'layers/mean_energies_%s' % i
                #print("Logging mean energies") #TODO rename these. They aren't energies, they're the outputs of the quadratic networks
                self.writer.add_scalar(mean_layer_string, enrg.mean(), step)
                if self.args.log_histograms  and \
                        step % self.args.histogram_logging_interval == 0:
                    hist_layer_string = 'layers/hist_energies_%s' % i
                    #print("Logging energy histograms")
                    self.writer.add_histogram(hist_layer_string, enrg, step)

        ## Pos or Neg total energies
        if positive_phase and step % self.args.scalar_logging_interval == 0:
            print('\nPos Energy: ' + str(energy.cpu().detach().numpy()))
            self.writer.add_scalar('train/PosSamplesEnergy', energy,
                                   self.global_step)
        elif step % self.args.scalar_logging_interval == 0: #i.e. if negative phase and appropriate step
            print('\nNeg Energy: ' + str(energy.cpu().detach().numpy()))
            self.writer.add_scalar('train/NegSamplesEnergy', energy,
                                   self.global_step)

        ## States and momenta (mean and histograms)
        if step % self.args.scalar_logging_interval == 0:
            for i, state in enumerate(states):
                mean_layer_string = 'layers/mean_states_%s' % i
                #print("Logging mean energies")
                self.writer.add_scalar(mean_layer_string, state.mean(), step)
                if self.args.log_histograms  and \
                        step % self.args.histogram_logging_interval == 0:
                    hist_layer_string = 'layers/hist_states_%s' % i
                    #print("Logging energy histograms")
                    self.writer.add_histogram(hist_layer_string, state, step)
        ## States and momenta (specific)
        if self.args.log_spec_neurons and step % self.args.scalar_logging_interval == 0:
            idxss = [[(0,0,13,13)], [(0,0,5,5)], [None], [None]]
            self.log_specific_states_and_momenta(states, outs, idxss, step)
        # End of data logging

        # Save latest energy outputs so you can schedule the phase lengths
        if positive_phase:
            self.latest_pos_enrg = energy.item()
        else:
            self.latest_neg_enrg = energy.item()


        # Prepare gradients and sample for next sampling step
        for state in states:
            state.grad.detach_()
            state.grad.zero_()
            state.data.clamp_(0, 1)

    def log_mean_energy_histories(self):
        mean_pos = sum(self.pos_history) / len(self.pos_history)
        mean_neg = sum(self.neg_history) / len(self.neg_history)
        self.writer.add_scalar('train/mean_neg_energy', mean_neg,
                               self.batch_num)
        self.writer.add_scalar('train/mean_pos_energy', mean_pos,
                               self.batch_num)

    def log_images(self, pos_img, pos_states, neg_states):
        shape = pos_img.shape
        neg_imgs_save = neg_states[0].reshape(shape).detach().to('cpu')
        utils.save_image(neg_imgs_save,
                         os.path.join(self.sample_log_dir,
                                      str(self.batch_num).zfill(
                                          6) + '.png'),
                         nrow=16, normalize=True, range=(0, 1))
        if self.args.save_pos_images:
            pos_imgs_save = pos_states[0].reshape(shape).detach().to('cpu')
            utils.save_image(pos_imgs_save,
                             os.path.join(self.sample_log_dir,
                                          'p0_' + str(
                                              self.batch_num).zfill(
                                              6) + '.png'),
                             nrow=16, normalize=True, range=(0, 1))

    def log_specific_states_and_momenta(self, states, outs, idxss, step):
        for j, (state, out, idxs, opt) in enumerate(zip(states, outs, idxss,
                                                  self.state_optimizers)):
            for idx in idxs:
                if idx is None:
                    break
                neuron_label = str(j)+'_'+str(idx)[:-1]
                #print("Logging specific state %s" % str(j)+'_'+str(idx))

                spec_state_val_string = 'spec/state_%s' % neuron_label
                spec_state_grad_string = 'spec/state_grad_%s' % neuron_label
                spec_state_enrg_string = 'spec/state_enrg_%s' % neuron_label

                state_value = state[idx]
                state_grad  = state.grad[idx]
                state_enrg  = out[idx]

                self.writer.add_scalar(spec_state_val_string, state_value, step)
                self.writer.add_scalar(spec_state_grad_string, state_grad, step)
                self.writer.add_scalar(spec_state_enrg_string, state_enrg, step)

                if opt.state_dict()['state']: # i.e. if momentum exists isn't empty, because it's only instantiated after 1 steps I think
                    spec_state_mom_string = 'spec/state_mom_%s' % neuron_label
                    key = list(opt.state_dict()['state'].keys())[0]
                    state_mom = opt.state_dict()['state'][key]['momentum_buffer']
                    state_mom = state_mom[idx]
                    self.writer.add_scalar(spec_state_mom_string, state_mom,
                                           step)

    def calc_energ_and_loss(self, neg_states, neg_id, pos_states, pos_id):

        # Get energies of positive and negative samples
        pos_energy, _ = self.model(pos_states, pos_id)
        neg_energy, _ = self.model(neg_states, neg_id)

        # Calculate the loss
        ## L2 penalty on energy magnitudes
        loss_l2 = self.args.l2_reg_energy_param * sum([pos_energy**2,
                                                       neg_energy**2])
        loss_ml = pos_energy - neg_energy  # Maximum likelihood loss
        loss = loss_ml + loss_l2
        loss = loss.mean()

        # Calculate gradients for the network params
        loss.backward()

        # Log loss to tensorboard
        self.writer.add_scalar('train/loss', loss.item(), self.global_step)

        return neg_energy, pos_energy, loss

    def update_weights(self, loss):

        # Stabilize Adam-optimized weight updates(?)
        clip_grad(self.parameters, self.optimizer)

        # Update the network params
        self.optimizer.step()

        # Print loss
        print(f'Loss: {loss.item():.5g}')

    def param_update_phase(self, neg_states, neg_id, pos_states, pos_id):

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

    def save_energies_to_histories(self):
        """Every training iteration (i.e. every positive-negative phase pair)
         record the mean negative and positive energies in the histories.
        """
        if len(self.pos_history) > self.max_history_len:
            self.pos_history.pop(0)
        self.pos_history.append(self.latest_pos_enrg)

        if len(self.neg_history) > self.max_history_len:
            self.neg_history.pop(0)
        self.neg_history.append(self.latest_neg_enrg)

    def neg_iterations_schedule_update(self):
        """Called every epoch to see whether the num it neg mean should be
        increased"""
        print("Num it neg: " + str(self.num_it_neg_mean))
        self.writer.add_scalar('train/num_it_neg', self.num_it_neg_mean,
                               self.epoch)

        if len(self.pos_history) < self.max_history_len or \
                len(self.neg_history) < self.max_history_len:
            pass
        elif self.neg_it_schedule_cooldown > 0:
            self.neg_it_schedule_cooldown -= 1
        else:
            mean_pos = sum(self.pos_history)/len(self.pos_history)
            mean_neg = sum(self.neg_history)/len(self.neg_history)
            if mean_neg > mean_pos + self.mean_neg_pos_margin \
                    and self.epoch > 10: # so never updates before 11th epoch

                # Scale up num it neg mean
                self.num_it_neg_mean = int(self.num_it_neg_mean * 1.25)

                # Log new num it neg mean
                self.writer.add_scalar('train/num_it_neg', self.num_it_neg_mean,
                                       self.epoch)

                # Reset cooldown timer and histories
                self.neg_it_schedule_cooldown = self.cooldown_len
                self.pos_history = []
                self.neg_history = []









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
        self.sample_log_dir = sample_log_dir
        self.parameters = model.parameters()
        self.optimizer = optim.Adam(self.parameters,
                                    lr=args.lr,
                                    betas=(0.9, 0.999))
        self.noises = lib.utils.generate_random_states(self.args.state_sizes,
                                                       self.device)
        self.global_step = 0
        self.batch_num = 0

        #for state in self.args.state_sizes =
        #TODO new state sizes with smaller image size vars(args)['state_sizes']

        # Load initializer network (initter)
        if args.initializer == 'ff_init':
            self.initter = nw.InitializerNetwork(args, writer, device)
            self.initter.to(device)
        else:
            self.initter = None

        # Load old networks and settings if loading an old model
        if self.args.load_model:
            loaded_model_name = str(self.args.load_model)
            path = 'exps/models/' + loaded_model_name + '.pt'
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['model_optimizer'])

            # Decide which current settings should override old settings
            new_args = checkpoint['args']
            for val in self.args.override_loaded:
                print('Overriding old value for %s' % val)
                vars(new_args)[val] = vars(self.args)[val]
            self.args = new_args

            # Turn off training
            vars(self.args)['no_train_model'] = True

            # Reload initter if it was used during training and currently using
            if self.args.initializer == 'ff_init':
                self.initter.load_state_dict(checkpoint['initializer'])
                self.initter.optimizer.load_state_dict(
                    checkpoint['initializer_optimizer'])

            self.batch_num = checkpoint['batch_num']
            print("Loaded model " + loaded_model_name + ' successfully')

            # Save new settings (doesn't overwrite old csv values)
            lib.utils.save_configs_to_csv(self.args, loaded_model_name+'viz')
        else:
            lib.utils.save_configs_to_csv(self.args, self.model.model_name)

        self.viz_batch_sizes = self.calc_viz_batch_sizes()

    def calc_viz_batch_sizes(self):
        """The batch size is now the number of pixels in the image, and
        there is only one channel because we only visualize one at a time."""
        if self.args.viz_type == 'standard':
            batch_sizes = self.args.state_sizes
        elif self.args.viz_type == 'neurons':
            batch_sizes = []
            for size in self.args.state_sizes:
                if len(size) == 4:
                    batch_sizes.append(size[2] * size[3])
                if len(size) == 2:
                    batch_sizes.append(size[1])
        elif self.args.viz_type == 'channels':
            batch_sizes = []
            for size in self.args.state_sizes:
                if len(size) == 4:
                    batch_sizes.append(size[1])
                if len(size) == 2:
                    batch_sizes.append(size[1])
        else:
            ValueError("Invalid CLI argument 'viz type'.")
        return batch_sizes

    def update_state_size_bs(self, sl_idx):
        self.model.batch_size = self.viz_batch_sizes[sl_idx]
        new_state_sizes = []
        for size in self.args.state_sizes:
            if len(size) == 4:
                new_state_sizes += [[self.viz_batch_sizes[sl_idx],
                                   size[1],
                                   size[2],
                                   size[3]]]
            if len(size) == 2:
                new_state_sizes += [[self.viz_batch_sizes[sl_idx],
                                   size[1]]]
        self.args.state_sizes = new_state_sizes
        self.model.args.state_sizes = new_state_sizes

    def calc_clamp_array_conv(self, state_layer_idx=None):

        if state_layer_idx is not None:
            size = self.args.state_sizes[state_layer_idx]
            clamp_array = torch.zeros(
                size=size,
                dtype=torch.uint8,
                device=self.device)
        else:
            size = None
            clamp_array = None

        if self.args.viz_type == 'neurons':
            # Sets to 1 the next pixel in each batch element
            mg = np.meshgrid(np.arange(0, size[2]),
                             np.arange(0, size[3]))
            idxs = list(zip(mg[1].flatten(), mg[0].flatten()))
            for i0 in range(size[0]):
                clamp_array[i0, 0][idxs[i0]] = 1.0
        elif self.args.viz_type == 'channels':
            # In batch i, sets the ith channel to 1.0
            for i in range(size[0]):
                clamp_array[i,i,:,:] = 1.0

        return clamp_array

    def visualize(self):
        """Clamps each neuron while sampling the others

        Goes through each of the state layers, and each of the channels a kind
        of negative phase where it settles for a long time and clamps a
        different neuron for each image."""

        if self.args.viz_type == 'standard':
            self.noises = lib.utils.generate_random_states(
                self.args.state_sizes,
                self.device)
            clamp_array = None
            self.visualization_phase()
        elif self.args.viz_type == 'channels':
            for state_layer_idx, size in enumerate(self.args.state_sizes[0:]):#, start=1):Lee
                if len(size) == 4:
                    print("Visualizing channels in state layer %s" % \
                          (state_layer_idx))
                    self.update_state_size_bs(state_layer_idx)
                    self.noises = lib.utils.generate_random_states(
                        self.args.state_sizes,
                        self.device)
                    clamp_array = self.calc_clamp_array_conv(state_layer_idx)
                    self.visualization_phase(state_layer_idx,
                                             channel_idx=None,
                                             clamp_array=clamp_array)
        elif self.args.viz_type == 'neurons':
            for state_layer_idx, size in enumerate(self.args.state_sizes[:]):#, start=1):Lee
                if state_layer_idx == 0:
                    continue
                if len(size) == 4:
                    for channel_idx in range(size[1]): #[1] for num of channels
                        print("Visualizing channel %s of state layer %s" % \
                              (str(channel_idx), state_layer_idx))
                        self.update_state_size_bs(state_layer_idx)
                        self.noises = lib.utils.generate_random_states(
                            self.args.state_sizes,
                            self.device)
                        clamp_array = self.calc_clamp_array_conv(state_layer_idx)
                        self.visualization_phase(state_layer_idx,
                                                 channel_idx,
                                                 clamp_array)
                elif len(size) == 2:
                    if self.args.viz_type == 'neurons':
                        # not if 'channels' because FC layers have no channels
                        print("Visualizing FC layer %s" % state_layer_idx)
                        self.update_state_size_bs(state_layer_idx)
                        self.noises  = lib.utils.generate_random_states(
                                                 self.args.state_sizes,
                                                 self.device)
                        clamp_array  = torch.eye(n=size[0], m=size[1],
                                                 dtype=torch.uint8,
                                                 device=self.device)
                        self.visualization_phase(state_layer_idx,
                                                 channel_idx=None,
                                                 clamp_array=clamp_array)
                    else:
                        print('Not visualizing FC layer because we\'re '
                              'visualizing channels')

    def visualization_phase(self, state_layer_idx=0, channel_idx=None,
                            clamp_array=None):

        states = lib.utils.generate_random_states(self.args.state_sizes,
                                                  self.device)
        id = None

        # Freeze network parameters and take grads w.r.t only the inputs
        lib.utils.requires_grad(states, True)
        lib.utils.requires_grad(self.parameters, False)
        self.model.eval()

        # Set up state optimizer if approp
        self.state_optimizers = lib.utils.get_state_optimizers(self.args,
                                                               states)

        # Viz phase sampling
        for k in tqdm(range(self.args.num_it_viz)):
            self.viz_sampler_step(states, id, state_layer_idx,
                                  clamp_array)
            if k % self.args.viz_img_logging_step_interval == 0:
                size = self.args.state_sizes[state_layer_idx]
                if clamp_array is None:
                    # Clamp array should be none during 'standard' viz
                    nrow = int(round(np.sqrt(size[0])))
                    ch_str = ''
                elif len(size) == 2:
                    nrow = int(round(np.sqrt(size[1])))
                    ch_str = 'fc' + '_'
                elif len(size) == 4:
                    nrow = size[2]
                    if channel_idx is None:
                        ch_str = ''
                    else:
                        ch_str = str(channel_idx) + '_'

                utils.save_image(states[0].detach().to('cpu'), #TODO fix so the size isn't hardcoded
                                 os.path.join(self.sample_log_dir,
                                              str(state_layer_idx) + '_' +
                                              ch_str +
                                              str(k).zfill(6)      + '.png'),
                                 nrow=nrow,
                                 normalize=True,
                                 range=(0, 1))
            self.global_step += 1

        # Stop calculting grads w.r.t. images
        for neg_state in states:
            neg_state.detach_()

        return states, id

    def viz_sampler_step(self, states, ids, state_layer_idx,
                         clamp_array):
        """In the state layer that's being visualized, the gradients come
        from the channel neuron that's being visualized. The gradients for
        other layers come from the energy gradient."""

        energies, outs = self.model(states, ids)  # Outputs energy of neg sample
        total_energy = energies.sum()

        if self.args.viz_type == 'standard':
            # Take gradient wrt states (before addition of noise)
            total_energy.backward()

        elif self.args.viz_type == 'neurons' or self.args.viz_type == 'channels':
            # Get the energy of the specific neuron you want to viz and get the
            # gradient that maximises its value
            feature_energy = outs[state_layer_idx] # -1 because energies is 0-indexed while the layers we want to visualize are 1-indexed
            feature_energy = feature_energy \
                             * self.args.energy_weight_mask[state_layer_idx]\
                             * 1. # Scales up the energy of the neuron/channel that we want to viz
            selected_feature_energy = torch.where(clamp_array,
                                                  feature_energy,
                                                  torch.zeros_like(feature_energy))
            selected_feature_energy = -selected_feature_energy.sum()
            # print(state_layer_idx)
            # print([fe.sum() for fe in outs])
            #selected_feature_energy.backward(retain_graph=True)

            # Take gradient wrt states (before addition of noise)
            total_energy.backward()

            # Zero the grads above the layer that we're visualizing
            # for s in states[state_layer_idx+1:]:
            #     s.grad.zero_() #TODO check what this is actually doing

        # The rest of the sampler step function is no different from the
        # negative step used in training
        torch.nn.utils.clip_grad_norm_(states,
                                       self.args.clip_state_grad_norm,
                                       norm_type=2)

        # Adding noise in the Langevin step (only for non conditional
        # layers in positive phase)
        if not self.args.state_optimizer == "sghmc":
            for layer_idx, (noise, state) in enumerate(zip(self.noises, states)):
                noise.normal_(0, self.args.sigma)
                state.data.add_(noise.data)


        # The gradient step in the Langevin/SGHMC step
        # It goes through each statelayer and steps back using its associated
        # optimizer. The gradients may pass through any network unless
        # you're using an appropriate energy mask that zeroes the grad through
        # that network in particular.
        for layer_idx, state in enumerate(states):
                self.state_optimizers[layer_idx].step()

        # Prepare gradients and sample for next sampling step
        for state in states:
            state.grad.detach_()
            state.grad.zero_()
            state.data.clamp_(0, 1)


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

def calc_enrg_masks(args):
    m = [divl(args.state_sizes[0][1:], x[1:]).item() for x in args.state_sizes]
    print(m)
    return m

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

        elif args.architecture == 'ConvBFN_med_2_dense_3layers':
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
                                         [args.batch_size, 64, 28, 28],
                                         [args.batch_size, 16, 28, 28],
                                         [args.batch_size, 16, 28, 28],
                                         [args.batch_size, 16, 28, 28],
                                         [args.batch_size, 16, 12, 12],
                                         [args.batch_size, 16, 12, 12],
                                         [args.batch_size, 16, 12, 12],
                                         #[args.batch_size, 16, 4, 4],#???Size???

                                         ]

            mod_connect_dict = {0: [],
                                1: [0],
                                2: [1],
                                3: [1,2],
                                4: [1,2,3],
                                5: [4],
                                6: [4,5],
                                7: [4,5,6]}
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
        elif args.architecture == 'ConvBFN_small_3_layers':
            vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
                                         [args.batch_size, 32, 28, 28],
                                         [args.batch_size, 32, 12, 12],
                                         [args.batch_size, 32, 4, 4]
                                         ]

            mod_connect_dict = {0: [],
                                1: [0],
                                2: [1],
                                3: [2]
                                }
            mod_kernel_dict = {0: [],
                                1: [3],
                                2: [7],
                                3: [7]
                                }
            mod_padding_dict = {0: [],
                                1: [1],
                                2: [1],
                                3: [1]
                                }
            mod_strides_dict = {0: [],
                                1: [1],
                                2: [2],
                                3: [2]
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
            vars(args)['energy_weight_mask'] = calc_enrg_masks(args)

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

    # Print final values for args
    for k, v in zip(vars(args).keys(), vars(args).values()):
        print(str(k) + '\t' * 2 + str(v))

    return args


def main():
    ### Parse CLI arguments.
    parser = argparse.ArgumentParser(description='Deep Attractor Network.')
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
    tgroup.add_argument('--pos_buffer_frac', type=float, default=0.0,
                        help='The fraction of images from the positive buffer to use to initialize negative samples.'
                             'Default: %(default)s.')
    tgroup.add_argument('--shuffle_pos_frac', type=float, default=0.0,
                        help='The fraction of images from the positive buffer that will be shuffled before initializing negative samples. The motivation for this is for experiments when a new image will initialize the state but the previous values for latent variables will be used, as would happen when an animal is presented with a new image.'
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
                             '[hardsig, relu, swish]'
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
    ngroup.add_argument('--energy_weight_min', type=float, default=0.001,
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
    ngroup.add_argument('--no_spec_norm_reg', action='store_true',
                        help='If true, networks are NOT subjected to ' +
                             'spectral norm regularisation. ' +
                             'Default: %(default)s.')
    parser.set_defaults(no_spec_norm_reg=False)


    vgroup = parser.add_argument_group('Visualization options')
    vgroup.add_argument('--viz', action='store_true',
                        help='Whether or not to do visualizations. The exact'
                             'type of visualization is defined in the'
                             '"viz_type" argument. Default: %(default)s.')
    parser.set_defaults(viz=False)
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
        model = nw.BengioFischerNetwork(args, device, model_name, writer).to(
        device)
    elif args.network_type == 'ConvBFN':
        model = nw.ConvBengioFischerNetwork(args, device, model_name, writer).to(
            device)
    elif args.network_type == 'VectorField':
        model = nw.VectorFieldNetwork(args, device, model_name, writer).to(
            device)
    elif args.network_type == 'DAN':
        model = nw.DeepAttractorNetwork(args, device, model_name, writer).to(
            device)
    elif args.network_type == 'EBMLV':
        model = nw.EBMLV(args, device, model_name, writer).to(
            device)
    elif args.network_type == 'SVF':
        model = nw.StructuredVectorFieldNetwork(args, device, model_name,
                                                writer).to(device)
    elif args.network_type == 'VFEBMLV':
        model = nw.VFEBMLV(args, device, model_name, writer).to(device)
    else:
        raise ValueError("Invalid CLI argument for argument 'network_type'. ")

    # Set up dataset
    data = Dataset(args)
    buffer = SampleBuffer(args, device=device)

    if not args.no_train_model:
        # Train the model
        tm = TrainingManager(args, model, data, buffer, writer, device,
                             sample_log_dir)
        tm.train()
    if args.viz:
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
divl = lambda l1, l2: torch.prod(torch.tensor(l1)).float()/torch.prod(torch.tensor(l2)).float()

if __name__ == '__main__':
    main()


###########################################################################
###########################################################################
###########################################################################




# if args.dataset == 'MNIST':
#         if args.architecture == 'mnist_1_layer_small':
#             vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
#                                          [args.batch_size, 3, 3, 3]]
#             vars(args)['arch_dict'] = {'num_ch': 32,
#                                        'num_sl': len(args.state_sizes) - 1,
#                                        'kernel_sizes': [3, 3],
#                                        'strides': [1,1],
#                                        'padding': 1}
#         elif args.architecture == 'mnist_2_layers_small':
#             vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
#                                          [args.batch_size, 9, 28, 28],
#                                          [args.batch_size, 9, 3, 3]]
#             vars(args)['arch_dict'] = {'num_ch': 16,
#                                        'num_sl': len(args.state_sizes) - 1,
#                                        'kernel_sizes': [3, 3],
#                                        'strides'3: [1,1],
#                                        'padding': 1}
#         elif args.architecture == 'mnist_2_layers_small_equal':
#             vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
#                                          [args.batch_size, 6, 16, 16],
#                                          [args.batch_size, 256, 3, 3]]
#             vars(args)['arch_dict'] = {'num_ch': 16,
#                                        'num_sl': len(args.state_sizes) - 1,
#                                        'kernel_sizes': [3, 3],
#                                        'strides': [1, 0],
#                                        'padding': 1}
#         elif args.architecture == 'mnist_3_layers_small':
#             vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
#                                          [args.batch_size, 9, 28, 28], #scale 1
#                                          [args.batch_size, 9, 3, 3], # scale 87.1
#                                          [args.batch_size, 9, 1, 1]] # scale 784
#             vars(args)['arch_dict'] = {'num_ch': 64,
#                                        'num_sl': len(args.state_sizes) - 1,
#                                        'kernel_sizes': [3, 3, 1],
#                                        'strides': [1,1,1],
#                                        'padding': 1}
#         elif args.architecture == 'mnist_3_layers_med':
#             vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
#                                          [args.batch_size, 64, 28, 28],
#                                          [args.batch_size, 32, 9, 9],
#                                          [args.batch_size, 10, 3, 3]]
#             vars(args)['arch_dict'] = {'num_ch': 64,
#                                        'num_sl': len(args.state_sizes) - 1,
#                                        'kernel_sizes': [3, 3, 3],
#                                        'strides': [1,1,1],
#                                        'padding': 0}
#         elif args.architecture == 'mnist_3_layers_large': # Have roughly equal amount of 'potential energy' (i.e. neurons) in each layer
#             vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
#                                          [args.batch_size, 8, 28, 28],  # 6272
#                                          [args.batch_size, 24, 16, 16], # 6144
#                                          [args.batch_size, 180, 6, 6]]  # 4608
#             vars(args)['arch_dict'] = {'num_ch': 64,
#                                        'num_sl': len(args.state_sizes) - 1,
#                                        'kernel_sizes': [3, 3, 3],
#                                        'strides': [1,1],
#                                        'padding': 1}
#         elif args.architecture == 'mnist_4_layers_med': # Have roughly equal amount of 'potential energy' (i.e. neurons) in each layer
#             vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
#                                          [args.batch_size, 16, 28, 28],  # 12544
#                                          [args.batch_size, 24, 16, 16],  # 6144
#                                          [args.batch_size, 32,  9,  9],  # 2592
#                                          [args.batch_size, 180, 3,  3]]  # 1620
#             vars(args)['arch_dict'] = {'num_ch': 64,
#                                        'num_sl': len(args.state_sizes) - 1,
#                                        'kernel_sizes': [3, 3, 3],
#                                        'strides': [1,1],
#                                        'padding': 1}
#             vars(args)['energy_weight_mask'] = [1, 2, 4.84, 7.743]
#         elif args.architecture == 'mnist_5_layers_med_fc_top2': # Have roughly equal amount of 'potential energy' (i.e. neurons) in each layer
#             vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
#                                          [args.batch_size, 16, 28, 28],  # 12544
#                                          [args.batch_size, 24, 16, 16],  # 6144
#                                          [args.batch_size, 32,  9,  9],  # 2592
#                                          [args.batch_size, 1024],
#                                          [args.batch_size, 256]]
#             mod_connect_dict = {0: [],
#                                 1: [0,1],
#                                 2: [1,2],
#                                 3: [2,3],
#                                 4: [3,4],
#                                 5: [4,5]}
#             vars(args)['arch_dict'] = {'num_ch': 64,
#                                        'num_sl': len(args.state_sizes) - 1,
#                                        'kernel_sizes': [3, 3, 3],
#                                        'strides': [1,1],
#                                        'padding': 1,
#                                        'mod_connect_dict': mod_connect_dict}
#             vars(args)['energy_weight_mask'] = [1, 2, 4.84, 12.25, 49]
#
#         elif args.architecture == 'mnist_2_layers_small_scl_UandD':
#             vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
#                                          [args.batch_size, 32, 28, 28],  # 25088
#                                          [args.batch_size, 32, 12, 12],  # 4608
#                                          ]
#
#             mod_connect_dict = {0: [],
#                                 1: [0,1],
#                                 2: [1,2]}
#
#             vars(args)['arch_dict'] = {'num_ch': 32,
#                                        'num_sl': len(args.state_sizes) - 1,
#                                        'kernel_sizes': [3, 3, 3],
#                                        'strides': [1,1],
#                                        'padding': 1,
#                                        'mod_connect_dict': mod_connect_dict}
#             vars(args)['energy_weight_mask'] = [0.184, 1.0]
#
#         elif args.architecture == 'mnist_3_layers_med_fc_top1': # Have roughly equal amount of 'potential energy' (i.e. neurons) in each layer
#             vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
#                                          [args.batch_size, 16, 28, 28],  # 12544
#                                          [args.batch_size, 32,  9,  9],  # 2592
#                                          [args.batch_size, 256]]
#
#             mod_connect_dict = {0: [],
#                                 1: [0,1],
#                                 2: [1,2],
#                                 3: [2,3]}
#
#             vars(args)['arch_dict'] = {'num_ch': 64,
#                                        'num_sl': len(args.state_sizes) - 1,
#                                        'kernel_sizes': [3, 3, 3],
#                                        'strides': [1,1],
#                                        'padding': 1,
#                                        'mod_connect_dict': mod_connect_dict}
#             vars(args)['energy_weight_mask'] = [1, 4.84, 49]
#
#         elif args.architecture == 'mnist_3_layers_med_fc_top1_upim_wild':
#             vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
#                                          [args.batch_size, 16, 48, 48],  # 36864
#                                          [args.batch_size, 32, 12, 12],  # 4608
#                                          [args.batch_size, 256]]
#
#             mod_connect_dict = {0: [],
#                                 1: [0,1],
#                                 2: [1,2],
#                                 3: [2,3]}
#
#             vars(args)['arch_dict'] = {'num_ch': 64,
#                                        'num_sl': len(args.state_sizes) - 1,
#                                        'kernel_sizes': [3, 3, 3],
#                                        'strides': [1,1],
#                                        'padding': 1,
#                                        'mod_connect_dict': mod_connect_dict}
#             vars(args)['energy_weight_mask'] = [1.0, 8.0, 144.0]
#         elif args.architecture == 'mnist_3_layers_large_fc_top1_upim_scl_UandD':
#             vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
#                                          [args.batch_size, 32, 48, 48],  # 73728
#                                          [args.batch_size, 32, 12, 12],  # 4608
#                                          [args.batch_size, 1024]]
#
#             mod_connect_dict = {0: [],
#                                 1: [0,1],
#                                 2: [1,2],
#                                 3: [2,3]}
#
#             vars(args)['arch_dict'] = {'num_ch': 64,
#                                        'num_sl': len(args.state_sizes) - 1,
#                                        'kernel_sizes': [3, 3, 3],
#                                        'strides': [1,1],
#                                        'padding': 1,
#                                        'mod_connect_dict': mod_connect_dict}
#             vars(args)['energy_weight_mask'] = [0.06, 1.0, 2.5]
#
#         elif args.architecture == 'mnist_3_layers_small_fc_top1_scl_UandD':
#             vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
#                                          [args.batch_size, 32, 28, 28],  # 25088
#                                          [args.batch_size, 32, 12, 12],  # 4608
#                                          [args.batch_size, 256]]
#
#             mod_connect_dict = {0: [],
#                                 1: [0,1],
#                                 2: [1,2],
#                                 3: [2,3]}
#
#             vars(args)['arch_dict'] = {'num_ch': 32,
#                                        'num_sl': len(args.state_sizes) - 1,
#                                        'kernel_sizes': [3, 3, 3],
#                                        'strides': [1,1],
#                                        'padding': 1,
#                                        'mod_connect_dict': mod_connect_dict}
#             vars(args)['energy_weight_mask'] = [0.184, 1.0, 18.0]
#
#         elif args.architecture == 'mnist_4_layers_med_fc_top1_upim':
#             vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
#                                          [args.batch_size, 16, 48, 48],  # 36864
#                                          [args.batch_size, 32, 12, 12],  # 4608
#                                          [args.batch_size, 1024],
#                                          [args.batch_size, 256]]
#
#             mod_connect_dict = {0: [],
#                                 1: [0,1],
#                                 2: [1,2],
#                                 3: [2,3],
#                                 4: [3,4]}
#
#             vars(args)['arch_dict'] = {'num_ch': 64,
#                                        'num_sl': len(args.state_sizes) - 1,
#                                        'kernel_sizes': [3, 3, 3],
#                                        'strides': [1,1],
#                                        'padding': 1,
#                                        'mod_connect_dict': mod_connect_dict}
#             vars(args)['energy_weight_mask'] = [1.0, 8.0, 36, 144.0]
#
#         elif args.architecture == 'mnist_5_layers_med_fc_top1_upim':
#             vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
#                                          [args.batch_size, 16, 48, 48],  # 36864
#                                          [args.batch_size, 32, 12, 12],  # 4608
#                                          [args.batch_size, 32, 6, 6],    # 1152
#                                          [args.batch_size, 1024],
#                                          [args.batch_size, 256]]
#
#
#             mod_connect_dict = {0: [],
#                                 1: [0,1],
#                                 2: [1,2],
#                                 3: [2,3],
#                                 4: [3,4],
#                                 5: [4,5]}
#
#             vars(args)['arch_dict'] = {'num_ch': 64,
#                                        'num_sl': len(args.state_sizes) - 1,
#                                        'kernel_sizes': [3, 3, 3],
#                                        'strides': [1,1],
#                                        'padding': 1,
#                                        'mod_connect_dict': mod_connect_dict}
#             vars(args)['energy_weight_mask'] = [1.0, 8.0, 32.0, 36, 144.0]
#


# Since changes to network

# elif args.architecture == 'DAN_very_small_3_layers':
#     vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
#                                  [args.batch_size, 100],
#                                  [args.batch_size, 50]]
#
#     mod_connect_dict = {0: [1],
#                         1: [0,2],
#                         2: [1,3]}
#
#     vars(args)['arch_dict'] = {'num_ch': 16,
#                                'num_ch_initter': 16,
#                                'num_sl': len(args.state_sizes) - 1,
#                                'kernel_sizes': [3, 3, 3],
#                                'strides': [1,1],
#                                'padding': 1,
#                                'mod_connect_dict': mod_connect_dict,
#                                'num_fc_channels': 64}
#     vars(args)['energy_weight_mask'] = [1.0, 7.84, 15.68]
#     elif args.architecture == 'DAN_large_5_layers_selftop': #untested
#         vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
#                                      [args.batch_size, 32, 16, 16],
#                                      [args.batch_size, 32, 8, 8],
#                                      [args.batch_size, 100],
#                                      [args.batch_size, 50]]
#
#         mod_connect_dict = {0: [1],
#                             1: [0,2],
#                             2: [1,3],
#                             3: [2,4],
#                             4: [3,4]}
#
#         vars(args)['arch_dict'] = {'num_ch': 64,
#                                    'num_ch_initter': 64,
#                                    'num_sl': len(args.state_sizes) - 1,
#                                    'kernel_sizes': [3, 3, 3],
#                                    'strides': [1,1],
#                                    'padding': 1,
#                                    'mod_connect_dict': mod_connect_dict,
#                                    'num_fc_channels': 64}
#         vars(args)['energy_weight_mask'] = [1.0, 0.09, 0.383, 7.84, 15.68]
#     elif args.architecture == 'EBMLV_very_small_4_layers_self':
#         vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
#                                      [args.batch_size, 16, 8, 8],
#                                      [args.batch_size, 100],
#                                      [args.batch_size, 50]]
#
#         mod_connect_dict = {0: [0,1],
#                             1: [0,1,2],
#                             2: [1,2,3],
#                             3: [2,3]}
#
#         vars(args)['arch_dict'] = {'num_ch': 16,
#                                    'num_ch_initter': 16,
#                                    'num_sl': len(args.state_sizes) - 1,
#                                    'kernel_sizes': [[3, 3], [3, 3], [3, 3]],
#                                    'strides': [1,1,1],
#                                    'padding': [[1,1], [1,1], [1,1]],
#                                    'mod_connect_dict': mod_connect_dict,
#                                    'num_fc_channels': 16}
#         vars(args)['energy_weight_mask'] = [1.0, 0.784, 7.84, 15.68]
#         #[1.0, 0.784, 7.84, 15.68] [1.0, 1.0, 1.0, 1.0]
#
#     elif args.architecture == 'EBMLV_very_small_4_layers_topnself':
#         vars(args)['state_sizes'] = [[args.batch_size,  1, 28, 28],
#                                      [args.batch_size, 16, 8, 8],
#                                      [args.batch_size, 100],
#                                      [args.batch_size, 50]]
#
#         mod_connect_dict = {0: [1],
#                             1: [0,1,2],
#                             2: [1,2,3],
#                             3: [2,3]}
#
#         vars(args)['arch_dict'] = {'num_ch': 16,
#                                    'num_ch_initter': 16,
#                                    'num_sl': len(args.state_sizes) - 1,
#                                    'kernel_sizes': [[3, 3], [3, 3], [3, 3]],
#                                    'strides': [1,1,1],
#                                    'padding': [[1,1], [1,1], [1,1]],
#                                    'mod_connect_dict': mod_connect_dict,
#                                    'num_fc_channels': 16}
#         vars(args)['energy_weight_mask'] = [1.0, 1.0, 1.0, 1.0]
#         #[1.0, 0.765, 7.84, 15.68] [1.0, 1.0, 1.0, 1.0]

# elif args.architecture == 'DAN_very_small_3_layers':
#     vars(args)['state_sizes'] = [[args.batch_size, 1, 28, 28],
#                                  [args.batch_size, 100],
#                                  [args.batch_size, 50]]
#
#     mod_connect_dict = {0: [1],
#                         1: [0, 2],
#                         2: [1, 3]}
#
#     vars(args)['arch_dict'] = {'num_ch': 16,
#                                'num_ch_initter': 16,
#                                'num_sl': len(args.state_sizes) - 1,
#                                'kernel_sizes': [[3, 3], [3, 3]],
#                                'strides': [[1, 1], [1, 1]],
#                                'padding': [[1, 1], [1, 1]],
#                                'mod_connect_dict': mod_connect_dict,
#                                'num_fc_channels': 64}
#     vars(args)['energy_weight_mask'] = [1.0, 7.84, 15.68]
#
# --cd_mixture
# --pos_buffer_frac
# 0.1
# --shuffle_pos_frac
# 0.4
