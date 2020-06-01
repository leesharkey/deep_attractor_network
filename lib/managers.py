import os
import numpy as np
import torch
from torch import optim
from tqdm import tqdm
import lib.networks.models as models
import lib.utils
import lib.networks.initializer as initializer
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from PIL import Image
import torchvision.transforms.functional


class Manager():
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
                                    betas=(0.0, 0.999))  # betas=(0.9, 0.999))
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size=1,
                                                   gamma=self.args.lr_decay_gamma)
        self.noises = lib.utils.generate_random_states(self.args.state_sizes,
                                                       self.device)
        self.global_step = 0
        self.batch_num = 0
        self.epoch = 0

        # Load initializer network (initter)
        if args.initializer == 'ff_init':
            self.initter = initializer.InitializerNetwork(args, writer, device,
                              layer_norm=self.args.initter_network_layer_norm,
                              weight_norm=self.args.initter_network_weight_norm)
            self.initter.to(device)
        else:
            self.initter = None

        # Load old networks and settings if loading an old model
        if self.args.load_model:
            self.loaded_model_name = str(self.args.load_model)
            path = 'exps/models/' + self.loaded_model_name + '.pt'
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
            print("Loaded model " + self.loaded_model_name + ' successfully')

            # Save new settings (doesn't overwrite old csv values)
            lib.utils.save_configs_to_csv(self.args, self.loaded_model_name+'loaded')
        else:
            lib.utils.save_configs_to_csv(self.args, self.model.model_name)

            self.num_it_neg_mean = self.args.num_it_neg
            self.num_it_neg = self.args.num_it_neg
            self.loaded_model_name = None

        # Print out param sizes to ensure you aren't using something stupidly
        # large
        param_sizes = [torch.prod(torch.tensor(sz)).item() for sz in
                       [prm.shape for prm in self.model.parameters()]]
        param_sizes.sort()
        top10_params = param_sizes[-10:]
        print("Top 10 network param sizes: \n %s" % str(top10_params))

    def make_save_dict(self):
        """Assembles the dictionary of objects to be saved"""
        save_dict = {'model': self.model.state_dict(),
                     'model_optimizer': self.optimizer.state_dict(),
                     'args': self.args,
                     'epoch': self.epoch,
                     'batch_num': self.batch_num,
                     'global_step': self.global_step}
        if self.args.initializer == 'ff_init':
            initter_sd = {'initializer': self.initter.state_dict(),
                          'initializer_optimizer': self.initter.optimizer.state_dict()}
            save_dict = {**save_dict, **initter_sd}
        return save_dict

    def save_net_and_settings(self):
        save_dict = self.make_save_dict()
        path = 'exps/models/' + self.model.model_name + '.pt'
        torch.save(save_dict, path)


class TrainingManager(Manager):
    def __init__(self, args, model, data, buffer, writer, device,
                 sample_log_dir):
        super().__init__(args, model, data, buffer, writer, device,
                 sample_log_dir)

        # Set the rest of the hyperparams for iteration scheduling
        self.pos_short_term_history = []
        self.pos_history = []
        self.neg_history = []
        self.max_history_len = 200
        self.mean_neg_pos_margin = 0 #somewhere between -100 and 200 seems sensible  ## mean_neg > mean_pos + self.mean_neg_pos_margin
        self.neg_it_schedule_cooldown = 0  # Always set this to 0
        self.cooldown_len = 5 #epochs
        self.latest_pos_enrg = None
        self.latest_neg_enrg = None
        self.num_it_neg_mean = self.args.num_it_neg
        self.num_it_neg = self.args.num_it_neg

    def train(self):
        self.save_net_and_settings()
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

                prev_states = [ps.clone().detach() for ps in pos_states] # In case pos init uses prev states

                if self.batch_num % self.args.img_logging_interval == 0:
                    self.log_images(pos_img, pos_states, neg_states)

                if self.epoch > 1:
                    self.save_energies_to_histories()
                    self.log_mean_energy_histories()

                # Save network(s) and settings
                if self.batch_num % self.args.model_save_interval == 0:
                    self.save_net_and_settings()

                self.batch_num += 1

                if self.num_it_neg_mean > 5000:
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
        self.save_net_and_settings()

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

        self.pos_short_term_history = []
        self.stop_pos_phase = False

        # Positive phase sampling
        for i in tqdm(range(self.args.num_it_pos)):
            self.sampler_step(pos_states, pos_id, positive_phase=True, pos_it=i,
                              step=self.global_step)
            if self.stop_pos_phase:
                break
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
        else:
            self.num_it_neg = self.num_it_neg_mean

        # Negative phase sampling
        for _ in tqdm(range(self.num_it_neg)):
            self.sampler_step(neg_states, neg_id, step=self.global_step)
            self.global_step += 1
            # #TODO remove when done debugging viz
            # neg_imgs_save = neg_states[0].detach().to('cpu')
            # utils.save_image(neg_imgs_save,
            #                  os.path.join(self.sample_log_dir,
            #                               str(self.batch_num).zfill(
            #                                   6) + str(_) + '.png'),
            #                  nrow=16, normalize=True, range=(0, 1))

        # Stop calculting grads w.r.t. images
        for neg_state in neg_states:
            neg_state.detach_()

        # Send negative samples to the negative buffer
        self.buffer.push(neg_states, neg_id)

        return neg_states, neg_id

    def sampler_step(self, states, ids, positive_phase=False, pos_it=None,
                     step=None):

        # Get total energy and energy outputs for indvdual neurons
        energy, outs, full_energies = self.model(states, ids, step)

        # Calculate the gradient wrt states for the Langevin step (before
        # addition of noise)
        if self.args.state_optimizer == 'sghmc':
            energy.backward()
        else:
            (-energy).backward()
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
            for i, (nrgy, out) in enumerate(zip(full_energies, outs)):
                mean_layer_string = 'layers/mean_bnry_%s' % i
                mean_full_enrg_string = 'layers/mean_energies_%s' % i
                self.writer.add_scalar(mean_layer_string, out.mean(), step)
                self.writer.add_scalar(mean_full_enrg_string, nrgy.mean(), step)

                if self.args.log_histograms  and \
                        step % self.args.histogram_logging_interval == 0:
                    hist_layer_string = 'layers/hist_bnrys_%s' % i
                    hist_full_enrg_string = 'layers/hist_energies_%s' % i

                    #print("Logging energy histograms")
                    self.writer.add_histogram(hist_layer_string, out, step)
                    self.writer.add_histogram(hist_full_enrg_string, nrgy, step)


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
            if self.args.truncate_pos_its:
                self.pos_iterations_trunc_update(pos_it)
        else:
            self.latest_neg_enrg = energy.item()


        # Prepare gradients and sample for next sampling step
        for i, state in enumerate(states):
            state.grad.detach_()
            state.grad.zero_()
            if self.args.states_activation == 'relu' and i>0:
                state.data.clamp_(0)
            else:
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
        pos_energy, _, _ = self.model(pos_states, pos_id)
        neg_energy, _, _ = self.model(neg_states, neg_id)

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
        self.clip_grad(self.parameters, self.optimizer)

        # Update the network params
        self.optimizer.step()

        # Print loss
        print(f'Loss: {loss.item():.5g}')

    def clip_grad(self, parameters, optimizer):
        with torch.no_grad():
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]

                    if 'step' not in state or state['step'] < 1:
                        continue

                    step = state['step']
                    exp_avg_sq = state['exp_avg_sq']
                    _, beta2 = group['betas']

                    bound = 3 * torch.sqrt(
                        exp_avg_sq / (1 - beta2 ** step)) + 0.1
                    p.grad.data.copy_(
                        torch.max(torch.min(p.grad.data, bound), -bound))

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

    def pos_iterations_trunc_update(self, current_iteration):
        if len(self.pos_short_term_history) > 15:
            self.pos_short_term_history.pop(0)
        self.pos_short_term_history.append(self.latest_pos_enrg)
        diff = max(self.pos_short_term_history) - \
               min(self.pos_short_term_history)
        if diff < self.args.truncate_pos_its_threshold \
                and current_iteration > 15:
            self.stop_pos_phase = True
            self.writer.add_scalar('train/trunc_pos_at', current_iteration,
                                   self.batch_num)
        else:
            self.stop_pos_phase = False

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
                    and self.epoch > 5: # so never updates before 6th epoch

                # Scale up num it neg mean
                self.num_it_neg_mean = int(self.num_it_neg_mean * 1.25)

                # Log new num it neg mean
                self.writer.add_scalar('train/num_it_neg', self.num_it_neg_mean,
                                       self.epoch)

                # Reset cooldown timer and histories
                self.neg_it_schedule_cooldown = self.cooldown_len
                self.pos_history = []
                self.neg_history = []


class VisualizationManager(Manager):
    """A new class because I need to redefine sampler step to clamp certain
    neurons, and need to define a new type of sampling phase."""

    def __init__(self, args, model, data, buffer, writer, device,
                 sample_log_dir):
        super().__init__(args, model, data, buffer, writer, device,
                         sample_log_dir)

        self.reset_opt_K_its = True
        self.reset_freq = 10000
        self.energy_scaler = 0.5

        # Defines what the index under investigation will be set as in the
        # clamp array. If 1, and clamp_value1or0 is 1, then during
        # channels_state inference the state value at
        # the index/channel being studied will be set to a value of1. If
        # clamp_idx1or0 is zero, then all the other indices will be set to a
        # value of 1. I think it makes most sense to set the clamp_index1or0
        # to 1 and the clamp_value1or0 to 1.
        self.clamp_idx_one_or_zero = 1.0
        self.clamp_value = 0.8

        for i, s in enumerate(self.args.state_sizes):
            vars(self.args)['state_sizes'][i][0] = 128

    def calc_clamp_array_conv(self, state_layer_idx=None, current_ch=None):
        if state_layer_idx is not None:
            size = self.args.state_sizes[state_layer_idx]
            if self.clamp_idx_one_or_zero == 1.0:
                clamp_array = torch.zeros(size=size,
                                          dtype=torch.uint8,
                                          device=self.device)
            else:
                clamp_array = torch.ones(size=size,
                                          dtype=torch.uint8,
                                          device=self.device)
        else:
            size = None
            clamp_array = None

        if self.args.viz_type in ['channels_energy', 'channels_state']:
            # In batch i, sets the ith channel to 1.0
            clamp_array[:,current_ch,13:19,13:19] = self.clamp_idx_one_or_zero

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
        elif self.args.viz_type in ['channels_energy', 'channels_state']:
            start_layer = self.args.viz_start_layer
            for state_layer_idx, size in enumerate(
                    self.args.state_sizes[start_layer:], start=start_layer):
                print("Visualizing channels in state layer %s" % \
                      (state_layer_idx))
                num_ch = size[1]
                for ch in range(num_ch):
                    print("Channel %s" % ch)
                    #self.update_state_size_bs(state_layer_idx)
                    self.noises = lib.utils.generate_random_states(
                        self.args.state_sizes,
                        self.device)
                    clamp_array = self.calc_clamp_array_conv(state_layer_idx,
                                                             current_ch=ch)
                    self.visualization_phase(state_layer_idx,
                                             channel_idx=ch,
                                             clamp_array=clamp_array)

    def visualization_phase(self, state_layer_idx=0, channel_idx=None,
                            clamp_array=None):


        states = lib.utils.generate_random_states(self.args.state_sizes,
                                                  self.device)
        # states = [s * 0.25 for s in states]
        # if self.args.initializer == 'ff_init':
        #     states_new = [states[0]]
        #     self.initter.eval()
        #     initted_states = self.initter.forward(states[0], [])
        #     states_new.extend(initted_states)
        #     states = states_new
        #     states = [s.detach() for s in states]

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
            if self.reset_opt_K_its and k % self.reset_freq == 0:
                print('Resetting state optimizers')
                self.state_optimizers = lib.utils.get_state_optimizers(
                    self.args, states)

            if self.args.viz_type == 'standard':
                self.viz_sampler_step_standard(states, id, state_layer_idx,
                                      clamp_array)
            elif self.args.viz_type == 'channels_energy':
                self.viz_sampler_step_channels_energy(states, id, state_layer_idx,
                                      clamp_array)
            elif self.args.viz_type == 'channels_state':
                self.viz_sampler_step_channels_state(states, id, state_layer_idx,
                                      clamp_array)
            if k % self.args.viz_img_logging_step_interval == 0:
                size = self.args.state_sizes[state_layer_idx]
                nrow = int(round(np.sqrt(size[0])))

                if self.args.viz_type is 'standard':
                    # Clamp array should be none during 'standard' viz
                    ch_str = ''
                else:
                    ch_str = '_ch' + str(channel_idx) + '_'

                utils.save_image(states[0].detach().to('cpu'),
                                 os.path.join(self.sample_log_dir,
                                              'lyr' + str(state_layer_idx) +
                                              ch_str +
                                              str(k).zfill(6)      + '.png'),
                                 nrow=16,
                                 normalize=True,
                                 range=(0, 1))
            self.global_step += 1

        # Stop calculting grads w.r.t. images
        for state in states:
            state.detach_()

        return states, id

    def viz_sampler_step_standard(self, states, ids, state_layer_idx,
                         clamp_array):
        """In the state layer that's being visualized, the gradients come
        from the channel neuron that's being visualized. The gradients for
        other layers come from the energy gradient."""

        energy, outs, energies = self.model(states, ids)  # Outputs energy of neg sample
        total_energy = energy.sum()
        total_energies = sum([e.sum() for e in energies])

        # Take gradient wrt states (before addition of noise)
        #total_energy.backward() #
        total_energies.backward()

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
        for i, state in enumerate(states):
            state.grad.detach_()
            state.grad.zero_()
            if self.args.states_activation == 'relu' and i>0:
                state.data.clamp_(0)
            else:
                state.data.clamp_(0, 1)

    def viz_sampler_step_channels_energy(self, states, ids, state_layer_idx,
                         clamp_array):
        """In the state layer that's being visualized, the gradients come
        from the channel neuron that's being visualized. The gradients for
        other layers come from the energy gradient."""

        energy, outs, energies = self.model(states, ids)  # Outputs energy of neg sample
        total_energy = energy.sum()
        total_energies = sum([e.sum() for e in energies])

        # Reshape energies
        energies = [enrg.view(state.shape) for enrg, state in zip(energies, states)]

        # Get the energy of the specific neuron you want to viz and get the
        # gradient that maximises its value
        feature_energy = energies[state_layer_idx]  # -1 because energies is 0-indexed while the layers we want to visualize are 1-indexed
        feature_energy = feature_energy \
                         * self.energy_scaler # Scales up the energy of the neuron/channel that we want to viz
        selected_feature_energy = torch.where(clamp_array,
                                              feature_energy,
                                              torch.zeros_like(feature_energy))
        selected_feature_energy = selected_feature_energy.sum()

        # Take gradient wrt states and then zero them
        total_energies.backward(retain_graph=True)
        stategrads = [st.grad.data.clone() for st in states]
        for s in states:
            s.grad.zero_()

        # Get the gradients of the selected feature
        selected_feature_energy.backward()

        # Fill in the gradients below the layer that you're currently
        # visualizing only if the gradients would be 0 otherwise.
        for i, state in enumerate(states):
            if torch.all(torch.eq(state.grad.data,
                   torch.zeros_like(state.grad.data))) and i < state_layer_idx:

                state.grad.data = stategrads[i]


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
        for i, state in enumerate(states):
            state.grad.detach_()
            state.grad.zero_()
            if self.args.states_activation == 'relu' and i>0:
                state.data.clamp_(0)
            else:
                state.data.clamp_(0, 1)


    def viz_sampler_step_channels_state(self, states, ids, state_layer_idx,
                         clamp_array):
        """In the state layer that's being visualized, the gradients come
        from the channel neuron that's being visualized. The gradients for
        other layers come from the energy gradient."""

        energy, outs, energies = self.model(states, ids)  # Outputs energy of neg sample
        total_energy = energy.sum()
        total_energies = sum([e.sum() for e in energies])

        total_energies.backward()

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
        for i, state in enumerate(states):
            state.grad.detach_()
            state.grad.zero_()
            if self.args.states_activation == 'relu' and i > 0:
                state.data.clamp_(0)
            else:
                state.data.clamp_(0, 1)

            # Clamp the state values to some chosen value
            if i == state_layer_idx:
                if self.clamp_value == 1.0:
                    clamped_values = torch.ones_like(state.data)
                    opp_clamped_values = torch.zeros_like(state.data)
                elif self.clamp_value == 0.0:
                    clamped_values = torch.zeros_like(state.data)
                    opp_clamped_values = torch.ones_like(state.data)
                elif self.clamp_value not in [0.0, 1.0]:
                    clamped_values = torch.ones_like(state.data) * \
                                     self.clamp_value
                    opp_clamped_values = torch.zeros_like(state.data)
                else:
                    raise ValueError("Invalid value for self.clamp_value")

                state.data = torch.where(clamp_array,
                                         clamped_values,
                                         opp_clamped_values)
            # elif i > state_layer_idx:
            #     state.data = torch.zeros_like(state.data)

class WeightVisualizationManager(Manager):
    def __init__(self, args, model, data, buffer, writer, device,
                 sample_log_dir):
        super().__init__(args, model, data, buffer, writer, device,
                 sample_log_dir)
        self.params = [p for p in self.model.parameters()]
        self.quad_nets = self.model.quadratic_nets

        # Get the forward network(s)
        forward_net  = self.model.quadratic_nets[1]
        backward_net = self.model.quadratic_nets[0]
        self.f_net_conv = None
        self.f_net_fc = None
        self.b_net_conv = None
        self.b_net_fc = None
        if hasattr(forward_net, 'densecctblock'):
            self.f_net_conv = forward_net.densecctblock
        if hasattr(backward_net, 'densecctblock'):
            self.b_net_conv = backward_net.densecctblock
        if hasattr(forward_net, 'fc_net'):
            self.f_net_fc = forward_net.fc_net
        if hasattr(backward_net, 'fc_net'):
            self.b_net_fc = backward_net.fc_net

        self.base_save_dir = 'exps/weight_visualizations'
        self.save_dir = self.base_save_dir + '/' + self.model.model_name

        if not os.path.isdir(self.base_save_dir):
            os.mkdir(self.base_save_dir)
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)


    def visualize_base_weights(self):
        lbls = ['densecctblock']
        nets = [self.f_net_conv]
        nets = [(lbl, net) for lbl, net in zip(lbls, nets) if net is not None]
        for lbl, net in nets:
            print("Visualizing %s" % lbl)
            self.visualize_cctblock(net, nrow=8, label=lbl)

    def standardize(self, tensor):
        std_t = torch.std(tensor)
        stdzd_t = (tensor / max([1e-14, std_t])) + 0.5
        return stdzd_t


    def visualize_cctblock(self, block, nrow, label):

        # Get the conv and/or transposed conv weight tensor
        if block.base_cctls[0][0].only_conv:
            conv = block.base_cctls[0][0].conv
            top = block.top_net
            weight = conv.weight
            bias = conv.bias

            stdzd_weights = self.standardize(weight)

            bias = bias.unsqueeze(1).unsqueeze(1).unsqueeze(1).transpose(0, 1)
            bias = [bias] * (
                torch.prod(torch.tensor(weight.shape[1:])))
            bias = torch.cat(bias, dim=0)
            bias = bias.view(weight.shape)
            bias = bias.to('cpu')
            stdzd_biases = self.standardize(bias)

            w_b = weight + bias
            stdzd_weights_and_biases = self.standardize(w_b)
        elif block.base_cctls[0][0].only_conv_t:
            # Untested
            conv_t = block.base_cctls[0][0].conv_t
            weight = conv_t.weight
            bias = conv_t.bias

            stdzd_weights = self.standardize(weight)

            bias = bias.unsqueeze(1).unsqueeze(1).unsqueeze(1).transpose(0, 1)
            bias = [bias] * (
                torch.prod(torch.tensor(weight.shape[1:])))
            bias = torch.cat(bias, dim=1)
            bias = bias.view(weight.shape)
            stdzd_biases = self.standardize(bias)

            w_b = weight + bias
            stdzd_weights_and_biases = self.standardize(w_b)
        else:
            conv = block.base_cctls[0][0].conv
            conv_t = block.base_cctls[0][0].conv_T

            # Combine their weights into one tensor
            weight = torch.cat([conv.weight.transpose(0, 1),
                                      conv_t.weight], dim=1)
            bias = torch.cat([conv.bias, conv_t.bias], dim=0)

            # Transform their weight block by the 1x1 conv that follows them
            # in the dense cct block
            transf_weight = block.top_net(weight)
            stdzd_weights = self.standardize(transf_weight)

            # Turn the vector of biases into a tensor with the same shape as the
            # weights
            bias = bias.unsqueeze(1).unsqueeze(1).unsqueeze(1).transpose(0, 1)
            bias = [bias] * (torch.prod(torch.tensor(weight.transpose(0,1).shape[1:])))
            bias = torch.cat(bias, dim=0)
            bias = bias.view(weight.shape)

            # Pass bias through the same 1x1 conv as the weights
            transf_bias = block.top_net(bias)
            stdzd_biases = self.standardize(transf_bias)

            # Combine weights and biases and pass through 1x1 conv
            w_b = weight + bias
            transf_w_b = block.top_net(w_b)
            stdzd_weights_and_biases = self.standardize(transf_w_b)

        utils.save_image(stdzd_weights,
                         self.save_dir + '/weights_%s.png' % label,
                         nrow=nrow, normalize=True, range=(0, 1))
        utils.save_image(stdzd_biases,
                         self.save_dir + '/biases_%s.png' % label,
                         nrow=nrow, normalize=True, range=(0, 1))
        utils.save_image(stdzd_weights_and_biases,
                         self.save_dir + '/weights_and_biases_%s.png' % label,
                         nrow=nrow, normalize=True, range=(0, 1))


    def visualize_weight_pretrained(self, name='densenet121'):
        save_dir = self.base_save_dir + '/' + name
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        imported = torch.hub.load('pytorch/vision:v0.5.0', name,
                                  pretrained=True)
        imported_w = [x for x in imported.parameters()][0]
        std = torch.std(imported_w)
        std_weights = (imported_w / std) + 0.5
        utils.save_image(std_weights, save_dir + '/weights.png', nrow=8,
                         normalize=True, range=(0, 1))








































































class ExperimentsManager(Manager):
    def __init__(self, args, model, data, buffer, writer, device,
                 sample_log_dir):
        super().__init__(args, model, data, buffer, writer, device,
                 sample_log_dir)

        # self.latest_pos_enrg = None
        # self.latest_neg_enrg = None
        self.num_it_neg_mean = self.args.num_it_neg
        self.num_it_neg = self.args.num_it_neg
        self.num_sl  = len(self.args.state_sizes)

        # Define the names of the save directories
        self.momenta_strs = ['momenta_%i' % i for i in range(self.num_sl)]
        self.states_strs  = ['state_%i' % i for i in range(self.num_sl)]
        self.grad_strs    = ['grad_%i' % i for i in range(self.num_sl)]
        self.bno_strs     = ['bno_%i' % i for i in range(self.num_sl)]
        self.energy_strs  = ['energy_%i' % i for i in range(self.num_sl)]
        self.img_str = 'images'
        # self.save_vars = ['momenta', 'all_states', 'binary_net_outputs',
        #                   'energies']
        self.base_save_dir = '/media/lee/DATA/DDocs/AI_neuro_work/DAN/exp_data'
        self.save_dir_model = self.base_save_dir + '/' + self.model.model_name + '_loaded' + self.loaded_model_name
        self.save_dir_exp = None
        self.data_save_dirs = self.momenta_strs
        self.data_save_dirs.extend(self.states_strs)
        self.data_save_dirs.extend(self.grad_strs)
        self.data_save_dirs.extend(self.bno_strs)
        self.data_save_dirs.extend(self.energy_strs)
        self.data_save_dirs.extend([self.img_str])

        # Make the base directories if they do not already exist
        if not os.path.isdir(self.base_save_dir):
            os.mkdir(self.base_save_dir)
        if not os.path.isdir(self.save_dir_model):
            os.mkdir(self.save_dir_model)

        self.global_step = 0

        # Get base image, which is just the image with the avg pixel values of
        # CIFAR10
        base_image_path = 'mean_cifar10_trainingset_pixels.png'
        base_image = Image.open(base_image_path)
        self.base_image = \
            torchvision.transforms.functional.to_tensor(base_image)
        base_im_batch = [self.base_image] * 128
        self.base_im_batch = torch.stack(base_im_batch)


    def observe_cifar_pos_phase(self):
        self.save_dir_exp = self.save_dir_model + '/' + 'observeCIFAR10' + '/'
        self.save_img = True
        if not os.path.isdir(self.save_dir_exp):
            os.mkdir(self.save_dir_exp)
        for data_dir in self.data_save_dirs:
            full_save_dir = self.save_dir_model + '/' + 'observeCIFAR10' + '/' + data_dir
            if not os.path.isdir(full_save_dir):
                os.mkdir(full_save_dir)

        from lib.data import Dataset
        # Use test set instead of train set and use fixed data batch
        self.data = Dataset(self.args, train_set=False, shuffle=False)
        pos_img, pos_id = next(iter(self.data.loader)) #Gets first batch only
        image_phase_list = [pos_img]


        # Determine how long each stim will be displayed for
        self.phase_lens = [7000]

        phase_idxs = []
        for phase, phase_len in enumerate(self.phase_lens):
            phase_idxs.extend([phase] * phase_len)

        print("Experiment records dynamics when presenting a batch of CIFAR10 images")
        self.observation_phase(image_phase_list=image_phase_list,
                               phase_idxs=phase_idxs)

    def orientations_present_single_gabor(self):
        exp_stim_path_base = "data/gabor_filters/single/"

        # To measure oscillations, use experiment with varied contrast & angle
        #exp_stim_stem = "contrast_and_angle"

        # To measure orientation preferences, use exp that varies only angle
        exp_stim_stem = "just_angle"

        exp_stim_path = exp_stim_path_base + exp_stim_stem

        self.save_dir_exp = self.save_dir_model + '/' + \
                            'orientations_present_single_gabor_' + \
                            exp_stim_stem + '/'
        self.save_img = True
        if not os.path.isdir(self.save_dir_exp):
            os.mkdir(self.save_dir_exp)
        for data_dir in self.data_save_dirs:
            full_save_dir = self.save_dir_exp + data_dir
            if not os.path.isdir(full_save_dir):
                os.mkdir(full_save_dir)


        from lib.data import Dataset
        # Because the sampler burns in depending on the statistics of the input
        # we use images from the test set to get the images for the burn in
        # phase. Use test set instead of train set and use fixed data batch.
        # Shuffle is set to false so that every experiment uses the same burn-
        # in stimuli.
        self.data = Dataset(self.args, train_set=False, shuffle=False)
        pos_img, pos_id = next(
            iter(self.data.loader))  # Gets first batch only

        # Images for first 50 timesteps are from data distribution
        image_phase_list = [pos_img]

        # The next image is a blank image
        image_phase_list.append(self.base_im_batch)

        # Get the experimental images

        (_, _, filenames) = next(os.walk(exp_stim_path))
        filenames = sorted(filenames)
        exp_stims = []
        for flnm in filenames:
            im = Image.open(os.path.join(exp_stim_path, flnm))
            im = torchvision.transforms.functional.to_tensor(im)
            exp_stims.append(im)
        exp_stims = torch.stack(exp_stims) # should have generated only 128 images

        # Then a gabor filter stimulus is displayed
        image_phase_list.append(exp_stims)

        # Then a blank image is displayed
        image_phase_list.append(self.base_im_batch)

        # Determine how long each stim will be displayed for
        #self.experiment_len = 200
        self.phase_lens = [50, 1000, 2500, 3000]

        phase_idxs = []
        for phase, phase_len in enumerate(self.phase_lens):
            phase_idxs.extend([phase] * phase_len)

        print("Experiment records dynamics when presenting a batch of "+
              "images of single gabor filters")
        self.observation_phase(image_phase_list=image_phase_list,
                               phase_idxs=phase_idxs)

    def initialize_states(self, pos_img=None, pos_id=None,
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

    def observation_phase(self, image_phase_list,
                          phase_idxs, prev_states=None):

        print('\nStarting observation phase...')
        # Get the loaded pos samples and put them on the correct device
        image_phase_list = [img.to(self.device) for img in image_phase_list]
        lib.utils.requires_grad(image_phase_list, True)

        # Gets the values of the pos states by running an inference phase
        # with the image state_layer clamped
        obs_states_init = self.initialize_states(pos_img=image_phase_list[0],
                                                 prev_states=prev_states)
        obs_states = [osi.clone().detach() for osi in obs_states_init]

        # Freeze network parameters and take grads w.r.t only the inputs
        lib.utils.requires_grad(obs_states, True)
        lib.utils.requires_grad(self.parameters, False)
        if self.args.initializer == 'ff_init':
            lib.utils.requires_grad(self.initter.parameters(), False)
        self.model.eval()

        # Get an optimizer for each statelayer
        self.state_optimizers = lib.utils.get_state_optimizers(self.args,
                                                               obs_states)

        # Observation phase sampling
        for pos_it in tqdm(range(sum(self.phase_lens))):
            obs_states[0] = image_phase_list[phase_idxs[pos_it]]
            obs_states = self.sampler_step(obs_states,
                                           ids=None,
                                           positive_phase=True,
                                           pos_it=pos_it,
                                           step=self.global_step)
            self.global_step += 1

        # Stop calculting grads w.r.t. images
        for obs_state in obs_states:
            obs_state.detach_()

    def sampler_step(self, states, ids, positive_phase=False, pos_it=None,
                     step=None):

        # Get total energy and energy outputs for indvdual neurons
        energy, outs, full_enrgs = self.model(states, ids, step)
        print("Energy: " + str(energy))
        # Calculate the gradient wrt states for the Langevin step (before
        # addition of noise)
        energy.backward()
        torch.nn.utils.clip_grad_norm_(states,
                                       self.args.clip_state_grad_norm,
                                       norm_type=2)

        # Adding noise in the Langevin step (only for non conditional
        # layers in positive phase)
        for layer_idx, (noise, state) in enumerate(zip(self.noises, states)):
            if positive_phase and layer_idx == 0:
                pass
            else:
                noise.normal_(0, self.args.sigma)
                # Note: Just set sigma to a very small value if you don't want to
                # add noise. It's so inconsequential that it's not worth the
                # if-statements to accommodate sigma==0.0
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
                mean_layer_string = 'layers/mean_bnry_%s' % i
                self.writer.add_scalar(mean_layer_string, enrg.mean(), step)
                if self.args.log_histograms  and \
                        step % self.args.histogram_logging_interval == 0:
                    hist_layer_string = 'layers/hist_bnrys_%s' % i
                    #print("Logging energy histograms")
                    self.writer.add_histogram(hist_layer_string, enrg, step)

        ## Pos or Neg total energies
        if positive_phase and step % self.args.scalar_logging_interval == 0:
            print('\nEnergy: ' + str(energy.cpu().detach().numpy()))
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
                if self.args.log_histograms and \
                        step % self.args.histogram_logging_interval == 0:
                    hist_layer_string = 'layers/hist_states_%s' % i
                    #print("Logging energy histograms")
                    self.writer.add_histogram(hist_layer_string, state, step)
        # End of tensorboard logging

        # Save latest energy outputs so you can schedule the phase lengths
        # if positive_phase:
        #     self.latest_pos_enrg = energy.item()
        # else:
        #     self.latest_neg_enrg = energy.item()

        # Prepare gradients and sample for next sampling step
        for i, state in enumerate(states):
            state.grad.detach_()
            state.grad.zero_()
            if self.args.states_activation == 'relu' and i>0:
                state.data.clamp_(0)
            else:
                state.data.clamp_(0, 1)

        # Save experimental data
        self.save_experimental_data(states, outs, full_enrgs, step,
                                    save_img=True)

        return states

    def save_experimental_data(self, states, outs, energies, step, save_img):
        # layerwise go through states etc. and save arrays
        start = 1
        end = 2
        for j, (state, out, enrg, opt) in enumerate(zip(states[start:end], outs[start:end],
                                                        energies[start:end],
                                                    self.state_optimizers[start:end]),
                                                    start=start): # LEE only saving the bottom two
            if opt.state_dict()['state']: # i.e. if momentum exists isn't empty, because it's only instantiated after 1 steps I think
                key = list(opt.state_dict()['state'].keys())[0]
                state_mom = opt.state_dict()['state'][key]['momentum']
            else:
                state_mom = torch.zeros_like(state)

            if state.grad is not None:
                grad = state.grad
            else:
                grad = torch.zeros_like(state)


            # Convert data from tensors to numpy arrays for saving
            grad      = grad.clone().detach().cpu().numpy()
            state_mom = state_mom.clone().detach().cpu().numpy()
            state     = state.clone().detach().cpu().numpy()
            bno       = out.clone().detach().cpu().numpy()
            enrg      = enrg.clone().detach().cpu().numpy()

            step_str = '%.6i' % step

            grad_str  = self.save_dir_exp + '/' + self.grad_strs[j] + '/' + step_str + '.npy'
            state_str = self.save_dir_exp + '/' + self.states_strs[j] + '/' + step_str + '.npy'
            bno_str   = self.save_dir_exp + '/' + self.bno_strs[j] + '/' + step_str + '.npy'
            enrg_str  = self.save_dir_exp + '/' + self.energy_strs[j] + '/' + step_str + '.npy'
            mom_str   = self.save_dir_exp + '/' + self.momenta_strs[j] + '/' + step_str + '.npy'

            np.save(grad_str,  grad)
            np.save(state_str, state)
            np.save(bno_str,   bno)
            np.save(enrg_str,  enrg)
            np.save(mom_str,   state_mom)

        if save_img:
            self.save_image(states, step)

    def save_image(self, states, step):
        img = states[0]
        shape = img.shape
        img_save = img.reshape(shape).detach().to('cpu')
        utils.save_image(img_save,
                         self.save_dir_exp + '/images/' + str(step) + '.png',
                         nrow=16, normalize=True, range=(0, 1))

    def obs_negative_phase(self):
        #TODO if necess
        print('\nStarting observational negative phase...')
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
        for _ in tqdm(range(self.num_it_neg)):
            self.sampler_step(neg_states, neg_id, step=self.global_step)
            self.global_step += 1

        # Stop calculting grads w.r.t. images
        for neg_state in neg_states:
            neg_state.detach_()

        # Send negative samples to the negative buffer
        # self.buffer.push(neg_states, neg_id)

        return neg_states, neg_id



class ExperimentalStimuliGenerationManager:
    # TODO function that creates a single long bar image
    def __init__(self,
                 base_image_path='mean_cifar10_trainingset_pixels.png',
                 save_path_root='data/gabor_filters/'):
        self.nstds = 3  # Number of standard deviation sigma of bounding box
        # Get base image

        if not os.path.isdir(save_path_root):
            os.mkdir(save_path_root)

        if not os.path.exists(base_image_path):
            self.get_dataset_avg_pixels(base_image_path=base_image_path)

        base_image = Image.open(base_image_path)
        self.base_image = \
            torchvision.transforms.functional.to_tensor(base_image)
        self.save_path_root = save_path_root
        self.center = np.array([int(32 / 2), int(32 / 2)])

    def get_dataset_avg_pixels(self, base_image_path):
        """Create base image that is the mean pixel value for all images in
        the training set."""
        dataset = datasets.CIFAR10('./data',
                                   download=True,
                                   transform=transforms.ToTensor())
        loader = DataLoader(dataset,
                            batch_size=1024,
                            shuffle=True,
                            drop_last=False,
                            )
        sum_pos_state = None
        counter = 0
        for pos_state, pos_id in loader:
            if sum_pos_state is None:
                sum_pos_state = torch.sum(pos_state, dim=0)
            else:
                sum_pos_state += torch.sum(pos_state, dim=0)
            counter += pos_state.shape[0]

        mean_pos_state = sum_pos_state / counter

        utils.save_image(mean_pos_state,
                         base_image_path,
                         nrow=1, normalize=True, range=(0, 1))

    def gabor(self, sigma, theta, Lambda, psi, gamma, contrast):
        """Gabor feature extraction."""
        sigma_x = sigma
        sigma_y = float(sigma) / gamma

        # Bounding box
        xmax = max(abs(self.nstds * sigma_x * np.cos(theta)),
                   abs(self.nstds * sigma_y * np.sin(theta)))
        xmax = np.ceil(max(1, xmax))
        ymax = max(abs(self.nstds * sigma_x * np.sin(theta)),
                   abs(self.nstds * sigma_y * np.cos(theta)))
        ymax = np.ceil(max(1, ymax))
        xmin = -xmax
        ymin = -ymax
        (y, x) = np.meshgrid(np.arange(ymin, ymax + 1),
                             np.arange(xmin, xmax + 1))

        # Rotation
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)

        gb = np.exp(-.5 * (
                    x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(
            2 * np.pi / Lambda * x_theta + psi)

        gb = gb * contrast
        gb = gb + 1.
        return gb

    def single_gabor_image(self,
                             loc=[0, 0],
                             sigma=0.7,
                             angle=0,
                             wavelength=3.,
                             spat_asp_ratio=0.4,
                             ellipicity=1,
                             contrast=1,
                             folder_name='FolderNameUnfilled',
                             save_image=True):

        loc = np.array(loc) + self.center
        for i in loc:
            if i > 32:
                raise ValueError(
                    'Location cannot be outside the bounds of image')

        # Get gabor filter
        filtr = self.gabor(sigma=sigma, theta=angle, gamma=spat_asp_ratio,
                             Lambda=wavelength, psi=ellipicity,
                           contrast=contrast)


        # Multiply base image by filter
        fshape = filtr.shape
        hf_h, hf_w = int(fshape[0] / 2), int(fshape[1] / 2)
        new_image = self.base_image.clone()

        patch = new_image[:, (loc[0] - hf_h):(loc[0] - hf_h + fshape[0]),
                (loc[1] - hf_w):(loc[1] - hf_w + fshape[1])]
        print(patch.shape, filtr.shape)
        new_image[:, (loc[0] - hf_h):(loc[0] - hf_h + fshape[0]),
        (loc[1] - hf_w):(loc[1] - hf_w + fshape[1])] = \
            np.multiply(patch, filtr)

        # Save image
        if save_image:
            loc_string = ''
            for l in loc:
                loc_string += str(l) + '-'
            loc_string = loc_string[:-1]
            save_string = 'gf_loc%s_sgm%s_th%s_w%s_rat%s_el%s_c%s' % (loc_string,
                                                          '%.5g' % sigma,
                                                          '%.5f' % angle,
                                                          '%.5g' % wavelength,
                                                          '%.5g' % spat_asp_ratio,
                                                          '%.5g' % ellipicity,
                                                          '%.5f' % contrast)
            save_string = os.path.join(self.save_path_root,
                                       folder_name, save_string)
            save_string += '.png'
            print(save_string)
            utils.save_image(new_image,
                             save_string,
                             nrow=1, normalize=True, range=(0, 1))
        return new_image

    def generate_single_gabor_dataset__contrast_and_angle(self):
        print("Careful, running this function might overwrite your "+
              "previously generated data. Cancel if you wish, or enter "+
              "any input")


        contrast_min = 0.0
        contrast_max = 2.80001
        contrast_incr = 0.4
        angle_min = 0.0
        angle_max = np.pi * 2
        angle_incr = (np.pi * 2) / 16

        # Make the folders to save the images in
        folder_name1 = 'single'
        folder_name2 = 'contrast_and_angle'
        full_folder_name = os.path.join(self.save_path_root,
                                        folder_name1,
                                        folder_name2)

        if not os.path.exists(full_folder_name):
            os.mkdir(os.path.join(self.save_path_root,
                                  folder_name1))
            os.mkdir(os.path.join(self.save_path_root,
                                  folder_name1,
                                  folder_name2))

        for c in np.arange(start=contrast_min, stop=contrast_max,
                       step=contrast_incr):
            for a in np.arange(start=angle_min, stop=angle_max, step=angle_incr):
                im = self.single_gabor_image(angle=a,
                                             contrast=c,
                                             folder_name=os.path.join(folder_name1,
                                                                      folder_name2))

    def generate_single_gabor_dataset__just_angle(self):
        print("Careful, running this function might overwrite your "+
              "previously generated data. Cancel if you wish, or enter "+
              "any input")


        contrast = 2.4
        angle_min = 0.0
        angle_max = np.pi * 2
        angle_incr = (np.pi * 2) / 128

        # Make the folders to save the images in
        folder_name1 = 'single'
        folder_name2 = 'just_angle'
        full_folder_name = os.path.join(self.save_path_root,
                                        folder_name1,
                                        folder_name2)

        if not os.path.exists(full_folder_name):
            if not os.path.exists(os.path.join(self.save_path_root,
                                  folder_name1)):
                os.mkdir(os.path.join(self.save_path_root,
                                      folder_name1))
            os.mkdir(os.path.join(self.save_path_root,
                                  folder_name1,
                                  folder_name2))

        for a in np.arange(start=angle_min, stop=angle_max, step=angle_incr):
            im = self.single_gabor_image(angle=a,
                                         contrast=contrast,
                                         folder_name=os.path.join(folder_name1,
                                                                 folder_name2))

    def generate_double_gabor_dataset__loc_and_angles(self):
        folder_name1 = 'double'
        folder_name2 = 'loc_and_angles'

        full_folder_name = os.path.join(self.save_path_root,
                                        folder_name1,
                                        folder_name2)

        if not os.path.exists(full_folder_name):
            os.mkdir(os.path.join(self.save_path_root,
                                  folder_name1))
            os.mkdir(os.path.join(self.save_path_root,
                                  folder_name1,
                                  folder_name2))

        angle_min = 0.0
        angle_max = np.pi * 2
        angle_incr = (np.pi * 2) / 10
        angle_range = np.arange(start=angle_min, stop=angle_max,
                                step=angle_incr)

        loc_min = -8
        loc_max = 9
        loc_incr = 2
        locs_range = list(range(loc_min, loc_max, loc_incr))
        locs = [[i,j] for i in locs_range for j in locs_range]

        single_base_im = self.single_gabor_image(save_image=False)
        centred_base_im = single_base_im - self.base_image
        for angle in angle_range:
            for loc in locs:
                # Create second gabor filter
                new_im = self.single_gabor_image(angle=angle,
                                                 loc=loc,
                                                 save_image=False)

                # Combine first and second images into one
                centred_new_im = new_im - self.base_image
                centred_combo = centred_new_im + centred_base_im
                new_im = centred_combo + self.base_image

                # Create name of image based on attributes
                loc_string = ''
                for l in loc:
                    loc_string += str(l) + '-'
                loc_string = loc_string[:-1]
                save_string = 'gf_loc%s_th%s' % (loc_string, '%.5f' % angle)
                save_string = os.path.join(self.save_path_root,
                                           folder_name1,
                                           folder_name2,
                                           save_string)
                save_string += '.png'
                print(save_string)

                # Save image
                utils.save_image(new_im,
                                 save_string,
                                 nrow=1, normalize=True, range=(0, 1))




shapes = lambda x : [y.shape for y in x]