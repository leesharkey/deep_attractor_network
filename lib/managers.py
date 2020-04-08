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
                                                   gamma=0.999)
        self.noises = lib.utils.generate_random_states(self.args.state_sizes,
                                                       self.device)
        self.global_step = 0
        self.batch_num = 0
        self.epoch = 0

        # Load initializer network (initter)
        if args.initializer == 'ff_init':
            self.initter = initializer.InitializerNetwork(args, writer, device)
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

            self.num_it_neg_mean = self.args.num_it_neg
            self.num_it_neg = self.args.num_it_neg

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
        self.max_history_len = 5000
        self.mean_neg_pos_margin = 5000
        self.neg_it_schedule_cooldown = 0
        self.cooldown_len = 10 #epochs
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

                prev_states = pos_states  # In case pos init uses prev states

                if self.batch_num % self.args.img_logging_interval == 0:
                    self.log_images(pos_img, pos_states, neg_states)

                self.save_energies_to_histories()
                self.log_mean_energy_histories()

                # Save network(s) and settings
                if self.batch_num % self.args.model_save_interval == 0:
                    self.save_net_and_settings()

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

    def sampler_step(self, states, ids, positive_phase=False, pos_it=None,
                     step=None):

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
                mean_layer_string = 'layers/mean_bnry_%s' % i
                self.writer.add_scalar(mean_layer_string, enrg.mean(), step)
                if self.args.log_histograms  and \
                        step % self.args.histogram_logging_interval == 0:
                    hist_layer_string = 'layers/hist_bnrys_%s' % i
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
            if self.args.truncate_pos_its:
                self.pos_iterations_trunc_update(pos_it)
        else:
            self.latest_neg_enrg = energy.item()


        # Prepare gradients and sample for next sampling step
        for state in states:
            state.grad.detach_()
            state.grad.zero_()
            state.data.clamp_(0, 1) #TODO when args.states_activation is relu: state.data.clamp_(0)

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
        if diff < 25 and current_iteration > 15:
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


class VisualizationManager(Manager):
    """A new class because I need to redefine sampler step to clamp certain
    neurons, and need to define a new type of sampling phase."""
    def __init__(self, args, model, data, buffer, writer, device, sample_log_dir):
        super(Manager, self).__init__()

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
            batch_sizes = None
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

                utils.save_image(states[0].detach().to('cpu'),
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
            #     s.grad.zero_()

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


class WeightVisualizationManager(Manager):
    def __init__(self, args, model, data, buffer, writer, device,
                 sample_log_dir):
        super().__init__(args, model, data, buffer, writer, device,
                 sample_log_dir)
        self.params = [p for p in self.model.parameters()]
        self.quad_nets = self.model.quadratic_nets
        self.forward_net  = self.model.quadratic_nets[0]
        self.backward_net = self.model.quadratic_nets[1]
        self.base_save_dir = 'exps/weight_visualizations'
        self.save_dir = self.base_save_dir + '/' + self.model.model_name

        if not os.path.isdir(self.base_save_dir):
            os.mkdir(self.base_save_dir)
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)


    def visualize_base_weights(self):
        nets = [('forward', self.forward_net), ('backward', self.backward_net)]
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
            weight = conv.weight
            bias = conv.bias

            stdzd_weights = self.standardize(weight)

            bias = bias.unsqueeze(1).unsqueeze(1).unsqueeze(1).transpose(0, 1)
            bias = [bias] * (
                torch.prod(torch.tensor(weight.shape[1:])))
            bias = torch.cat(bias, dim=0)
            bias = bias.view(weight.shape)
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


class ExperimentalStimuliGenerationManager:
    def __init__(self,
                 base_image_path='mean_cifar10_trainingset_pixels.png',
                 save_path_root='data/gabor_filters/'):
        self.nstds = 3  # Number of standard deviation sigma of bounding box
        # Get base image
        base_image = Image.open(base_image_path)
        self.base_image = \
            torchvision.transforms.functional.to_tensor(base_image)
        self.save_path_root = save_path_root
        self.center = np.array([int(32 / 2), int(32 / 2)])

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
        contrast_max = 3.0
        contrast_incr = 0.3
        angle_min = 0.0
        angle_max = np.pi * 2
        angle_incr = (np.pi * 2) / 15

        for c in np.arange(start=contrast_min, stop=contrast_max,
                       step=contrast_incr):
            for a in np.arange(start=angle_min, stop=angle_max, step=angle_incr):
                im = self.single_gabor_image(angle=a,
                                             contrast=c,
                                             folder_name='single/contrast_and_angle')

    def generate_double_gabor_dataset__loc_and_angles(self):
        folder_name = 'double/loc_and_angles'

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
                new_im = self.single_gabor_image(angle=angle,
                                                 loc=loc,
                                                 save_image=False)
                centred_new_im = new_im - self.base_image
                centred_combo = centred_new_im + centred_base_im
                new_im = centred_combo + self.base_image

                loc_string = ''
                for l in loc:
                    loc_string += str(l) + '-'
                loc_string = loc_string[:-1]
                save_string = 'gf_loc%s_th%s' % (loc_string, '%.5f' % angle)
                save_string = os.path.join(self.save_path_root,
                                           folder_name, save_string)
                save_string += '.png'
                print(save_string)
                utils.save_image(new_im,
                                 save_string,
                                 nrow=1, normalize=True, range=(0, 1))

    @staticmethod
    def get_dataset_avg_pixels(self):
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
                         'mean_cifar10_trainingset_pixels.png',
                         nrow=1, normalize=True, range=(0, 1))