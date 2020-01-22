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
        self.state_sizes = args.states_sizes
        self.parameters = model.parameters()
        self.optimizer = optim.Adam(self.parameters, lr=args.lr, betas=(0.0, 0.999))


        # initter_optimizer = optim.Adam(parameters, lr=1e-3, betas=(0.0, 0.999))

        self.noises = [torch.randn(self.state_sizes[i][0],
                                   self.state_sizes[i][1],
                                   self.state_sizes[i][2],
                                   self.state_sizes[i][3],
                                   device=self.device)
                       for i in range(len(self.state_sizes))]

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


    def pre_train_initializer(self):
        i = 0
        for batch, (pos_img, pos_id) in self.data.loader:
            print("Pretraining step")
            pos_states, pos_id = self.positive_phase(pos_img, pos_id)
            i += 1
            if i > self.args.num_pretraining_batches:
                break
        print("\nPretraining of initializer complete")


    def train(self):
        save_dict = self.make_save_dict()
        path = 'exps/models/' +self.model.model_name + '.pt'
        torch.save(save_dict, path)
        for batch, (pos_img, pos_id) in self.data.loader:

            pos_states, pos_id = self.positive_phase(pos_img, pos_id)

            neg_states, neg_id = self.negative_phase()

            self.param_update_phase(neg_states, neg_id, pos_states, pos_id)

            if self.batch_num % self.args.img_logging_interval == 0:
                utils.save_image(
                    neg_states[0].detach().to('cpu'),
                    os.path.join(self.sample_log_dir,
                                 str(self.batch_num).zfill(5) + '.png'),
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

    def positive_phase(self, pos_img, pos_id):

        print('\nStarting positive phase...')
        # Get the loaded pos samples and put them on the correct device
        pos_img, pos_id = pos_img.to(self.device), pos_id.to(self.device)

        # Gets the values of the pos images by initting with ff net and then
        # running a short inference phase
        pos_states = [pos_img]
        #requires_grad(pos_states, True)
        if self.args.initializer == 'ff_init':
            self.initter.train()
            pos_states_init = self.initter.forward(pos_img, pos_id)
            pos_states_new = [psi.clone().detach() for psi in pos_states_init]
            pos_states.extend(pos_states_new)
        elif self.args.initializer == 'zeros':
            pos_states.extend(
                [torch.zeros(size, device=self.device, requires_grad=True)
                 for size in self.state_sizes[1:]])
        else:
            pos_states.extend(
                [torch.randn(size, device=self.device, requires_grad=True)
                 for size in self.state_sizes[1:]])
        ################################################
        # Freeze network parameters and take grads w.r.t only the inputs

        requires_grad(pos_states, True)
        requires_grad(self.parameters, False)
        if self.args.initializer == 'ff_init':
            requires_grad(self.initter.parameters(), False)
        self.model.eval()

        if self.args.state_optimizer is not 'langevin':
            self.state_optimizer = get_state_optimizer(self.args, pos_states)

        # Positive phase sampling (will be made into initialisation func later)
        for k in tqdm(range(self.args.num_it_pos)):
            self.sampler_step(pos_states, pos_id, positive_phase=True)
            self.global_step += 1

        # Stop calculting grads w.r.t. images
        for pos_state in pos_states:
            pos_state.detach_()

        # Update initializer network if present
        if self.args.initializer == 'ff_init':
            requires_grad(self.initter.parameters(), True)
            loss = self.initter.update_weights(outs=pos_states_init,
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
        requires_grad(neg_states, True)
        requires_grad(self.parameters, False)
        self.model.eval()
        if self.args.state_optimizer is not 'langevin':
            self.state_optimizer = get_state_optimizer(self.args, neg_states)

        # Set up state optimizer if approp

        # Negative phase sampling
        for k in tqdm(range(self.args.num_it_neg)):
            self.sampler_step(neg_states, neg_id)
            self.global_step += 1

        # Stop calculting grads w.r.t. images
        for neg_state in neg_states:
            neg_state.detach_()

        # Send negative samples to the buffer (on cpu)
        self.buffer.push(neg_states, neg_id)
        return neg_states, neg_id

    def sampler_step(self, states, ids, positive_phase=False):

        for noise in self.noises:
            noise.normal_(0, self.args.sigma)

        # Adding noise in the Langevin step, but only to latent variables
        # in positive phase
        for i, (noise, state) in enumerate(zip(self.noises, states)):
            if positive_phase and i == 0:
                pass
            else:
                state.data.add_(noise.data)

        energies = self.model(states, ids)  # Outputs energy of neg sample
        total_energy = energies.sum()
        if positive_phase:
            print('\nPos Energy: ' + str(total_energy.cpu().detach().numpy()))
            self.writer.add_scalar('train/PosSamplesEnergy', total_energy,
                                   self.global_step)
        else:
            print('\nNeg Energy: ' + str(total_energy.cpu().detach().numpy()))
            self.writer.add_scalar('train/NegSamplesEnergy', total_energy,
                                   self.global_step)

        total_energy.backward()
        torch.nn.utils.clip_grad_norm_(states,
                                       self.args.clip_state_grad_norm,
                                       norm_type=2)

        for i, state in enumerate(states):
            # The gradient step in the Langevin step (only for upper layers)
            if positive_phase and i == 0:
                pass
            else:
                if self.args.state_optimizer is not 'langevin':
                    self.state_optimizer.step()
                else:
                    state.data.add_(-self.args.sampling_step_size,
                                state.grad.data)

            # Prepare gradients and sample for next sampling step
            state.grad.detach_()
            state.grad.zero_()
            state.data.clamp_(0, 1)

    def calc_energ_and_loss(self, neg_states, neg_id, pos_states, pos_id):
        # Get energies of positive and negative samples
        pos_energy = self.model(pos_states, pos_id)
        neg_energy = self.model(neg_states, neg_id)

        # Calculate the loss and the gradients for the network params
        loss_l2 = self.args.l2_reg_energy_param * (
                pos_energy ** 2 + neg_energy ** 2)  # L2 penalty on energy magnitudes
        loss_ml = pos_energy - neg_energy  # Maximum likelihood loss
        loss = loss_ml + loss_l2
        loss = loss.mean()
        loss.backward()
        self.writer.add_scalar('train/loss', loss.item(), self.global_step)
        return neg_energy, pos_energy, loss

    def update_weights(self, loss):
        clip_grad(self.parameters, self.optimizer)

        # Update the network params
        self.optimizer.step()
        for energy_weight in self.model.energy_weights.parameters():
            energy_weight.data.clamp_(self.args.energy_weight_min)

        self.data.loader.set_description(f'loss: {loss.item():.5f}')

    def param_update_phase(self, neg_states, neg_id, pos_states, pos_id):

        # Put model in training mode and prepare network parameters for updates
        requires_grad(self.parameters, True)
        self.model.train()  # Not to be confused with self.TrainingManager.train
        self.model.zero_grad()

        neg_energy, pos_energy, loss = \
            self.calc_energ_and_loss(neg_states, neg_id,
                                     pos_states, pos_id)

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

def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag

def get_state_optimizer(args, params):
    if args.state_optimizer == 'langevin':
        return None
    if args.state_optimizer == 'sgd':
        return optim.SGD(params, args.sampling_step_size)
    if args.state_optimizer == 'sgd_momentum':
        return optim.SGD(params, args.sampling_step_size,
                         momentum=args.momentum_param,
                         dampening=args.dampening_param)
    if args.state_optimizer == 'nesterov':
        return optim.SGD(params, args.sampling_step_size,
                         momentum=args.momentum_param, nesterov=True)
    if args.state_optimizer == 'adam':
        return optim.Adam(params, args.sampling_step_size, betas=(0.9,0.999))


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
        vars(args)['states_sizes'] = [[args.batch_size, 3, 32, 32],
                                     [args.batch_size, 9, 16, 16]]  #,#[args.batch_size, 18, 8, 8]]

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
    ngroup.add_argument('--states_sizes', type=list, nargs='+', default=[[]],
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
                        ' Default: %(default)s.')
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
    mgroup.add_argument('--histogram_logging_interval', type=int, default=100,
                        help='The size of the intervals between the logging ' +
                             'of histogram data.') #On Euler do around 1000
    mgroup.add_argument('--scalar_logging_interval', type=int, default=1,
                        help='The size of the intervals between the logging ' +
                             'of scalar data.') #On Euler do around 100
    mgroup.add_argument('--img_logging_interval', type=int, default=100,
                        help='The size of the intervals between the logging ' +
                             'of image samples.')
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


    # Train the model
    tm = TrainingManager(args, model, data, buffer, writer, device, sample_log_dir)
    if args.initializer == 'ff_init' and args.pretrain_initializer:
        tm.pre_train_initializer()
    if not args.no_train_model:
        tm.train()

shapes = lambda x : [y.shape for y in x]
nancheck = lambda x : (x != x).any()
listout = lambda y : [x for x in y]
gradcheck = lambda  y : [x.requires_grad for x in y]
leafcheck = lambda  y : [x.is_leaf for x in y]
existgradcheck = lambda  y : [(x.grad is not None) for x in y]
existgraddatacheck = lambda  y : [(x.grad.data is not None) for x in y]


if __name__ == '__main__':
    main()