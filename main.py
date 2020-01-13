import argparse
import random
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from lib.networks import IGEBM
from lib.data import SampleBuffer, sample_buffer, sample_data, Dataset
import lib.utils



def train(model, data, writer, alpha=1, step_size=10, sample_step=60, device='cuda'):


    buffer = SampleBuffer()

    noise = torch.randn(128, 3, 32, 32, device=device)  # TODO change for samples

    parameters = model.parameters()
    optimizer = optim.Adam(parameters, lr=1e-4, betas=(0.0, 0.999))

    for i, (pos_img, pos_id) in data.loader:
        # Get the loaded pos samples and put them on the correct device
        pos_img, pos_id = pos_img.to(device), pos_id.to(device)

        # Initialize the chain (either as noise or from buffer)
        neg_img, neg_id = sample_buffer(buffer=buffer,
                                        batch_size=pos_img.shape[0],
                                        p=0.95)
        # Freeze network parameters and take grads w.r.t only the inputs
        neg_img.requires_grad = True
        requires_grad(parameters, False)
        model.eval()

        # Negative phase sampling
        for k in tqdm(range(sample_step)):
            if noise.shape[0] != neg_img.shape[0]:
                noise = torch.randn(neg_img.shape[0], 3, 32, 32, device=device)

            noise.normal_(0, 0.005)
            neg_img.data.add_(noise.data)  # Adding noise to the Langevin step

            neg_out = model(neg_img, neg_id)  # Outputs energy of neg sample
            print(neg_out.sum())
            neg_out.sum().backward()
            neg_img.grad.data.clamp_(-0.01, 0.01) #TODO change to clip by norm, not just clipping values like implemented. torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip) https://stackoverflow.com/questions/54716377/how-to-do-gradient-clipping-in-pytorch

            # The gradient step in the Langevin step
            neg_img.data.add_(-step_size, neg_img.grad.data)

            # Prepare gradients and sample for next sampling step
            neg_img.grad.detach_()
            neg_img.grad.zero_()
            neg_img.data.clamp_(0, 1)

        # Stop calculting grads w.r.t. images
        neg_img = neg_img.detach()

        # Put model in training mode and prepare network parameters for updates
        requires_grad(parameters, True)
        model.train()
        model.zero_grad()

        # Get energies of positive and negative samples
        pos_out = model(pos_img, pos_id)
        neg_out = model(neg_img, neg_id)

        # Calculate the loss and the gradients for the network params
        loss = alpha * (pos_out ** 2 + neg_out ** 2)  # L2 penalty on energy magnitudes
        loss = loss + (pos_out - neg_out)  # Contrastive loss + L2 penalty
        loss = loss.mean()
        loss.backward()

        clip_grad(parameters, optimizer)

        # Update the network params
        optimizer.step()

        # Send negative samples to the buffer (on cpu)
        buffer.push(neg_img, neg_id)

        data.loader.set_description(f'loss: {loss.item():.5f}')

        #if i % 1 == 0:
        if i % 100 == 0:
            # TODO When sampling latents, this will need to save
            #  only the image part
            utils.save_image(
                neg_img.detach().to('cpu'),
                f'samples/{str(i).zfill(5)}.png',
                nrow=16,
                normalize=True,
                range=(0, 1),
            )


def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag


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


def main():
    ### Parse CLI arguments.
    parser = argparse.ArgumentParser(description='MNIST classification with ' +
                                     'EP-trained Hopfield neural networks.')

    sgroup = parser.add_argument_group('Sampling options')
    sgroup.add_argument('--alphas', type=float, nargs='+', default=1e-3,
                        help='Individual learning rates for each layer. '+
                             ' Default: %(default)s. '+
                             'When randomizing, the following options define'+
                             'a range of indices and the random value assigned'+
                             'to the argument will be 10 to the power of the'+
                             'float selected from the range. ' +
                             'Options: [-1.25, -0.25]. ' +
                             'The next option defines the multiplier for the'+
                             ' alphas in subsequent layers: ' +
                             'Opt2: {0.05, 0.5}.')
    sgroup.add_argument('--epsilon', type=float, default=1e-2,
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
    tgroup.add_argument('--l2_reg_param_w', type=float, default=0.0,
                        help='Scaling parameter for the L2 regularisation ' +
                             'term placed on the weight values. Default: ' +
                             '%(default)s.'+
                             'When randomizing, the following options define' +
                             'a range of indices and the random value assigned' +
                             'to the argument will be 10 to the power of the' +
                             'float selected from the range. '+
                             'Options: [-6, -2].')
    tgroup.add_argument('--l2_reg_param_energy', type=float, default=1.0,
                        help='Scaling parameter for the L2 regularisation ' +
                             'term placed on the weight values. Default: ' +
                             '%(default)s.'+
                             'When randomizing, the following options define' +
                             'a range of indices and the random value assigned' +
                             'to the argument will be 10 to the power of the' +
                             'float selected from the range. '+
                             'Options: [-6, -2].')
    tgroup.add_argument('--state_gradient_clipping', action='store_true',
                        help='Clips state gradient updates. Default: ' +
                             '%(default)s.')
    parser.set_defaults(state_gradient_clipping=False)
    tgroup.add_argument('--state_gradient_clipping_val', type=float, default=2.,
                        help='The maximum norm value to clip ' +
                             'the state gradients at. Default: ' +
                             '%(default)s.')


    ngroup = parser.add_argument_group('Network and states options')
    ngroup.add_argument('--activation', type=str, default="hardsig",
                        help='The activation function. Options: ' +
                             '[hardsig, relu, swish, leaky_relu]'
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
    ngroup.add_argument('--sigma', type=float, default=0.0,
                        help='Sets the scale of the noise '
                             'in the network.'+
                             'When randomizing, the following options define' +
                             'a range of indices and the random value assigned' +
                             'to the argument will be 10 to the power of the' +
                             'float selected from the range. '+
                             'Options: [-3, 0].')
    ngroup.add_argument('--size_layers', type=int, nargs='+', default=[6, 4, 3],
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


    mgroup = parser.add_argument_group('Miscellaneous options')
    mgroup.add_argument('--use_cuda', action='store_true',
                        help='Flag to enable GPU usage.')
    mgroup.add_argument('--weights_and_biases_histograms', action='store_true',
                        help='Plots the weights and biases in tensorboard.')
    mgroup.add_argument('--randomize_args', type=str, nargs='+', default=[],
                        help='List of CLI args to pass to the random arg ' +
                             'generator. Default: %(default)s.',
                        required=False)

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

    args = parser.parse_args()
    # Generate random args, if any
    if args.randomize_args is not []:
        args = lib.utils.random_arg_generator(parser, args)

    # Determine the correct device
    vars(args)['use_cuda'] = args.use_cuda and torch.cuda.is_available()

    # Give a very short description of what is special about this run
    if args.require_special_name:
        vars(args)['special_name'] = input("Special name: ") or "None"

    # Print final values for args
    for k, v in zip(vars(args).keys(), vars(args).values()):
        print(str(k) + '\t' * 2 + str(v))

    # Set up the tensorboard summary writer
    model_name = lib.utils.datetimenow() + '__rndidx_' + str(np.random.randint(0,99999))
    print(model_name)
    writer = SummaryWriter(args.tensorboard_log_dir + '/' + model_name)

    # Set up model
    model = IGEBM(10).to('cuda') #TODO change this to the model I want

    # Set up dataset
    data = Dataset(args)

    # Train the model
    train(model, data, writer, sample_step=10)

if __name__ == '__main__':
    main()