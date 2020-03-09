"""Based on https://github.com/rosinality/igebm-pytorch/blob/master/model.py
which is available under an MIT OSI licence."""

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn import utils
import lib.utils
import lib.custom_swish_activation as cust_actv
from lib import activations #TODO clean up scripts later

#SGHMC
#https://pysgmcmc.readthedocs.io/en/pytorch/_modules/pysgmcmc/optimizers/sghmc.html


def get_swish(): # TODO put in utils
    sigmoid = torch.nn.Sigmoid()
    return lambda x : x * sigmoid(x)


def get_leaky_hard_sigmoid(): # TODO put in utils
    hsig = torch.nn.Hardtanh(min_val=0.0)
    return lambda x : hsig(x) + 0.01*x


def conv_output_shape(h_w, kernel_size=1, stride=1, padding=0, dilation=1):
    """
    Thanks to DuaneNielsen
    https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/5
    """
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor( ((h_w[0] + (2 * padding) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    w = floor( ((h_w[1] + (2 * padding) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    return h, w


class SpectralNorm:
    def __init__(self, name, bound=False):
        self.name = name
        self.bound = bound

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        size = weight.size()
        weight_mat = weight.contiguous().view(size[0], -1)

        with torch.no_grad():
            v = weight_mat.t() @ u
            v = v / v.norm()
            u = weight_mat @ v
            u = u / u.norm()

        sigma = u @ weight_mat @ v

        if self.bound:
            weight_sn = weight / (sigma + 1e-6) * torch.clamp(sigma, max=1)

        else:
            weight_sn = weight / sigma

        return weight_sn, u

    @staticmethod
    def apply(module, name, bound):
        fn = SpectralNorm(name, bound)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', weight)
        input_size = weight.size(0)
        u = weight.new_empty(input_size).normal_()
        module.register_buffer(name, weight)
        module.register_buffer(name + '_u', u)

        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight_sn, u = self.compute_weight(module)
        setattr(module, self.name, weight_sn)
        setattr(module, self.name + '_u', u)


def spectral_norm(module, init=True, std=1, bound=False):
    if init:
        nn.init.normal_(module.weight, 0, std)

    if hasattr(module, 'bias') and module.bias is not None:
        module.bias.data.zero_()

    SpectralNorm.apply(module, 'weight', bound=bound)

    return module


class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, n_class=None, downsample=False):
        super().__init__()

        self.conv1 = spectral_norm(
            nn.Conv2d(
                in_channel,
                out_channel,
                3,
                padding=1,
                bias=False if n_class is not None else True,
            )
        )

        self.conv2 = spectral_norm(
            nn.Conv2d(
                out_channel,
                out_channel,
                3,
                padding=1,
                bias=False if n_class is not None else True,
            ), std=1e-10, bound=True
        )

        self.class_embed = None

        if n_class is not None:
            class_embed = nn.Embedding(n_class, out_channel * 2 * 2)
            class_embed.weight.data[:, : out_channel * 2] = 1
            class_embed.weight.data[:, out_channel * 2 :] = 0

            self.class_embed = class_embed

        self.skip = None

        if in_channel != out_channel or downsample:
            self.skip = nn.Sequential(
                spectral_norm(nn.Conv2d(in_channel, out_channel, 1, bias=False))
            )

        self.downsample = downsample

    def forward(self, input, class_id=None):
        out = input

        out = self.conv1(out)

        if self.class_embed is not None:
            embed = self.class_embed(class_id).view(input.shape[0], -1, 1, 1)
            weight1, weight2, bias1, bias2 = embed.chunk(4, 1)
            out = weight1 * out + bias1

        out = F.leaky_relu(out, negative_slope=0.2)
        out = self.conv2(out)
        if self.class_embed is not None:
            out = weight2 * out + bias2

        if self.skip is not None:
            skip = self.skip(input)
        else:
            skip = input

        out = out + skip

        if self.downsample:
            out = F.avg_pool2d(out, 2)

        out = F.leaky_relu(out, negative_slope=0.2)
        return out

class ConvFCMixturetoTwoDim(nn.Module):
    """
    Takes a mixture of four dimensional inputs and two dimensional inputs
    and combines them and outputs a two dimensional output, the size of the
    current layer
    """
    def __init__(self, args, layer_idx):
        super().__init__()
        self.args = args
        self.pad_mode = 'zeros'
        self.layer_idx = layer_idx
        self.act = lib.utils.get_activation_function(args)
        self.base_fc_out_size = 256 # TODO this seems like a hack

        # Gets the indices of the state_layers that will be input to this net.
        self.input_idxs = self.args.arch_dict['mod_connect_dict'][layer_idx]

        # Size values of state_layer of this network
        self.state_layer_size = self.args.state_sizes[layer_idx][1]

        # The summed number of conv channels of all the input state_layers
        self.in_conv_sizes = [self.args.state_sizes[j]
                                     for j in self.input_idxs
                                     if len(self.args.state_sizes[j]) == 4]
        self.num_in_conv_channels = sum([size[1] for size in self.in_conv_sizes])

        # Defines interpolater that reshapes all conv inputs to same size H x W
        self.mean_4dim_size = int(torch.mean(torch.tensor(
            [size[2] for size in self.in_conv_sizes]).float()))
        self.mean_4dim_size = [self.mean_4dim_size, self.mean_4dim_size] #TODO inclined to change this to max
        self.interp = Interpolate(size=self.mean_4dim_size, mode='bilinear')

        # Define base convs (includes avg pooling to downsample)
        base_conv = nn.Conv2d(
            in_channels=self.num_in_conv_channels,
            out_channels=self.args.arch_dict['num_ch'],
            kernel_size=self.args.arch_dict['kernel_sizes'][layer_idx][0],
            padding=self.args.arch_dict['padding'][layer_idx][0],
            stride=self.args.arch_dict['strides'][0],
            padding_mode=self.pad_mode,
            bias=True)
        if not self.args.no_spec_norm_reg:
            base_conv = spectral_norm(base_conv)
        self.base_conv = nn.Sequential(base_conv,
                                       nn.AvgPool2d(kernel_size=2,
                                                    count_include_pad=True))

        # Get size of base conv outputs
        self.base_conv_outshapes = []
        outshape0, outshape1 = \
            conv_output_shape(self.mean_4dim_size,
                             kernel_size= self.args.arch_dict['kernel_sizes'][layer_idx][0],
                             padding=self.args.arch_dict['padding'][layer_idx][0],
                             stride=self.args.arch_dict['strides'][0])
        outshape = (outshape0, outshape1)
        self.base_conv_outshapes.append([conv_output_shape(outshape,
                                                     kernel_size=2,
                                                     padding=0,
                                                     stride=2)]) #for avg pool2d
        self.base_conv_outsizes = [torch.prod(torch.tensor(bcos)) * \
                                   self.args.arch_dict['num_ch']
                              for bcos in self.base_conv_outshapes] # TODO this scales poorly with num channels. Consider adding another conv layer with smaller output when network is working

        # Get num of fc neuron
        self.in_fc_sizes = [self.args.state_sizes[j][1]
                            for j in self.input_idxs
                            if len(self.args.state_sizes[j]) == 2]
        self.num_fc_inps = len(self.in_fc_sizes)
        self.in_fc_neurons = sum(self.in_fc_sizes)

        # Define base FCs
        self.base_fc_layers = nn.ModuleList([])
        for in_size in self.in_fc_sizes:
            fc_layer = nn.Linear(in_size, self.base_fc_out_size)
            if not self.args.no_spec_norm_reg:
                fc_layer = spectral_norm(fc_layer, bound=True)
            self.base_fc_layers.append(fc_layer) #TODO consider changing this to sequential and adding actv instead of doing actvs in forward (do once you've got network working)

        # Define energy FC (take flattened convs and base FCs as input)
        self.energy_inp_size = (self.base_fc_out_size * self.num_fc_inps) + \
            sum(self.base_conv_outsizes)
        energy_layer = nn.Linear(self.energy_inp_size,
                                 self.state_layer_size)
        if not self.args.no_spec_norm_reg:
            energy_layer = spectral_norm(energy_layer, bound=True)
        self.energy_layer = energy_layer

    def forward(self, pre_states, inputs, class_id=None):
        # Get 4-d inputs and pass through base conv layers, then flatten output
        inps_4d = [inp for inp in inputs if len(inp.shape) == 4]
        reshaped_4dim_inps = [self.interp(inp) for inp in inps_4d]
        reshaped_4dim_inps = torch.cat(reshaped_4dim_inps, dim=1)
        base_conv_out = self.act(self.base_conv(reshaped_4dim_inps))
        resized_base_conv_out = base_conv_out.view(base_conv_out.shape[0], -1)

        # Get 2-d inputs and pass through base FC layers
        inps_2d = [inp for inp in inputs if len(inp.shape) == 2]
        fc_outs = [self.act(self.base_fc_layers[i](inp))
                   for (i, inp) in enumerate(inps_2d)]
        fc_outs_cat = torch.cat(fc_outs, dim=1)

        # Combine outputs of base fc & base conv layers and get energy output
        energy_input = torch.cat([resized_base_conv_out, fc_outs_cat], dim=1)
        out = self.act(self.energy_layer(energy_input))
        quadr_out = 0.5 * torch.einsum('ba,ba->b',
                                       out.view(out.shape[0], -1),
                                       pre_states.view(pre_states.shape[0],-1))
        return quadr_out, out

class ConvFCMixturetoFourDim(nn.Module):
    """
    Takes a mixture of four dimensional and two dimensional inputs and
    outputs a four dimensional output of the same shape as the current state.

    """
    def __init__(self, args, layer_idx):
        super().__init__()
        self.args = args
        self.pad_mode = 'zeros'
        self.layer_idx = layer_idx
        self.act = lib.utils.get_activation_function(args)
        self.num_fc_channels = self.args.arch_dict['num_fc_channels']

        # Gets the indices of the state_layers that will be input to this net.
        self.input_idxs = self.args.arch_dict['mod_connect_dict'][layer_idx]

        # Size values of state_layer of this network
        self.state_layer_ch = self.args.state_sizes[layer_idx][1]
        self.state_layer_h_w = self.args.state_sizes[layer_idx][2:]

        self.interp = Interpolate(size=self.state_layer_h_w, mode='bilinear')

        # The summed number of channels of all the input state_layers
        self.in_conv_channels = []
        self.in_conv_channels = sum([self.args.state_sizes[j][1]
                                     for j in self.input_idxs
                                     if len(self.args.state_sizes[j]) == 4])

        self.in_fc_sizes   = [self.args.state_sizes[j][1]
                                  for j in self.input_idxs
                                  if len(self.args.state_sizes[j]) == 2]
        self.in_fc_neurons = sum(self.in_fc_sizes)
        fc_channel_fracs   = \
            [int(round(self.num_fc_channels*(sz/sum(self.in_fc_sizes))))
             for sz in self.in_fc_sizes]

        # Define base convs (no max pooling)
        self.base_conv = spectral_norm(nn.Conv2d(
            in_channels=self.in_conv_channels,
            out_channels=self.args.arch_dict['num_ch'],
            kernel_size=self.args.arch_dict['kernel_sizes'][layer_idx][0],
            padding=self.args.arch_dict['padding'][layer_idx][0],
            stride=self.args.arch_dict['strides'][0],
            padding_mode=self.pad_mode,
            bias=True))

        # Define base FCs (then reshape their output to something that fits a conv)

        self.base_actv_fc_layers = nn.ModuleList([])
        for (in_size, out_size) in zip(self.in_fc_sizes, fc_channel_fracs):
            base_fc_layer = nn.Linear(in_size, out_size)
            if not self.args.no_spec_norm_reg:
                base_fc_layer = spectral_norm(base_fc_layer, bound=True)
            self.base_actv_fc_layers.append(nn.Sequential(
                                        base_fc_layer,
                                        self.act,
                                        Reshape(-1, out_size, 1, 1), # makes each neuron a 'channel'
                                        self.interp) # Spreads each 1D channel across the whole sheet of latent neurons
            )

        # Define energy convs (take output of base convs and FCs as input
        energy_conv = nn.Conv2d(
            in_channels=self.args.arch_dict['num_ch'] +
                        self.in_conv_channels +
                        self.num_fc_channels,
            out_channels=self.state_layer_ch,
            kernel_size=self.args.arch_dict['kernel_sizes'][layer_idx][1],
            padding=self.args.arch_dict['padding'][layer_idx][1],
            padding_mode=self.pad_mode,
            bias=True)
        if not self.args.no_spec_norm_reg:
            energy_conv = spectral_norm(energy_conv, std=1e-10, bound=True)
        self.energy_conv = energy_conv

    def forward(self, pre_states, inputs, class_id=None):
        inps_4d = [inp for inp in inputs if len(inp.shape) == 4]
        inps_2d = [inp for inp in inputs if len(inp.shape) == 2]
        reshaped_4dim_inps = [self.interp(inp) for inp in inps_4d
                             if inp.shape[2:] != self.state_layer_h_w]
        reshaped_inps = torch.cat(reshaped_4dim_inps, dim=1)
        base_conv_out = self.base_conv(reshaped_inps)

        fc_outs = [self.base_actv_fc_layers[i](inp)
                   for (i, inp) in enumerate(inps_2d)]
        fc_outs = torch.cat(fc_outs, dim=1)
        energy_input = torch.cat([reshaped_inps, base_conv_out, fc_outs],dim=1)
        out = self.act(self.energy_conv(energy_input)) #TODO put act in sequential above
        quadr_out = 0.5 * torch.einsum('ba,ba->b',
                                       out.view(
                                           int(out.shape[0]), -1),
                                       pre_states.view(
                                           int(pre_states.shape[0]), -1))
        return quadr_out, out


class DenseConv(nn.Module):
    def __init__(self, args, layer_idx):
        super().__init__()
        self.args = args
        self.pad_mode = 'zeros'
        self.layer_idx = layer_idx
        self.act = lib.utils.get_activation_function(args)

        # Gets the indices of the state_layers that will be input to this net.
        self.input_idxs = self.args.arch_dict['mod_connect_dict'][layer_idx]

        # The summed number of channels of all the input state_layers
        self.in_channels = sum([self.args.state_sizes[j][1]
                                for j in self.input_idxs])

        # Size values of state_layer of this network
        self.state_layer_ch = self.args.state_sizes[layer_idx][1]
        self.state_layer_h_w = self.args.state_sizes[layer_idx][2:]
        print("Input channels in %i: %i" % (layer_idx, self.in_channels))
        print("Input h and w  in %i: %r" % (layer_idx, self.state_layer_h_w))

        # Network
        self.interp = Interpolate(size=self.state_layer_h_w, mode='bilinear')
        base_conv = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.args.arch_dict['num_ch'],
                kernel_size=self.args.arch_dict['kernel_sizes'][layer_idx][0],
                padding=self.args.arch_dict['padding'][layer_idx][0],
                stride=self.args.arch_dict['strides'][0],
                padding_mode=self.pad_mode,
                bias=True)
        energy_conv = nn.Conv2d(
            in_channels=self.args.arch_dict['num_ch'] + self.in_channels,
            out_channels=self.state_layer_ch,
            kernel_size=self.args.arch_dict['kernel_sizes'][layer_idx][1],
            padding=self.args.arch_dict['padding'][layer_idx][1],
            padding_mode=self.pad_mode,
            bias=True)

        if not self.args.no_spec_norm_reg:
            base_conv   = spectral_norm(base_conv)
            energy_conv = spectral_norm(energy_conv, std=1e-10, bound=True)

        self.base_conv = base_conv
        self.energy_conv = energy_conv

    def forward(self, pre_states, inputs, class_id=None):
        reshaped_inps = [self.interp(inp) for inp in inputs
                         if inp.shape[2:] != self.state_layer_h_w]
        reshaped_inps = torch.cat(reshaped_inps, dim=1)
        base_out = self.act(self.base_conv(reshaped_inps))
        energy_input = torch.cat([reshaped_inps, base_out], dim=1)
        out = self.act(self.energy_conv(energy_input))
        quadr_out = 0.5 * torch.einsum('ba,ba->b', out.view(int(out.shape[0]), -1),
                                     pre_states.view(int(pre_states.shape[0]), -1))
        return quadr_out, out


class FCFC(nn.Module):
    def __init__(self, args, layer_idx):
        super().__init__()
        self.args = args
        self.act = lib.utils.get_activation_function(args)
        self.layer_idx = layer_idx

        # Get the indices of the state_layers that will be input to this net
        # and their sizes. They may be flattened conv outputs.
        self.input_idxs = self.args.arch_dict['mod_connect_dict'][layer_idx]
        self.in_sizes = [torch.prod(torch.tensor(self.args.state_sizes[j][1:]))
                         for j in self.input_idxs]

        self.out_size = self.args.state_sizes[layer_idx][1]
        self.base_fc_layers = nn.ModuleList([
            spectral_norm(nn.Linear(in_size, self.out_size), bound=True)
            for in_size in self.in_sizes])

        fc1 = nn.Linear(self.out_size * len(self.in_sizes), self.out_size)
        fc2 = nn.Linear(self.out_size, self.out_size)
        if not self.args.no_spec_norm_reg:
            fc1 = spectral_norm(fc1, bound=True)
            fc2 = spectral_norm(fc2, bound=True)
        self.energy_actv_fc_layer = nn.Sequential(
            fc1,
            self.act,
            fc2,
            self.act)

    def forward(self, pre_states, actv_post_states, class_id=None): #I think I might have misnamed the pre and post states.
        inputs = actv_post_states
        reshaped_inps = [inp.view(inp.shape[0], -1) for inp in inputs]
        base_outs = [self.act(base_fc(inp))
                     for base_fc, inp in zip(self.base_fc_layers,
                                             reshaped_inps)]
        energy_input = torch.cat(base_outs, dim=1)
        out = self.energy_actv_fc_layer(energy_input)
        quadr_out = 0.5 * torch.einsum('ba,ba->b', out,
                                     pre_states.view(pre_states.shape[0], -1))
        return quadr_out, out


class LinearLayer(nn.Module):
    def __init__(self, args, layer_idx):
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx

        # Get the indices of the state_layers that will be input to this net
        # and their sizes. They may be flattened conv outputs.
        self.input_idxs = self.args.arch_dict['mod_connect_dict'][layer_idx]
        self.in_sizes = [torch.prod(torch.tensor(self.args.state_sizes[j][1:]))
                         for j in self.input_idxs]

        self.out_size = int(torch.prod(torch.tensor(
            self.args.state_sizes[layer_idx][1:]))) # Size of current statelayer

        layers = [nn.Linear(in_size, self.out_size, bias=False)
                  for in_size in self.in_sizes]
        if not self.args.no_spec_norm_reg:
            layers = [spectral_norm(layer, bound=True) for layer in layers]
        self.layers = nn.ModuleList(layers)

    def forward(self, pre_states, actv_post_states, class_id=None):
        reshaped_inps = [inp.view(inp.shape[0], -1) for inp in actv_post_states]
        out_list = [lin_layer(inp) for lin_layer, inp in zip(self.layers,
                                                         reshaped_inps)]
        quadr_out = sum([0.5 * torch.einsum('ba,ba->b', out,
                               pre_states.view(pre_states.shape[0], -1))
                    for out in out_list])

        if len(out_list) > 1:
            out = torch.stack(out_list, dim=1)
            out = torch.sum(out, dim=1)
        else:
            out = out_list[0]
        return quadr_out, out

# Also need classes for
# mix of fc and conv to 4dim and
# mix to 2dim.

class LinearConvFCMixturetoTwoDim(nn.Module):
    """
    Takes a mixture of four dimensional inputs and two dimensional inputs
    and combines them and outputs a two dimensional output, the size of the
    current layer
    """
    def __init__(self, args, layer_idx):
        super().__init__()
        self.args = args
        self.pad_mode = 'zeros'
        self.layer_idx = layer_idx
        self.out_ch = 1
        #self.act = lib.utils.get_activation_function(args)
        #self.base_fc_out_size = 256 # TODO this seems like a hack

        # Gets the indices of the state_layers that will be input to this net.
        self.input_idxs = self.args.arch_dict['mod_connect_dict'][layer_idx]

        # Size values of state_layer of this network
        self.state_layer_size = self.args.state_sizes[layer_idx][1]

        # The summed number of conv channels of all the input state_layers
        self.in_conv_sizes = [self.args.state_sizes[j]
                                     for j in self.input_idxs
                                     if len(self.args.state_sizes[j]) == 4]
        self.num_in_conv_channels = sum([size[1] for size in self.in_conv_sizes])

        # Defines interpolater that reshapes all conv inputs to same size H x W
        self.max_4dim_size = int(torch.max(torch.tensor(
            [size[2] for size in self.in_conv_sizes]).float()))
        self.max_4dim_size = [self.max_4dim_size, self.max_4dim_size]
        self.interp = Interpolate(size=self.max_4dim_size, mode='bilinear')

        # Define base convs (includes avg pooling to downsample)
        base_conv = nn.Conv2d(
            in_channels=self.num_in_conv_channels,
            out_channels=self.out_ch, #TODO consider making a hyperparam
            kernel_size=self.args.arch_dict['kernel_sizes'][layer_idx][0],
            padding=self.args.arch_dict['padding'][layer_idx][0],
            stride=self.args.arch_dict['strides'][0],
            padding_mode=self.pad_mode,
            bias=True)
        if not self.args.no_spec_norm_reg:
            base_conv = spectral_norm(base_conv)
        self.base_conv = base_conv
        # self.base_conv = nn.Sequential(base_conv,
        #                                nn.AvgPool2d(kernel_size=2, #TODO consider making this a hyperparam
        #                                             count_include_pad=True))

        # Get size of base conv outputs
        self.base_conv_outshapes = []
        outshape0, outshape1 = \
            conv_output_shape(self.max_4dim_size,
                             kernel_size=self.args.arch_dict['kernel_sizes'][layer_idx][0],
                             padding=self.args.arch_dict['padding'][layer_idx][0],
                             stride=self.args.arch_dict['strides'][0])
        outshape = (outshape0, outshape1)

        # [conv_output_shape(outshape,
        #                    kernel_size=2,
        #                    padding=0,
        #                    stride=2)]

        self.base_conv_outshapes.append(outshape) #for avg pool2d
        self.base_conv_outsizes = [torch.prod(torch.tensor(bcos))# * \
                                   #self.args.arch_dict['num_ch']
                                   for bcos in self.base_conv_outshapes] # TODO this scales poorly with num channels. Consider adding another conv layer with smaller output when network is working
        self.linker_net = nn.Linear(self.base_conv_outsizes[0],
                                    self.state_layer_size) #todo probs don't need self.base_conv_outsizes to be a list

        # Get num of fc neuron
        self.in_fc_sizes = [self.args.state_sizes[j][1]
                            for j in self.input_idxs
                            if len(self.args.state_sizes[j]) == 2]
        self.num_fc_inps = len(self.in_fc_sizes)
        self.in_fc_neurons = sum(self.in_fc_sizes)

        # Define base FCs
        self.base_fc_layers = nn.ModuleList([])
        for in_size in self.in_fc_sizes:
            fc_layer = nn.Linear(in_size, self.state_layer_size)
            if not self.args.no_spec_norm_reg:
                fc_layer = spectral_norm(fc_layer, bound=True)
            self.base_fc_layers.append(fc_layer) #TODO consider changing this to sequential and adding actv instead of doing actvs in forward (do once you've got network working)

        # # Define energy FC (take flattened convs and base FCs as input)
        # self.energy_inp_size = (self.base_fc_out_size * self.num_fc_inps) + \
        #     sum(self.base_conv_outsizes)
        # energy_layer = nn.Linear(self.energy_inp_size,
        #                          self.state_layer_size)
        # if not self.args.no_spec_norm_reg:
        #     energy_layer = spectral_norm(energy_layer, bound=True)
        # self.energy_layer = energy_layer

    def forward(self, pre_states, inputs, class_id=None):
        # Get 4-d inputs and pass through base conv layers, then flatten output
        inps_4d = [inp for inp in inputs if len(inp.shape) == 4]
        reshaped_4dim_inps = [self.interp(inp) for inp in inps_4d]
        reshaped_4dim_inps = torch.cat(reshaped_4dim_inps, dim=1)
        base_conv_out = self.base_conv(reshaped_4dim_inps)
        resized_base_conv_out = base_conv_out.view(base_conv_out.shape[0], -1)
        base_conv_out = self.linker_net(resized_base_conv_out)

        # Get 2-d inputs and pass through base FC layers
        inps_2d = [inp for inp in inputs if len(inp.shape) == 2]
        if inps_2d:
            fc_outs = [self.base_fc_layers[i](inp)
                       for (i, inp) in enumerate(inps_2d)]
            fc_outs_cat = torch.cat(fc_outs, dim=1)
            out = fc_outs_cat + base_conv_out
        else:
            out = base_conv_out
        # out = torch.sum(all_outs_cat, dim=1)

        # Combine outputs of base fc & base conv layers and get energy output
        # energy_input = torch.cat([resized_base_conv_out, fc_outs_cat], dim=1)
        # out = self.energy_layer(energy_input)
        quadr_out = 0.5 * torch.einsum('ba,ba->b',
                                       out.view(out.shape[0], -1),
                                       pre_states.view(pre_states.shape[0],-1))
        return quadr_out, out

class LinearConvFCMixturetoFourDim(nn.Module):
    """
    Takes a mixture of four dimensional and two dimensional inputs and
    outputs a four dimensional output of the same shape as the current state.

    """
    def __init__(self, args, layer_idx):
        super().__init__()
        self.args = args
        self.pad_mode = 'zeros'
        self.layer_idx = layer_idx
        # self.act = lib.utils.get_activation_function(args)
        # self.num_fc_channels = self.args.arch_dict['num_fc_channels']

        # Gets the indices of the state_layers that will be input to this net.
        self.input_idxs = self.args.arch_dict['mod_connect_dict'][layer_idx]

        # Size values of state_layer of this network
        self.state_layer_ch = self.args.state_sizes[layer_idx][1]
        self.state_layer_h_w = self.args.state_sizes[layer_idx][2:]
        self.total_nrns = self.state_layer_ch * \
                          torch.prod(torch.tensor(self.state_layer_h_w))

        self.interp = Interpolate(size=self.state_layer_h_w, mode='bilinear')

        # The summed number of channels of all the input state_layers
        self.in_conv_channels = []
        self.in_conv_channels = sum([self.args.state_sizes[j][1]
                                     for j in self.input_idxs
                                     if len(self.args.state_sizes[j]) == 4])

        self.in_fc_sizes   = [self.args.state_sizes[j][1]
                                  for j in self.input_idxs
                                  if len(self.args.state_sizes[j]) == 2]
        self.in_fc_neurons = sum(self.in_fc_sizes)
        # fc_channel_fracs   = \
        #     [int(round(self.num_fc_channels*(sz/sum(self.in_fc_sizes))))
        #      for sz in self.in_fc_sizes]

        # Define base convs (no max pooling)
        if self.in_conv_channels > 0:
            self.base_conv = spectral_norm(nn.Conv2d(
                in_channels=self.in_conv_channels,
                out_channels=self.state_layer_ch,
                kernel_size=self.args.arch_dict['kernel_sizes'][layer_idx][0],
                padding=self.args.arch_dict['padding'][layer_idx][0],
                stride=self.args.arch_dict['strides'][0],
                padding_mode=self.pad_mode,
                bias=True))

        # Define base FCs (then reshape their output to something that fits a conv)

        self.base_actv_fc_layers = nn.ModuleList([])
        for in_size in self.in_fc_sizes:
            base_fc_layer = nn.Linear(in_size, self.total_nrns.item())
            if not self.args.no_spec_norm_reg:
                base_fc_layer = spectral_norm(base_fc_layer, bound=True)
            self.base_actv_fc_layers.append(nn.Sequential(
                                        base_fc_layer,
                                        Reshape(-1,
                                                self.state_layer_ch,
                                                self.state_layer_h_w[0],
                                                self.state_layer_h_w[1])) # Spreads each 1D channel across the whole sheet of latent neurons
            )

        # # Define energy convs (take output of base convs and FCs as input
        # energy_conv = nn.Conv2d(
        #     in_channels=self.args.arch_dict['num_ch'] +
        #                 self.in_conv_channels +
        #                 self.num_fc_channels,
        #     out_channels=self.state_layer_ch,
        #     kernel_size=self.args.arch_dict['kernel_sizes'][layer_idx][1],
        #     padding=self.args.arch_dict['padding'][layer_idx][1],
        #     padding_mode=self.pad_mode,
        #     bias=True)
        # if not self.args.no_spec_norm_reg:
        #     energy_conv = spectral_norm(energy_conv, std=1e-10, bound=True)
        # self.energy_conv = energy_conv

    def forward(self, pre_states, inputs, class_id=None):
        inps_4d = [inp for inp in inputs if len(inp.shape) == 4]
        inps_2d = [inp for inp in inputs if len(inp.shape) == 2]
        fc_outs = [self.base_actv_fc_layers[i](inp)
                   for (i, inp) in enumerate(inps_2d)]
        fc_outs = torch.cat(fc_outs, dim=1)

        if inps_4d:
            reshaped_4dim_inps = [self.interp(inp) for inp in inps_4d
                                  if inp.shape[2:] != self.state_layer_h_w]
            reshaped_inps = torch.cat(reshaped_4dim_inps, dim=1)
            base_conv_out = self.base_conv(reshaped_inps)
            out = base_conv_out + fc_outs
        else:
            # all_out = torch.stack([base_conv_out, fc_outs])
            # out = torch.sum(all_out, dim=0)
            out = fc_outs

        quadr_out = 0.5 * torch.einsum('ba,ba->b',
                                       out.view(
                                           int(out.shape[0]), -1),
                                       pre_states.view(
                                           int(pre_states.shape[0]), -1))
        return quadr_out, out


class LinearConv(nn.Module):
    def __init__(self, args, layer_idx):
        super().__init__()
        self.args = args
        self.pad_mode = 'zeros'
        self.layer_idx = layer_idx
        # self.act = lib.utils.get_activation_function(args)

        # Gets the indices of the state_layers that will be input to this net.
        self.input_idxs = self.args.arch_dict['mod_connect_dict'][layer_idx]

        # The summed number of channels of all the input state_layers
        self.in_channels = sum([self.args.state_sizes[j][1]
                                for j in self.input_idxs])

        # Size values of state_layer of this network
        self.state_layer_ch = self.args.state_sizes[layer_idx][1]
        self.state_layer_h_w = self.args.state_sizes[layer_idx][2:]
        print("Input channels in %i: %i" % (layer_idx, self.in_channels))
        print("Input h and w  in %i: %r" % (layer_idx, self.state_layer_h_w))

        # Network
        self.interp = Interpolate(size=self.state_layer_h_w, mode='bilinear')
        base_conv = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.state_layer_ch,
                kernel_size=self.args.arch_dict['kernel_sizes'][layer_idx][0],
                padding=self.args.arch_dict['padding'][layer_idx][0],
                stride=self.args.arch_dict['strides'][0],
                padding_mode=self.pad_mode,
                bias=True)

        if not self.args.no_spec_norm_reg:
            base_conv = spectral_norm(base_conv)

        self.base_conv = base_conv

    def forward(self, pre_states, inputs, class_id=None):
        reshaped_inps = [self.interp(inp) for inp in inputs
                         if inp.shape[2:] != self.state_layer_h_w]
        reshaped_inps = torch.cat(reshaped_inps, dim=1)
        out = self.base_conv(reshaped_inps)
        # energy_input = torch.cat([reshaped_inps, base_out], dim=1)
        # out = self.act(self.energy_conv(energy_input))
        quadr_out = 0.5 * torch.einsum('ba,ba->b',
                                     out.view(int(out.shape[0]), -1),
                                     pre_states.view(int(pre_states.shape[0]),
                                                     -1))
        return quadr_out, out

##############################################################################
##############################################################################


class VectorFieldNetwork(nn.Module):
    """Defines the vector field studied by Scellier et al. (2018)

    @misc{scellier2018generalization,
        title={Generalization of Equilibrium Propagation to Vector Field Dynamics},
        author={Benjamin Scellier and Anirudh Goyal and Jonathan Binas and Thomas Mesnard and Yoshua Bengio},
        year={2018},
        eprint={1808.04873},
        archivePrefix={arXiv},
        primaryClass={cs.LG}
    }
    https://arxiv.org/abs/1808.04873

    The network is a relaxation of the continuous Hopfield-like network (CHN)
    studied by Bengio and Fischer (2015) and later in
    Equilibrium Propagation (Scellier et al. 2017) and other works. It
    no longer required symmetric weights as in the CHN.
    """
    def __init__(self, args, device, model_name, writer, n_class=None):
        super().__init__()
        self.args = args
        self.device = device
        self.writer = writer
        self.model_name = model_name
        self.num_state_layers = len(self.args.state_sizes[1:])

        self.quadratic_nets = nn.ModuleList([
            LinearLayer(args, i) for i in range(len(self.args.state_sizes))
        ])

        # self.weights = nn.ModuleList(
        #     [nn.Linear(torch.prod(torch.tensor(l1[1:])),
        #                torch.prod(torch.tensor(l2[1:])), bias=False)
        #      for l1, l2 in zip(args.state_sizes[:-1], args.state_sizes[1:])])
        self.biases = nn.ModuleList([nn.Linear(torch.prod(torch.tensor(l[1:])),
                                               1, bias=False)
                       for l in args.state_sizes])
        for bias in self.biases:
            torch.nn.init.zeros_(bias.weight)

        if self.args.states_activation == "hardsig":
            self.actvn = activations.get_hard_sigmoid()
        elif self.args.states_activation == "relu":
            self.actvn = activations.get_relu()
        elif self.args.states_activation == "swish":
            self.actvn = activations.get_swish()

    def forward(self, states, class_id=None, step=None):

        # Squared norm
        sq_nrm = sum([(0.5 * (layer.view(layer.shape[0], -1) ** 2)).sum() for layer in states])

        # Linear terms
        linear_terms = - sum([bias(self.actvn(layer.view(layer.shape[0], -1))).sum()
                              for layer, bias in
                              zip(states, self.biases)])

        # Quadratic terms
        quadr_outs = []
        outs = []
        for i, (pre_state, net) in enumerate(zip(states, self.quadratic_nets)):
            post_inp_idxs = self.args.arch_dict['mod_connect_dict'][i]
            pos_inp_states = [states[j] for j in post_inp_idxs]
            quadr_out, out = net(pre_state, pos_inp_states)
            quadr_out = self.args.energy_weight_mask[i] * quadr_out #TODO temporary, just to see if this messes it up with larger archi layers
            quadr_outs.append(quadr_out)
            outs.append(out)


        # if self.args.log_histograms and step is not None and \
        #     step % self.args.histogram_logging_interval == 0:
        #     for i, enrg in enumerate(outs):
        #         layer_string = 'train/energies_%s' % i
        #         self.writer.add_histogram(layer_string, enrg, step)
        # reshaped_outs = [out.view(out.shape[0], -1) for out in outs]
        # reshaped_outs = [out * mask.view(out.shape[0], -1)
        #         for out, mask in zip(reshaped_outs, self.energy_weight_masks)]
        quadratic_terms = - sum(sum(quadr_outs))

        energy = sq_nrm + linear_terms + quadratic_terms

        return energy, outs


class DeepAttractorNetwork(nn.Module):
    """Defines the Deep Attractor Network (Sharkey 2019)

    The network is a generalisation of the vector field network used in
    Scellier et al. (2018) relaxation of the continuous Hopfield-like network
    (CHN) studied by Bengio and Fischer (2015) and later in
    Equilibrium Propagation (Scellier et al. 2017) and other works. It
    no longer required symmetric weights as in the CHN.
    """
    def __init__(self, args, device, model_name, writer, n_class=None):
        super().__init__()
        self.args = args
        self.device = device
        self.writer = writer
        self.model_name = model_name
        self.num_state_layers = len(self.args.state_sizes[1:])

        # Define the networks that output the quadratic terms
        self.quadratic_nets = nn.ModuleList([])
        for i, size in enumerate(self.args.state_sizes):
            inp_idxs = self.args.arch_dict['mod_connect_dict'][i]
            state_sizes = [self.args.state_sizes[j] for j in inp_idxs]
            state_sizes.append(size)
            all_same_dim = all([len(state_sizes[0])==len(sz) for sz in state_sizes])
            if len(size) == 4:
                if all_same_dim:
                    net = DenseConv(self.args, i)
                else:
                    net = ConvFCMixturetoFourDim(self.args, i)
            elif len(size) == 2:
                if all_same_dim:
                    net = FCFC(self.args, i)
                else:
                    net = ConvFCMixturetoTwoDim(self.args, i)
            self.quadratic_nets.append(net)

        # Define the biases that determine the linear term
        self.biases = nn.ModuleList(
            [nn.Linear(torch.prod(torch.tensor(l[1:])), 1, bias=False)
             for l in args.state_sizes])
        for bias in self.biases:
            torch.nn.init.zeros_(bias.weight)

        if self.args.states_activation == "hardsig":
            self.state_actv = activations.get_hard_sigmoid()
        elif self.args.states_activation == "relu":
            self.state_actv = activations.get_relu()
        elif self.args.states_activation == "swish":
            self.state_actv = activations.get_swish()

    def forward(self, states, class_id=None, step=None):

        # Squared norm
        # sq_nrm = sum(
        #     [(0.5 * (layer.view(layer.shape[0], -1) ** 2)).sum() for layer in
        #      states])
        sq_terms = []
        for i, layer in enumerate(states):
            sq_term = 0.5 * (layer.view(layer.shape[0], -1) ** 2).sum()
            #sq_term = self.args.energy_weight_mask[i] * sq_term
            sq_terms.append(sq_term)
        sq_nrm = sum(sq_terms)

        # Linear terms
        lin_terms = []
        for i, (layer, bias) in enumerate(zip(states, self.biases)):
            lin_term = bias(self.state_actv(layer.view(layer.shape[0], -1)))
            lin_term = lin_term.sum()
            #lin_term = self.args.energy_weight_mask[i] * lin_term
            lin_terms.append(lin_term)
        lin_terms = - sum(lin_terms)

        # linear_terms = - sum([bias(self.state_actv(layer.view(layer.shape[0], -1))).sum()
        #                       for layer, bias in
        #                       zip(states, self.biases)])

        # Quadratic terms
        quadr_outs = []
        outs = []
        for i, (pre_state, net) in enumerate(zip(states, self.quadratic_nets)):
            post_inp_idxs = self.args.arch_dict['mod_connect_dict'][i]
            pos_inp_states = [states[j] for j in post_inp_idxs]
            pos_inp_states = [self.state_actv(state)
                              for state in pos_inp_states]
            quadr_out, out = net(pre_state, pos_inp_states)
            quadr_out = self.args.energy_weight_mask[i] * quadr_out
            quadr_outs.append(quadr_out)
            outs.append(out)

        quadratic_terms = - sum(sum(quadr_outs))  # Note the minus here

        # Get the final energy
        energy = sq_nrm + lin_terms + quadratic_terms

        return energy, outs


class BengioFischerNetwork(nn.Module):
    """Defines the attractor network studied by Bengio and Fischer

    Bengio, Y. and Fischer, A. (2015). Early inference in energy-based models
    approximates back-propagation. Technical Report arXiv:1510.02777,
    Universite de Montreal.

    Bengio, Y., Mesnard, T., Fischer, A., Zhang, S., and Wu, Y. (2015).
    STDP as presynaptic activity times rate of change of postsynaptic activity.
    arXiv:1509.05936.

    The network is a continuous Hopfield-like network. It was first
    studied by Bengio and Fischer (2015), and has since been used in
    Equilibrium Propagation (Scellier et al. 2017) and other works.
    """
    def __init__(self, args, device, model_name, writer, n_class=None):
        super().__init__()
        self.args = args
        self.device = device
        self.writer = writer
        self.model_name = model_name
        self.num_state_layers = len(self.args.state_sizes[1:])


        self.weights = nn.ModuleList(
            [nn.Linear(torch.prod(torch.tensor(l1[1:])),
                       torch.prod(torch.tensor(l2[1:])), bias=False)
             for l1, l2 in zip(args.state_sizes[:-1], args.state_sizes[1:])])
        self.biases = nn.ModuleList([nn.Linear(torch.prod(torch.tensor(l[1:])),
                                               1, bias=False)
                       for l in args.state_sizes])
        for bias in self.biases:
            torch.nn.init.zeros_(bias.weight)

        if self.args.activation == "hardsig":
            self.actvn = activations.get_hard_sigmoid()
        elif self.args.activation == "relu":
            self.actvn = activations.get_relu()
        elif self.args.activation == "swish":
            self.actvn = activations.get_swish()

    def forward(self, states, class_id=None, step=None):

        # Squared norm
        sq_nrm = sum([(0.5 * (layer.view(layer.shape[0], -1) ** 2)).sum() for layer in states])

        # Linear terms
        linear_terms = - sum([bias(self.actvn(layer.view(layer.shape[0], -1))).sum()
                              for layer, bias in
                              zip(states, self.biases)])

        quadratic_terms = - sum([0.5 * sum(torch.einsum('ba,ba->b',
                                                        W(self.actvn(pre.view(pre.shape[0], -1))),
                                                        self.actvn(post.view(post.shape[0], -1))))
                                 for pre, W, post in
                                 zip(states[:-1], self.weights, states[1:])])
        return sq_nrm + linear_terms + quadratic_terms, None

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class InitializerNetwork(torch.nn.Module):
    def __init__(self, args, writer, device):
        super(InitializerNetwork, self).__init__()
        self.args = args
        self.writer = writer
        self.device = device
        self.input_size = self.args.state_sizes[0]
        self.output_sizes = args.state_sizes[1:]
        self.swish = get_swish()
        self.criterion = nn.MSELoss()
        self.criteria = []
        self.num_ch = self.args.arch_dict['num_ch_initter']
        self.encs = nn.ModuleList([])
        self.sides = nn.ModuleList([])
        self.in_channels = self.args.state_sizes[0][1]

        # Define the base encoder
        if len(self.args.state_sizes[1])==4:
            self.enc_base = nn.Sequential(nn.BatchNorm2d(self.in_channels),
                                          cust_actv.Swish_module(),
                                          nn.Conv2d(in_channels=self.in_channels,
                                                    out_channels=self.num_ch,
                                                    kernel_size=3,
                                                    padding=1,
                                                    bias=True),
                                          nn.BatchNorm2d(self.num_ch),
                                          cust_actv.Swish_module(),
                                          nn.Conv2d(
                                              in_channels=self.num_ch,
                                              out_channels=self.num_ch,
                                              kernel_size=3,
                                              padding=1,
                                              bias=True)
                                          )
        elif len(self.args.state_sizes[1])==2:
            img_size = int(torch.prod(torch.tensor(self.args.state_sizes[0][1:])))
            self.enc_base = nn.Sequential(
                                      Reshape(self.args.batch_size, img_size),
                                      nn.Linear(in_features=img_size,
                                                out_features=img_size),
                                      cust_actv.Swish_module(),
                                      nn.BatchNorm1d(img_size))


        # Define the rest of the encoders
        for i in range(1, len(self.args.state_sizes)):
            # encs should take as input the image size and output the statesize for that statelayer
            if len(self.args.state_sizes[i]) == 4: #TODO this is a hack for the new type of networks with dim4 in sl0
                self.encs.append(
                                nn.Sequential(
                                          nn.BatchNorm2d(self.num_ch),
                                          cust_actv.Swish_module(),
                                          nn.Conv2d(in_channels=self.num_ch,
                                                    out_channels=self.num_ch,
                                                    kernel_size=3,
                                                    padding=1,
                                                    bias=True),
                                          Interpolate(size=self.args.state_sizes[i][2:],
                                                      mode='bilinear')).to(self.device))
            elif len(self.args.state_sizes[i]) == 2:
                prev_sl_shape = self.args.state_sizes[i - 1]
                if (self.args.network_type == 'VectorField' and \
                    len(prev_sl_shape) == 4) or (i==1):
                    prev_size = self.args.state_sizes[i - 1][1:]
                    prev_size = int(torch.prod(torch.tensor(prev_size)))
                elif len(prev_sl_shape) == 4:
                    prev_size = self.args.state_sizes[i - 1][2:]
                    prev_size.append(self.num_ch)
                    prev_size = int(torch.prod(torch.tensor(prev_size)))
                elif len(prev_sl_shape) == 2:
                    prev_size = self.args.state_sizes[i - 1][1]

                new_size = (self.args.batch_size, prev_size)
                # new_size.extend(self.args.state_sizes[i][1])
                self.encs.append(
                    nn.Sequential(
                        Reshape(self.args.batch_size, prev_size),
                        nn.Linear(in_features=prev_size,
                                  out_features=self.args.state_sizes[i][1]),
                        cust_actv.Swish_module(),
                        nn.BatchNorm1d(self.args.state_sizes[i][1])))

            # Define the side branches that split off from the encoder and
            # output the state layer initializations at each statelayer

            # Sides should output the statesize for that statelayer and input is same size as output.
            # so long as the input to the side is the same as the size of the
            # base conv (i.e. the statesize), I can just use the same settings
            # as for the base+energy convs
            if len(self.args.state_sizes[i]) == 4:
                self.sides.append(
                                nn.Sequential(nn.BatchNorm2d(self.num_ch),
                                          cust_actv.Swish_module(),
                                          nn.Conv2d(in_channels=self.num_ch,
                                                     out_channels=self.num_ch,
                                                     kernel_size=3,
                                                     padding=1, bias=True),
                                          nn.BatchNorm2d(self.num_ch),
                                          cust_actv.Swish_module(),
                                          nn.Conv2d(in_channels=self.num_ch,
                                                     out_channels=self.num_ch,
                                                     kernel_size=3,
                                                     padding=1, bias=True),
                                          nn.Conv2d(in_channels=self.num_ch,
                                                     out_channels=self.args.state_sizes[i][1],
                                                     kernel_size=3,
                                                     padding=1, bias=True)).to(self.device))#adjust kernel size so that output is b,9,16,16
            elif len(self.args.state_sizes[i]) == 2:
                self.sides.append(
                    nn.Sequential(
                        nn.Linear(in_features=self.args.state_sizes[i][1],
                                  out_features=self.args.state_sizes[i][1]),
                        cust_actv.Swish_module(),
                        nn.BatchNorm1d(self.args.state_sizes[i][1]),
                        nn.Linear(in_features=self.args.state_sizes[i][1],
                                  out_features=self.args.state_sizes[i][1]),
                        cust_actv.Swish_module()
                    )
                )

        self.optimizer = optim.SGD(self.parameters(),
                                   nesterov=True,
                                   momentum=0.6,
                                   lr=self.args.initter_network_lr)
        self.lh_sig = get_leaky_hard_sigmoid()

    def forward(self, x, x_id):
        print("Initializing with FF net")
        hids = []
        inp = self.enc_base(x)
        for enc_i in self.encs:
            hid_i = enc_i(inp)
            hids.append(hid_i)
            inp = hid_i

        outs = []
        for side_i, hid_i in zip(self.sides, hids):
            out_i = side_i(hid_i)

            # (Leakily) Clamp the outputs to (approximately) [0,1]
            outs.append(self.lh_sig(out_i))

        return outs

    def update_weights(self, outs, targets, step):
        self.optimizer.zero_grad()
        self.criteria = [self.criterion(o, t) for o,t in zip(outs,targets)]
        loss = torch.sum(torch.stack(self.criteria))
        loss.backward()
        self.optimizer.step()
        print("\nInitializer loss: " + '%.4g' % loss.item())
        # if step % self.args.scalar_logging_interval == 0:
        self.writer.add_scalar('Initializer/total_loss', loss.item(),
                               step)
        for i, l in enumerate(self.criteria, start=1):
            name = 'Initializer/loss_layer_%i' % i
            self.writer.add_scalar(name, l.item(),
                                   step)
        return loss











###### Function to help me debug
shapes = lambda x : [y.shape for y in x]
nancheck = lambda x : (x != x).any()
listout = lambda y : [x for x in y]






class EBMLV(nn.Module):
    """Defines the EBM with latent variables (Sharkey 2019)

    The network is a relaxtion of the DAN, which is in turn a generalisation
     of the vector field network used in
     Scellier et al. (2018) relaxation of the continuous Hopfield-like network (CHN)
    studied by Bengio and Fischer (2015) and later in
    Equilibrium Propagation (Scellier et al. 2017) and other works. It
    no longer required symmetric weights as in the CHN and the network outputs
     are also interpreted differently. In the DAN, VFN, and CHN, the nonlinear
     parts define a target for the state value, and the difference between the
     target and the state defines the dynamics. In the EBMLV, the outputs of the
     networks are truly just energy functions.
    """
    def __init__(self, args, device, model_name, writer, n_class=None):
        super().__init__()
        self.args = args
        self.device = device
        self.writer = writer
        self.model_name = model_name
        self.num_state_layers = len(self.args.state_sizes[1:])

        self.quadratic_nets = nn.ModuleList([])

        for i, size in enumerate(self.args.state_sizes):
            inp_idxs = self.args.arch_dict['mod_connect_dict'][i]
            state_sizes = [self.args.state_sizes[j] for j in inp_idxs]
            state_sizes.append(size)
            all_same_dim = all([len(state_sizes[0])==len(sz) for sz in state_sizes])
            if len(size) == 4:
                if all_same_dim:
                    net = DenseConv(self.args, i)
                else:
                    net = ConvFCMixturetoFourDim(self.args, i)
            elif len(size) == 2:
                if all_same_dim:
                    net = FCFC(self.args, i)
                else:
                    net = ConvFCMixturetoTwoDim(self.args, i)
            self.quadratic_nets.append(net)

        # self.biases = nn.ModuleList([nn.Linear(torch.prod(torch.tensor(l[1:])),
        #                                        1, bias=False)
        #                for l in args.state_sizes])
        # for bias in self.biases:
        #     torch.nn.init.zeros_(bias.weight)

        if self.args.states_activation == "hardsig":
            self.state_actv = activations.get_hard_sigmoid()
        elif self.args.states_activation == "relu":
            self.state_actv = activations.get_relu()
        elif self.args.states_activation == "swish":
            self.state_actv = activations.get_swish()

    def forward(self, states, class_id=None, step=None):

        # # Squared norm
        # sq_nrm = sum([(0.5 * (layer.view(layer.shape[0], -1) ** 2)).sum() for layer in states])
        #
        # # # Linear terms
        # # linear_terms = - sum([bias(self.state_actv(layer.view(layer.shape[0], -1))).sum()
        # #                       for layer, bias in
        # #                       zip(states, self.biases)])
        #
        # # Quadratic terms
        # enrgs = []
        # outs = []
        # for i, (pre_state, net) in enumerate(zip(states, self.quadratic_nets)):
        #     post_inp_idxs = self.args.arch_dict['mod_connect_dict'][i]
        #     pos_inp_states = [states[j] for j in post_inp_idxs]
        #     pos_inp_states = [self.state_actv(state)
        #                       for state in pos_inp_states]
        #     enrg, out = net(pre_state, pos_inp_states)
        #     enrg = self.args.energy_weight_mask[i] * enrg
        #     enrgs.append(enrg)
        #     outs.append(out)
        #
        # energy = sum(sum(enrgs))  # Note there is no minus sign here where it
        # # exists in the DAN

        # Squared norm
        sq_nrm = sum([(0.5 * (layer.view(layer.shape[0], -1) ** 2)).sum() for layer in states])

        # # # Linear terms
        # linear_terms = - sum([bias(self.state_actv(layer.view(layer.shape[0], -1))).sum()
        #                       for layer, bias in
        #                       zip(states, self.biases)])

        # Quadratic terms
        enrgs = []
        outs = []
        for i, (pre_state, net) in enumerate(zip(states, self.quadratic_nets)):
            post_inp_idxs = self.args.arch_dict['mod_connect_dict'][i]
            pos_inp_states = [states[j] for j in post_inp_idxs]
            pos_inp_states = [self.state_actv(state)
                              for state in pos_inp_states]
            enrg, out = net(pre_state, pos_inp_states)
            #enrg = self.args.energy_weight_mask[i] * enrg
            enrgs.append(enrg)
            outs.append(out)

        quadratic_terms = - sum(sum(enrgs))  # Note the minus here
        energy = quadratic_terms #sq_nrm + quadratic_terms #linear_terms +

        return energy, outs


class StructuredVectorFieldNetwork(nn.Module):
    """Like the VectorFieldNetwork but allows for conv layers
    """
    def __init__(self, args, device, model_name, writer, n_class=None):
        super().__init__()
        self.args = args
        self.device = device
        self.writer = writer
        self.model_name = model_name
        self.num_state_layers = len(self.args.state_sizes[1:])

        # Define the networks that output the quadratic terms
        self.quadratic_nets = nn.ModuleList([])
        for i, size in enumerate(self.args.state_sizes):
            inp_idxs = self.args.arch_dict['mod_connect_dict'][i]
            state_sizes = [self.args.state_sizes[j] for j in inp_idxs]
            state_sizes.append(size)
            all_same_dim = all([len(state_sizes[0])==len(sz)
                                for sz in state_sizes])
            if len(size) == 4:
                if all_same_dim:
                    net = LinearConv(self.args, i)
                else:
                    net = LinearConvFCMixturetoFourDim(self.args, i)
            elif len(size) == 2:
                if all_same_dim:
                    net = LinearLayer(self.args, i)
                else:
                    net = LinearConvFCMixturetoTwoDim(self.args, i)
            self.quadratic_nets.append(net)

        # Define the biases that determine the linear term
        self.biases = nn.ModuleList(
            [nn.Linear(torch.prod(torch.tensor(l[1:])), 1, bias=False)
             for l in args.state_sizes])
        for bias in self.biases:
            torch.nn.init.zeros_(bias.weight)

        if self.args.states_activation == "hardsig":
            self.state_actv = activations.get_hard_sigmoid()
        elif self.args.states_activation == "relu":
            self.state_actv = activations.get_relu()
        elif self.args.states_activation == "swish":
            self.state_actv = activations.get_swish()

    def forward(self, states, class_id=None, step=None):

        # Squared norm
        # sq_nrm = sum(
        #     [(0.5 * (layer.view(layer.shape[0], -1) ** 2)).sum() for layer in
        #      states])
        sq_terms = []
        for i, layer in enumerate(states):
            sq_term = 0.5 * (layer.view(layer.shape[0], -1) ** 2).sum()
            #sq_term = self.args.energy_weight_mask[i] * sq_term
            sq_terms.append(sq_term)
        sq_nrm = sum(sq_terms)

        # # Linear terms
        # lin_terms = []
        # for i, (layer, bias) in enumerate(zip(states, self.biases)):
        #     lin_term = bias(self.state_actv(layer.view(layer.shape[0], -1)))
        #     lin_term = lin_term.sum()
        #     #lin_term = self.args.energy_weight_mask[i] * lin_term
        #     lin_terms.append(lin_term)
        # lin_terms = - sum(lin_terms)

        # linear_terms = - sum([bias(self.state_actv(layer.view(layer.shape[0], -1))).sum()
        #                       for layer, bias in
        #                       zip(states, self.biases)])

        # Quadratic terms
        quadr_outs = []
        outs = []
        for i, (pre_state, net) in enumerate(zip(states, self.quadratic_nets)):
            post_inp_idxs = self.args.arch_dict['mod_connect_dict'][i]
            pos_inp_states = [states[j] for j in post_inp_idxs]
            pos_inp_states = [self.state_actv(state)
                              for state in pos_inp_states]
            quadr_out, out = net(pre_state, pos_inp_states)
            #quadr_out = self.args.energy_weight_mask[i] * quadr_out
            quadr_outs.append(quadr_out)
            outs.append(out)

        quadratic_terms = sum(sum(quadr_outs))  # Note no the minus here

        # Get the final energy
        energy = sq_nrm + quadratic_terms #+ lin_terms

        return energy, outs
