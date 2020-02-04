"""Based on https://github.com/rosinality/igebm-pytorch/blob/master/model.py
which is available under an MIT OSI licence."""

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn import utils
import lib.utils
import lib.custom_swish_activation as cust_actv


def get_swish():
    sigmoid = torch.nn.Sigmoid()
    return lambda x : x * sigmoid(x)


def get_leaky_hard_sigmoid():
    hsig = torch.nn.Hardtanh(min_val=0.0)
    return lambda x : hsig(x) + 0.01*x


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


class IGEBM(nn.Module):
    def __init__(self, args, device, model_name, n_class=None):
        super().__init__()
        self.args = args
        self.device = device
        self.model_name = model_name

        # Template conv params:
        #  torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1,
        #  padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv1 = spectral_norm(nn.Conv2d(3, 128, 3, padding=1), std=1)

        self.blocks = nn.ModuleList(
            [     #in_channel, out_channel, n_class
                ResBlock(128, 128, n_class, downsample=True),
                ResBlock(128, 128, n_class),
                ResBlock(128, 256, n_class, downsample=True),
                ResBlock(256, 256, n_class),
                ResBlock(256, 256, n_class, downsample=True),
                ResBlock(256, 256, n_class),
            ]
        )

        self.linear = nn.Linear(256, 1)

    def forward(self, input, class_id=None):
        out = self.conv1(input)

        out = F.leaky_relu(out, negative_slope=0.2)

        for block in self.blocks:
            out = block(out, class_id)

        out = F.relu(out)
        out = out.view(out.shape[0], out.shape[1], -1).sum(2)
        out = self.linear(out)

        return out

class DenseConv(nn.Module):
    def __init__(self, args, layer_idx):
        super().__init__()
        self.args = args
        self.pad_mode = 'zeros'
        self.layer_idx = layer_idx
        self.act = lib.utils.get_activation_function(args)

        # The indices of the state_layers that will be input to this net.
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
        self.base_conv = spectral_norm(nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.args.arch_dict['num_ch'],
                kernel_size=self.args.arch_dict['kernel_sizes'][0],
                padding=self.args.arch_dict['padding'],
                stride=self.args.arch_dict['strides'][0],
                padding_mode=self.pad_mode,
                bias=True))
        self.energy_conv = spectral_norm(nn.Conv2d(
                in_channels=self.args.arch_dict['num_ch'] + self.in_channels,
                out_channels=self.state_layer_ch,
                kernel_size=self.args.arch_dict['kernel_sizes'][0],
                padding=self.args.arch_dict['padding'],
                padding_mode=self.pad_mode,
                bias=True), std=1e-10, bound=True)

    def forward(self, inputs, class_id=None):
        reshaped_inps = [self.interp(inp) for inp in inputs
                         if inp.shape[2:] != self.state_layer_h_w]
        reshaped_inps = torch.cat(reshaped_inps, dim=1)
        base_out = self.act(self.base_conv(reshaped_inps))
        energy_input = torch.cat([reshaped_inps, base_out], dim=1)
        out = self.act(self.energy_conv(energy_input))
        return out


class FCFC(nn.Module):
    def __init__(self, args, layer_idx):
        super().__init__()
        self.args = args
        self.act = lib.utils.get_activation_function(args)
        self.layer_idx = layer_idx

        # The indices of the state_layers that will be input to this net and
        # their sizes. They may be flattened conv outputs.
        self.input_idxs = self.args.arch_dict['mod_connect_dict'][layer_idx]
        self.in_sizes = [torch.prod(torch.tensor(self.args.state_sizes[j][1:]))
                         for j in self.input_idxs]

        self.out_size = self.args.state_sizes[layer_idx][1]
        self.base_fc_layers = nn.ModuleList([spectral_norm(
            nn.Linear(in_size, self.out_size)) for in_size in self.in_sizes])
        self.energy_fc_layer = nn.Sequential(
            spectral_norm(nn.Linear(self.out_size * len(self.in_sizes),
                                    self.out_size)),
            self.act,
            spectral_norm(nn.Linear(self.out_size, self.out_size)),
            self.act)

    def forward(self, inputs, class_id=None):
        reshaped_inps = [inp.view(self.args.batch_size, -1) for inp in inputs]
        base_outs = [self.act(base_fc(inp))
                     for base_fc, inp in zip(self.base_fc_layers,
                                             reshaped_inps)]
        energy_input = torch.cat(base_outs, dim=1)
        out = self.energy_fc_layer(energy_input)
        return out

class DeepAttractorNetwork(nn.Module):
    """Define the attractor network

    Define its inputs, how the inputs are processed, what it outputs, any
    upsampling or downsampling
    """
    def __init__(self, args, device, model_name, writer, n_class=None):
        super().__init__()
        self.args = args
        self.device = device
        self.writer = writer
        self.model_name = model_name
        self.num_state_layers = len(self.args.state_sizes[1:])
        self.act = lib.utils.get_activation_function(args)
        self.num_state_neurons = torch.sum(
            torch.tensor([torch.prod(torch.tensor(ss[1:]))
                 for ss in self.args.state_sizes[1:]]))
        self.energy_weights = nn.Linear(int(self.num_state_neurons), 1, bias=False)
        nn.init.uniform_(self.energy_weights.weight, a=0.1, b=0.9)
        self.energy_weight_masks = None
        self.calc_energy_weight_masks()
        # TODO
        # set up the conv and fc layers in the DAN
        num_convs = len([s for s in self.args.state_sizes[1:] if len(s)==4])
        num_fcs = len([s for s in self.args.state_sizes if len(s)==2])
        self.convs = [DenseConv(args, i) for i in range(1,num_convs+1)]
        self.fcs = [FCFC(args, j) for j in range(num_convs+1,
                                                 num_convs+1+num_fcs)]
        self.nets = nn.ModuleList(self.convs)
        self.nets.extend(self.fcs)

    def forward(self, states, class_id=None, step=None):
        outs = []
        for i, (state, net) in enumerate(zip(states, self.nets), start=1):
            inp_idxs = self.args.arch_dict['mod_connect_dict'][i]
            inp_states = [states[j] for j in inp_idxs]
            out = self.nets[i-1](inp_states)
            outs.append(out)

        if self.args.log_histograms and step is not None and \
            step % self.args.histogram_logging_interval == 0:
            for i, enrg in enumerate(outs):
                layer_string = 'train/energies_%s' % i
                self.writer.add_histogram(layer_string, enrg, step)

        outs = [out.view(self.args.batch_size, -1) for out in outs]
        # TODO consider placing a hook here
        outs = [out * mask.view(self.args.batch_size, -1)
                for out, mask in zip(outs, self.energy_weight_masks)]
        #print(self.energy_weight_masks[0])
        outs = torch.cat(outs, dim=1)
        #print(outs)
        energy = self.energy_weights(outs)
        #print(list(self.energy_weights.parameters()))
        return energy


    def calc_energy_weight_masks(self):
        # TODO
        #  masks should accoutn for wherever the energy gradient
        #  is 0, so that swish activation isn’t fixed to a part of the
        #  slope where it is positive.

        self.energy_weight_masks = []
        for i, m in enumerate(self.args.energy_weight_mask):
            energy_weight_mask = m * torch.ones(
                tuple(self.args.state_sizes[i + 1]),
                # +1 because we don't apply the energy masks to the image
                requires_grad=False,
                device=self.device)  # TODO still don't know if these ever change value in training
            self.energy_weight_masks.append(energy_weight_mask)


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
        self.encs = []
        self.sides = []
        self.in_channels = self.args.state_sizes[0][1]

        self.enc_base = nn.Sequential(nn.BatchNorm2d(self.in_channels),#TODO consider removing
                                  cust_actv.Swish_module(),
                                  nn.Conv2d(in_channels=self.in_channels,
                                            out_channels=64,
                                            kernel_size=3,
                                            padding=1,
                                            bias=True))
        for i in range(1, len(self.args.state_sizes)):
            # encs should take as input the image size and output the statesize for that statelayer
            self.encs.append(
                            nn.Sequential(
                                      nn.BatchNorm2d(64),
                                      cust_actv.Swish_module(),
                                      nn.Conv2d(in_channels=64,
                                                out_channels=64,
                                                kernel_size=3,
                                                padding=1,
                                                bias=True),
                                      Interpolate(size=self.args.state_sizes[i][2:],
                                                  mode='bilinear')).to(self.device))

            # Sides should output the statesize for that statelayer and input is same size as output.
            # so long as the input to the side is the same as the size of the
            # base conv (i.e. the statesize), I can just use the same settings
            # as for the base+energy convs
            self.sides.append(
                            nn.Sequential(nn.BatchNorm2d(64),
                                      cust_actv.Swish_module(),
                                      nn.Conv2d(in_channels=64,
                                                 out_channels=64,
                                                 kernel_size=3,
                                                 padding=1, bias=True),
                                      nn.BatchNorm2d(64),
                                      cust_actv.Swish_module(),
                                      nn.Conv2d(in_channels=64,
                                                 out_channels=64,
                                                 kernel_size=3,
                                                 padding=1, bias=True),
                                      nn.Conv2d(in_channels=64,
                                                 out_channels=self.args.state_sizes[i][1],
                                                 kernel_size=3,
                                                 padding=1, bias=True)).to(self.device))#adjust kernel size so that output is b,9,16,16


        self.optimizer = optim.Adam(self.parameters(),
                                    lr=self.args.initter_network_lr)


        self.lh_sig = get_leaky_hard_sigmoid()

    # def forward(self, x):
    #     # x is input image
    #     x = x.permute(1,0)
    #     x = self.swish_bn_layer(x, self.first_ff, self.first_bn).detach()
    #     side_outs = [None] * (len(self.args.size_layers)-1)
    #     for i in range(len(self.args.size_layers)-1):
    #         x = self.swish_bn_layer(x, self.ffs[i], self.bnorms[i])
    #         side_outs[i] = self.hardsig_layer(x, self.sides[i])
    #     side_outs = [so.permute(1, 0) for so in side_outs]
    #     return side_outs

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
        loss.backward(retain_graph=True)
        self.optimizer.step()
        if step % self.args.scalar_logging_interval == 0:
            print("\nInitializer loss: " + '%.4g' % loss.item())
            self.writer.add_scalar('train/initter_loss', loss.item(),
                                   step)
        return loss


###### Function to help me debug
shapes = lambda x : [y.shape for y in x]
nancheck = lambda x : (x != x).any()
listout = lambda y : [x for x in y]


# if self.args.architecture == 'mnist_1_layer_small':
#     self.num_ch = 32
#     self.num_sl = 1
#     self.args.arch_dict['kernel_sizes'] = [3, 3]
#     self.padding = 1
# elif self.args.architecture == 'mnist_2_layers_small':
#     self.num_ch = 32
#     self.num_sl = 2
#     self.args.arch_dict['kernel_sizes'] = [3, 3]
#     self.padding = 1
# elif args.architecture == 'mnist_2_layers_small_equal':
#     self.num_ch = 16
#     self.num_sl = 2
#     self.args.arch_dict['kernel_sizes'] = [3, 3]
#     self.padding = 1
#     self.strides = [1, 0]
# elif self.args.architecture == 'mnist_3_layers_med':
#     self.num_ch = 64
#     self.num_sl = 3
#     self.args.arch_dict['kernel_sizes'] = [3, 3, 3]
#     self.padding = 0
# elif self.args.architecture == 'mnist_3_layers_large':
#     self.num_ch = 64
#     self.num_sl = 3
#     self.args.arch_dict['kernel_sizes'] = [3, 3, 3]
#     self.padding = 1
# elif self.args.architecture == 'cifar10_2_layers':
#     self.num_ch = 64
#     self.num_sl = 2
#     self.args.arch_dict['kernel_sizes'] = [3, 3]
#     self.padding = 1