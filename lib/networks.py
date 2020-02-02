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
        self.pad_mode = 'zeros'

        self.act1 = lib.utils.get_activation_function(args)

        self.base_convs = nn.ModuleList([
            spectral_norm(nn.Conv2d(in_channels=self.args.state_sizes[i][1] + \
                                                self.args.state_sizes[i+1][1],
                                    out_channels=self.args.arch_dict['num_ch'],
                                    kernel_size=self.args.arch_dict['kernel_sizes'][0],
                                    padding=self.args.arch_dict['padding'],
                                    stride=self.args.arch_dict['strides'][0],
                                    padding_mode=self.pad_mode,
                                    bias=True))
            for i in range(len(self.args.state_sizes[1:]))]) ########cifar10 yields [128, 64, 32, 32]#yields [128, 64, 16, 16]

        self.energy_convs = nn.ModuleList([ #TODO check stride works
            spectral_norm(nn.Conv2d(in_channels=self.args.arch_dict['num_ch'] +
                                                self.args.state_sizes[i][1] +
                                                self.args.state_sizes[i+1][1], #Num channels in base conv plus num ch in statelayer plus num channels in prev statelayer
                                    out_channels=self.args.state_sizes[i+1][1],
                                    kernel_size=self.args.arch_dict['kernel_sizes'][1],
                                    padding=self.args.arch_dict['padding'],
                                    padding_mode=self.pad_mode,
                                    bias=True))
            for i in range(len(self.args.state_sizes[1:]))])

        self.interps = nn.ModuleList([Interpolate(size=size[2:], mode='bilinear')
                                      for size in self.args.state_sizes[1:]])

        # I'm including energy weights because it might give the network a
        # chance to silence some persistently high energy state neurons during
        # learning but to unsilence them when they start being learnable (e.g.
        # when the layers below have learned properly.
        # These will be clamped to be at least a small positive value
        # TODO consider unclamping this after you watch its updates during
        #  training and the progress of the energy outputs of each layer
        self.num_state_neurons = torch.sum(
            torch.tensor([torch.prod(torch.tensor(ss[1:]))
                 for ss in self.args.state_sizes[1:]]))
        self.energy_weights = nn.Linear(int(self.num_state_neurons), 1, bias=False)
        nn.init.uniform_(self.energy_weights.weight, a=0.5, b=1.5)

        self.energy_weight_masks = None
        self.calc_energy_weight_masks()

    def calc_energy_weight_masks(self):

        # TODO
        #  masks should accoutn for wherever the energy gradient
        #  is 0, so that swish activation isn’t fixed to a part of the
        #  slope where it is positive.

        self.energy_weight_masks = []
        for i, m in enumerate(self.args.energy_weight_mask):
            energy_weight_mask = m * torch.ones(tuple(self.args.state_sizes[i+1]), #+1 because we don't apply the energy masks to the image
                                                requires_grad=False,
                                                device=self.device) #TODO still don't know if these ever change value in training
            self.energy_weight_masks.append(energy_weight_mask)

    def forward(self, states, class_id=None, step=None):

        num_state_layers = len(states[1:])

        inputs = []
        for i, state in enumerate(states[:-1]):
            resized_prev = self.interps[i](states[i]) # resizes previous state (or image when i==0) to the size of the next statelayer
            input_i = torch.cat([resized_prev, states[i+1]], dim=1)
            inputs.append(input_i)

        base_outs = [None] * num_state_layers #Not totally sure this is necessary
        for i in range(num_state_layers):
            base_outs[i] = self.act1(self.base_convs[i](inputs[i]))

        outs = [None] * num_state_layers
        for i in range(num_state_layers):
            energy_input_i = torch.cat([inputs[i], base_outs[i]], dim=1)
            outs[i] = self.act1(self.energy_convs[i](energy_input_i))


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