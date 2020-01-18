"""Based on https://github.com/rosinality/igebm-pytorch/blob/master/model.py
which is available under an MIT OSI licence."""

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn import utils
import lib.custom_swish_activation as cust_actv


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
    def __init__(self, args, n_class=None):
        super().__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1,
        # padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
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
    """Define the attractor network. It takes the list of state layers as
    input, passes each through a base conv net, the output of which is passed
    to layers ahead and behind (where appropriate). If ahead, these are
     downsampled so that they have the same dimensions as the statelayer ahead;
     if behind, these are upsampling so that they have the same dimensions as
     behind. The concatenated result is passed through another conv net then a
     relu then another (simply linear) conv net, and
     then the values of the state layer are added (making it a pre-activation resnet), which
     outputs a tensor that has the same dimensions as the state layer, defining
     the energy for each of the state layer neurons. These are all summed in
     order to get the total energy."""
    def __init__(self, args, device, model_name, n_class=None):
        super().__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1,
        # padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        # An example that works:
        # nn.Conv2d(3,4,3,padding=1,bias=False)(torch.rand(128,3,32,32))
        # Fails: nn.Conv2d(3,4,3,padding=1,bias=False)(torch.rand(128,32,32,3))
        self.args = args
        self.device = device
        self.model_name = model_name
        self.state_sizes = args.states_sizes
        if args.activation == 'leaky_relu':
            self.act1 = torch.nn.LeakyReLU()
        elif args.activation == "relu":
            self.act1 = torch.nn.ReLU()
        elif args.activation == "swish":
            self.act1 = cust_actv.Swish_module()
        self.act2 = torch.nn.LeakyReLU()


        # Base convs are common to all state layers
        self.base_convs = nn.ModuleList([
            spectral_norm(nn.Conv2d(in_channels=3,
                                    out_channels=64,
                                    kernel_size=3, padding=1, bias=True)), #yields [128, 64, 32, 32]
            spectral_norm(nn.Conv2d(in_channels=9,
                                    out_channels=64,
                                    kernel_size=3, padding=1, bias=True)), #yields [128, 64, 16, 16]
            spectral_norm(nn.Conv2d(in_channels=18,
                                    out_channels=64,
                                    kernel_size=3, padding=1, bias=True))])  #yields [128, 64, 8, 8]
        self.intermed_convs = nn.ModuleList([
            spectral_norm(nn.Conv2d(in_channels=(64*2)+3,
                                    out_channels=64,
                                    kernel_size=3, padding=1, bias=True)),
            spectral_norm(nn.Conv2d(in_channels=(64*3)+9,
                                    out_channels=64,
                                    kernel_size=3, padding=1, bias=True)),
            spectral_norm(nn.Conv2d(in_channels=(64*2)+18,
                                    out_channels=64,
                                    kernel_size=3, padding=1, bias=True))])

        self.energy_convs = nn.ModuleList([
            spectral_norm(nn.Conv2d(in_channels=(64*3)+3,
                                    out_channels=3,
                                    kernel_size=3, padding=1, bias=True)),
            spectral_norm(nn.Conv2d(in_channels=(64*4)+9,
                                    out_channels=9,
                                    kernel_size=3, padding=1, bias=True)),
            spectral_norm(nn.Conv2d(in_channels=(64*3)+18,
                                    out_channels=18,
                                    kernel_size=3, padding=1, bias=True))])

        self.interps = nn.ModuleList([Interpolate(size=size[2:], mode='bilinear')
                                      for size in self.state_sizes])

        # I'm including energy weights because it might give the network a
        # chance to silence some persistently high energy state neurons during
        # learning but to unsilence them when they start being learnable (e.g.
        # when the layers below have learned properly.
        # These will be clamped to be at least a small positive value
        self.num_state_neurons = torch.sum(
            torch.tensor([torch.prod(torch.tensor(ss[1:]))
                 for ss in self.state_sizes]))
        self.energy_weights = nn.Linear(int(self.num_state_neurons), 1, bias=False)
        nn.init.uniform_(self.energy_weights.weight, a=0.5, b=1.5)

        self.energy_weight_masks = []
        for i, m in enumerate(self.args.energy_weight_mask):
            energy_weight_mask = m * torch.ones(tuple(self.args.states_sizes[i]),
                                                device=self.device)
            self.energy_weight_masks.append(energy_weight_mask)

    def forward(self, state_layers, class_id=None):
        num_state_layers = len(state_layers)
        base_outs = [None] * num_state_layers #Not totally sure this is necessary
        for i in range(len(state_layers)):
            base_outs[i] = self.act1(self.base_convs[i](state_layers[i]))

        outs = [None] * num_state_layers
        for i in range(num_state_layers):
            if i == 0:
                from_next = self.interps[i](base_outs[i + 1])
                intermed_input = torch.cat([state_layers[i],
                                           base_outs[i],
                                           from_next],
                                           dim=1)
            elif i == num_state_layers - 1:
                from_previous = self.interps[i](base_outs[i - 1])
                intermed_input = torch.cat([state_layers[i],
                                          from_previous,
                                          base_outs[i]], dim=1)
            else:
                from_previous = self.interps[i](base_outs[i - 1])
                from_next = self.interps[i](base_outs[i + 1])
                intermed_input = torch.cat([state_layers[i],
                                          from_previous,
                                          base_outs[i],
                                          from_next], dim=1)
            intermed_out = self.act1(self.intermed_convs[i](intermed_input))
            energy_input = torch.cat([intermed_input, intermed_out], dim=1)
            outs[i] = self.act1(self.energy_convs[i](energy_input)) #TODO consider having no bias in this layer in order to keep the energies close to 0.

        outs = [out.view(self.args.batch_size, -1) for out in outs]
        outs = [out * mask.view(self.args.batch_size, -1)
                for out, mask in zip(outs, self.energy_weight_masks)]
        outs = torch.cat(outs, dim=1)
        energy = self.energy_weights(outs)
        return energy


class InitializerNetwork(torch.nn.Module):
    def __init__(self, args, writer, device):
        super(InitializerNetwork, self).__init__()
        self.args = args
        self.writer = writer
        self.device = device
        self.input_size = self.args.states_sizes[0]
        self.output_sizes = args.states_sizes[1:]
        self.swish = get_swish()
        #self.first_ff = nn.Linear(self.input_size, self.input_size).to(self.device)
        #self.first_bn = nn.BatchNorm1d(self.input_size).to(self.device)
        # self.ffs = nn.ModuleList([nn.Linear(size1, size2).to(self.device)
        #                               for (size1, size2) in
        #                               zip(args.size_layers[:-1],
        #                                   args.size_layers[1:])])
        # self.bnorms = nn.ModuleList([nn.BatchNorm1d(size2).to(self.device)
        #                              for size2 in
        #                              args.size_layers[1:]])
        # self.sides = nn.ModuleList([nn.Linear(size2, size2).to(self.device)
        #                              for size2 in
        #                              args.size_layers[1:]])
        self.criterion = nn.MSELoss()
        self.criteria = []
        #bnorm #https://discuss.pytorch.org/t/example-on-how-to-use-batch-norm/216/2
        x = torch.rand(size=tuple(self.args.states_sizes[0]))
        self.enc1 = nn.Sequential(nn.BatchNorm2d(3),
                                  cust_actv.Swish_module(),
                                  nn.Conv2d(in_channels=3,
                                            out_channels=64,
                                            kernel_size=3,
                                            padding=1, bias=True),
                                  nn.BatchNorm2d(64),
                                  cust_actv.Swish_module(),
                                  nn.Conv2d(in_channels=64,
                                            out_channels=64,
                                            kernel_size=3,
                                            padding=1, bias=True),
                                  Interpolate(size=self.args.states_sizes[1][2:],
                                              mode='bilinear'))
        self.enc2 = nn.Sequential(nn.BatchNorm2d(64),
                                  cust_actv.Swish_module(),
                                  nn.Conv2d(in_channels=64,
                                            out_channels=64,
                                            kernel_size=3,
                                            padding=1, bias=True),
                                  Interpolate(size=self.args.states_sizes[2][2:],
                                              mode='bilinear'))
        self.side1 = nn.Sequential(nn.BatchNorm2d(64),
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
                                             out_channels=9,
                                             kernel_size=3,
                                             padding=1, bias=True))#adjust kernel size so that output is b,9,16,16
        self.side2 = nn.Sequential(nn.BatchNorm2d(64),
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
                                             out_channels=18,
                                             kernel_size=3, #adjust kernel size so that output is b,9,16,16
                                             padding=1, bias=True))

        print('boop')
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
        hid1 = self.enc1(x)
        hid2 = self.enc2(hid1)

        out1 = self.side1(hid1) #TODO clamp these to 0,1
        out2 = self.side2(hid2)

        # (Leakily) Clamp the outputs to (approximately) [0,1]
        out1 = self.lh_sig(out1)
        out2 = self.lh_sig(out2)
        return out1, out2

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

def get_swish():
    sigmoid = torch.nn.Sigmoid()
    return lambda x : x * sigmoid(x)

def get_leaky_hard_sigmoid():
    hsig = torch.nn.Hardtanh(min_val=0.0)
    return lambda x : hsig(x) + 0.01*x

###### Function to help me debug
shapes = lambda x : [y.shape for y in x]
nancheck = lambda x : (x != x).any()
listout = lambda y : [x for x in y]