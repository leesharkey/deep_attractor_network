"""Based on https://github.com/rosinality/igebm-pytorch/blob/master/model.py
which is available under an MIT OSI licence."""

import torch
from torch import nn, optim
from torch.nn import functional as F
import lib.utils as utils
import lib.custom_components.custom_swish_activation as cust_actv



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
        x = self.interp(x, size=self.size, mode=self.mode)#, align_corners=False)#removed because move to nearest neighbour mode.
        return x



class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

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


class CCTLayer(nn.Module):
    """A network formed of possibly two networks, a CNN and a transposed CNN

    In the case where there are two networks, the input is fed to both
    networks separately and their result is concatenated together along the
    channel dimension before output.

    The motivation for having two directions
    is that for networks with arbitrary connectivity,
    there is not necessarily an obvious direction that information needs
    to be processed in, unlike in standard feed forward networks,
    such as the discriminator or generator networks of GANs which use conv
    and transposed conv nets respectively. In the DAN (of the 2nd kind),
    information flows both ways along connections. If, despite the capability
    to use both the conv and the transposed conv components of the network,
    the direction that information needs to flow in a particular
    case demands only one, the network should be able to learn to ignore
    the other channels. This feat is to be facilitated by the 1 by 1
    convolutions in the CCTBlock, which typically follow (and often precede)
    a CCTLayer.

    In the case where there is only one network, the output is the same number
    of channels as when there are two networks.

    """
    def __init__(self, args, in_channels, out_channels, kernel_size,
                 padding=None, only_conv=False, only_conv_t=False):
        super().__init__()

        if only_conv and only_conv_t:
            raise ValueError("Cannot set CCTLayer to be only_conv AND only"
                             "conv_t. Only one can be true.")
        self.only_conv   = only_conv
        self.only_conv_t = only_conv_t
        spec_norm_reg = args.arch_dict['spec_norm_reg']

        # Define padding values
        # Padding of 0 is for when the architecture is compressing or expanding
        # the volume of the output compared with the input. These will only
        # be used in base CCTLayers
        # Padding of 1 with KS==3 OR padding of 3 with KS==7 is to preserve
        # volume of output compared with the input. These will be used either
        # in base CCTLayers or in the main dense blocks, since dense blocks
        # require volume preservation.
        if padding == 0:
           self.padding = padding
        elif kernel_size == 1:
            self.padding = 0
        elif kernel_size == 3:
            self.padding = 1
        elif kernel_size == 7:
            self.padding = 3
        elif kernel_size == 11:
            self.padding = 5
        else:
            self.padding = padding
        self.kernel_size = kernel_size
        self.output_pad = 0

        if self.only_conv:
            self.conv = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=self.kernel_size,
                                  padding=self.padding,
                                  stride=1,
                                  padding_mode='zeros')
            if spec_norm_reg:
                self.conv   = spectral_norm(self.conv)
        elif self.only_conv_t:
            self.conv_T = nn.ConvTranspose2d(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=self.kernel_size,
                                             padding=self.padding,
                                             stride=1,
                                             padding_mode='zeros',
                                             output_padding=self.output_pad)
            if spec_norm_reg:
                self.conv_T = spectral_norm(self.conv_T)
        else:
            out_channels_half = out_channels // 2
            self.conv = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels_half,
                                  kernel_size=self.kernel_size,
                                  padding=self.padding,
                                  stride=1,
                                  padding_mode='zeros')
            self.conv_T = nn.ConvTranspose2d(in_channels=in_channels,
                                             out_channels=out_channels_half,
                                             kernel_size=self.kernel_size,
                                             padding=self.padding,
                                             stride=1,
                                             padding_mode='zeros',
                                             output_padding=self.output_pad)
            if spec_norm_reg:
                self.conv   = spectral_norm(self.conv)
                self.conv_T = spectral_norm(self.conv_T)

    def forward(self, inp):
        if self.only_conv:
            out = self.conv(inp)
        elif self.only_conv_t:
            out = self.conv_T(inp)
        else:
            out_conv = self.conv(inp)
            out_conv_T = self.conv_T(inp)
            out = torch.cat([out_conv, out_conv_T], dim=1)
        return out


class CCTBlock(nn.Module):
    """"""
    def __init__(self, args, in_channels, out_channels, kernel_size,
                 only_conv=False, only_conv_t=False, padding=None):
        super().__init__()

        self.one_by_one_conv = nn.Conv2d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         padding_mode='zeros')
        self.cct_layer = CCTLayer(args,
                                  out_channels,
                                  out_channels,
                                  kernel_size,
                                  padding=padding,
                                  only_conv=only_conv,
                                  only_conv_t=only_conv_t)
        self.act = utils.get_activation_function(args)
        # TODO
        #  if no batch norm
        #  else:
        #  put in batch norm
        self.block = nn.Sequential(self.one_by_one_conv,
                                   self.cct_layer,
                                   self.act)

    def forward(self, inp):
        return self.block(inp)

class DenseCCTMiddle(nn.Module):
    """

    Takes as input the concatenated input states

    Does not include base layer or final layer of the full DenseCCTBlock"""

    def __init__(self, args, in_channels, growth_rate, num_layers,
                 kernel_size, only_conv=False, only_conv_t=False,):
        super().__init__()
        self.cctb_blocks = nn.ModuleList([])
        for i in range(num_layers):
            num_in_ch = in_channels + (i * growth_rate)
            cctb = CCTBlock(args,
                            in_channels=num_in_ch,
                            out_channels=growth_rate,
                            kernel_size=kernel_size,
                            only_conv=only_conv,
                            only_conv_t=only_conv_t,)
            self.cctb_blocks.append(cctb)
    def forward(self, inp):
        outs = [inp]
        for block in self.cctb_blocks:
            inps = torch.cat(outs, dim=1)
            out = block(inps)
            outs.append(out)
        out = torch.cat(outs, dim=1)
        return out


class DenseCCTBlock(nn.Module):
    """
    Takes as input the separate input states, passes them through a
    cct layer, interps then concats the outputs, passes the concatted tensor
    through an activation, then a 1x1 conv, before passing it to the main
    DenseCCTMiddle layers, if there are any, otherwise just outputs, and if
    there are any, then takes the output of the DenseCCTMiddle and passes it
    through a final 1x1 conv.

    Does not include base layer or final layer of the full DenseCCTBlock"""

    def __init__(self, args, state_layer_idx):
        super().__init__()
        self.args = args
        base_ch      = args.arch_dict['num_ch_base']
        growth_rate  = args.arch_dict['growth_rate']
        num_layers   = args.arch_dict['mod_num_lyr_dict'][state_layer_idx]
        inp_idxs     = args.arch_dict['mod_connect_dict'][state_layer_idx]
        cct_statuses = args.arch_dict['mod_cct_status_dict'][state_layer_idx]
        base_kern_pads = args.arch_dict['base_kern_pad_dict'][state_layer_idx]
        kern = args.arch_dict['main_kern_dict'][state_layer_idx]
        out_shape    = args.state_sizes[state_layer_idx]

        # Throw away info for FC nets
        inp_idxs         = args.arch_dict['mod_connect_dict'][state_layer_idx]
        inp_state_shapes = [self.args.state_sizes[j] for j in inp_idxs]
        inp_state_shapes = [ii for (ii, cct_s) in zip(inp_state_shapes, cct_statuses) if cct_s != 3]
        base_kern_pads   = [bkp for (bkp, cct_s) in zip(base_kern_pads, cct_statuses) if cct_s != 3]
        cct_statuses     = [cct_s for cct_s in cct_statuses if cct_s != 3]


        self.act = utils.get_activation_function(args)


        # Makes the base CCTLayers
        self.base_cctls = nn.ModuleList([])
        #TODO if these aren't the same length, raise an issue.
        for (cct_status, shape, kp) in zip(cct_statuses,
                                           inp_state_shapes,
                                           base_kern_pads):
            if cct_status == 1:
                only_conv = True
                only_conv_t = False
            elif cct_status == 2:
                only_conv = False
                only_conv_t = True
            else:
                only_conv = only_conv_t = False
            cctl = nn.Sequential(CCTLayer(args,
                                          in_channels=shape[1],
                                          out_channels=base_ch,
                                          kernel_size=kp[0],
                                          padding = kp[1],
                                          only_conv = only_conv,
                                          only_conv_t = only_conv_t),
                                 Interpolate(out_shape[2:],
                                             mode='nearest'),#newsince20200408
                                 self.act)
            self.base_cctls.append(cctl)

        # If the num_layers is 0, the final 1x1 conv input channel is
        # base_ch * num inputs since it just takes the concated outputs of the
        # base ccts
        # elif the num layers is greater than 0, the final 1x1 cct has
        # num_out_ch for the number of in channels, since it's taking the
        # output of DenseCCTMiddle.

        # If the cct_statuses are all 1 (or all 2), then the status input to the
        # DenseCCTMiddle is also 1 (or 2). But if there is any mixture,
        # then the layers in DenseCCTMiddle are also mixtures, even though the
        # bases get their preferences

        all_same_cct_status = all([cct_statuses[0] == cct_st
                                    for cct_st in cct_statuses])
        if all_same_cct_status:
            if cct_statuses[0] == 1:
                only_conv = True
                only_conv_t = False
            elif cct_statuses[0] == 2:
                only_conv = False
                only_conv_t = True
        else:
            only_conv = only_conv_t = False

        if num_layers > 0:
            in_channels = base_ch * len(inp_state_shapes)
            num_out_ch  = in_channels + (num_layers * growth_rate)
            dcctm = DenseCCTMiddle(args,
                                   in_channels=in_channels,
                                   growth_rate=growth_rate,
                                   num_layers=num_layers,
                                   kernel_size=kern,
                                   only_conv=only_conv,
                                   only_conv_t=only_conv_t)
            final_1x1_conv = nn.Conv2d(in_channels=num_out_ch,
                                       out_channels=out_shape[1],
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)
            self.top_net = nn.Sequential(dcctm,
                                         final_1x1_conv)
        else:
            num_out_ch = base_ch * len(inp_state_shapes)
            final_1x1_conv = nn.Conv2d(in_channels=num_out_ch,
                                       out_channels=base_ch,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)
            final_conv = CCTLayer(args,
                                  in_channels=base_ch,
                                  out_channels=out_shape[1],
                                  kernel_size=kern,
                                  only_conv=True)

            self.top_net = nn.Sequential(final_1x1_conv,
                                         final_conv)


    def forward(self, pre_state, inps):
        outs = []
        for base_cctl, inp in zip(self.base_cctls, inps):
            out = base_cctl(inp)
            outs.append(out)
        out = torch.cat(outs, dim=1)
        out = self.top_net(out)

        # quadr_out = 0.5 * torch.einsum('ba,ba->b',
        #                                out.view(out.shape[0], -1),
        #                                pre_state.view(pre_state.shape[0],-1))
        out = torch.einsum('ba,ba->ba',
                                 out.view(out.shape[0], -1),
                                 pre_state.view(pre_state.shape[0],-1))
        quadr_out = out.sum(dim=1) #almost identical outputs with some differences for unknown numerical reasons

        return quadr_out, out

class FC2(nn.Module):
    def __init__(self, args, state_layer_idx):
        super().__init__()
        self.args = args
        self.act = utils.get_activation_function(args)
        self.state_layer_idx = state_layer_idx
        self.internal_size = 64

        # Get the indices of the state_layers that will be input to this net
        # and their sizes. They may be flattened conv outputs.
        input_idxs   = self.args.arch_dict['mod_connect_dict'][state_layer_idx]
        cct_statuses = args.arch_dict['mod_cct_status_dict'][state_layer_idx]
        num_layers   = args.arch_dict['mod_num_lyr_dict'][state_layer_idx]

        self.input_idxs = [ii for (ii, cct_s) in zip(input_idxs, cct_statuses) if cct_s == 3]
        self.in_sizes = [torch.prod(torch.tensor(self.args.state_sizes[j][1:]))
                         for j in self.input_idxs]
        self.in_size = sum(self.in_sizes)
        self.out_size = torch.prod(torch.tensor(self.args.state_sizes[state_layer_idx][1:])).item()

        # Define the networks
        fc_1 = nn.Linear(self.in_size, self.internal_size)
        layers = [fc_1, self.act]
        ## if num layers is not 0, then add internal layers
        for l in range(num_layers):
            fc_i = nn.Linear(self.internal_size,
                             self.internal_size)
            layers.append(fc_i)
            layers.append(self.act)
        fc_end = nn.Linear(self.internal_size, self.out_size)
        layers.append(fc_end)
        self.net = nn.Sequential(*layers)

    def forward(self, pre_states, actv_post_states, class_id=None): #I think I might have misnamed the pre and post states.
        inputs = actv_post_states
        reshaped_inps = [inp.view(inp.shape[0], -1) for inp in inputs]
        inputs = torch.cat(reshaped_inps, dim=1)
        out = self.net(inputs)
        quadr_out = 0.5 * torch.einsum('ba,ba->b', out,
                                     pre_states.view(pre_states.shape[0], -1))
        return quadr_out, out

class ContainerFCandDenseCCTBlock(nn.Module):
    """
    Takes as input the separate input states, passes them through a
    DenseCCTBlock and FC network if both are indicated.
    """

    def __init__(self, args, state_layer_idx):
        super().__init__()
        self.args = args

        inp_idxs     = args.arch_dict['mod_connect_dict'][state_layer_idx]
        self.cct_statuses = args.arch_dict['mod_cct_status_dict'][state_layer_idx]
        if 0 in self.cct_statuses or 1 in self.cct_statuses or 2 in self.cct_statuses:
            self.densecctblock = DenseCCTBlock(args, state_layer_idx)
        else:
            self.densecctblock = None
        if 3 in self.cct_statuses:
            self.fc_net = FC2(args, state_layer_idx)
        else:
            self.fc_net = None

    def forward(self, pre_state, inps):
        #split inputs
        conv_inps = [inp for (inp, cct_s) in zip(inps, self.cct_statuses) if cct_s != 3]
        fc_inps = [inp for (inp, cct_s) in zip(inps, self.cct_statuses) if cct_s == 3]
        quadr_outs = []
        outs = []
        if self.densecctblock is not None:
            quadr_out, out = self.densecctblock(pre_state, conv_inps)
            quadr_outs.append(quadr_out)
            outs.append(out)
        if self.fc_net is not None:
            quadr_out, out = self.fc_net(pre_state, fc_inps)
            quadr_outs.append(quadr_out)
            outs.append(out)

        quadr_outs = torch.stack(quadr_outs)
        outs = torch.stack(outs)

        quadr_out = torch.sum(quadr_outs, dim=0)
        out       = torch.sum(outs, dim=0)


        #quadr_out = out.sum(dim=1) #almost identical outputs with some differences for unknown numerical reasons

        return quadr_out, out











# DAN2 above
#################################################



















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
        self.act = utils.get_activation_function(args)
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
        self.max_4dim_size = int(torch.max(torch.tensor(
            [size[2] for size in self.in_conv_sizes]).float()))
        self.max_4dim_size = [self.max_4dim_size, self.max_4dim_size]
        self.interp = Interpolate(size=self.max_4dim_size, mode='bilinear')

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
        outshape = \
            utils.conv_output_shape(self.max_4dim_size,
                             kernel_size= self.args.arch_dict['kernel_sizes'][layer_idx][0],
                             padding=self.args.arch_dict['padding'][layer_idx][0],
                             stride=self.args.arch_dict['strides'][0])
        outshape = utils.conv_output_shape(outshape,
                                             kernel_size=2,
                                             padding=0,
                                             stride=2)
        self.base_conv_outshapes.append([outshape,outshape]) #for avg pool2d
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
        out = self.energy_layer(energy_input)
        if not self.args.no_end_layer_activation:
            out = self.act(out)
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
        self.act = utils.get_activation_function(args)
        self.num_fc_channels = self.args.arch_dict['num_fc_channels']

        # Gets the indices of the state_layers that will be input to this net.
        self.input_idxs = self.args.arch_dict['mod_connect_dict'][layer_idx]

        # Size values of state_layer of this network
        self.state_layer_ch = self.args.state_sizes[layer_idx][1]
        self.state_layer_h_w = self.args.state_sizes[layer_idx][2:]

        self.interp = Interpolate(size=self.state_layer_h_w, mode='nearest')

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

        out = self.energy_conv(energy_input)
        if not self.args.no_end_layer_activation:
            out = self.act(out) #TODO put act in sequential above
        quadr_out = 0.5 * torch.einsum('ba,ba->b',
                                       out.view(
                                           int(out.shape[0]), -1),
                                       pre_states.view(
                                           int(pre_states.shape[0]), -1))
        return quadr_out, out


class ConvFCMixturetoFourDim_Type2(nn.Module):
    """
    Takes a mixture of four dimensional and two dimensional inputs and
    outputs a four dimensional output of the same shape as the current state.

    """
    def __init__(self, args, layer_idx):
        super().__init__()
        self.args = args
        self.pad_mode = 'zeros'
        self.layer_idx = layer_idx
        self.act = utils.get_activation_function(args)
        #self.num_fc_channels = self.args.arch_dict['num_fc_channels']

        # Gets the indices of the state_layers that will be input to this net.
        self.input_idxs = self.args.arch_dict['mod_connect_dict'][layer_idx]

        # Size values of state_layer of this network
        self.state_layer_ch = self.args.state_sizes[layer_idx][1]
        self.state_layer_h_w = self.args.state_sizes[layer_idx][2:]
        # self.state_layer_ttl_nrns = torch.prod(torch.tensor(
        #     self.args.state_sizes[layer_idx][1:])).item()
        self.state_layer_ttl_nrns = self.args.state_sizes[layer_idx][1:].item()

        self.interp = Interpolate(size=self.state_layer_h_w, mode='nearest')

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
        self.base_conv = spectral_norm(nn.Conv2d(
            in_channels=self.in_conv_channels,
            out_channels=self.state_layer_ch,# this used to be self.args.arch_dict['num_ch'], but I decided that it would work better architecturally if I could have several nonlinearities applied to both the 4d and 2d inputs, and this would only work by decoupling the number of internal channels in this layer from the number of channels in the other layers, because I need this to be of a manageable size for FC layers to interface with. I'll only be making such layers when the number of channels in the state layer is manageable for the FC layers, so using the num of ch in this sl seemed like a reasonable number to use.
            kernel_size=self.args.arch_dict['kernel_sizes'][layer_idx][0],
            padding=self.args.arch_dict['padding'][layer_idx][0],
            stride=self.args.arch_dict['strides'][0],
            padding_mode=self.pad_mode,
            bias=True))

        # Define base FCs (then reshape their output to something that fits a conv)
        base_fc = nn.Linear(self.in_fc_neurons, self.state_layer_ttl_nrns)
        if not self.args.no_spec_norm_reg:
            self.base_fc = spectral_norm(base_fc, bound=True)
        self.base_fc_to_4dim = nn.Sequential(base_fc,
                                             self.act,
                                             Reshape(-1,
                                                     self.state_layer_ch,
                                                     self.state_layer_h_w[0],
                                                     self.state_layer_h_w[1]))

        # self.base_actv_fc_layers = nn.ModuleList([])
        # for (in_size, out_size) in zip(self.in_fc_sizes, fc_channel_fracs):
        #     base_fc_layer = nn.Linear(in_size, out_size)
        #     if not self.args.no_spec_norm_reg:
        #         base_fc_layer = spectral_norm(base_fc_layer, bound=True)
        #     self.base_actv_fc_layers.append(nn.Sequential(
        #                                 base_fc_layer,
        #                                 self.act,
        #                                 Reshape(-1, out_size, 1, 1), # makes each neuron a 'channel'
        #                                 self.interp) # Spreads each 1D channel across the whole sheet of latent neurons
        #     )

        # Define energy convs (take output of base convs and FCs as input
        energy_conv = nn.Conv2d(
            in_channels=(self.state_layer_ch*2)+self.in_conv_channels,  # One for base conv, one for fc layers, and one for
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
        reshaped_4dim_inps = torch.cat(reshaped_4dim_inps, dim=1)
        base_conv_out = self.base_conv(reshaped_4dim_inps)

        fc_inputs = torch.cat(inps_2d, dim=1)
        fc_4dimout = self.base_fc_to_4dim(fc_inputs)
        # fc_outs = [self.base_actv_fc_layers[i](inp)
        #            for (i, inp) in enumerate(inps_2d)]
        # fc_outs = torch.cat(fc_outs, dim=1)
        energy_input = torch.cat([reshaped_4dim_inps, base_conv_out,
                                  fc_4dimout], dim=1)
        out = self.energy_conv(energy_input)
        if not self.args.no_end_layer_activation:
            out = self.act(out) #TODO put act in sequential above
        quadr_out = 0.5 * torch.einsum('ba,ba->b',
                                       out.view(
                                           int(out.shape[0]), -1),
                                       pre_states.view(
                                           int(pre_states.shape[0]), -1))
        return quadr_out, out


class ConvFCMixturetoFourDim_Type3(nn.Module):
    """
    Takes a mixture of four dimensional and two dimensional inputs and
    outputs a four dimensional output of the same shape as the current state.

    """
    def __init__(self, args, layer_idx):
        super().__init__()
        self.args = args
        self.pad_mode = 'zeros'
        self.layer_idx = layer_idx
        self.act = utils.get_activation_function(args)
        #self.num_fc_channels = self.args.arch_dict['num_fc_channels']

        # Gets the indices of the state_layers that will be input to this net.
        self.input_idxs = self.args.arch_dict['mod_connect_dict'][layer_idx]

        # Size values of state_layer of this network
        self.state_layer_ch = self.args.state_sizes[layer_idx][1]
        self.state_layer_h_w = self.args.state_sizes[layer_idx][2:]
        # self.state_layer_ttl_nrns = torch.prod(torch.tensor(
        #     self.args.state_sizes[layer_idx][1:])).item()
        self.state_layer_ttl_nrns = self.args.state_sizes[layer_idx][1:].item()

        self.interp = Interpolate(size=self.state_layer_h_w, mode='nearest')

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
        self.base_conv = spectral_norm(nn.Conv2d(
            in_channels=self.in_conv_channels,
            out_channels=self.state_layer_ch,# this used to be self.args.arch_dict['num_ch'], but I decided that it would work better architecturally if I could have several nonlinearities applied to both the 4d and 2d inputs, and this would only work by decoupling the number of internal channels in this layer from the number of channels in the other layers, because I need this to be of a manageable size for FC layers to interface with. I'll only be making such layers when the number of channels in the state layer is manageable for the FC layers, so using the num of ch in this sl seemed like a reasonable number to use.
            kernel_size=self.args.arch_dict['kernel_sizes'][layer_idx][0],
            padding=self.args.arch_dict['padding'][layer_idx][0],
            stride=self.args.arch_dict['strides'][0],
            padding_mode=self.pad_mode,
            bias=True))

        # Define base FCs (then reshape their output to something that fits a conv)
        base_fc = nn.Linear(self.in_fc_neurons, self.state_layer_ttl_nrns)
        if not self.args.no_spec_norm_reg:
            self.base_fc = spectral_norm(base_fc, bound=True)
        self.base_fc_to_4dim = nn.Sequential(base_fc,
                                             self.act,
                                             Reshape(-1,
                                                     self.state_layer_ch,
                                                     self.state_layer_h_w[0],
                                                     self.state_layer_h_w[1]))

        # self.base_actv_fc_layers = nn.ModuleList([])
        # for (in_size, out_size) in zip(self.in_fc_sizes, fc_channel_fracs):
        #     base_fc_layer = nn.Linear(in_size, out_size)
        #     if not self.args.no_spec_norm_reg:
        #         base_fc_layer = spectral_norm(base_fc_layer, bound=True)
        #     self.base_actv_fc_layers.append(nn.Sequential(
        #                                 base_fc_layer,
        #                                 self.act,
        #                                 Reshape(-1, out_size, 1, 1), # makes each neuron a 'channel'
        #                                 self.interp) # Spreads each 1D channel across the whole sheet of latent neurons
        #     )

        # Define energy convs (take output of base convs and FCs as input
        energy_conv = nn.Conv2d(
            in_channels=(self.state_layer_ch*2)+self.in_conv_channels,  # One for base conv, one for fc layers, and one for
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
        reshaped_4dim_inps = torch.cat(reshaped_4dim_inps, dim=1)
        base_conv_out = self.base_conv(reshaped_4dim_inps)

        fc_inputs = torch.cat(inps_2d, dim=1)
        fc_4dimout = self.base_fc_to_4dim(fc_inputs)
        # fc_outs = [self.base_actv_fc_layers[i](inp)
        #            for (i, inp) in enumerate(inps_2d)]
        # fc_outs = torch.cat(fc_outs, dim=1)
        energy_input = torch.cat([reshaped_4dim_inps, base_conv_out,
                                  fc_4dimout], dim=1)
        out = self.energy_conv(energy_input)
        if not self.args.no_end_layer_activation:
            out = self.act(out) #TODO put act in sequential above
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
        self.act = utils.get_activation_function(args)

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
        out = self.energy_conv(energy_input)
        if not self.args.no_end_layer_activation:
            out = self.act(out)
        quadr_out = 0.5 * torch.einsum('ba,ba->b', out.view(int(out.shape[0]), -1),
                                     pre_states.view(int(pre_states.shape[0]), -1))
        return quadr_out, out


class FCFC(nn.Module):
    def __init__(self, args, layer_idx):
        super().__init__()
        self.args = args
        self.act = utils.get_activation_function(args)
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
        if not self.args.no_end_layer_activation:
            self.energy_actv_fc_layer = nn.Sequential(
                fc1,
                self.act,
                fc2,
                self.act)
        else:
            self.energy_actv_fc_layer = nn.Sequential(
                fc1,
                self.act,
                fc2)


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

#########################################################################


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
        self.out_ch = 64

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
            out_channels=self.out_ch,
            kernel_size=self.args.arch_dict['kernel_sizes'][layer_idx][0],
            padding=self.args.arch_dict['padding'][layer_idx][0],
            stride=self.args.arch_dict['strides'][0],
            padding_mode=self.pad_mode,
            bias=True)
        if not self.args.no_spec_norm_reg:
            base_conv = spectral_norm(base_conv)
        self.base_conv = base_conv

        # Get size of base conv outputs
        self.base_conv_outshapes = []
        outshape = \
            utils.conv_output_shape(self.max_4dim_size,
                             kernel_size=self.args.arch_dict['kernel_sizes'][layer_idx][0],
                             padding=self.args.arch_dict['padding'][layer_idx][0],
                             stride=self.args.arch_dict['strides'][0])
        outshape = (outshape, outshape)

        # [utils.conv_output_shape(outshape,
        #                    kernel_size=2,
        #                    padding=0,
        #                    stride=2)]

        self.base_conv_outshapes.append(outshape) #for avg pool2d
        self.base_conv_outsizes = [torch.prod(torch.tensor(bcos))
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
        # self.act = utils.get_activation_function(args)
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
        # self.act = utils.get_activation_function(args)

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
