"""Based on https://github.com/rosinality/igebm-pytorch/blob/master/model.py
which is available under an MIT OSI licence."""

import torch
from torch import nn, optim
from torch.nn import functional as F
import lib.utils as utils
import lib.custom_components.custom_swish_activation as cust_actv


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



class SSNBlock(nn.Module):
    """
    Takes as input the separate input states, passes them through a
    set of conv layers corresponding to the EE-EI-IE-II connections.

    If the mod connect dict says there is a connection in a certain direction,
    it means that the E block of the input is fed as input to the conv nets of
    the I and the E of the current block.

    Consists of
    A conv net that connect EE networks within and between state layers
    A conv net that connect EI networks within and between state layers
    A conv net that connects IE networks within a state layer
    A conv net that connect II networks within a state layer

    We're going to stick with the EI constraints for layer 0 because we
    want to be able to say that the excitatory connections encode positive
    correlations and the inhibitory connections the opposite. It would be
    weird to have differing roles for the correlations between latent variables
    and visible variables.

    """

    def __init__(self, args, state_layer_idx):
        super().__init__()
        self.args = args
        self.state_layer_idx = state_layer_idx
        state_size = args.state_sizes[state_layer_idx]
        state_ch = state_size[1]
        state_hw = state_size[2]
        self.inp_idxs      = args.arch_dict['mod_connect_dict'][state_layer_idx]
        exc_kern_pads = args.arch_dict['exc_kern_pad_dict'][state_layer_idx]
        inh_kern_pads = args.arch_dict['inh_kern_pad_dict'][state_layer_idx]

        inp_sizes = [args.state_sizes[i] for i in self.inp_idxs]

        self.convs = torch.nn.ModuleDict({})

        self.supralinear_act = lambda x: torch.tensor(args.supra_k,
                                                      device=self.args.device) * \
                                         (torch.nn.ReLU()(x)**torch.tensor(args.supra_n,
                                                      device=self.args.device))

        # Special case for image layer (state layer 0) that has inhibitory
        # projections
        if state_layer_idx == 0:
            for i, inp_size in enumerate(inp_sizes):
                ch_inp = inp_size[1]
                hw_inp = inp_size[2]

                # Make excitatory conv net
                e_kern, e_pad = exc_kern_pads[i]
                e_conv = nn.Conv2d(in_channels=ch_inp,
                                  out_channels=state_ch,
                                  kernel_size=e_kern,
                                  stride=1,
                                  padding=e_pad,
                                  padding_mode='zeros')
                ## init exc with gamma distrib
                gamma_alpha = torch.ones_like(
                    e_conv.weight.data) * args.weights_gamma_alpha
                gamma_beta = torch.ones_like(
                    e_conv.weight.data) * args.weights_gamma_beta
                gamma_init = torch.distributions.gamma.Gamma(gamma_alpha,
                                                             gamma_beta)
                e_conv.weight.data = torch.nn.Parameter(gamma_init.sample().to(self.args.device)) * torch.tensor(args.weights_mag_scale, device=self.args.device)
                e_conv.weight.data = e_conv.weight.data / torch.sqrt(torch.prod(torch.tensor(e_conv.weight.data.shape)).float())

                # Make inhibitory conv net
                i_kern, i_pad = inh_kern_pads[i]
                i_conv = nn.Conv2d(in_channels=ch_inp,
                                  out_channels=state_ch,
                                  kernel_size=i_kern,
                                  stride=1,
                                  padding=i_pad,
                                  padding_mode='zeros')
                ## init inh with gamma distrib
                gamma_alpha = torch.ones_like(
                    i_conv.weight.data) * args.weights_gamma_alpha
                gamma_beta = torch.ones_like(
                    i_conv.weight.data) * args.weights_gamma_beta
                gamma_init = torch.distributions.gamma.Gamma(gamma_alpha,
                                                             gamma_beta)
                i_conv.weight.data = torch.nn.Parameter(-gamma_init.sample().to(self.args.device)) * torch.tensor(args.weights_mag_scale, device=self.args.device)
                i_conv.weight.data = i_conv.weight.data / torch.sqrt(torch.prod(torch.tensor(i_conv.weight.data.shape)).float())


                # Make labels for nets
                i_label = 'I_%i_%i' % (self.inp_idxs[i], state_layer_idx)
                e_label = 'E_%i_%i' % (self.inp_idxs[i], state_layer_idx)

                ## if different size to current statelayer, resize output
                if state_hw != hw_inp: # TODO experiment with flipping the order of the interp and the conv and see which is faster and which produces better results. I predict that interping the output will be faster
                    max_hw = int(max([state_hw, hw_inp]))
                    interp = Interpolate(size=[max_hw,max_hw],
                                         mode='nearest')
                    e_conv = nn.Sequential(e_conv,
                                           interp)
                    i_conv = nn.Sequential(i_conv,
                                           interp)


                # Add conv nets to moduledict
                self.convs.update({e_label: e_conv,
                                   i_label: i_conv})
        elif state_layer_idx >= 1:
            #TODO change so approp for normal inter and intra connections
            for i, inp_size in enumerate(inp_sizes):
                ch_inp = inp_size[1]
                hw_inp = inp_size[2]

                # Make EE conv net
                e_kern, e_pad = exc_kern_pads[i]
                ee_conv = nn.Conv2d(in_channels=ch_inp,
                                  out_channels=state_ch,
                                  kernel_size=e_kern,
                                  stride=1,
                                  padding=e_pad,
                                  padding_mode='zeros',
                                  bias=False)
                ## init exc with gamma distrib
                gamma_alpha = torch.ones_like(
                    ee_conv.weight.data) * args.weights_gamma_alpha
                gamma_beta = torch.ones_like(
                    ee_conv.weight.data) * args.weights_gamma_beta
                gamma_init = torch.distributions.gamma.Gamma(gamma_alpha,
                                                             gamma_beta)
                ee_conv.weight.data = torch.nn.Parameter(gamma_init.sample().to(self.args.device)) * torch.tensor(args.weights_mag_scale, device=self.args.device)
                ee_conv.weight.data = ee_conv.weight.data / torch.sqrt(torch.prod(torch.tensor(ee_conv.weight.data.shape)).float())

                # Make EI conv net
                ei_kern, ei_pad = exc_kern_pads[i]
                ei_conv = nn.Conv2d(in_channels=ch_inp,
                                  out_channels=state_ch,
                                  kernel_size=ei_kern,
                                  stride=1,
                                  padding=ei_pad,
                                  padding_mode='zeros',
                                  bias=False)
                ## init EI with gamma distrib
                gamma_alpha = torch.ones_like(
                    ei_conv.weight.data) * args.weights_gamma_alpha
                gamma_beta = torch.ones_like(
                    ei_conv.weight.data) * args.weights_gamma_beta
                gamma_init = torch.distributions.gamma.Gamma(gamma_alpha,
                                                             gamma_beta)
                ei_conv.weight.data = torch.nn.Parameter(gamma_init.sample().to(self.args.device)) * torch.tensor(args.weights_mag_scale, device=self.args.device)
                ei_conv.weight.data = ei_conv.weight.data / torch.sqrt(torch.prod(torch.tensor(ei_conv.weight.data.shape)).float())

                # Make labels for exc nets
                ei_label = 'EI_%i_%i' % (self.inp_idxs[i], state_layer_idx)
                ee_label = 'EE_%i_%i' % (self.inp_idxs[i], state_layer_idx)


                # If different size to current statelayer, resize output
                if state_hw != hw_inp: # TODO experiment with flipping the order of the interp and the conv and see which is faster and which produces better results. I predict that interping the output will be faster
                    # max_hw = int(max([state_hw, hw_inp]))
                    interp = Interpolate(size=[state_hw, state_hw],
                                         mode='nearest')
                    ee_conv = nn.Sequential(ee_conv,
                                           interp)
                    ei_conv = nn.Sequential(ei_conv,
                                           interp)


                if state_layer_idx == self.inp_idxs[i]:  # i.e. if self connection

                    # Make IE conv net
                    ie_kern, ie_pad = inh_kern_pads[i]
                    ie_conv = nn.Conv2d(in_channels=ch_inp,
                                        out_channels=state_ch,
                                        kernel_size=ie_kern,
                                        stride=1,
                                        padding=ie_pad,
                                        padding_mode='zeros',
                                        bias=False)
                    ## init IE with gamma distrib
                    gamma_alpha = torch.ones_like(
                        ie_conv.weight.data) * args.weights_gamma_alpha
                    gamma_beta = torch.ones_like(
                        ie_conv.weight.data) * args.weights_gamma_beta
                    gamma_init = torch.distributions.gamma.Gamma(gamma_alpha,
                                                                 gamma_beta)
                    ie_conv.weight.data = torch.nn.Parameter(
                        -gamma_init.sample().to(
                            self.args.device)) * torch.tensor(
                        args.weights_mag_scale, device=self.args.device)
                    ie_conv.weight.data = ie_conv.weight.data / torch.sqrt(torch.prod(
                        torch.tensor(ie_conv.weight.data.shape)).float())

                    # Make II conv net
                    ii_kern, ii_pad = inh_kern_pads[i]
                    ii_conv = nn.Conv2d(in_channels=ch_inp,
                                        out_channels=state_ch,
                                        kernel_size=ii_kern,
                                        stride=1,
                                        padding=ii_pad,
                                        padding_mode='zeros',
                                        bias=False)
                    ## init II with gamma distrib
                    gamma_alpha = torch.ones_like(
                        ii_conv.weight.data) * args.weights_gamma_alpha
                    gamma_beta = torch.ones_like(
                        ii_conv.weight.data) * args.weights_gamma_beta
                    gamma_init = torch.distributions.gamma.Gamma(gamma_alpha,
                                                                 gamma_beta)
                    ii_conv.weight.data = torch.nn.Parameter(
                        -gamma_init.sample().to(
                            self.args.device)) * torch.tensor(
                        args.weights_mag_scale, device=self.args.device)
                    ii_conv.weight.data = ii_conv.weight.data / torch.sqrt(torch.prod(
                        torch.tensor(ii_conv.weight.data.shape)).float())

                    # Make labels for exc nets
                    ie_label = 'IE_%i_%i' % (self.inp_idxs[i], state_layer_idx)
                    ii_label = 'II_%i_%i' % (self.inp_idxs[i], state_layer_idx)

                    # If different size to current statelayer, resize output
                    if state_hw != hw_inp:  # TODO experiment with flipping the order of the interp and the conv and see which is faster and which produces better results. I predict that interping the output will be faster
                        # max_hw = int(max([state_hw, hw_inp]))
                        interp = Interpolate(size=[state_hw, state_hw],
                                             mode='nearest')
                        ie_conv = nn.Sequential(ie_conv,
                                               interp)
                        ii_conv = nn.Sequential(ii_conv,
                                               interp)

                    # Remove self connecting params
                    utils.remove_autapse_params(args, ii_conv.weight.data)
                    utils.remove_autapse_params(args, ee_conv.weight.data)

                    # Add inh conv nets to moduledict
                    self.convs.update({ie_label: ie_conv,
                                       ii_label: ii_conv})
                # Add exc conv nets to moduledict
                self.convs.update({ee_label: ee_conv,
                                   ei_label: ei_conv})


    def forward(self, layer_states, inps):

        if self.state_layer_idx == 0:
            layer_state = layer_states[0]
            full_pre_quad_outs = []
            for j, (e_inp, i_inp) in enumerate(inps):
                i_label = 'I_%i_%i' % (self.inp_idxs[j], self.state_layer_idx)
                e_label = 'E_%i_%i' % (self.inp_idxs[j], self.state_layer_idx)
                e_out = self.convs[e_label](self.supralinear_act(e_inp))
                i_out = self.convs[i_label](self.supralinear_act(i_inp)) #LEE used to not have supralinear act here. Added so that the weights feeding onto image would be adequate
                full_pre_quad_outs.append(e_out)
                full_pre_quad_outs.append(i_out)

            full_pre_quad_outs = torch.stack(full_pre_quad_outs)
            full_pre_quad_out = torch.sum(full_pre_quad_outs, dim=0)

            # Make the prequad terms into quadratic terms
            full_quad_out = torch.mul(full_pre_quad_out,  # elementwise mult
                                      layer_state)

            # Reshape to size of state layer
            full_quad_outs = [full_quad_out.view(layer_state.shape)]
            full_pre_quad_outs = [full_pre_quad_out.view(layer_state.shape)]

        else: #TODO change references to presyn to post syn (note that not all pre uses are presyn here)
            full_pre_quad_outs = [[],[]]
            full_quad_outs = [[],[]]

            # For each input to this statelayer's networks,
            # put the input into the conv net
            for j, (e_inp, i_inp) in enumerate(inps):
                ee_label = 'EE_%i_%i' % (self.inp_idxs[j], self.state_layer_idx)
                ei_label = 'EI_%i_%i' % (self.inp_idxs[j], self.state_layer_idx)

                ee_out = self.convs[ee_label](self.supralinear_act(e_inp))
                ei_out = self.convs[ei_label](self.supralinear_act(e_inp))

                full_pre_quad_outs[0].append(ee_out)
                full_pre_quad_outs[1].append(ei_out)

                # If the input is the current statelayer, then do
                # what inhibitory connection it has. There aren't inter-layer
                # inhibitory connections, only intra-layer.
                if self.state_layer_idx == self.inp_idxs[j]:
                    ie_label = 'IE_%i_%i' % (self.inp_idxs[j], self.state_layer_idx)
                    ii_label = 'II_%i_%i' % (self.inp_idxs[j], self.state_layer_idx)

                    ie_out = self.convs[ie_label](self.supralinear_act(i_inp))
                    ii_out = self.convs[ii_label](self.supralinear_act(i_inp))

                    full_pre_quad_outs[0].append(ie_out)
                    full_pre_quad_outs[1].append(ii_out)
                    # print('boop')

            # For E then I states: stack, sum, multiply with presyn state
            for k, nrn_type_block in enumerate(full_pre_quad_outs):
                block = torch.stack(nrn_type_block)
                full_pre_quad_outs[k] = torch.sum(block, dim=0)

                # Make the prequad terms into quadratic terms
                full_quad_outs[k] = torch.mul(full_pre_quad_outs[k],
                                           # elementwise mult
                                           layer_states[k])#.view(layer_states[k].shape[0],-1))

                # Reshape to size of state layer
                full_quad_outs[k] = full_quad_outs[k].view(layer_states[k].shape)
                full_pre_quad_outs[k] = full_pre_quad_outs[k].view(layer_states[k].shape)

        return full_quad_outs, full_pre_quad_outs

