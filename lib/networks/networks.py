"""Based on https://github.com/rosinality/igebm-pytorch/blob/master/model.py
which is available under an MIT OSI licence."""

import torch
from torch import nn, optim
from torch.nn import functional as F
import lib.utils
import lib.custom_components.custom_swish_activation as cust_actv
from lib.custom_components import activations
import lib.networks.layers


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


        self.actvn = lib.utils.get_activation_function(args)


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


class ConvBengioFischerNetwork(nn.Module):
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
        self.pad_mode = 'zeros'
        self.num_state_layers = len(self.args.state_sizes[1:])
        self.connexions = self.args.arch_dict['mod_connect_dict']
        self.f_keys = []
        self.f_nets = nn.ModuleDict({})
        self.b_nets = nn.ModuleDict({})
        for i in self.args.arch_dict['mod_connect_dict'].keys():
            inp_idxs = self.connexions[i]
            if any([j > i for j in inp_idxs]):
                raise ValueError("In Conv BFN, the mod connect dict only " +
                                 "defines feedforward connections. At least one " +
                                 "indice here is describing a feedback " +
                                 "connection, which is not allowed.")
            if inp_idxs:
                for j_ind, j in enumerate(inp_idxs):
                    f_key = '%i--%i' % (j, i)
                    #print(f_key)
                    self.f_keys.append((f_key))

                    f_in_ch  = self.args.state_sizes[j][1]
                    f_out_ch = self.args.state_sizes[i][1]

                    f_net = nn.Conv2d(in_channels=f_in_ch,
                                      out_channels=f_out_ch,
                                      kernel_size=self.args.arch_dict[
                                          'kernel_sizes'][i][j_ind],
                                      padding=self.args.arch_dict[
                                          'padding'][i][j_ind],
                                      stride=self.args.arch_dict[
                                          'strides'][i][j_ind],
                                      padding_mode='zeros')

                    if not self.args.no_spec_norm_reg:
                        f_net = spectral_norm(f_net)
                    self.f_nets.update({f_key: f_net})

                    if self.args.no_forced_symmetry:
                        b_key = '%i--%i' % (i, j)
                        if self.args.arch_dict['strides'][i][j_ind] == 2:
                            output_padding = 1
                        else:
                            output_padding = 0
                        b_net = nn.ConvTranspose2d(in_channels=f_out_ch,
                                                   out_channels=f_in_ch,
                                                   kernel_size=self.args.arch_dict[
                                                       'kernel_sizes'][i][j_ind],
                                                   padding=self.args.arch_dict[
                                                       'padding'][i][j_ind],
                                                   output_padding=output_padding,
                                                   stride=self.args.arch_dict[
                                                       'strides'][i][j_ind]
                                                   )
                        if not self.args.no_spec_norm_reg:
                            b_net = spectral_norm(b_net)
                        self.b_nets.update({b_key: b_net})

        self.biases = nn.ModuleList([nn.Linear(torch.prod(torch.tensor(l[1:])),
                                               1, bias=False)
                       for l in args.state_sizes])
        for bias in self.biases:
            torch.nn.init.zeros_(bias.weight)

        self.actvn = lib.utils.get_activation_function(args)

    def forward(self, states, class_id=None, step=None):

        # Squared norm
        sq_nrm = sum(
            [(0.5 * (layer.view(layer.shape[0], -1) ** 2)).sum() for layer in
             states])

        # Quadratic terms
        outs = []
        labs = []
        for l in range(self.num_state_layers+1):
                outs.append([None] * (self.num_state_layers+1))
                labs.append([None] * (self.num_state_layers + 1))

        for (i, out_st) in enumerate(states):
            inp_states = [(k, states[k]) for k in self.connexions[i] if self.connexions[i]]
            for (j_ind, (j, inp_st)) in enumerate(inp_states):
                f_key = '%i--%i' % (j, i)
                f_net = self.f_nets[f_key]
                f_out = f_net(inp_st)
                if self.args.arch_dict['strides'][i][j_ind] == 2:
                    output_padding = 1
                else:
                    output_padding = 0
                if self.args.no_forced_symmetry:
                    b_key = '%i--%i' % (i, j)
                    b_net = self.b_nets[b_key]
                    b_out = b_net(out_st)
                else:
                    b_out = nn.functional.conv_transpose2d(out_st,
                                                           weight=f_net.weight.transpose(2,3), #TODO check if it should be transposed at all.
                                                           padding=self.args.arch_dict['padding'][i][j_ind],
                                                           output_padding=output_padding,
                                                           stride=self.args.arch_dict['strides'][i][j_ind])
                labs[j][i] = '%i--%i' % (j, i)
                labs[i][j] = '%i--%i' % (i, j)

                outs[j][i] = f_out
                outs[i][j] = b_out

        # Transpose list of lists
        outs = list(map(list, zip(*outs)))
        labs = list(map(list, zip(*labs)))

        # Remove empty entries in list of list
        outs = [[o for o in out if o is not None] for out in outs]
        #print([shapes(out) for out in outs])
        outs = [torch.stack(out) for out in outs]
        outs = [torch.sum(out, dim=0) for out in outs]

        quadratic_terms = - sum([0.5 * sum(torch.einsum('ba,ba->b',
                                                        w_inps.view(state.shape[0], -1),
                                                        self.actvn(   state.view(state.shape[0], -1))))
                                 for w_inps, state in
                                 zip(outs, states)])

        # Linear terms
        linear_terms = - sum([bias(self.actvn(layer.view(layer.shape[0], -1))).sum()
                              for layer, bias in
                              zip(states, self.biases)]) #TODO init that uses biases


        return sq_nrm + linear_terms + quadratic_terms, outs

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

        self.actvn = lib.utils.get_activation_function(args)

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
                    net = ConvFCMixturetoFourDim_Type2(self.args, i) #LEE this has changed
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

        self.actvn = lib.utils.get_activation_function(args)

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
            #quadr_out = self.args.energy_weight_mask[i] * quadr_out
            quadr_outs.append(quadr_out)
            outs.append(out)

        quadratic_terms = - sum(sum(quadr_outs))  # Note the minus here

        # Get the final energy
        energy = sq_nrm + lin_terms + quadratic_terms

        return energy, outs



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

        self.actvn = lib.utils.get_activation_function(args)

    def forward(self, states, class_id=None, step=None):

        # # Squared norm
        # sq_nrm = sum([(0.5 * (layer.view(layer.shape[0], -1) ** 2)).sum() for layer in states])

        # # Linear terms
        linear_terms = sum([bias(self.state_actv(layer.view(layer.shape[0], -1))).sum()
                              for layer, bias in
                              zip(states, self.biases)])

        # Quadratic terms
        enrgs = []
        outs = []
        for i, (pre_state, net) in enumerate(zip(states, self.quadratic_nets)):
            post_inp_idxs = self.args.arch_dict['mod_connect_dict'][i]
            pos_inp_states = [states[j] for j in post_inp_idxs]
            # pos_inp_states = [self.state_actv(state)
            #                   for state in pos_inp_states]
            enrg, out = net(pre_state, pos_inp_states)
            enrg = self.args.energy_weight_mask[i] * enrg
            enrgs.append(enrg)
            outs.append(out)

        quadratic_terms = sum(sum(enrgs))
        energy = quadratic_terms + linear_terms

        return energy, outs

class VFEBMLV(nn.Module):
    """Defines the vector field studied by Scellier et al. (2018)

        Uses same archi set as VF but implements it as an EBMLV

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

        self.actvn = lib.utils.get_activation_function(args)

    def forward(self, states, class_id=None, step=None):

        # Squared norm
        #sq_nrm = sum([(0.5 * (layer.view(layer.shape[0], -1) ** 2)).sum() for layer in states])

        #Linear terms #TODO IF YOU RETRUN TO THIS, PUT LIN/UNARY TERMS BACK IN BECAUSE OTHER MRFs have them
        linear_terms = sum([bias(self.actvn(layer.view(layer.shape[0], -1))).sum()
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

        quadratic_terms = sum(sum(quadr_outs))

        energy = quadratic_terms + linear_terms

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

        self.actvn = lib.utils.get_activation_function(args)

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

        quadratic_terms = - sum(sum(quadr_outs))  # Note no the minus here

        # Get the final energy
        energy = sq_nrm + quadratic_terms #+ lin_terms

        return energy, outs



###### Function to help me debug
shapes = lambda x : [y.shape for y in x]
nancheck = lambda x : (x != x).any()
listout = lambda y : [x for x in y]





