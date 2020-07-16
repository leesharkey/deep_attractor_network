import torch
from torch import nn
import lib.utils
from lib.networks.layers import *


class BaseModel(nn.Module):
    def __init__(self, args, device, model_name, writer, n_class=None):
        super().__init__()
        self.args = args
        self.device = device
        self.writer = writer
        self.model_name = model_name
        self.pad_mode = 'zeros'
        self.num_state_layers = len(self.args.state_sizes[1:])
        self.st_act = \
            lib.utils.get_activation_function(args, states_activation=True)

        # Initialize biases
        self.biases = nn.ModuleList([
            nn.Linear(torch.prod(torch.tensor(l[1:])), 1, bias=False)
                       for l in args.state_sizes])
        for bias in self.biases:
            torch.nn.init.zeros_(bias.weight)



class DeepAttractorNetworkTakeTwo(BaseModel):
    """Defines the Deep Attractor Network of the second kind(Sharkey 2019) #TODO change doc strings
    # TODO change name of DAN to NRF

    The network is a generalisation of the vector field network used in
    Scellier et al. (2018) relaxation of the continuous Hopfield-like network
    (CHN) studied by Bengio and Fischer (2015) and later in
    Equilibrium Propagation (Scellier et al. 2017) and other works. It
    no longer required symmetric weights as in the CHN.
    """
    def __init__(self, args, device, model_name, writer, n_class=None):
        super().__init__(args, device, model_name, writer)
        self.args = args
        self.device = device
        self.writer = writer
        self.model_name = model_name
        self.num_state_layers = len(self.args.state_sizes[1:])

        # Define the networks that output the quadratic terms
        self.quadratic_nets = nn.ModuleList([])
        for i in range(len(self.args.state_sizes)):
            net = ContainerFCandDenseCCTBlock(
                    args, i, weight_norm=self.args.model_weight_norm)
            self.quadratic_nets.append(net)

        if self.args.ff_dynamics:
            self.forward = self.forward_ff_dynamics

            # Define new biases that are just blocks of params
            # self.biases = nn.ParameterList([])
            # for state_size in self.args.state_sizes:
            #     b = nn.Parameter(torch.randn(state_size[1:]) * 1e-9)
            #     self.biases.append(b)
            # for bias in self.biases:
            #     torch.nn.init.zeros_(bias.weight)

            # Define the ranges that should be masked by the 'gradient' of the
            # states_activation
            if self.args.states_activation == 'hardsig':
                self.mask_func = lambda state: torch.where((state < 1.), #& \
                                                      #(state > 0.),
                                                      torch.ones_like(state),
                                                      torch.zeros_like(state))
            elif self.args.states_activation == 'hardtanh':
                self.mask_func = lambda state: torch.where((state > -1.) & \
                                                      (state < 1.),
                                                      torch.ones_like(state),
                                                      torch.zeros_like(state))
            elif self.args.states_activation == 'relu':
                self.mask_func = lambda state: torch.where((state > 0.),
                                                      torch.ones_like(state),
                                                      torch.zeros_like(state))

        else:
            self.forward = self.forward_bp_dynamics

    #noBP dynamics:
    # a function that if args.ff_dynamics, sets self.forward as
    # forward_ff_dynamics and if not then sets is as
    # calculate_energy (what it is currently)

    # Also to use calculate_energy during ff_dynamics, but only for weights
    # update or if I want to know the energy during the run

    def forward_ff_dynamics(self, states, energy_calc=False):
        """This calculates the gradient of the energy function without
           actually using backprop.

           Should be much faster. The main difference with the BP dynamics
           is that it will exclude a term, since the gradient of E wrt s_i
           is some function of several {s_j}s that includes s_i.

           The equation it should calculate is
           $\frac{ds_i}{dt}=-s_i+\rho'(s_i)(f_\theta(s)+b_i)"""

        grads = []
        energy = torch.tensor(0)
        energies = []

        for i, (state, net, bias) in enumerate(zip(states,
                                                   self.quadratic_nets,
                                                   self.biases)):

            # Get the right input states
            post_inp_idxs = self.args.arch_dict['mod_connect_dict'][i]
            pos_inp_states = [states[j] for j in post_inp_idxs]
            pos_inp_states = [self.st_act(state)
                              for state in pos_inp_states]


            _, full_pre_quadr_out = net(state, pos_inp_states)
            full_pre_quadr_out = 0.5 * full_pre_quadr_out

            network_terms = full_pre_quadr_out + bias.weight.view(full_pre_quadr_out.shape[1:])
            #network_terms = full_pre_quadr_out + bias

            # Apply the mask that acts as the \rho gradient
            # mask = self.mask_func(state).byte()
            # grad = -state + torch.where(mask,
            #                             network_terms,
            #                             torch.zeros_like(network_terms))
            # This doesn't work because they don't actually use the grad of rho
            # because it causes dead neurons, like I found with the mask.
            # It works if you just add the updates and clip the values.
            grad = -state + network_terms
            grads.append(grad)

            if energy_calc:
                full_sq_term = 0.5 * (state.view(state.shape[0], -1) ** 2)
                quad_bias_terms = \
                    torch.mul(network_terms.view(state.shape[0], -1),
                              state.view(state.shape[0], -1))
                energy = full_sq_term - quad_bias_terms
                energies.append(energy)


        if energy_calc:
            full_energies = energies
            energy = sum([e.sum() for e in energies])

            print("Energy: %f" % energy.item())
        else:
            full_energies = [torch.zeros_like(state) for state in states]

        return -energy, grads, full_energies



    def forward_bp_dynamics(self, states, energy_calc=True):

        # Takes extra effort to calculate energy of each nrn individually
        # Squared norm
        sq_terms = []
        full_sq_terms = []
        for i, state in enumerate(states):
            full_sq_term = 0.5 * (state.view(state.shape[0], -1) ** 2)
            sq_term = full_sq_term.sum()

            full_sq_terms.append(full_sq_term)
            sq_terms.append(sq_term)
        sq_nrm = sum(sq_terms)
        #print("sq: " + str(sq_nrm==sum([st.sum() for st in full_sq_terms])))

        # Linear terms
        lin_terms = []
        full_lin_terms = []
        for i, (layer, bias) in enumerate(zip(states, self.biases)):
            full_lin_term = torch.mul(bias.weight, layer.view(layer.shape[0], -1))
            full_lin_term = -full_lin_term
            full_lin_terms.append(full_lin_term)
            lin_terms.append(full_lin_term.sum())
        lin_term = sum(lin_terms)
        #print("lt: " + str(lin_term==sum([lt.sum() for lt in full_lin_terms])))

        # Quadratic terms
        full_quadr_outs = []
        full_pre_quadr_outs = []
        for i, (pre_state, net) in enumerate(
                zip(states, self.quadratic_nets)):
            post_inp_idxs = self.args.arch_dict['mod_connect_dict'][i]
            pos_inp_states = [states[j] for j in post_inp_idxs]
            pos_inp_states = [self.st_act(state)
                              for state in pos_inp_states]
            full_quadr_out, full_pre_quadr_out = net(pre_state, pos_inp_states)
            full_quadr_out     = -full_quadr_out #minus
            full_pre_quadr_out = -full_pre_quadr_out
            full_quadr_outs.append(full_quadr_out)
            full_pre_quadr_outs.append(full_pre_quadr_out)

        quadratic_term = sum([fqo.sum() for fqo in full_quadr_outs])

        #print("qt: " + str(quadratic_term == sum([qt.sum() for qt in full_quadr_outs])))

        # Get the final energy
        energy = sq_nrm + lin_term + quadratic_term

        #Full version
        full_energies = [sq + qt.view(qt.shape[0],-1) + lt
                         for sq, qt, lt in zip(full_sq_terms,
                                               full_quadr_outs,
                                               full_lin_terms)]

        return energy, full_pre_quadr_outs, full_energies




























class BengioFischerNetwork(BaseModel):
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
        super().__init__(args, device, model_name, writer)
        self.args = args
        self.device = device
        self.writer = writer
        self.model_name = model_name
        self.num_state_layers = len(self.args.state_sizes[1:])

        self.weights = nn.ModuleList(
            [nn.Linear(torch.prod(torch.tensor(l1[1:])),
                       torch.prod(torch.tensor(l2[1:])), bias=False)
             for l1, l2 in zip(args.state_sizes[:-1], args.state_sizes[1:])])

        self.st_act = lib.utils.get_activation_function(args)


    def forward(self, states, class_id=None, step=None):

        # Squared norm
        sq_nrm = sum([(0.5 * (layer.view(layer.shape[0], -1) ** 2)).sum() for layer in states])

        # Linear terms
        linear_terms = - sum([bias(self.st_act(layer.view(layer.shape[0], -1))).sum()
                              for layer, bias in
                              zip(states, self.biases)])

        quadratic_terms = - sum([0.5 * sum(torch.einsum('ba,ba->b',
                                                        W(self.st_act(pre.view(pre.shape[0], -1))),
                                                        self.st_act(post.view(post.shape[0], -1))))
                                 for pre, W, post in
                                 zip(states[:-1], self.weights, states[1:])])
        return sq_nrm + linear_terms + quadratic_terms, None


class ConvBengioFischerNetwork(BaseModel):
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
        super().__init__(args, device, model_name, writer)

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

                    if self.args.arch_dict['spec_norm_reg']:
                        f_net = spectral_norm(f_net)
                    self.f_nets.update({f_key: f_net})

                    if self.args.no_forced_symmetry:
                        b_key = '%i--%i' % (i, j)
                        if self.args.arch_dict['strides'][i][j_ind] == 2:
                            output_padding = 1
                        else:
                            output_padding = 0
                        resize = Interpolate(size=self.args.state_sizes[j][2],
                                             mode='nearest')
                        back_conv = nn.Conv2d(in_channels=f_out_ch,
                                           out_channels=f_in_ch,
                                           kernel_size=self.args.arch_dict[
                                               'kernel_sizes'][i][j_ind],
                                           padding=self.args.arch_dict[
                                               'padding'][i][j_ind],
                                           stride=self.args.arch_dict[
                                               'strides'][i][j_ind])
                        if self.args.arch_dict['spec_norm_reg']:
                            back_conv = spectral_norm(back_conv)
                        b_net = nn.Sequential(resize,
                                              back_conv)
                        # b_net = nn.ConvTranspose2d(in_channels=f_out_ch,
                        #                            out_channels=f_in_ch,
                        #                            kernel_size=self.args.arch_dict[
                        #                                'kernel_sizes'][i][j_ind],
                        #                            padding=self.args.arch_dict[
                        #                                'padding'][i][j_ind],
                        #                            output_padding=output_padding,
                        #                            stride=self.args.arch_dict[
                        #                                'strides'][i][j_ind])

                        self.b_nets.update({b_key: b_net})

    def forward(self, states, class_id=None, step=None):

        # Squared norm
        sq_nrm = sum([(0.5 * (layer.view(layer.shape[0], -1) ** 2)).sum()
                      for layer in states])

        # Quadratic terms
        outs = []
        labs = []
        for l in range(self.num_state_layers+1):
                outs.append([None] * (self.num_state_layers + 1))
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
                                                           weight=f_net.weight.transpose(2,3),
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
        outs = [torch.stack(out) for out in outs]
        outs = [torch.sum(out, dim=0) for out in outs]

        quadratic_terms = - sum([0.5 * sum(torch.einsum('ba,ba->b',
                                                        w_inps.view(state.shape[0], -1),
                                                        self.st_act(   state.view(state.shape[0], -1))))
                                 for w_inps, state in
                                 zip(outs, states)])

        # Linear terms
        linear_terms = - sum([bias(self.st_act(layer.view(layer.shape[0], -1))).sum()
                              for layer, bias in
                              zip(states, self.biases)])

        return sq_nrm + linear_terms + quadratic_terms, outs


class VectorFieldNetwork(BaseModel):
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
        super().__init__(args, device, model_name, writer)
        self.args = args
        self.device = device
        self.writer = writer
        self.model_name = model_name
        self.num_state_layers = len(self.args.state_sizes[1:])

        self.quadratic_nets = nn.ModuleList([
            LinearLayer(args, i) for i in range(len(self.args.state_sizes))
        ])

    def forward(self, states, class_id=None, step=None):

        # Squared norm
        sq_nrm = sum([(0.5 * (layer.view(layer.shape[0], -1) ** 2)).sum() for layer in states])

        # Linear terms
        linear_terms = - sum([bias(self.st_act(layer.view(layer.shape[0], -1))).sum()
                              for layer, bias in
                              zip(states, self.biases)])

        # Quadratic terms
        quadr_outs = []
        outs = []
        for i, (pre_state, net) in enumerate(zip(states, self.quadratic_nets)):
            post_inp_idxs = self.args.arch_dict['mod_connect_dict'][i]
            pos_inp_states = [states[j] for j in post_inp_idxs]
            quadr_out, out = net(pre_state, pos_inp_states)
            quadr_outs.append(quadr_out)
            outs.append(out)

        quadratic_terms = - sum(sum(quadr_outs))

        energy = sq_nrm + linear_terms + quadratic_terms

        return energy, outs


class DeepAttractorNetwork(BaseModel):
    """Defines the Deep Attractor Network (Sharkey 2019)

    The network is a generalisation of the vector field network used in
    Scellier et al. (2018) relaxation of the continuous Hopfield-like network
    (CHN) studied by Bengio and Fischer (2015) and later in
    Equilibrium Propagation (Scellier et al. 2017) and other works. It
    no longer required symmetric weights as in the CHN.
    """
    def __init__(self, args, device, model_name, writer, n_class=None):
        super().__init__(args, device, model_name, writer)
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
                    net = ConvFCMixturetoFourDim_Type2(self.args, i) #N.b. II
            elif len(size) == 2:
                if all_same_dim:
                    net = FCFC(self.args, i)
                else:
                    net = ConvFCMixturetoTwoDim(self.args, i)
            self.quadratic_nets.append(net)


    def forward(self, states, class_id=None, step=None):

        # Squared norm
        sq_terms = []
        for i, layer in enumerate(states):
            sq_term = 0.5 * (layer.view(layer.shape[0], -1) ** 2).sum()
            sq_terms.append(sq_term)
        sq_nrm = sum(sq_terms)

        # Linear terms
        lin_terms = []
        for i, (layer, bias) in enumerate(zip(states, self.biases)):
            lin_term = bias(self.st_act(layer.view(layer.shape[0], -1)))
            lin_term = lin_term.sum()
            lin_terms.append(lin_term)
        lin_terms = - sum(lin_terms)


        # Quadratic terms
        quadr_outs = []
        outs = []
        for i, (pre_state, net) in enumerate(zip(states, self.quadratic_nets)):
            post_inp_idxs = self.args.arch_dict['mod_connect_dict'][i]
            pos_inp_states = [states[j] for j in post_inp_idxs]
            pos_inp_states = [self.st_act(state)
                              for state in pos_inp_states]
            quadr_out, out = net(pre_state, pos_inp_states)
            quadr_outs.append(quadr_out)
            outs.append(out)

        quadratic_terms = - sum(sum(quadr_outs))  # Note the minus here

        # Get the final energy
        energy = sq_nrm + lin_terms + quadratic_terms

        return energy, outs



class FactorHyperGraph(BaseModel):
    """Defines the Factor Hypergraph
    # TODO change name of DAN to NRF

    The network is a generalisation of the vector field network used in
    Scellier et al. (2018) relaxation of the continuous Hopfield-like network
    (CHN) studied by Bengio and Fischer (2015) and later in
    Equilibrium Propagation (Scellier et al. 2017) and other works. It
    no longer required symmetric weights as in the CHN.
    """
    def __init__(self, args, device, model_name, writer, n_class=None):
        super().__init__(args, device, model_name, writer)
        self.args = args
        self.device = device
        self.writer = writer
        self.model_name = model_name
        self.num_state_layers = len(self.args.state_sizes[1:])

        # Define the networks that output the cubic terms
        self.cubic_nets = nn.ModuleList([])
        for i in range(len(self.args.state_sizes)):
            net = CubicContainerFCandDenseCCTBlock(
                    args, i, weight_norm=self.args.model_weight_norm)
            self.cubic_nets.append(net)

    def forward(self, states, class_id=None, step=None):

        # Takes extra effort to calculate energy of each nrn individually
        # Squared norm
        sq_terms = []
        full_sq_terms = []
        for i, state in enumerate(states):
            full_sq_term = 0.5 * (state.view(state.shape[0], -1) ** 2)
            sq_term = full_sq_term.sum()

            full_sq_terms.append(full_sq_term)
            sq_terms.append(sq_term)
        sq_nrm = sum(sq_terms)
        #print("sq: " + str(sq_nrm==sum([st.sum() for st in full_sq_terms])))

        # Linear terms
        lin_terms = []
        full_lin_terms = []
        for i, (layer, bias) in enumerate(zip(states, self.biases)):
            full_lin_term = torch.mul(bias.weight, layer.view(layer.shape[0], -1))
            full_lin_term = -full_lin_term
            full_lin_terms.append(full_lin_term)
            lin_terms.append(full_lin_term.sum())
        lin_term = sum(lin_terms)
        #print("lt: " + str(lin_term==sum([lt.sum() for lt in full_lin_terms])))

        # Quadratic terms
        full_cubic_outs = []
        full_pre_cubic_outs = []
        for i, (pre_state, net) in enumerate(
                zip(states, self.cubic_nets)):

            # Input states for quadratic term
            post_inp_idxs = self.args.arch_dict['mod_connect_dict'][i]
            pos_inp_states = [states[j] for j in post_inp_idxs]
            pos_inp_states = [self.st_act(state)
                              for state in pos_inp_states]

            # Input states for cubic term
            cubic_post_inp_idxs = self.args.arch_dict['cubic_mod_connect_dict'][i]
            cubic_pos_inp_states = [states[j] for j in cubic_post_inp_idxs]
            cubic_pos_inp_states = [self.st_act(state)
                              for state in cubic_pos_inp_states]



            full_cubic_out, full_pre_cubic_out = net(pre_state, pos_inp_states,
                                                     cubic_pos_inp_states)
            full_cubic_out     = -full_cubic_out #minus
            full_pre_cubic_out = -full_pre_cubic_out
            full_cubic_outs.append(full_cubic_out)
            full_pre_cubic_outs.append(full_pre_cubic_out)

        cubic_term = sum([fco.sum() for fco in full_cubic_outs])

        #print("qt: " + str(quadratic_term == sum([qt.sum() for qt in full_quadr_outs])))

        # Get the final energy
        energy = sq_nrm + lin_term + cubic_term

        #Full version
        full_energies = [sq + qt.view(qt.shape[0],-1) + lt
                         for sq, qt, lt in zip(full_sq_terms,
                                               full_cubic_outs,
                                               full_lin_terms)]

        return energy, full_pre_cubic_outs, full_energies



class StructuredVectorFieldNetwork(BaseModel):
    """Like the VectorFieldNetwork but allows for conv layers
    """
    def __init__(self, args, device, model_name, writer, n_class=None):
        super().__init__(args, device, model_name, writer)
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
            self.quadratic_ne1ts.append(net)

    def forward(self, states, class_id=None, step=None):

        # Squared norm
        sq_terms = []
        for i, layer in enumerate(states):
            sq_term = 0.5 * (layer.view(layer.shape[0], -1) ** 2).sum()
            sq_terms.append(sq_term)
        sq_nrm = sum(sq_terms)

        # Linear terms
        lin_terms = []
        for i, (layer, bias) in enumerate(zip(states, self.biases)):
            lin_term = bias(self.state_actv(layer.view(layer.shape[0], -1)))
            lin_term = lin_term.sum()
            lin_terms.append(lin_term)
        lin_terms = - sum(lin_terms)

        # Quadratic terms
        quadr_outs = []
        outs = []
        for i, (pre_state, net) in enumerate(zip(states, self.quadratic_nets)):
            post_inp_idxs = self.args.arch_dict['mod_connect_dict'][i]
            pos_inp_states = [states[j] for j in post_inp_idxs]
            pos_inp_states = [self.state_actv(state)
                              for state in pos_inp_states]
            quadr_out, out = net(pre_state, pos_inp_states)
            quadr_outs.append(quadr_out)
            outs.append(out)

        quadratic_terms = - sum(sum(quadr_outs))

        # Get the final energy
        energy = sq_nrm + quadratic_terms + lin_terms

        return energy, outs



###### Function to help me debug
shapes = lambda x : [y.shape for y in x]
nancheck = lambda x : (x != x).any()
listout = lambda y : [x for x in y]





