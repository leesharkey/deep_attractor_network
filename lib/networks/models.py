import torch
from torch import nn
import lib.utils
from lib.networks.layers import *

class BaseModel(nn.Module):
    def __init__(self, args, device, session_name, model_name, unique_id,
                 writer, n_class=None):
        super().__init__()
        self.args = args
        self.device = device
        self.writer = writer
        self.session_name = session_name
        self.model_name = model_name
        self.unique_id = unique_id
        self.pad_mode = 'zeros'
        self.num_state_layers = len(self.args.state_sizes[1:])
        self.st_act = \
            lib.utils.get_activation_function(args, states_activation=True)

        # Initialize biases

        """Biases should be kept at exactly 0 all the time. this is because
        Rubin et al 2015:
        We have previously discussed one mechanism underlying our 
        model (Ahmadian et al., 2013). It is based on the fact that a 
        cortical neuron’s firing rate is well described by raising its input,
        as reflected in its depolarization from rest, to a power greater 
        than 1."""
        self.biases = nn.ModuleList([
            torch.nn.ModuleList([nn.Linear(torch.prod(torch.tensor(l[1:])), 1, bias=False)]*2)
                       for l in args.state_sizes])
        for bias1, bias2 in self.biases:
            torch.nn.init.ones_(bias1.weight) * 0.5
            torch.nn.init.ones_(bias2.weight) * 0.5
            # torch.nn.init.zeros_(bias1.weight)
            # torch.nn.init.zeros_(bias2.weight)
            # bias1.weight.data = -bias1.weight.data # LEE not that these are now negative, motivated by the fact that the mean firing rate of neurons is supposed to be driven by fluctuations and the mean input is subthreshold
            # bias2.weight.data = -bias2.weight.data

class SSNEBM(BaseModel):
    """Defines the SSN.
    """
    def __init__(self, args, device, session_name, model_name, unique_id,
                 writer, n_class=None):
        super().__init__(args, device, session_name, model_name, unique_id,
                 writer, n_class=None)
        # self.args = args
        # self.device = device
        # self.writer = writer
        # self.model_name = model_name
        # self.num_state_layers = len(self.args.state_sizes[1:])

        # Define the networks that output the quadratic terms
        self.quadratic_nets = nn.ModuleList([])
        for i in range(len(self.args.state_sizes)):
            net = SSNBlock(args, i)
            self.quadratic_nets.append(net)

        if self.args.ff_dynamics:
            self.forward = self.forward_ff_dynamics

            # # Define the ranges that should be masked by the 'gradient' of the
            # # states_activation
            # if self.args.states_activation == 'relu':
            #     self.mask_func = lambda state: torch.where((state > 0.),
            #                                           torch.ones_like(state),
            #                                           torch.zeros_like(state))

        else:
            self.forward = self.forward_bp_dynamics

    def forward_ff_dynamics(self, states, energy_calc=False):
        """This calculates the gradient of the energy function without
           actually using backprop.
           #TODO change these docstrings
           Should be much faster. The main difference with the BP dynamics
           is that it will exclude a term, since the gradient of E wrt s_i
           is some function of several {s_j}s that includes s_i.

           The equation it should calculate is
           $\frac{ds_i}{dt}=-s_i+\rho'(s_i)(f_\theta(s)+b_i)"""

        grads = []
        energy = torch.tensor(0)
        energies = []
        network_terms = []

        for layer_idx, (layer_states, net, layer_biases) in enumerate(zip(states,
                                                   self.quadratic_nets,
                                                   self.biases)):

            # Get the right input states
            post_inp_idxs = self.args.arch_dict['mod_connect_dict'][layer_idx]
            pos_inp_states = [states[j] for j in post_inp_idxs]
            _, full_pre_quadr_out = net(layer_states, pos_inp_states)
            # full_pre_quadr_out = 0.5 * full_pre_quadr_out

            layer_grads = []
            layer_energies = []
            layer_network_terms = []
            for ei_ind, state in enumerate(layer_states):
                #i.e. don't try to update inh states for layer 0 because there is none
                if layer_idx == 0 and ei_ind > 0:
                    break
                elif layer_idx == 0 and ei_ind == 0:
                    network_term = full_pre_quadr_out[ei_ind] + \
                                   layer_biases[ei_ind].weight.view(
                                       full_pre_quadr_out[ei_ind].shape[
                                       1:])  # LEE reinstated biases
                else: # only have bias terms on the visible terms
                    network_term = full_pre_quadr_out[ei_ind] #+ layer_biases[ei_ind].weight.view(full_pre_quadr_out[ei_ind].shape[1:])

                grad = -state + network_term

                layer_grads.append(grad)
                layer_network_terms.append(network_term)

                if energy_calc:
                    full_sq_term = 0.5 * (state.view(state.shape[0], -1) ** 2)
                    quad_bias_terms = \
                        torch.mul(network_term.view(state.shape[0], -1),
                                  state.view(state.shape[0], -1))
                    energy = full_sq_term - quad_bias_terms
                    layer_energies.append(energy)
            grads.append(layer_grads)
            energies.append(layer_energies)
            network_terms.append(layer_network_terms)



        if energy_calc:
            full_energies = energies
            energy = sum([sum([e.sum() for e in layer_energies]) for layer_energies in energies])

            print("Energy: %f" % energy.item())
        else:
            full_energies = [torch.zeros_like(state) for state in states]

        return energy, grads, network_terms, full_energies



    def forward_bp_dynamics(self, states, energy_calc=True):

        # Takes extra effort to calculate energy of each nrn individually
        # Squared norm
        sq_terms = []
        full_sq_terms = []
        for layer_idx, layer_states in enumerate(states):
            layer_full_sq_terms = []
            layer_sq_terms = []
            for ei_ind, state in enumerate(layer_states):
                if layer_idx == 0 and ei_ind > 0:
                    break
                full_sq_term = 0.5 * (state.view(state.shape[0], -1) ** 2)
                sq_term = full_sq_term.sum()
                layer_full_sq_terms.append(full_sq_term)
                layer_sq_terms.append(sq_term)
            full_sq_terms.append(layer_full_sq_terms)
            sq_terms.append(layer_sq_terms)
        sq_nrm = sum([sum(layer_sq) for layer_sq in sq_terms])
        #print("sq: " + str(sq_nrm==sum([st.sum() for st in full_sq_terms])))

        # Linear terms
        lin_terms = []
        full_lin_terms = []
        for layer_idx, (layer_states, layer_biases) in enumerate(zip(states, self.biases)):
            layer_lin_terms = []
            layer_full_lin_terms = []
            for ei_ind, (state, bias) in enumerate(zip(layer_states, layer_biases)):
                if layer_idx == 0 and ei_ind > 0:
                    break
                elif layer_idx == 0 and ei_ind == 0:
                    full_lin_term = torch.mul(bias.weight, state.view(state.shape[0], -1))
                    full_lin_term = -full_lin_term
                else: # only have bias terms on the visible terms
                    full_lin_term = torch.zeros_like(state.view(state.shape[0], -1))
                layer_full_lin_terms.append(full_lin_term)
                layer_lin_terms.append(full_lin_term.sum())

            full_lin_terms.append(layer_full_lin_terms)
            lin_terms.append(layer_full_lin_terms)
        lin_term = sum([sum(layer_lt).sum() for layer_lt in lin_terms])
        #lin_term = 0 #sum([sum(layer_lt).sum() for layer_lt in lin_terms])
        # LEE reinstated bias terms
        # print("lt: " + str(lin_term==sum([lt.sum() for lt in full_lin_terms])))

        # Quadratic terms
        full_quadr_outs = []
        full_pre_quadr_outs = []
        for i, (layer_states, net) in enumerate(
                zip(states, self.quadratic_nets)):
            # Get the right input states
            post_inp_idxs = self.args.arch_dict['mod_connect_dict'][i]
            pos_inp_states = [states[j] for j in post_inp_idxs]
            full_quadr_out, full_pre_quadr_out = net(layer_states, pos_inp_states)
            full_quadr_out     = [-fqo for fqo in full_quadr_out] #minus
            full_pre_quadr_out = [-fpqo for fpqo in full_pre_quadr_out]
            full_quadr_outs.append(full_quadr_out)
            full_pre_quadr_outs.append(full_pre_quadr_out)

        quadratic_term = sum([sum([fqo.sum() for fqo in layer_fqo])
                              for layer_fqo in full_quadr_outs])

        #print("qt: " + str(quadratic_term == sum([qt.sum() for qt in full_quadr_outs])))

        # Get the final energy
        energy = sq_nrm + quadratic_term + lin_term #LEE reinstated biases

        #Full version
        full_energies = [[sq + qt.view(qt.shape[0],-1) + lt #LEE removed biases
                         for sq, qt, lt in zip(layer_full_sq_terms,
                                               layer_full_quadr_outs,
                                               layer_full_lin_terms)]
                         for (layer_full_sq_terms,
                              layer_full_quadr_outs,
                              layer_full_lin_terms) in  zip(full_sq_terms,
                                                            full_quadr_outs,
                                                            full_lin_terms)]

        return energy, full_pre_quadr_outs, full_energies




















































































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

        return energy, grads, full_energies



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









###### Function to help me debug
shapes = lambda x : [y.shape for y in x]
nancheck = lambda x : (x != x).any()
listout = lambda y : [x for x in y]





