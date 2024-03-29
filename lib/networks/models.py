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


class NRFLV(BaseModel):
    """Defines the NRFLV

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

        # Full version
        full_energies = [sq + qt.view(qt.shape[0],-1) + lt
                         for sq, qt, lt in zip(full_sq_terms,
                                               full_quadr_outs,
                                               full_lin_terms)]

        return energy, full_pre_quadr_outs, full_energies