# vim: foldmethod=marker
# Copied directly from PYSGMCMC library source code. For documentation, see:
# https://pysgmcmc.readthedocs.io/en/pytorch/_modules/pysgmcmc/optimizers/sghmc.html
import torch
from torch.optim import Optimizer
import numpy as np
import scipy.stats as sps


class SGHMC(Optimizer):
    """ Stochastic Gradient Hamiltonian Monte-Carlo Sampler that uses a burn-in
        procedure to adapt its own hyperparameters during the initial stages
        of sampling.

        See [1] for more details on this burn-in procedure.\n
        See [2] for more details on Stochastic Gradient Hamiltonian Monte-Carlo.

        [1] J. T. Springenberg, A. Klein, S. Falkner, F. Hutter
            In Advances in Neural Information Processing Systems 29 (2016).\n
            `Bayesian Optimization with Robust Bayesian Neural Networks. <http://aad.informatik.uni-freiburg.de/papers/16-NIPS-BOHamiANN.pdf>`_
        [2] T. Chen, E. B. Fox, C. Guestrin
            In Proceedings of Machine Learning Research 32 (2014).\n
            `Stochastic Gradient Hamiltonian Monte Carlo <https://arxiv.org/pdf/1402.4102.pdf>`_
    """
    name = "SGHMC"

    def __init__(self,
                 params,
                 lr: float=1e-2,
                 num_burn_in_steps: int=0, #used to be 3000 #50
                 noise: float=0.,
                 mdecay: float=0.05,
                 scale_grad: float=1.,
                 min_sq_sigma: float=1e-5,
                 args=None,
                 state_layer=None) -> None:
        """ Set up a SGHMC Optimizer.

        Parameters
        ----------
        params : iterable
            Parameters serving as optimization variable.
        lr: float, optional
            Base learning rate for this optimizer.
            Must be tuned to the specific function being minimized.
            Default: `1e-2`.
        num_burn_in_steps: int, optional
            Number of burn-in steps to perform. In each burn-in step, this
            sampler will adapt its own internal parameters to decrease its error.
            Set to `0` to turn scale adaption off.
            Default: `3000`.
        noise: float, optional
            (Constant) per-parameter noise level.
            Default: `0.`.
        mdecay:float, optional
            (Constant) momentum decay per time-step.
            Default: `0.05`.
        scale_grad: float, optional
            Value that is used to scale the magnitude of the noise used
            during sampling. In a typical batches-of-data setting this usually
            corresponds to the number of examples in the entire dataset.
            Default: `1.0`.

        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if num_burn_in_steps < 0:
            raise ValueError("Invalid num_burn_in_steps: {}".format(num_burn_in_steps))

        defaults = dict(
            lr=lr, scale_grad=float(scale_grad),
            num_burn_in_steps=num_burn_in_steps,
            mdecay=mdecay,
            noise=noise
        )
        super().__init__(params, defaults)

        self.args = args
        self.batch_size = 128  # TODO soft-code
        self.min_sq_sigma = min_sq_sigma
        if len(self.args.mom_clip_vals) == 1:
            self.momenta_clip_norm_vals = self.args.mom_clip_vals * len(self.args.state_sizes)
        self.state_layer_idx = state_layer
        self.printing_grad_mom_info = False

        self.bump_scaler = 1e-3
        self.inv_M_sds = {32: 0.55,
                          16: 0.3,
                          8:  0.1}
        # self.inv_M_sds = {32: 0.6,
        #                   16: 0.45,
        #                   8:  0.3}
        # self.inv_M_sds_chs = {32: 0.01,
        #                       16: 0.01,
        #                        8:  0.01}


        #self.inv_M_sds_lat = 0.4
        self.inv_M_weights = []
        self.inv_M_conv = None

        if self.args.non_diag_inv_mass:  #impose circular correlations
            # between channels

            for group in self.param_groups:
                for parameter in group["params"]:

                    # Makes a gaussian kernel that defines a certain inverse mass
                    # matrix on the dynamics
                    num_ch = parameter.shape[1]
                    if num_ch < 6:
                        break
                    channels = np.arange(num_ch)
                    sd = self.inv_M_sds[num_ch]
                    mean_0 = num_ch // 2
                    gaussian = sps.norm(mean_0, sd)
                    densities = gaussian.pdf(channels)

                    # import matplotlib.pyplot as plot
                    # plot.plot(np.arange(len(densities)), densities)
                    # plot.savefig("circulargaussian.png")
                    #plot.clf()

                    #densities = densities / np.max(densities)  # Normalize
                    densities = np.roll(densities, num_ch//2)  # centre on ch 0
                    densities[densities < 1e-15] = 0.0  # zero-out the smallest


                    for ch in channels:
                        densities_rolled = np.roll(densities, ch)
                        weights = torch.tensor(densities_rolled)
                        self.inv_M_weights.append(weights)
                    self.inv_M_weights = torch.stack(self.inv_M_weights)
                    self.inv_M_weights = self.inv_M_weights.to(parameter.device)



                    self.inv_M_weights = \
                        torch.nn.Parameter(
                            self.inv_M_weights[:,:, None,None].float())
                    self.inv_M_conv = torch.nn.Conv2d(in_channels=num_ch,
                                                      out_channels=num_ch,
                                                      kernel_size=1,
                                                      padding=0,
                                                      bias=False)
                    self.inv_M_conv.weight = self.inv_M_weights


    def step(self, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:

                if parameter.grad is None:
                    continue

                state = self.state[parameter]

                #  State initialization {{{ #

                if len(state) == 0:
                    state["iteration"] = 0
                    state["tau"] = torch.ones_like(parameter)
                    state["g"] = torch.ones_like(parameter)
                    state["v_hat"] = torch.ones_like(parameter)
                    state["momentum"] = torch.zeros_like(parameter).normal_(mean=0, std=1e-5)
                    #torch.zeros_like(parameter)#.normal_(mean=0, std=1.)
                #  }}} State initialization #

                state["iteration"] += 1

                #  Readability {{{ #
                mdecay, noise, lr = group["mdecay"], group["noise"], group["lr"]
                scale_grad = torch.tensor(group["scale_grad"])
                tau, g, v_hat = state["tau"], state["g"], state["v_hat"]
                momentum = state["momentum"]
                gradient = parameter.grad.data
                #  }}} Readability #

                r_t = 1. / (tau + 1.)
                minv_t = 1. / torch.sqrt(v_hat)

                #  Burn-in updates {{{ #
                if state["iteration"] <= group["num_burn_in_steps"]:
                    # Update state
                    tau.add_(1. - tau * (g * g / v_hat))
                    g.add_(-g * r_t + r_t * gradient)
                    v_hat.add_(-v_hat * r_t + r_t * (gradient ** 2))

                #  }}} Burn-in updates #

                lr_scaled = lr / torch.sqrt(scale_grad)

                #  Draw random sample {{{ #

                ns_term1 = 2. * (lr_scaled ** 2) * mdecay * minv_t
                ns_term2 = 2. * (lr_scaled ** 3) * (minv_t ** 2) * noise
                ns_term3  = (lr_scaled ** 4)
                noise_scale = (ns_term1 - ns_term2 - ns_term3) #LEE was minus

                #noise_scale = torch.ones_like(minv_t) * 0.005
                print("%i neg: %s;  NSterms: %f  %f  %f" % (self.state_layer_idx, str(noise_scale.mean().item() < 0.0), ns_term1.mean(), ns_term2.mean(), ns_term3))
                # print("%i NS2: %f" % (self.state_layer_idx, ))
                # print("%i NS3: %f " % (self.state_layer_idx))
                # print("%i Noise scale is negative: %s\n\n" % (self.state_layer_idx,
                #                                               ))

                sigma = torch.sqrt(torch.clamp(noise_scale,
                                               min=self.min_sq_sigma))

                sample_t = torch.normal(mean=0., std=sigma)
                #  }}} Draw random sample #

                # Note: for the mostpart, minv_t should basically be identical
                # (currently during viz mean is identical for all channels
                # up until the 3rd decimal point)
                # for all variables, so it shouldn't differentially affect them
                # on different training/visualization runs.
                #minv_t.mean(dim=[0,2,3])
                #minv_t.var(dim=[0,2,3])


                #  SGHMC Update {{{ #
                if self.args.non_diag_inv_mass and self.inv_M_conv is not None:
                    friction = mdecay * self.inv_M_conv(momentum)
                else:
                    friction = mdecay * momentum

                mom_summand = \
                    - (lr ** 2) * minv_t * gradient - friction + sample_t
                momentum_t = momentum.add_(mom_summand)


                if self.args.non_diag_inv_mass and self.inv_M_conv is not None:
                    momentum_t = self.inv_M_conv(momentum_t)

                if self.args.mom_clip:
                    momentum_t = tensor_norm_clip(momentum_t,
                        self.momenta_clip_norm_vals[self.state_layer_idx])




                if self.printing_grad_mom_info:
                    print("\nMomentum norm %f" % torch.norm(momentum_t, 2))
                    print("Mom summand mean %f ; var %f" % (mom_summand.mean(), mom_summand.var()))
                    print("Noise mean %f ; var %f" % (sample_t.mean(), sample_t.var()))

                    print("Momentum mean %f ; var %f" % (momentum.mean(), momentum.var()))
                    print("Gradient mean %f ; var %f" % (gradient.mean(), gradient.var()))

                parameter.data.add_(momentum_t)

                # Add a very small amount of noise to all variables (ensures no
                # uniform colours like blackness or whiteness)
                baseline_noise_level = torch.ones_like(parameter) * 1e-7
                parameter.data.add_(torch.normal(0., baseline_noise_level))
                #  }}} SGHMC Update #

        return loss

def tensor_norm_clip(tensor, clip_val):
    norm = torch.norm(tensor, 2)
    if norm > clip_val:
        tensor = tensor.clone().detach()
        return tensor * clip_val / norm
    else:
        return tensor


def alteredstep(self, closure=None):
    loss = None

    if closure is not None:
        loss = closure()

    for group in self.param_groups:
        for parameter in group["params"]:

            if parameter.grad is None:
                continue

            state = self.state[parameter]

            # num_dims = torch.prod(torch.tensor(parameter.shape)) #lee

            #  State initialization {{{ #

            if len(state) == 0:
                state["iteration"] = 0
                state["tau"] = torch.ones_like(parameter)
                state["g"] = torch.ones_like(parameter)
                state["v_hat"] = torch.ones_like(parameter)
                state["momentum"] = \
                    torch.zeros_like(parameter)#.normal_(mean=0, std=1.)
            #  }}} State initialization #

            state["iteration"] += 1

            #  Readability {{{ #
            mdecay, noise, lr = group["mdecay"], group["noise"], group["lr"]
            scale_grad = torch.tensor(group["scale_grad"])

            tau, g, v_hat = state["tau"], state["g"], state["v_hat"]
            momentum = state["momentum"]


            gradient = parameter.grad.data
            #  }}} Readability #

            r_t = 1. / (tau + 1.)
            minv_t = 1. / torch.sqrt(v_hat)

            #  Burn-in updates {{{ #
            if state["iteration"] <= group["num_burn_in_steps"]:
                # Update state
                tau.add_(1. - tau * (g * g / v_hat))
                g.add_(-g * r_t + r_t * gradient)
                v_hat.add_(-v_hat * r_t + r_t * (gradient ** 2))

            #  }}} Burn-in updates #

            lr_scaled = lr / torch.sqrt(scale_grad)


            # average the variances over batches #lee

            #normal one
            # minv_t = minv_t.mean(dim=0) # average the variances over batches #lee
            # minv_t = torch.stack(self.batch_size * [minv_t])
            # end normal one

            minv_t = torch.ones_like(minv_t) * 16. #minv_t.mean()

            # mv = [minv_t * (10/i) for i in range(1,self.batch_size+1)]
            # minv_t = torch.stack(mv)

            #  Draw random sample {{{ #

            # noise_scale = (
            #     2. * (lr_scaled ** 2) * mdecay * minv_t - #LEE was minus
            #     2. * (lr_scaled ** 3) * (minv_t ** 2) * noise -
            #     (lr_scaled ** 4)
            # )
            noise_scale = torch.ones_like(minv_t) * 0.005

            sigma = torch.sqrt(torch.clamp(noise_scale,
                                           min=self.min_sq_sigma))

            sample_t = torch.normal(mean=0., std=sigma)
            #  }}} Draw random sample #

            # Note: for the mostpart, minv_t should basically be identical
            # (currently during viz mean is identical for all channels
            # up until the 3rd decimal point)
            # for all variables, so it shouldn't differentially affect them
            # on different training/visualization runs.
            #minv_t.mean(dim=[0,2,3])
            #minv_t.var(dim=[0,2,3])


            if self.args.maxminstate_to_zeromom:
                if self.args.states_activation == 'hardsig':
                    mask = (parameter > 0.) #& (parameter < 1.)
                    momentum = torch.where(mask,
                                             momentum,
                                             momentum * self.bump_scaler)
                elif self.args.states_activation in \
                        ['relu', 'leaky_relu', 'swish']:
                    momentum = torch.where(
                        parameter > 0.0,
                        momentum,
                        momentum * self.bump_scaler)
                elif self.args.states_activation == 'hardtanh': #not like you'd ever use this, but including it for completeness
                    momentum = torch.where(
                        parameter > -1.0,
                        momentum,
                        momentum * self.bump_scaler)


            #  SGHMC Update {{{ #
            if self.args.non_diag_inv_mass and self.inv_M_conv is not None:
                friction = mdecay * self.inv_M_conv(momentum)
            else:
                friction = mdecay * momentum

            mom_summand = \
                - (lr ** 2) * minv_t * gradient - friction + sample_t
            momentum_t = momentum.add_(mom_summand)


            if self.args.non_diag_inv_mass and self.inv_M_conv is not None:
                momentum_t = self.inv_M_conv(momentum_t)

            if self.args.mom_clip:
                momentum_t = tensor_norm_clip(momentum_t,
                    self.momenta_clip_norm_vals[self.state_layer_idx])




            if self.printing_grad_mom_info:
                print("\nMomentum norm %f" % torch.norm(momentum_t, 2))
                print("Mom summand mean %f ; var %f" % (mom_summand.mean(), mom_summand.var()))
                print("Noise mean %f ; var %f" % (sample_t.mean(), sample_t.var()))

                print("Momentum mean %f ; var %f" % (momentum.mean(), momentum.var()))
                print("Gradient mean %f ; var %f" % (gradient.mean(), gradient.var()))

            parameter.data.add_(momentum_t)

            # Add a very small amount of noise to all variables (ensures no
            # uniform colours like blackness or whiteness)
            baseline_noise_level = torch.ones_like(parameter) * 1e-6
            parameter.data.add_(torch.normal(0., baseline_noise_level))
            #  }}} SGHMC Update #

    return loss