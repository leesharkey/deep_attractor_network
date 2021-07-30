import torch
from torch import nn, optim
import lib.custom_components.custom_swish_activation as cust_actv
import lib.networks.layers as layers
import lib.utils as utils


class InitializerNetwork(torch.nn.Module):
    def __init__(self, args, writer, device, layer_norm=False,
                 weight_norm=False):
        super(InitializerNetwork, self).__init__()
        self.args = args
        self.writer = writer
        self.device = device
        self.input_size = self.args.state_sizes[0]
        self.in_channels = self.args.state_sizes[0][1]
        self.output_sizes = args.state_sizes[1:]
        self.criterion = nn.MSELoss()
        self.criteria = []
        self.num_ch = self.args.arch_dict['num_ch_initter']
        #self.encs = nn.ModuleList([])
        self.sides = nn.ModuleList([])
        self.growth_rate = self.num_ch #was 16
        self.num_layers_dense = len(self.output_sizes) + 2

        # Define the activation functions for the network and the state vars
        sigmoid = torch.nn.Sigmoid()
        self.swish = lambda x: x * sigmoid(x)
        rects = ['relu', 'leaky_relu', 'swish']
        if self.args.states_activation == 'hardsig':
            hsig = torch.nn.Hardtanh(min_val=0.0)
            self.initter_states_act = lambda x: hsig(x) + 0.01 * x
        if self.args.states_activation == 'hardtanh':
            hsig = torch.nn.Hardtanh()
            self.initter_states_act = lambda x: hsig(x) + 0.01 * x
        elif self.args.states_activation in rects:
            self.initter_states_act = torch.nn.LeakyReLU()

        # Define the encoder
        self.enc = layers.DenseCCTMiddle(args,
                                         in_channels=self.in_channels,
                                         growth_rate=self.growth_rate,
                                         num_layers=self.num_layers_dense,
                                         kernel_size=7,
                                         layer_norm=layer_norm,
                                         weight_norm=weight_norm)
        ch_out = self.in_channels + (self.num_layers_dense * self.growth_rate)


        # Define the side networks
        for sz in self.output_sizes:
            if sz[3] < self.input_size[3]:
                k1 = 7
                p1 = 0
                step_down1 = True
                outshape1 = utils.conv_output_shape(self.input_size[3],
                                                   kernel_size=k1,
                                                   stride=1,
                                                   padding=p1)
                outshape2 = utils.conv_output_shape(outshape1,
                                                   kernel_size=k1,
                                                   stride=1,
                                                   padding=p1)
                if outshape2 < sz[3]:
                    k2 = 7
                    p2 = 0
                    step_down2 = True #TODO get rid of stepdown2 if initter changes turn out okay
                else:
                    k2 = 7
                    p2 = 3
                    step_down2 = False
            else:
                k1 = 7
                p1 = 3
                k2 = 7
                p2 = 3
                step_down1 = False
                step_down2 = False
            self.sides.append(
                nn.Sequential(layers.CCTBlock(args,
                                              in_channels=ch_out,
                                               out_channels=max(self.num_ch,
                                                                sz[1]*2),
                                               kernel_size=k1,
                                               padding=p1,
                                               only_conv=step_down1,
                                               layer_norm=layer_norm,
                                               weight_norm=weight_norm),
                              layers.Interpolate(size=sz[2:],
                                                 mode='nearest'),
                              layers.CCTBlock(args,
                                              in_channels=max(self.num_ch,
                                                              sz[1] * 2),
                                              out_channels=max(self.num_ch,
                                                               sz[1]),
                                              kernel_size=7,
                                              padding=3,
                                              layer_norm=layer_norm,
                                              weight_norm=weight_norm),
                              nn.Conv2d(in_channels=max(self.num_ch, sz[1]),
                                        out_channels=sz[1],
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)).to(self.device))



        # Define the optimizer
        # self.optimizer = optim.Adam(self.parameters(),
        #                             lr=self.args.initter_network_lr,
        #                             betas=(0.9, 0.999))
        self.optimizer = optim.SGD(self.parameters(),
                                   nesterov=True,
                                   momentum=0.85,
                                   lr=self.args.initter_network_lr)

    def forward(self, inp, x_id):
        print("Initializing with FF net")
        out = self.enc(inp)
        outs = []
        for side in self.sides:
            out_i = side(out)

            # (Leakily) Clamp the outputs to (approximately) [0,1]
            outs.append(self.initter_states_act(out_i))

        return outs

    def update_weights(self, outs, targets, step):
        self.optimizer.zero_grad()
        self.criteria = [self.criterion(o, t) for o,t in zip(outs,targets)]
        loss = torch.sum(torch.stack(self.criteria))
        loss.backward()
        self.optimizer.step()
        print("\nInitializer loss: " + '%.4g' % loss.item())
        # if step % self.args.scalar_logging_interval == 0:
        self.writer.add_scalar('Initializer/total_loss', loss.item(),
                               step)
        for i, l in enumerate(self.criteria, start=1):
            name = 'Initializer/loss_layer_%i' % i
            self.writer.add_scalar(name, l.item(),
                                   step)
        return loss

