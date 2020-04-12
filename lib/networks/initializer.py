import torch
from torch import nn, optim
import lib.custom_components.custom_swish_activation as cust_actv
import lib.networks.layers as layers
import lib.utils as utils


class InitializerNetwork_first_version(torch.nn.Module):
    def __init__(self, args, writer, device):
        super(InitializerNetwork_first_version, self).__init__()
        self.args = args
        self.writer = writer
        self.device = device
        self.input_size = self.args.state_sizes[0]
        self.output_sizes = args.state_sizes[1:]
        self.criterion = nn.MSELoss()
        self.criteria = []
        self.num_ch = self.args.arch_dict['num_ch_initter']
        self.encs = nn.ModuleList([])
        self.sides = nn.ModuleList([])
        self.in_channels = self.args.state_sizes[0][1]

        sigmoid = torch.nn.Sigmoid()
        self.swish = lambda x: x * sigmoid(x)

        if self.args.states_activation == 'relu':
            self.initter_states_act = torch.nn.LeakyReLU()
        else:
            hsig = torch.nn.Hardtanh(min_val=0.0)
            self.initter_states_act = lambda x: hsig(x) + 0.01 * x


        # Define the base encoder
        if len(self.args.state_sizes[1])==4:
            self.enc_base = nn.Sequential(nn.BatchNorm2d(self.in_channels),
                                          cust_actv.Swish_module(),
                                          nn.Conv2d(in_channels=self.in_channels,
                                                    out_channels=self.num_ch,
                                                    kernel_size=3,
                                                    padding=1,
                                                    bias=True),
                                          nn.BatchNorm2d(self.num_ch),
                                          cust_actv.Swish_module(),
                                          nn.Conv2d(
                                              in_channels=self.num_ch,
                                              out_channels=self.num_ch,
                                              kernel_size=3,
                                              padding=1,
                                              bias=True)
                                          )
        elif len(self.args.state_sizes[1])==2:
            img_size = int(torch.prod(torch.tensor(self.args.state_sizes[0][1:])))
            self.enc_base = nn.Sequential(
                                      layers.Reshape(self.args.batch_size, img_size),
                                      nn.Linear(in_features=img_size,
                                                out_features=img_size),
                                      cust_actv.Swish_module(),
                                      nn.BatchNorm1d(img_size))


        # Define the rest of the encoders
        for i in range(1, len(self.args.state_sizes)):
            # encs should take as input the image size and output the statesize for that statelayer
            if len(self.args.state_sizes[i]) == 4:
                self.encs.append(
                                nn.Sequential(
                                          nn.BatchNorm2d(self.num_ch),
                                          cust_actv.Swish_module(),
                                          nn.Conv2d(in_channels=self.num_ch,
                                                    out_channels=self.num_ch,
                                                    kernel_size=3,
                                                    padding=1,
                                                    bias=True),
                                          layers.Interpolate(size=self.args.state_sizes[i][2:],
                                                      mode='bilinear')).to(self.device))
            elif len(self.args.state_sizes[i]) == 2:
                prev_sl_shape = self.args.state_sizes[i - 1]
                if (self.args.network_type == 'VectorField' and \
                    len(prev_sl_shape) == 4) or (i==1):
                    prev_size = self.args.state_sizes[i - 1][1:]
                    prev_size = int(torch.prod(torch.tensor(prev_size)))
                elif len(prev_sl_shape) == 4:
                    prev_size = self.args.state_sizes[i - 1][2:]
                    prev_size.append(self.num_ch)
                    prev_size = int(torch.prod(torch.tensor(prev_size)))
                elif len(prev_sl_shape) == 2:
                    prev_size = self.args.state_sizes[i - 1][1]

                new_size = (self.args.batch_size, prev_size)
                # new_size.extend(self.args.state_sizes[i][1])
                self.encs.append(
                    nn.Sequential(
                        layers.Reshape(self.args.batch_size, prev_size),
                        nn.Linear(in_features=prev_size,
                                  out_features=self.args.state_sizes[i][1]),
                        cust_actv.Swish_module(),
                        nn.BatchNorm1d(self.args.state_sizes[i][1])))

            # Define the side branches that split off from the encoder and
            # output the state layer initializations at each statelayer

            # Sides should output the statesize for that statelayer and input is same size as output.
            # so long as the input to the side is the same as the size of the
            # base conv (i.e. the statesize), I can just use the same settings
            # as for the base+energy convs
            if len(self.args.state_sizes[i]) == 4:
                self.sides.append(
                                nn.Sequential(nn.BatchNorm2d(self.num_ch),
                                          cust_actv.Swish_module(),
                                          nn.Conv2d(in_channels=self.num_ch,
                                                     out_channels=self.num_ch,
                                                     kernel_size=3,
                                                     padding=1, bias=True),
                                          nn.BatchNorm2d(self.num_ch),
                                          cust_actv.Swish_module(),
                                          nn.Conv2d(in_channels=self.num_ch,
                                                     out_channels=self.num_ch,
                                                     kernel_size=3,
                                                     padding=1, bias=True),
                                          nn.Conv2d(in_channels=self.num_ch,
                                                     out_channels=self.args.state_sizes[i][1],
                                                     kernel_size=3,
                                                     padding=1, bias=True)).to(self.device))#adjust kernel size so that output is b,9,16,16
            elif len(self.args.state_sizes[i]) == 2:
                self.sides.append(
                    nn.Sequential(
                        nn.Linear(in_features=self.args.state_sizes[i][1],
                                  out_features=self.args.state_sizes[i][1]),
                        cust_actv.Swish_module(),
                        nn.BatchNorm1d(self.args.state_sizes[i][1]),
                        nn.Linear(in_features=self.args.state_sizes[i][1],
                                  out_features=self.args.state_sizes[i][1]),
                        cust_actv.Swish_module()
                    )
                )

        self.optimizer = optim.SGD(self.parameters(),
                                   nesterov=True,
                                   momentum=0.6,
                                   lr=self.args.initter_network_lr)

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


class InitializerNetwork(torch.nn.Module):
    def __init__(self, args, writer, device):
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
        self.growth_rate = 16
        self.num_layers_dense = len(self.output_sizes)

        # Define the activation functions for the network and the state vars
        sigmoid = torch.nn.Sigmoid()
        self.swish = lambda x: x * sigmoid(x)
        rects = ['relu', 'leaky_relu', 'swish']
        if self.args.states_activation == 'hardsig':
            hsig = torch.nn.Hardtanh(min_val=0.0)
            self.initter_states_act = lambda x: hsig(x) + 0.01 * x
        elif self.args.states_activation in rects:
            self.initter_states_act = torch.nn.LeakyReLU()

        # Define the encoder
        self.enc = layers.DenseCCTMiddle(args, #TODO batchnorm option
                                         in_channels=self.in_channels,
                                         growth_rate=self.growth_rate,
                                         num_layers=self.num_layers_dense,
                                         kernel_size=7)
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
                                               only_conv=step_down1),
                              layers.Interpolate(size=sz[2:],
                                                 mode='nearest'),
                              layers.CCTBlock(args,
                                              in_channels=max(self.num_ch,
                                                              sz[1] * 2),
                                              out_channels=max(self.num_ch,
                                                               sz[1]),
                                              kernel_size=7,
                                              padding=3),
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

