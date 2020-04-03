import torch
from torch import nn, optim
import lib.custom_components.custom_swish_activation as cust_actv
from lib.networks.layers import Reshape, Interpolate


class InitializerNetwork(torch.nn.Module):
    def __init__(self, args, writer, device):
        super(InitializerNetwork, self).__init__()
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

        hsig = torch.nn.Hardtanh(min_val=0.0)
        self.lh_sig = lambda x: hsig(x) + 0.01 * x

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
                                      Reshape(self.args.batch_size, img_size),
                                      nn.Linear(in_features=img_size,
                                                out_features=img_size),
                                      cust_actv.Swish_module(),
                                      nn.BatchNorm1d(img_size))


        # Define the rest of the encoders
        for i in range(1, len(self.args.state_sizes)):
            # encs should take as input the image size and output the statesize for that statelayer
            if len(self.args.state_sizes[i]) == 4: #TODO this is a hack for the new type of networks with dim4 in sl0
                self.encs.append(
                                nn.Sequential(
                                          nn.BatchNorm2d(self.num_ch),
                                          cust_actv.Swish_module(),
                                          nn.Conv2d(in_channels=self.num_ch,
                                                    out_channels=self.num_ch,
                                                    kernel_size=3,
                                                    padding=1,
                                                    bias=True),
                                          Interpolate(size=self.args.state_sizes[i][2:],
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
                        Reshape(self.args.batch_size, prev_size),
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
            outs.append(self.lh_sig(out_i))

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




